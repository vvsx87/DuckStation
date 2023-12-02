// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_sw_backend.h"
#include "system.h"

#include "util/gpu_device.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/intrin.h"
#include "common/log.h"

#include <algorithm>

Log_SetChannel(GPUSWBackend);

static constexpr u16*& m_vram = GPU::m_vram_ptr;

ALWAYS_INLINE_RELEASE static u16 GetPixel(const u32 x, const u32 y)
{
  return m_vram[VRAM_WIDTH * y + x];
}
ALWAYS_INLINE_RELEASE static void SetPixel(const u32 x, const u32 y, const u16 value)
{
  m_vram[VRAM_WIDTH * y + x] = value;
}

GPUSWBackend::GPUSWBackend() : GPUBackend()
{
}

GPUSWBackend::~GPUSWBackend() = default;

bool GPUSWBackend::Initialize()
{
  if (!GPUBackend::Initialize())
    return false;

  std::memset(m_vram, 0, VRAM_SIZE);
  SetDisplayTextureFormat();
  return true;
}

void GPUSWBackend::Shutdown()
{
  GPUBackend::Shutdown();
}

bool GPUSWBackend::DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display)
{
  // TODO: FIXME
  // ignore the host texture for software mode, since we want to save vram here
  return GPUBackend::DoState(sw, nullptr, update_display);
}

void GPUSWBackend::Reset(bool clear_vram)
{
  m_drawing_area = {};

  if (clear_vram)
    std::memset(m_vram, 0, VRAM_SIZE);
}

void GPUSWBackend::ReadVRAM(u32 x, u32 y, u32 width, u32 height)
{
}

void GPUSWBackend::DrawPolygon(const GPUBackendDrawPolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const DrawTriangleFunction DrawFunction = GetDrawTriangleFunction(
    rc.shading_enable, rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable, dithering_enable);

  (this->*DrawFunction)(cmd, &cmd->vertices[0], &cmd->vertices[1], &cmd->vertices[2]);
  if (rc.quad_polygon)
    (this->*DrawFunction)(cmd, &cmd->vertices[2], &cmd->vertices[1], &cmd->vertices[3]);
}

void GPUSWBackend::DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const DrawTriangleFunction DrawFunction = GetDrawTriangleFunction(
    rc.shading_enable, rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable, dithering_enable);

  // Need to cut out the irrelevant bits.
  // TODO: In _theory_ we could use the fixed-point parts here.
  GPUBackendDrawPolygonCommand::Vertex vertices[4];
  for (u32 i = 0; i < cmd->num_vertices; i++)
  {
    const GPUBackendDrawPrecisePolygonCommand::Vertex& src = cmd->vertices[i];
    GPUBackendDrawPolygonCommand::Vertex& dst = vertices[i];

    dst.x = src.native_x;
    dst.y = src.native_y;
    dst.color = src.color;
    dst.texcoord = src.texcoord;
  }

  (this->*DrawFunction)(cmd, &vertices[0], &vertices[1], &vertices[2]);
  if (rc.quad_polygon)
    (this->*DrawFunction)(cmd, &vertices[2], &vertices[1], &vertices[3]);
}

void GPUSWBackend::DrawSprite(const GPUBackendDrawSpriteCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};

  const DrawSpriteFunction DrawFunction =
    GetDrawSpriteFunction(rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable);

  (this->*DrawFunction)(cmd);
}

void GPUSWBackend::DrawLine(const GPUBackendDrawLineCommand* cmd)
{
  const DrawLineFunction DrawFunction =
    GetDrawLineFunction(cmd->rc.shading_enable, cmd->rc.transparency_enable, cmd->IsDitheringEnabled());

  for (u16 i = 1; i < cmd->num_vertices; i++)
    (this->*DrawFunction)(cmd, &cmd->vertices[i - 1], &cmd->vertices[i]);
}

constexpr GPUSWBackend::DitherLUT GPUSWBackend::ComputeDitherLUT()
{
  DitherLUT lut = {};
  for (u32 i = 0; i < DITHER_MATRIX_SIZE; i++)
  {
    for (u32 j = 0; j < DITHER_MATRIX_SIZE; j++)
    {
      for (u32 value = 0; value < DITHER_LUT_SIZE; value++)
      {
        const s32 dithered_value = (static_cast<s32>(value) + DITHER_MATRIX[i][j]) >> 3;
        lut[i][j][value] = static_cast<u8>((dithered_value < 0) ? 0 : ((dithered_value > 31) ? 31 : dithered_value));
      }
    }
  }
  return lut;
}

static constexpr GPUSWBackend::DitherLUT s_dither_lut = GPUSWBackend::ComputeDitherLUT();

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable, bool dithering_enable>
void ALWAYS_INLINE_RELEASE GPUSWBackend::ShadePixel(const GPUBackendDrawCommand* cmd, u32 x, u32 y, u8 color_r,
                                                    u8 color_g, u8 color_b, u8 texcoord_x, u8 texcoord_y)
{
  VRAMPixel color;
  if constexpr (texture_enable)
  {
    // Apply texture window
    texcoord_x = (texcoord_x & cmd->window.and_x) | cmd->window.or_x;
    texcoord_y = (texcoord_y & cmd->window.and_y) | cmd->window.or_y;

    VRAMPixel texture_color;
    switch (cmd->draw_mode.texture_mode)
    {
      case GPUTextureMode::Palette4Bit:
      {
        const u16 palette_value =
          GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 4)) % VRAM_WIDTH,
                   (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
        const u16 palette_index = (palette_value >> ((texcoord_x % 4) * 4)) & 0x0Fu;

        texture_color.bits =
          GetPixel((cmd->palette.GetXBase() + ZeroExtend32(palette_index)) % VRAM_WIDTH, cmd->palette.GetYBase());
      }
      break;

      case GPUTextureMode::Palette8Bit:
      {
        const u16 palette_value =
          GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 2)) % VRAM_WIDTH,
                   (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
        const u16 palette_index = (palette_value >> ((texcoord_x % 2) * 8)) & 0xFFu;
        texture_color.bits =
          GetPixel((cmd->palette.GetXBase() + ZeroExtend32(palette_index)) % VRAM_WIDTH, cmd->palette.GetYBase());
      }
      break;

      default:
      {
        texture_color.bits = GetPixel((cmd->draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x)) % VRAM_WIDTH,
                                      (cmd->draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT);
      }
      break;
    }

    if (texture_color.bits == 0)
      return;

    if constexpr (raw_texture_enable)
    {
      color.bits = texture_color.bits;
    }
    else
    {
      const u32 dither_y = (dithering_enable) ? (y & 3u) : 2u;
      const u32 dither_x = (dithering_enable) ? (x & 3u) : 3u;

      color.bits = (ZeroExtend16(s_dither_lut[dither_y][dither_x][(u16(texture_color.r) * u16(color_r)) >> 4]) << 0) |
                   (ZeroExtend16(s_dither_lut[dither_y][dither_x][(u16(texture_color.g) * u16(color_g)) >> 4]) << 5) |
                   (ZeroExtend16(s_dither_lut[dither_y][dither_x][(u16(texture_color.b) * u16(color_b)) >> 4]) << 10) |
                   (texture_color.bits & 0x8000u);
    }
  }
  else
  {
    const u32 dither_y = (dithering_enable) ? (y & 3u) : 2u;
    const u32 dither_x = (dithering_enable) ? (x & 3u) : 3u;

    // Non-textured transparent polygons don't set bit 15, but are treated as transparent.
    color.bits = (ZeroExtend16(s_dither_lut[dither_y][dither_x][color_r]) << 0) |
                 (ZeroExtend16(s_dither_lut[dither_y][dither_x][color_g]) << 5) |
                 (ZeroExtend16(s_dither_lut[dither_y][dither_x][color_b]) << 10) | (transparency_enable ? 0x8000u : 0);
  }

  const VRAMPixel bg_color{GetPixel(static_cast<u32>(x), static_cast<u32>(y))};
  if constexpr (transparency_enable)
  {
    if (color.bits & 0x8000u || !texture_enable)
    {
      // Based on blargg's efficient 15bpp pixel math.
      u32 bg_bits = ZeroExtend32(bg_color.bits);
      u32 fg_bits = ZeroExtend32(color.bits);
      switch (cmd->draw_mode.transparency_mode)
      {
        case GPUTransparencyMode::HalfBackgroundPlusHalfForeground:
        {
          bg_bits |= 0x8000u;
          color.bits = Truncate16(((fg_bits + bg_bits) - ((fg_bits ^ bg_bits) & 0x0421u)) >> 1);
        }
        break;

        case GPUTransparencyMode::BackgroundPlusForeground:
        {
          bg_bits &= ~0x8000u;

          const u32 sum = fg_bits + bg_bits;
          const u32 carry = (sum - ((fg_bits ^ bg_bits) & 0x8421u)) & 0x8420u;

          color.bits = Truncate16((sum - carry) | (carry - (carry >> 5)));
        }
        break;

        case GPUTransparencyMode::BackgroundMinusForeground:
        {
          bg_bits |= 0x8000u;
          fg_bits &= ~0x8000u;

          const u32 diff = bg_bits - fg_bits + 0x108420u;
          const u32 borrow = (diff - ((bg_bits ^ fg_bits) & 0x108420u)) & 0x108420u;

          color.bits = Truncate16((diff - borrow) & (borrow - (borrow >> 5)));
        }
        break;

        case GPUTransparencyMode::BackgroundPlusQuarterForeground:
        {
          bg_bits &= ~0x8000u;
          fg_bits = ((fg_bits >> 2) & 0x1CE7u) | 0x8000u;

          const u32 sum = fg_bits + bg_bits;
          const u32 carry = (sum - ((fg_bits ^ bg_bits) & 0x8421u)) & 0x8420u;

          color.bits = Truncate16((sum - carry) | (carry - (carry >> 5)));
        }
        break;

        default:
          break;
      }

      // See above.
      if constexpr (!texture_enable)
        color.bits &= ~0x8000u;
    }
  }

  const u16 mask_and = cmd->params.GetMaskAND();
  if ((bg_color.bits & mask_and) != 0)
    return;

  SetPixel(static_cast<u32>(x), static_cast<u32>(y), color.bits | cmd->params.GetMaskOR());
}

template<bool texture_enable, bool raw_texture_enable, bool transparency_enable>
void GPUSWBackend::DrawSprite(const GPUBackendDrawSpriteCommand* cmd)
{
  const s32 origin_x = cmd->x;
  const s32 origin_y = cmd->y;
  const auto [r, g, b] = UnpackColorRGB24(cmd->color);
  const auto [origin_texcoord_x, origin_texcoord_y] = UnpackTexcoord(cmd->texcoord);

  for (u32 offset_y = 0; offset_y < cmd->height; offset_y++)
  {
    const s32 y = origin_y + static_cast<s32>(offset_y);
    if (y < static_cast<s32>(m_drawing_area.top) || y > static_cast<s32>(m_drawing_area.bottom) ||
        (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u)))
    {
      continue;
    }

    const u8 texcoord_y = Truncate8(ZeroExtend32(origin_texcoord_y) + offset_y);

    for (u32 offset_x = 0; offset_x < cmd->width; offset_x++)
    {
      const s32 x = origin_x + static_cast<s32>(offset_x);
      if (x < static_cast<s32>(m_drawing_area.left) || x > static_cast<s32>(m_drawing_area.right))
        continue;

      const u8 texcoord_x = Truncate8(ZeroExtend32(origin_texcoord_x) + offset_x);

      ShadePixel<texture_enable, raw_texture_enable, transparency_enable, false>(
        cmd, static_cast<u32>(x), static_cast<u32>(y), r, g, b, texcoord_x, texcoord_y);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Polygon and line rasterization ported from Mednafen
//////////////////////////////////////////////////////////////////////////

#define COORD_FBS 12
#define COORD_MF_INT(n) ((n) << COORD_FBS)
#define COORD_POST_PADDING 12

static ALWAYS_INLINE_RELEASE s64 MakePolyXFP(s32 x)
{
  return ((u64)x << 32) + ((1ULL << 32) - (1 << 11));
}

static ALWAYS_INLINE_RELEASE s64 MakePolyXFPStep(s32 dx, s32 dy)
{
  s64 ret;
  s64 dx_ex = (u64)dx << 32;

  if (dx_ex < 0)
    dx_ex -= dy - 1;

  if (dx_ex > 0)
    dx_ex += dy - 1;

  ret = dx_ex / dy;

  return (ret);
}

static ALWAYS_INLINE_RELEASE s32 GetPolyXFP_Int(s64 xfp)
{
  return (xfp >> 32);
}

template<bool shading_enable, bool texture_enable>
bool ALWAYS_INLINE_RELEASE GPUSWBackend::CalcIDeltas(i_deltas& idl, const GPUBackendDrawPolygonCommand::Vertex* A,
                                                     const GPUBackendDrawPolygonCommand::Vertex* B,
                                                     const GPUBackendDrawPolygonCommand::Vertex* C)
{
#define CALCIS(x, y) (((B->x - A->x) * (C->y - B->y)) - ((C->x - B->x) * (B->y - A->y)))

  s32 denom = CALCIS(x, y);

  if (!denom)
    return false;

  if constexpr (shading_enable)
  {
    idl.dr_dx = (u32)(CALCIS(r, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dr_dy = (u32)(CALCIS(x, r) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.dg_dx = (u32)(CALCIS(g, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dg_dy = (u32)(CALCIS(x, g) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.db_dx = (u32)(CALCIS(b, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.db_dy = (u32)(CALCIS(x, b) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
  }

  if constexpr (texture_enable)
  {
    idl.du_dx = (u32)(CALCIS(u, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.du_dy = (u32)(CALCIS(x, u) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;

    idl.dv_dx = (u32)(CALCIS(v, y) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
    idl.dv_dy = (u32)(CALCIS(x, v) * (1 << COORD_FBS) / denom) << COORD_POST_PADDING;
  }

  return true;

#undef CALCIS
}

template<bool shading_enable, bool texture_enable>
void ALWAYS_INLINE_RELEASE GPUSWBackend::AddIDeltas_DX(i_group& ig, const i_deltas& idl, u32 count /*= 1*/)
{
  if constexpr (shading_enable)
  {
    ig.r += idl.dr_dx * count;
    ig.g += idl.dg_dx * count;
    ig.b += idl.db_dx * count;
  }

  if constexpr (texture_enable)
  {
    ig.u += idl.du_dx * count;
    ig.v += idl.dv_dx * count;
  }
}

template<bool shading_enable, bool texture_enable>
void ALWAYS_INLINE_RELEASE GPUSWBackend::AddIDeltas_DY(i_group& ig, const i_deltas& idl, u32 count /*= 1*/)
{
  if constexpr (shading_enable)
  {
    ig.r += idl.dr_dy * count;
    ig.g += idl.dg_dy * count;
    ig.b += idl.db_dy * count;
  }

  if constexpr (texture_enable)
  {
    ig.u += idl.du_dy * count;
    ig.v += idl.dv_dy * count;
  }
}

template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
         bool dithering_enable>
void GPUSWBackend::DrawSpan(const GPUBackendDrawCommand* cmd, s32 y, s32 x_start, s32 x_bound, i_group ig,
                            const i_deltas& idl)
{
  if (cmd->params.interlaced_rendering && cmd->params.active_line_lsb == (Truncate8(static_cast<u32>(y)) & 1u))
    return;

  s32 x_ig_adjust = x_start;
  s32 w = x_bound - x_start;
  s32 x = TruncateGPUVertexPosition(x_start);

  if (x < static_cast<s32>(m_drawing_area.left))
  {
    s32 delta = static_cast<s32>(m_drawing_area.left) - x;
    x_ig_adjust += delta;
    x += delta;
    w -= delta;
  }

  if ((x + w) > (static_cast<s32>(m_drawing_area.right) + 1))
    w = static_cast<s32>(m_drawing_area.right) + 1 - x;

  if (w <= 0)
    return;

  AddIDeltas_DX<shading_enable, texture_enable>(ig, idl, x_ig_adjust);
  AddIDeltas_DY<shading_enable, texture_enable>(ig, idl, y);

  do
  {
    const u32 r = ig.r >> (COORD_FBS + COORD_POST_PADDING);
    const u32 g = ig.g >> (COORD_FBS + COORD_POST_PADDING);
    const u32 b = ig.b >> (COORD_FBS + COORD_POST_PADDING);
    const u32 u = ig.u >> (COORD_FBS + COORD_POST_PADDING);
    const u32 v = ig.v >> (COORD_FBS + COORD_POST_PADDING);

    ShadePixel<texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
      cmd, static_cast<u32>(x), static_cast<u32>(y), Truncate8(r), Truncate8(g), Truncate8(b), Truncate8(u),
      Truncate8(v));

    x++;
    AddIDeltas_DX<shading_enable, texture_enable>(ig, idl);
  } while (--w > 0);
}

template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
         bool dithering_enable>
void GPUSWBackend::DrawTriangle(const GPUBackendDrawCommand* cmd, const GPUBackendDrawPolygonCommand::Vertex* v0,
                                const GPUBackendDrawPolygonCommand::Vertex* v1,
                                const GPUBackendDrawPolygonCommand::Vertex* v2)
{
  u32 core_vertex;
  {
    u32 cvtemp = 0;

    if (v1->x <= v0->x)
    {
      if (v2->x <= v1->x)
        cvtemp = (1 << 2);
      else
        cvtemp = (1 << 1);
    }
    else if (v2->x < v0->x)
      cvtemp = (1 << 2);
    else
      cvtemp = (1 << 0);

    if (v2->y < v1->y)
    {
      std::swap(v2, v1);
      cvtemp = ((cvtemp >> 1) & 0x2) | ((cvtemp << 1) & 0x4) | (cvtemp & 0x1);
    }

    if (v1->y < v0->y)
    {
      std::swap(v1, v0);
      cvtemp = ((cvtemp >> 1) & 0x1) | ((cvtemp << 1) & 0x2) | (cvtemp & 0x4);
    }

    if (v2->y < v1->y)
    {
      std::swap(v2, v1);
      cvtemp = ((cvtemp >> 1) & 0x2) | ((cvtemp << 1) & 0x4) | (cvtemp & 0x1);
    }

    core_vertex = cvtemp >> 1;
  }

  if (v0->y == v2->y)
    return;

  // TODO: This could possibly be removed now.
  if (static_cast<u32>(std::abs(v2->x - v0->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(std::abs(v2->x - v1->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(std::abs(v1->x - v0->x)) >= MAX_PRIMITIVE_WIDTH ||
      static_cast<u32>(v2->y - v0->y) >= MAX_PRIMITIVE_HEIGHT)
  {
    return;
  }

  s64 base_coord = MakePolyXFP(v0->x);
  s64 base_step = MakePolyXFPStep((v2->x - v0->x), (v2->y - v0->y));
  s64 bound_coord_us;
  s64 bound_coord_ls;
  bool right_facing;

  if (v1->y == v0->y)
  {
    bound_coord_us = 0;
    right_facing = (bool)(v1->x > v0->x);
  }
  else
  {
    bound_coord_us = MakePolyXFPStep((v1->x - v0->x), (v1->y - v0->y));
    right_facing = (bool)(bound_coord_us > base_step);
  }

  if (v2->y == v1->y)
    bound_coord_ls = 0;
  else
    bound_coord_ls = MakePolyXFPStep((v2->x - v1->x), (v2->y - v1->y));

  i_deltas idl;
  if (!CalcIDeltas<shading_enable, texture_enable>(idl, v0, v1, v2))
    return;

  const GPUBackendDrawPolygonCommand::Vertex* vertices[3] = {v0, v1, v2};

  i_group ig;
  if constexpr (texture_enable)
  {
    ig.u = (COORD_MF_INT(vertices[core_vertex]->u) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
    ig.v = (COORD_MF_INT(vertices[core_vertex]->v) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  }

  ig.r = (COORD_MF_INT(vertices[core_vertex]->r) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  ig.g = (COORD_MF_INT(vertices[core_vertex]->g) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;
  ig.b = (COORD_MF_INT(vertices[core_vertex]->b) + (1 << (COORD_FBS - 1))) << COORD_POST_PADDING;

  AddIDeltas_DX<shading_enable, texture_enable>(ig, idl, -vertices[core_vertex]->x);
  AddIDeltas_DY<shading_enable, texture_enable>(ig, idl, -vertices[core_vertex]->y);

  struct TriangleHalf
  {
    u64 x_coord[2];
    u64 x_step[2];

    s32 y_coord;
    s32 y_bound;

    bool dec_mode;
  } tripart[2];

  u32 vo = 0;
  u32 vp = 0;
  if (core_vertex != 0)
    vo = 1;
  if (core_vertex == 2)
    vp = 3;

  {
    TriangleHalf* tp = &tripart[vo];
    tp->y_coord = vertices[0 ^ vo]->y;
    tp->y_bound = vertices[1 ^ vo]->y;
    tp->x_coord[right_facing] = MakePolyXFP(vertices[0 ^ vo]->x);
    tp->x_step[right_facing] = bound_coord_us;
    tp->x_coord[!right_facing] = base_coord + ((vertices[vo]->y - vertices[0]->y) * base_step);
    tp->x_step[!right_facing] = base_step;
    tp->dec_mode = vo;
  }

  {
    TriangleHalf* tp = &tripart[vo ^ 1];
    tp->y_coord = vertices[1 ^ vp]->y;
    tp->y_bound = vertices[2 ^ vp]->y;
    tp->x_coord[right_facing] = MakePolyXFP(vertices[1 ^ vp]->x);
    tp->x_step[right_facing] = bound_coord_ls;
    tp->x_coord[!right_facing] =
      base_coord + ((vertices[1 ^ vp]->y - vertices[0]->y) *
                    base_step); // base_coord + ((vertices[1].y - vertices[0].y) * base_step);
    tp->x_step[!right_facing] = base_step;
    tp->dec_mode = vp;
  }

  for (u32 i = 0; i < 2; i++)
  {
    s32 yi = tripart[i].y_coord;
    s32 yb = tripart[i].y_bound;

    u64 lc = tripart[i].x_coord[0];
    u64 ls = tripart[i].x_step[0];

    u64 rc = tripart[i].x_coord[1];
    u64 rs = tripart[i].x_step[1];

    if (tripart[i].dec_mode)
    {
      while (yi > yb)
      {
        yi--;
        lc -= ls;
        rc -= rs;

        s32 y = TruncateGPUVertexPosition(yi);

        if (y < static_cast<s32>(m_drawing_area.top))
          break;

        if (y > static_cast<s32>(m_drawing_area.bottom))
          continue;

        DrawSpan<shading_enable, texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
          cmd, yi, GetPolyXFP_Int(lc), GetPolyXFP_Int(rc), ig, idl);
      }
    }
    else
    {
      while (yi < yb)
      {
        s32 y = TruncateGPUVertexPosition(yi);

        if (y > static_cast<s32>(m_drawing_area.bottom))
          break;

        if (y >= static_cast<s32>(m_drawing_area.top))
        {

          DrawSpan<shading_enable, texture_enable, raw_texture_enable, transparency_enable, dithering_enable>(
            cmd, yi, GetPolyXFP_Int(lc), GetPolyXFP_Int(rc), ig, idl);
        }

        yi++;
        lc += ls;
        rc += rs;
      }
    }
  }
}

enum
{
  Line_XY_FractBits = 32
};
enum
{
  Line_RGB_FractBits = 12
};

struct line_fxp_coord
{
  u64 x, y;
  u32 r, g, b;
};

struct line_fxp_step
{
  s64 dx_dk, dy_dk;
  s32 dr_dk, dg_dk, db_dk;
};

static ALWAYS_INLINE_RELEASE s64 LineDivide(s64 delta, s32 dk)
{
  delta = (u64)delta << Line_XY_FractBits;

  if (delta < 0)
    delta -= dk - 1;
  if (delta > 0)
    delta += dk - 1;

  return (delta / dk);
}

template<bool shading_enable, bool transparency_enable, bool dithering_enable>
void GPUSWBackend::DrawLine(const GPUBackendDrawLineCommand* cmd, const GPUBackendDrawLineCommand::Vertex* p0,
                            const GPUBackendDrawLineCommand::Vertex* p1)
{
  const s32 i_dx = std::abs(p1->x - p0->x);
  const s32 i_dy = std::abs(p1->y - p0->y);
  const s32 k = (i_dx > i_dy) ? i_dx : i_dy;
  if (i_dx >= MAX_PRIMITIVE_WIDTH || i_dy >= MAX_PRIMITIVE_HEIGHT)
    return;

  if (p0->x >= p1->x && k > 0)
    std::swap(p0, p1);

  line_fxp_step step;
  if (k == 0)
  {
    step.dx_dk = 0;
    step.dy_dk = 0;

    if constexpr (shading_enable)
    {
      step.dr_dk = 0;
      step.dg_dk = 0;
      step.db_dk = 0;
    }
  }
  else
  {
    step.dx_dk = LineDivide(p1->x - p0->x, k);
    step.dy_dk = LineDivide(p1->y - p0->y, k);

    if constexpr (shading_enable)
    {
      step.dr_dk = (s32)((u32)(p1->r - p0->r) << Line_RGB_FractBits) / k;
      step.dg_dk = (s32)((u32)(p1->g - p0->g) << Line_RGB_FractBits) / k;
      step.db_dk = (s32)((u32)(p1->b - p0->b) << Line_RGB_FractBits) / k;
    }
  }

  line_fxp_coord cur_point;
  cur_point.x = ((u64)p0->x << Line_XY_FractBits) | (1ULL << (Line_XY_FractBits - 1));
  cur_point.y = ((u64)p0->y << Line_XY_FractBits) | (1ULL << (Line_XY_FractBits - 1));

  cur_point.x -= 1024;

  if (step.dy_dk < 0)
    cur_point.y -= 1024;

  if constexpr (shading_enable)
  {
    cur_point.r = (p0->r << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
    cur_point.g = (p0->g << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
    cur_point.b = (p0->b << Line_RGB_FractBits) | (1 << (Line_RGB_FractBits - 1));
  }

  for (s32 i = 0; i <= k; i++)
  {
    // Sign extension is not necessary here for x and y, due to the maximum values that ClipX1 and ClipY1 can contain.
    const s32 x = (cur_point.x >> Line_XY_FractBits) & 2047;
    const s32 y = (cur_point.y >> Line_XY_FractBits) & 2047;

    if ((!cmd->params.interlaced_rendering || cmd->params.active_line_lsb != (Truncate8(static_cast<u32>(y)) & 1u)) &&
        x >= static_cast<s32>(m_drawing_area.left) && x <= static_cast<s32>(m_drawing_area.right) &&
        y >= static_cast<s32>(m_drawing_area.top) && y <= static_cast<s32>(m_drawing_area.bottom))
    {
      const u8 r = shading_enable ? static_cast<u8>(cur_point.r >> Line_RGB_FractBits) : p0->r;
      const u8 g = shading_enable ? static_cast<u8>(cur_point.g >> Line_RGB_FractBits) : p0->g;
      const u8 b = shading_enable ? static_cast<u8>(cur_point.b >> Line_RGB_FractBits) : p0->b;

      ShadePixel<false, false, transparency_enable, dithering_enable>(cmd, static_cast<u32>(x), static_cast<u32>(y), r,
                                                                      g, b, 0, 0);
    }

    cur_point.x += step.dx_dk;
    cur_point.y += step.dy_dk;

    if constexpr (shading_enable)
    {
      cur_point.r += step.dr_dk;
      cur_point.g += step.dg_dk;
      cur_point.b += step.db_dk;
    }
  }
}

void GPUSWBackend::FlushRender()
{
}

void GPUSWBackend::DrawingAreaChanged()
{
}

GPUSWBackend::DrawLineFunction GPUSWBackend::GetDrawLineFunction(bool shading_enable, bool transparency_enable,
                                                                 bool dithering_enable)
{
  static constexpr DrawLineFunction funcs[2][2][2] = {
    {{&GPUSWBackend::DrawLine<false, false, false>, &GPUSWBackend::DrawLine<false, false, true>},
     {&GPUSWBackend::DrawLine<false, true, false>, &GPUSWBackend::DrawLine<false, true, true>}},
    {{&GPUSWBackend::DrawLine<true, false, false>, &GPUSWBackend::DrawLine<true, false, true>},
     {&GPUSWBackend::DrawLine<true, true, false>, &GPUSWBackend::DrawLine<true, true, true>}}};

  return funcs[u8(shading_enable)][u8(transparency_enable)][u8(dithering_enable)];
}

GPUSWBackend::DrawSpriteFunction GPUSWBackend::GetDrawSpriteFunction(bool texture_enable, bool raw_texture_enable,
                                                                     bool transparency_enable)
{
  static constexpr DrawSpriteFunction funcs[2][2][2] = {
    {{&GPUSWBackend::DrawSprite<false, false, false>, &GPUSWBackend::DrawSprite<false, false, true>},
     {&GPUSWBackend::DrawSprite<false, false, false>, &GPUSWBackend::DrawSprite<false, false, true>}},
    {{&GPUSWBackend::DrawSprite<true, false, false>, &GPUSWBackend::DrawSprite<true, false, true>},
     {&GPUSWBackend::DrawSprite<true, true, false>, &GPUSWBackend::DrawSprite<true, true, true>}}};

  return funcs[u8(texture_enable)][u8(raw_texture_enable)][u8(transparency_enable)];
}

GPUSWBackend::DrawTriangleFunction GPUSWBackend::GetDrawTriangleFunction(bool shading_enable, bool texture_enable,
                                                                         bool raw_texture_enable,
                                                                         bool transparency_enable,
                                                                         bool dithering_enable)
{
  static constexpr DrawTriangleFunction funcs[2][2][2][2][2] = {
    {{{{&GPUSWBackend::DrawTriangle<false, false, false, false, false>,
        &GPUSWBackend::DrawTriangle<false, false, false, false, true>},
       {&GPUSWBackend::DrawTriangle<false, false, false, true, false>,
        &GPUSWBackend::DrawTriangle<false, false, false, true, true>}},
      {{&GPUSWBackend::DrawTriangle<false, false, false, false, false>,
        &GPUSWBackend::DrawTriangle<false, false, false, false, false>},
       {&GPUSWBackend::DrawTriangle<false, false, false, true, false>,
        &GPUSWBackend::DrawTriangle<false, false, false, true, false>}}},
     {{{&GPUSWBackend::DrawTriangle<false, true, false, false, false>,
        &GPUSWBackend::DrawTriangle<false, true, false, false, true>},
       {&GPUSWBackend::DrawTriangle<false, true, false, true, false>,
        &GPUSWBackend::DrawTriangle<false, true, false, true, true>}},
      {{&GPUSWBackend::DrawTriangle<false, true, true, false, false>,
        &GPUSWBackend::DrawTriangle<false, true, true, false, false>},
       {&GPUSWBackend::DrawTriangle<false, true, true, true, false>,
        &GPUSWBackend::DrawTriangle<false, true, true, true, false>}}}},
    {{{{&GPUSWBackend::DrawTriangle<true, false, false, false, false>,
        &GPUSWBackend::DrawTriangle<true, false, false, false, true>},
       {&GPUSWBackend::DrawTriangle<true, false, false, true, false>,
        &GPUSWBackend::DrawTriangle<true, false, false, true, true>}},
      {{&GPUSWBackend::DrawTriangle<true, false, false, false, false>,
        &GPUSWBackend::DrawTriangle<true, false, false, false, false>},
       {&GPUSWBackend::DrawTriangle<true, false, false, true, false>,
        &GPUSWBackend::DrawTriangle<true, false, false, true, false>}}},
     {{{&GPUSWBackend::DrawTriangle<true, true, false, false, false>,
        &GPUSWBackend::DrawTriangle<true, true, false, false, true>},
       {&GPUSWBackend::DrawTriangle<true, true, false, true, false>,
        &GPUSWBackend::DrawTriangle<true, true, false, true, true>}},
      {{&GPUSWBackend::DrawTriangle<true, true, true, false, false>,
        &GPUSWBackend::DrawTriangle<true, true, true, false, false>},
       {&GPUSWBackend::DrawTriangle<true, true, true, true, false>,
        &GPUSWBackend::DrawTriangle<true, true, true, true, false>}}}}};

  return funcs[u8(shading_enable)][u8(texture_enable)][u8(raw_texture_enable)][u8(transparency_enable)]
              [u8(dithering_enable)];
}

void GPUSWBackend::SetDisplayTextureFormat()
{
  static constexpr const std::array formats_for_16bit = {GPUTexture::Format::RGB565, GPUTexture::Format::RGBA5551,
                                                         GPUTexture::Format::RGBA8, GPUTexture::Format::BGRA8};
  static constexpr const std::array formats_for_24bit = {GPUTexture::Format::RGBA8, GPUTexture::Format::BGRA8,
                                                         GPUTexture::Format::RGB565, GPUTexture::Format::RGBA5551};
  for (const GPUTexture::Format format : formats_for_16bit)
  {
    if (g_gpu_device->SupportsTextureFormat(format))
    {
      m_16bit_display_format = format;
      break;
    }
  }
  for (const GPUTexture::Format format : formats_for_24bit)
  {
    if (g_gpu_device->SupportsTextureFormat(format))
    {
      m_24bit_display_format = format;
      break;
    }
  }
}

GPUTexture* GPUSWBackend::GetDisplayTexture(u32 width, u32 height, GPUTexture::Format format)
{
  if (!m_private_display_texture || m_private_display_texture->GetWidth() != width ||
      m_private_display_texture->GetHeight() != height || m_private_display_texture->GetFormat() != format)
  {
    ClearDisplayTexture();
    m_private_display_texture.reset();
    m_private_display_texture =
      g_gpu_device->FetchTexture(width, height, 1, 1, 1, GPUTexture::Type::Texture, format, nullptr, 0);
    if (!m_private_display_texture)
      Log_ErrorPrintf("Failed to create %ux%u %u texture", width, height, static_cast<u32>(format));
  }

  return m_private_display_texture.get();
}

template<GPUTexture::Format out_format, typename out_type>
static void CopyOutRow16(const u16* src_ptr, out_type* dst_ptr, u32 width);

template<GPUTexture::Format out_format, typename out_type>
static out_type VRAM16ToOutput(u16 value);

template<>
ALWAYS_INLINE u16 VRAM16ToOutput<GPUTexture::Format::RGBA5551, u16>(u16 value)
{
  return (value & 0x3E0) | ((value >> 10) & 0x1F) | ((value & 0x1F) << 10);
}

template<>
ALWAYS_INLINE u16 VRAM16ToOutput<GPUTexture::Format::RGB565, u16>(u16 value)
{
  return ((value & 0x3E0) << 1) | ((value & 0x20) << 1) | ((value >> 10) & 0x1F) | ((value & 0x1F) << 11);
}

template<>
ALWAYS_INLINE u32 VRAM16ToOutput<GPUTexture::Format::RGBA8, u32>(u16 value)
{
  const u32 value32 = ZeroExtend32(value);
  const u32 r = (value32 & 31u) << 3;
  const u32 g = ((value32 >> 5) & 31u) << 3;
  const u32 b = ((value32 >> 10) & 31u) << 3;
  const u32 a = ((value >> 15) != 0) ? 255 : 0;
  return ZeroExtend32(r) | (ZeroExtend32(g) << 8) | (ZeroExtend32(b) << 16) | (ZeroExtend32(a) << 24);
}

template<>
ALWAYS_INLINE u32 VRAM16ToOutput<GPUTexture::Format::BGRA8, u32>(u16 value)
{
  const u32 value32 = ZeroExtend32(value);
  const u32 r = (value32 & 31u) << 3;
  const u32 g = ((value32 >> 5) & 31u) << 3;
  const u32 b = ((value32 >> 10) & 31u) << 3;
  return ZeroExtend32(b) | (ZeroExtend32(g) << 8) | (ZeroExtend32(r) << 16) | (0xFF000000u);
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGBA5551, u16>(const u16* src_ptr, u16* dst_ptr, u32 width)
{
  u32 col = 0;

#if defined(CPU_ARCH_SSE)
  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  for (; col < aligned_width; col += 8)
  {
    const __m128i single_mask = _mm_set1_epi16(0x1F);
    __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr));
    src_ptr += 8;
    __m128i a = _mm_and_si128(value, _mm_set1_epi16(static_cast<s16>(static_cast<u16>(0x3E0))));
    __m128i b = _mm_and_si128(_mm_srli_epi16(value, 10), single_mask);
    __m128i c = _mm_slli_epi16(_mm_and_si128(value, single_mask), 10);
    value = _mm_or_si128(_mm_or_si128(a, b), c);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr), value);
    dst_ptr += 8;
  }
#elif defined(CPU_ARCH_NEON)
  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  for (; col < aligned_width; col += 8)
  {
    const uint16x8_t single_mask = vdupq_n_u16(0x1F);
    uint16x8_t value = vld1q_u16(src_ptr);
    src_ptr += 8;
    uint16x8_t a = vandq_u16(value, vdupq_n_u16(0x3E0));
    uint16x8_t b = vandq_u16(vshrq_n_u16(value, 10), single_mask);
    uint16x8_t c = vshlq_n_u16(vandq_u16(value, single_mask), 10);
    value = vorrq_u16(vorrq_u16(a, b), c);
    vst1q_u16(dst_ptr, value);
    dst_ptr += 8;
  }
#endif

  for (; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGBA5551, u16>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGB565, u16>(const u16* src_ptr, u16* dst_ptr, u32 width)
{
  u32 col = 0;

#if defined(CPU_ARCH_SSE)
  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  for (; col < aligned_width; col += 8)
  {
    const __m128i single_mask = _mm_set1_epi16(0x1F);
    __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr));
    src_ptr += 8;
    __m128i a = _mm_slli_epi16(_mm_and_si128(value, _mm_set1_epi16(static_cast<s16>(static_cast<u16>(0x3E0)))), 1);
    __m128i b = _mm_slli_epi16(_mm_and_si128(value, _mm_set1_epi16(static_cast<s16>(static_cast<u16>(0x20)))), 1);
    __m128i c = _mm_and_si128(_mm_srli_epi16(value, 10), single_mask);
    __m128i d = _mm_slli_epi16(_mm_and_si128(value, single_mask), 11);
    value = _mm_or_si128(_mm_or_si128(_mm_or_si128(a, b), c), d);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr), value);
    dst_ptr += 8;
  }
#elif defined(CPU_ARCH_NEON)
  const u32 aligned_width = Common::AlignDownPow2(width, 8);
  const uint16x8_t single_mask = vdupq_n_u16(0x1F);
  for (; col < aligned_width; col += 8)
  {
    uint16x8_t value = vld1q_u16(src_ptr);
    src_ptr += 8;
    uint16x8_t a = vshlq_n_u16(vandq_u16(value, vdupq_n_u16(0x3E0)), 1); // (value & 0x3E0) << 1
    uint16x8_t b = vshlq_n_u16(vandq_u16(value, vdupq_n_u16(0x20)), 1);  // (value & 0x20) << 1
    uint16x8_t c = vandq_u16(vshrq_n_u16(value, 10), single_mask);       // ((value >> 10) & 0x1F)
    uint16x8_t d = vshlq_n_u16(vandq_u16(value, single_mask), 11);       // ((value & 0x1F) << 11)
    value = vorrq_u16(vorrq_u16(vorrq_u16(a, b), c), d);
    vst1q_u16(dst_ptr, value);
    dst_ptr += 8;
  }
#endif

  for (; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGB565, u16>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::RGBA8, u32>(const u16* src_ptr, u32* dst_ptr, u32 width)
{
  for (u32 col = 0; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::RGBA8, u32>(*(src_ptr++));
}

template<>
ALWAYS_INLINE void CopyOutRow16<GPUTexture::Format::BGRA8, u32>(const u16* src_ptr, u32* dst_ptr, u32 width)
{
  for (u32 col = 0; col < width; col++)
    *(dst_ptr++) = VRAM16ToOutput<GPUTexture::Format::BGRA8, u32>(*(src_ptr++));
}

template<GPUTexture::Format display_format>
void GPUSWBackend::CopyOut15Bit(u32 src_x, u32 src_y, u32 width, u32 height, u32 field, bool interlaced,
                                bool interleaved)
{
  using OutputPixelType =
    std::conditional_t<display_format == GPUTexture::Format::RGBA8 || display_format == GPUTexture::Format::BGRA8, u32,
                       u16>;

  GPUTexture* texture = GetDisplayTexture(width, height, display_format);
  if (!texture)
    return;

  u32 dst_stride = GPU_MAX_DISPLAY_WIDTH * sizeof(OutputPixelType);
  u8* dst_ptr = m_display_texture_buffer.data() + (interlaced ? (field != 0 ? dst_stride : 0) : 0);

  const bool mapped =
    (!interlaced && texture->Map(reinterpret_cast<void**>(&dst_ptr), &dst_stride, 0, 0, width, height));

  const u32 output_stride = dst_stride;
  const u8 interlaced_shift = BoolToUInt8(interlaced);
  const u8 interleaved_shift = BoolToUInt8(interleaved);

  // Fast path when not wrapping around.
  if ((src_x + width) <= VRAM_WIDTH && (src_y + height) <= VRAM_HEIGHT)
  {
    const u32 rows = height >> interlaced_shift;
    dst_stride <<= interlaced_shift;

    const u16* src_ptr = &m_vram[src_y * VRAM_WIDTH + src_x];
    const u32 src_step = VRAM_WIDTH << interleaved_shift;
    for (u32 row = 0; row < rows; row++)
    {
      CopyOutRow16<display_format>(src_ptr, reinterpret_cast<OutputPixelType*>(dst_ptr), width);
      src_ptr += src_step;
      dst_ptr += dst_stride;
    }
  }
  else
  {
    const u32 rows = height >> interlaced_shift;
    dst_stride <<= interlaced_shift;

    const u32 end_x = src_x + width;
    for (u32 row = 0; row < rows; row++)
    {
      const u16* src_row_ptr = &m_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
      OutputPixelType* dst_row_ptr = reinterpret_cast<OutputPixelType*>(dst_ptr);

      for (u32 col = src_x; col < end_x; col++)
        *(dst_row_ptr++) = VRAM16ToOutput<display_format, OutputPixelType>(src_row_ptr[col % VRAM_WIDTH]);

      src_y += (1 << interleaved_shift);
      dst_ptr += dst_stride;
    }
  }

  if (mapped)
    texture->Unmap();
  else
    texture->Update(0, 0, width, height, m_display_texture_buffer.data(), output_stride);

  SetDisplayTexture(texture, 0, 0, width, height);
}

void GPUSWBackend::CopyOut15Bit(GPUTexture::Format display_format, u32 src_x, u32 src_y, u32 width, u32 height,
                                u32 field, bool interlaced, bool interleaved)
{
  switch (display_format)
  {
    case GPUTexture::Format::RGBA5551:
      CopyOut15Bit<GPUTexture::Format::RGBA5551>(src_x, src_y, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::RGB565:
      CopyOut15Bit<GPUTexture::Format::RGB565>(src_x, src_y, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::RGBA8:
      CopyOut15Bit<GPUTexture::Format::RGBA8>(src_x, src_y, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::BGRA8:
      CopyOut15Bit<GPUTexture::Format::BGRA8>(src_x, src_y, width, height, field, interlaced, interleaved);
      break;
    default:
      break;
  }
}

template<GPUTexture::Format display_format>
void GPUSWBackend::CopyOut24Bit(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 field, bool interlaced,
                                bool interleaved)
{
  using OutputPixelType =
    std::conditional_t<display_format == GPUTexture::Format::RGBA8 || display_format == GPUTexture::Format::BGRA8, u32,
                       u16>;

  GPUTexture* texture = GetDisplayTexture(width, height, display_format);
  if (!texture)
    return;

  u32 dst_stride = Common::AlignUpPow2<u32>(width * sizeof(OutputPixelType), 4);
  u8* dst_ptr = m_display_texture_buffer.data() + (interlaced ? (field != 0 ? dst_stride : 0) : 0);
  const bool mapped =
    (!interlaced && texture->Map(reinterpret_cast<void**>(&dst_ptr), &dst_stride, 0, 0, width, height));

  const u32 output_stride = dst_stride;
  const u8 interlaced_shift = BoolToUInt8(interlaced);
  const u8 interleaved_shift = BoolToUInt8(interleaved);
  const u32 rows = height >> interlaced_shift;
  dst_stride <<= interlaced_shift;

  if ((src_x + width) <= VRAM_WIDTH && (src_y + (rows << interleaved_shift)) <= VRAM_HEIGHT)
  {
    const u8* src_ptr = reinterpret_cast<const u8*>(&m_vram[src_y * VRAM_WIDTH + src_x]) + (skip_x * 3);
    const u32 src_stride = (VRAM_WIDTH << interleaved_shift) * sizeof(u16);
    for (u32 row = 0; row < rows; row++)
    {
      if constexpr (display_format == GPUTexture::Format::RGBA8)
      {
        const u8* src_row_ptr = src_ptr;
        u8* dst_row_ptr = reinterpret_cast<u8*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = *(src_row_ptr++);
          *(dst_row_ptr++) = 0xFF;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::BGRA8)
      {
        const u8* src_row_ptr = src_ptr;
        u8* dst_row_ptr = reinterpret_cast<u8*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = src_row_ptr[2];
          *(dst_row_ptr++) = src_row_ptr[1];
          *(dst_row_ptr++) = src_row_ptr[0];
          *(dst_row_ptr++) = 0xFF;
          src_row_ptr += 3;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::RGB565)
      {
        const u8* src_row_ptr = src_ptr;
        u16* dst_row_ptr = reinterpret_cast<u16*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = ((static_cast<u16>(src_row_ptr[0]) >> 3) << 11) |
                             ((static_cast<u16>(src_row_ptr[1]) >> 2) << 5) | (static_cast<u16>(src_row_ptr[2]) >> 3);
          src_row_ptr += 3;
        }
      }
      else if constexpr (display_format == GPUTexture::Format::RGBA5551)
      {
        const u8* src_row_ptr = src_ptr;
        u16* dst_row_ptr = reinterpret_cast<u16*>(dst_ptr);
        for (u32 col = 0; col < width; col++)
        {
          *(dst_row_ptr++) = ((static_cast<u16>(src_row_ptr[0]) >> 3) << 10) |
                             ((static_cast<u16>(src_row_ptr[1]) >> 3) << 5) | (static_cast<u16>(src_row_ptr[2]) >> 3);
          src_row_ptr += 3;
        }
      }

      src_ptr += src_stride;
      dst_ptr += dst_stride;
    }
  }
  else
  {
    for (u32 row = 0; row < rows; row++)
    {
      const u16* src_row_ptr = &m_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
      OutputPixelType* dst_row_ptr = reinterpret_cast<OutputPixelType*>(dst_ptr);

      for (u32 col = 0; col < width; col++)
      {
        const u32 offset = (src_x + (((skip_x + col) * 3) / 2));
        const u16 s0 = src_row_ptr[offset % VRAM_WIDTH];
        const u16 s1 = src_row_ptr[(offset + 1) % VRAM_WIDTH];
        const u8 shift = static_cast<u8>(col & 1u) * 8;
        const u32 rgb = (((ZeroExtend32(s1) << 16) | ZeroExtend32(s0)) >> shift);

        if constexpr (display_format == GPUTexture::Format::RGBA8)
        {
          *(dst_row_ptr++) = rgb | 0xFF000000u;
        }
        else if constexpr (display_format == GPUTexture::Format::BGRA8)
        {
          *(dst_row_ptr++) = (rgb & 0x00FF00) | ((rgb & 0xFF) << 16) | ((rgb >> 16) & 0xFF) | 0xFF000000u;
        }
        else if constexpr (display_format == GPUTexture::Format::RGB565)
        {
          *(dst_row_ptr++) = ((rgb >> 3) & 0x1F) | (((rgb >> 10) << 5) & 0x7E0) | (((rgb >> 19) << 11) & 0x3E0000);
        }
        else if constexpr (display_format == GPUTexture::Format::RGBA5551)
        {
          *(dst_row_ptr++) = ((rgb >> 3) & 0x1F) | (((rgb >> 11) << 5) & 0x3E0) | (((rgb >> 19) << 10) & 0x1F0000);
        }
      }

      src_y += (1 << interleaved_shift);
      dst_ptr += dst_stride;
    }
  }

  if (mapped)
    texture->Unmap();
  else
    texture->Update(0, 0, width, height, m_display_texture_buffer.data(), output_stride);

  SetDisplayTexture(texture, 0, 0, width, height);
}

void GPUSWBackend::CopyOut24Bit(GPUTexture::Format display_format, u32 src_x, u32 src_y, u32 skip_x, u32 width,
                                u32 height, u32 field, bool interlaced, bool interleaved)
{
  switch (display_format)
  {
    case GPUTexture::Format::RGBA5551:
      CopyOut24Bit<GPUTexture::Format::RGBA5551>(src_x, src_y, skip_x, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::RGB565:
      CopyOut24Bit<GPUTexture::Format::RGB565>(src_x, src_y, skip_x, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::RGBA8:
      CopyOut24Bit<GPUTexture::Format::RGBA8>(src_x, src_y, skip_x, width, height, field, interlaced, interleaved);
      break;
    case GPUTexture::Format::BGRA8:
      CopyOut24Bit<GPUTexture::Format::BGRA8>(src_x, src_y, skip_x, width, height, field, interlaced, interleaved);
      break;
    default:
      break;
  }
}

void GPUSWBackend::ClearDisplay()
{
  std::memset(m_display_texture_buffer.data(), 0, m_display_texture_buffer.size());
}

void GPUSWBackend::UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd)
{
  if (!g_settings.debugging.show_vram)
  {
    SetDisplayParameters(cmd->display_width, cmd->display_height, cmd->display_origin_left, cmd->display_origin_top,
                         cmd->display_vram_width, cmd->display_vram_height, cmd->display_aspect_ratio);

    if (cmd->display_disabled)
    {
      ClearDisplayTexture();
      return;
    }

    const u32 vram_offset_y = cmd->display_vram_top;
    const u32 display_width = cmd->display_vram_width;
    const u32 display_height = cmd->display_vram_height;

    if (cmd->interlaced_display_enabled)
    {
      const u32 field = cmd->interlaced_display_field;
      if (cmd->display_24bit)
      {
        CopyOut24Bit(m_24bit_display_format, cmd->X, vram_offset_y + field, cmd->display_vram_left - cmd->X,
                     display_width, display_height, field, true, cmd->interlaced_display_interleaved);
      }
      else
      {
        CopyOut15Bit(m_16bit_display_format, cmd->display_vram_left, vram_offset_y + field, display_width,
                     display_height, field, true, cmd->interlaced_display_interleaved);
      }
    }
    else
    {
      if (cmd->display_24bit)
      {
        CopyOut24Bit(m_24bit_display_format, cmd->X, vram_offset_y, cmd->display_vram_left - cmd->X, display_width,
                     display_height, 0, false, false);
      }
      else
      {
        CopyOut15Bit(m_16bit_display_format, cmd->display_vram_left, vram_offset_y, display_width, display_height, 0,
                     false, false);
      }
    }
  }
  else
  {
    CopyOut15Bit(m_16bit_display_format, 0, 0, VRAM_WIDTH, VRAM_HEIGHT, 0, false, false);
    SetDisplayParameters(VRAM_WIDTH, VRAM_HEIGHT, 0, 0, VRAM_WIDTH, VRAM_HEIGHT,
                         static_cast<float>(VRAM_WIDTH) / static_cast<float>(VRAM_HEIGHT));
  }
}

std::unique_ptr<GPUBackend> GPUBackend::CreateSoftwareBackend()
{
  std::unique_ptr<GPUSWBackend> gpu(std::make_unique<GPUSWBackend>());
  if (!gpu->Initialize())
    return nullptr;

  return gpu;
}
