// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_sw_backend.h"
#include "gpu.h"
#include "gpu_sw_rasterizer.h"
#include "system.h"

#include "util/gpu_device.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/intrin.h"
#include "common/log.h"

#include <algorithm>

Log_SetChannel(GPUSWBackend);

GPUSWBackend::GPUSWBackend() = default;

GPUSWBackend::~GPUSWBackend() = default;

bool GPUSWBackend::Initialize()
{
  if (!GPUBackend::Initialize())
    return false;

  GPU_SW_Rasterizer::SelectImplementation();

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
  if (clear_vram)
    std::memset(g_vram, 0, sizeof(g_vram));

  GPU_SW_Rasterizer::g_drawing_area = {};
}

void GPUSWBackend::ReadVRAM(u32 x, u32 y, u32 width, u32 height)
{
}

void GPUSWBackend::DrawPolygon(const GPUBackendDrawPolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const GPU_SW_Rasterizer::DrawTriangleFunction DrawFunction = GPU_SW_Rasterizer::GetDrawTriangleFunction(
    rc.shading_enable, rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable, dithering_enable);

  DrawFunction(cmd, &cmd->vertices[0], &cmd->vertices[1], &cmd->vertices[2]);
  if (rc.quad_polygon)
    DrawFunction(cmd, &cmd->vertices[2], &cmd->vertices[1], &cmd->vertices[3]);
}

void GPUSWBackend::DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};
  const bool dithering_enable = rc.IsDitheringEnabled() && cmd->draw_mode.dither_enable;

  const GPU_SW_Rasterizer::DrawTriangleFunction DrawFunction = GPU_SW_Rasterizer::GetDrawTriangleFunction(
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

  DrawFunction(cmd, &vertices[0], &vertices[1], &vertices[2]);
  if (rc.quad_polygon)
    DrawFunction(cmd, &vertices[2], &vertices[1], &vertices[3]);
}

void GPUSWBackend::DrawSprite(const GPUBackendDrawSpriteCommand* cmd)
{
  const GPURenderCommand rc{cmd->rc.bits};

  const GPU_SW_Rasterizer::DrawRectangleFunction DrawFunction =
    GPU_SW_Rasterizer::GetDrawRectangleFunction(rc.texture_enable, rc.raw_texture_enable, rc.transparency_enable);

  DrawFunction(cmd);
}

void GPUSWBackend::DrawLine(const GPUBackendDrawLineCommand* cmd)
{
  const GPU_SW_Rasterizer::DrawLineFunction DrawFunction = GPU_SW_Rasterizer::GetDrawLineFunction(
    cmd->rc.shading_enable, cmd->rc.transparency_enable, cmd->IsDitheringEnabled());

  for (u16 i = 1; i < cmd->num_vertices; i++)
    DrawFunction(cmd, &cmd->vertices[i - 1], &cmd->vertices[i]);
}

void GPUSWBackend::DrawingAreaChanged(const Common::Rectangle<u32>& new_drawing_area)
{
  GPU_SW_Rasterizer::g_drawing_area = new_drawing_area;
}

void GPUSWBackend::FlushRender()
{
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

    const u16* src_ptr = &g_vram[src_y * VRAM_WIDTH + src_x];
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
      const u16* src_row_ptr = &g_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
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
    const u8* src_ptr = reinterpret_cast<const u8*>(&g_vram[src_y * VRAM_WIDTH + src_x]) + (skip_x * 3);
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
      const u16* src_row_ptr = &g_vram[(src_y % VRAM_HEIGHT) * VRAM_WIDTH];
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
