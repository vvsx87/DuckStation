// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "gpu.h"
#include "gpu_backend.h"

#include "util/gpu_device.h"

#include "common/heap_array.h"

#include <array>
#include <memory>
#include <vector>

// TODO: Move to cpp

class GPUSWBackend final : public GPUBackend
{
public:
  GPUSWBackend();
  ~GPUSWBackend() override;

  bool Initialize() override;
  void Shutdown() override;

  // this is actually (31 * 255) >> 4) == 494, but to simplify addressing we use the next power of two (512)
  static constexpr u32 DITHER_LUT_SIZE = 512;
  using DitherLUT = std::array<std::array<std::array<u8, 512>, DITHER_MATRIX_SIZE>, DITHER_MATRIX_SIZE>;
  static constexpr DitherLUT ComputeDitherLUT();

protected:
  bool DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display) override;
  void Reset(bool clear_vram) override;

  void ReadVRAM(u32 x, u32 y, u32 width, u32 height) override;

  void DrawPolygon(const GPUBackendDrawPolygonCommand* cmd) override;
  void DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd) override;
  void DrawLine(const GPUBackendDrawLineCommand* cmd) override;
  void DrawSprite(const GPUBackendDrawSpriteCommand* cmd) override;
  void FlushRender() override;
  void DrawingAreaChanged() override;

  void ClearDisplay() override;
  void UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd) override;

private:
  union VRAMPixel
  {
    u16 bits;

    BitField<u16, u8, 0, 5> r;
    BitField<u16, u8, 5, 5> g;
    BitField<u16, u8, 10, 5> b;
    BitField<u16, bool, 15, 1> c;

    void Set(u8 r_, u8 g_, u8 b_, bool c_ = false)
    {
      bits = (ZeroExtend16(r_)) | (ZeroExtend16(g_) << 5) | (ZeroExtend16(b_) << 10) | (static_cast<u16>(c_) << 15);
    }

    void ClampAndSet(u8 r_, u8 g_, u8 b_, bool c_ = false)
    {
      Set(std::min<u8>(r_, 0x1F), std::min<u8>(g_, 0x1F), std::min<u8>(b_, 0x1F), c_);
    }

    void SetRGB24(u32 rgb24, bool c_ = false)
    {
      bits = Truncate16(((rgb24 >> 3) & 0x1F) | (((rgb24 >> 11) & 0x1F) << 5) | (((rgb24 >> 19) & 0x1F) << 10)) |
             (static_cast<u16>(c_) << 15);
    }

    void SetRGB24(u8 r8, u8 g8, u8 b8, bool c_ = false)
    {
      bits = (ZeroExtend16(r8 >> 3)) | (ZeroExtend16(g8 >> 3) << 5) | (ZeroExtend16(b8 >> 3) << 10) |
             (static_cast<u16>(c_) << 15);
    }

    void SetRGB24Dithered(u32 x, u32 y, u8 r8, u8 g8, u8 b8, bool c_ = false)
    {
      const s32 offset = DITHER_MATRIX[y & 3][x & 3];
      r8 = static_cast<u8>(std::clamp<s32>(static_cast<s32>(ZeroExtend32(r8)) + offset, 0, 255));
      g8 = static_cast<u8>(std::clamp<s32>(static_cast<s32>(ZeroExtend32(g8)) + offset, 0, 255));
      b8 = static_cast<u8>(std::clamp<s32>(static_cast<s32>(ZeroExtend32(b8)) + offset, 0, 255));
      SetRGB24(r8, g8, b8, c_);
    }

    u32 ToRGB24() const
    {
      const u32 r_ = ZeroExtend32(r.GetValue());
      const u32 g_ = ZeroExtend32(g.GetValue());
      const u32 b_ = ZeroExtend32(b.GetValue());

      return ((r_ << 3) | (r_ & 7)) | (((g_ << 3) | (g_ & 7)) << 8) | (((b_ << 3) | (b_ & 7)) << 16);
    }
  };

  static constexpr std::tuple<u8, u8> UnpackTexcoord(u16 texcoord)
  {
    return std::make_tuple(static_cast<u8>(texcoord), static_cast<u8>(texcoord >> 8));
  }

  static constexpr std::tuple<u8, u8, u8> UnpackColorRGB24(u32 rgb24)
  {
    return std::make_tuple(static_cast<u8>(rgb24), static_cast<u8>(rgb24 >> 8), static_cast<u8>(rgb24 >> 16));
  }

  //////////////////////////////////////////////////////////////////////////
  // Rasterization
  //////////////////////////////////////////////////////////////////////////
  template<bool texture_enable, bool raw_texture_enable, bool transparency_enable, bool dithering_enable>
  void ShadePixel(const GPUBackendDrawCommand* cmd, u32 x, u32 y, u8 color_r, u8 color_g, u8 color_b, u8 texcoord_x,
                  u8 texcoord_y);

  template<bool texture_enable, bool raw_texture_enable, bool transparency_enable>
  void DrawSprite(const GPUBackendDrawSpriteCommand* cmd);

  using DrawSpriteFunction = void (GPUSWBackend::*)(const GPUBackendDrawSpriteCommand* cmd);
  DrawSpriteFunction GetDrawSpriteFunction(bool texture_enable, bool raw_texture_enable, bool transparency_enable);

  //////////////////////////////////////////////////////////////////////////
  // Polygon and line rasterization ported from Mednafen
  //////////////////////////////////////////////////////////////////////////
  struct i_deltas
  {
    u32 du_dx, dv_dx;
    u32 dr_dx, dg_dx, db_dx;

    u32 du_dy, dv_dy;
    u32 dr_dy, dg_dy, db_dy;
  };

  struct i_group
  {
    u32 u, v;
    u32 r, g, b;
  };

  template<bool shading_enable, bool texture_enable>
  bool CalcIDeltas(i_deltas& idl, const GPUBackendDrawPolygonCommand::Vertex* A,
                   const GPUBackendDrawPolygonCommand::Vertex* B, const GPUBackendDrawPolygonCommand::Vertex* C);

  template<bool shading_enable, bool texture_enable>
  void AddIDeltas_DX(i_group& ig, const i_deltas& idl, u32 count = 1);

  template<bool shading_enable, bool texture_enable>
  void AddIDeltas_DY(i_group& ig, const i_deltas& idl, u32 count = 1);

  template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
           bool dithering_enable>
  void DrawSpan(const GPUBackendDrawCommand* cmd, s32 y, s32 x_start, s32 x_bound, i_group ig,
                const i_deltas& idl);

  template<bool shading_enable, bool texture_enable, bool raw_texture_enable, bool transparency_enable,
           bool dithering_enable>
  void DrawTriangle(const GPUBackendDrawCommand* cmd, const GPUBackendDrawPolygonCommand::Vertex* v0,
                    const GPUBackendDrawPolygonCommand::Vertex* v1, const GPUBackendDrawPolygonCommand::Vertex* v2);

  using DrawTriangleFunction = void (GPUSWBackend::*)(const GPUBackendDrawCommand* cmd,
                                                      const GPUBackendDrawPolygonCommand::Vertex* v0,
                                                      const GPUBackendDrawPolygonCommand::Vertex* v1,
                                                      const GPUBackendDrawPolygonCommand::Vertex* v2);
  DrawTriangleFunction GetDrawTriangleFunction(bool shading_enable, bool texture_enable, bool raw_texture_enable,
                                               bool transparency_enable, bool dithering_enable);

  template<bool shading_enable, bool transparency_enable, bool dithering_enable>
  void DrawLine(const GPUBackendDrawLineCommand* cmd, const GPUBackendDrawLineCommand::Vertex* p0,
                const GPUBackendDrawLineCommand::Vertex* p1);

  using DrawLineFunction = void (GPUSWBackend::*)(const GPUBackendDrawLineCommand* cmd,
                                                  const GPUBackendDrawLineCommand::Vertex* p0,
                                                  const GPUBackendDrawLineCommand::Vertex* p1);
  DrawLineFunction GetDrawLineFunction(bool shading_enable, bool transparency_enable, bool dithering_enable);

  template<GPUTexture::Format display_format>
  void CopyOut15Bit(u32 src_x, u32 src_y, u32 width, u32 height, u32 field, bool interlaced, bool interleaved);
  void CopyOut15Bit(GPUTexture::Format display_format, u32 src_x, u32 src_y, u32 width, u32 height, u32 field,
                    bool interlaced, bool interleaved);

  template<GPUTexture::Format display_format>
  void CopyOut24Bit(u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height, u32 field, bool interlaced,
                    bool interleaved);
  void CopyOut24Bit(GPUTexture::Format display_format, u32 src_x, u32 src_y, u32 skip_x, u32 width, u32 height,
                    u32 field, bool interlaced, bool interleaved);

  void SetDisplayTextureFormat();
  GPUTexture* GetDisplayTexture(u32 width, u32 height, GPUTexture::Format format);

  FixedHeapArray<u8, GPU_MAX_DISPLAY_WIDTH * GPU_MAX_DISPLAY_HEIGHT * sizeof(u32)> m_display_texture_buffer;
  GPUTexture::Format m_16bit_display_format = GPUTexture::Format::RGB565;
  GPUTexture::Format m_24bit_display_format = GPUTexture::Format::RGBA8;
  std::unique_ptr<GPUTexture> m_private_display_texture; // TODO: Move to base.
};
