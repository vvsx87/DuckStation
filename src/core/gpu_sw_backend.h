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

protected:
  bool DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display) override;
  void Reset(bool clear_vram) override;

  void ReadVRAM(u32 x, u32 y, u32 width, u32 height) override;

  void DrawPolygon(const GPUBackendDrawPolygonCommand* cmd) override;
  void DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd) override;
  void DrawLine(const GPUBackendDrawLineCommand* cmd) override;
  void DrawSprite(const GPUBackendDrawSpriteCommand* cmd) override;
  void FlushRender() override;
  void DrawingAreaChanged(const Common::Rectangle<u32>& new_drawing_area) override;

  void ClearDisplay() override;
  void UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd) override;

private:
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
