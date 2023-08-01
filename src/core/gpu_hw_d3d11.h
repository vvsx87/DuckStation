// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "gpu/d3d11/shader_cache.h"
#include "gpu/d3d11/stream_buffer.h"
#include "gpu/d3d11_texture.h"
#include "gpu_hw.h"
#include <array>
#include <d3d11.h>
#include <memory>
#include <tuple>
#include <wrl/client.h>

class GPU_HW_D3D11 final : public GPU_HW
{
public:
  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

  GPU_HW_D3D11();
  ~GPU_HW_D3D11() override;

  GPURenderer GetRendererType() const override;

  bool Initialize() override;

protected:
  void UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, bool set_mask, bool check_mask) override;

private:
  void SetCapabilities();

  bool CreateTextureBuffer();

  D3D11::StreamBuffer m_texture_stream_buffer;

  ComPtr<ID3D11ShaderResourceView> m_texture_stream_buffer_srv_r16ui;
};
