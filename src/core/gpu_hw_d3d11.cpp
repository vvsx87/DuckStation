// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_hw_d3d11.h"
#include "common/assert.h"
#include "common/log.h"

Log_SetChannel(GPU_HW_D3D11);

GPU_HW_D3D11::GPU_HW_D3D11() = default;

GPU_HW_D3D11::~GPU_HW_D3D11() = default;

GPURenderer GPU_HW_D3D11::GetRendererType() const
{
  return GPURenderer::HardwareD3D11;
}

bool GPU_HW_D3D11::Initialize()
{
  SetCapabilities();

  if (!GPU_HW::Initialize())
    return false;

  if (!CreateTextureBuffer())
  {
    Log_ErrorPrintf("Failed to create texture buffer");
    return false;
  }

  RestoreGraphicsAPIState();
  return true;
}

void GPU_HW_D3D11::SetCapabilities()
{
  const u32 max_texture_size = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  const u32 max_texture_scale = max_texture_size / VRAM_WIDTH;

  ID3D11Device* device = D3D11Device::GetD3DDevice();

  m_max_resolution_scale = max_texture_scale;
  m_supports_dual_source_blend = true;
  m_supports_per_sample_shading = (device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_10_1);
  m_supports_adaptive_downsampling = true;
  m_supports_disable_color_perspective = true;

  m_max_multisamples = 1;
  for (u32 multisamples = 2; multisamples < D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT; multisamples++)
  {
    UINT num_quality_levels;
    if (SUCCEEDED(
          device->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, multisamples, &num_quality_levels)) &&
        num_quality_levels > 0)
    {
      m_max_multisamples = multisamples;
    }
  }
}

bool GPU_HW_D3D11::CreateTextureBuffer()
{
  ID3D11Device* device = D3D11Device::GetD3DDevice();

  if (!m_texture_stream_buffer.Create(device, D3D11_BIND_SHADER_RESOURCE, VRAM_UPDATE_TEXTURE_BUFFER_SIZE))
    return false;

  const CD3D11_SHADER_RESOURCE_VIEW_DESC srv_desc(D3D11_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R16_UINT, 0,
                                                  VRAM_UPDATE_TEXTURE_BUFFER_SIZE / sizeof(u16));
  const HRESULT hr = device->CreateShaderResourceView(m_texture_stream_buffer.GetD3DBuffer(), &srv_desc,
                                                      m_texture_stream_buffer_srv_r16ui.ReleaseAndGetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Creation of texture buffer SRV failed: 0x%08X", hr);
    return false;
  }

  return true;
}

void GPU_HW_D3D11::UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, bool set_mask, bool check_mask)
{
  if (IsUsingSoftwareRendererForReadbacks())
    UpdateSoftwareRendererVRAM(x, y, width, height, data, set_mask, check_mask);

  const Common::Rectangle<u32> bounds = GetVRAMTransferBounds(x, y, width, height);
  GPU_HW::UpdateVRAM(bounds.left, bounds.top, bounds.GetWidth(), bounds.GetHeight(), data, set_mask, check_mask);

  if (!check_mask)
  {
    const TextureReplacementTexture* rtex = g_texture_replacements.GetVRAMWriteReplacement(width, height, data);
    if (rtex && BlitVRAMReplacementTexture(rtex, x * m_resolution_scale, y * m_resolution_scale,
                                           width * m_resolution_scale, height * m_resolution_scale))
    {
      return;
    }
  }

  const u32 num_pixels = width * height;
  const auto map_result =
    m_texture_stream_buffer.Map(D3D11Device::GetD3DContext(), sizeof(u16), num_pixels * sizeof(u16));
  std::memcpy(map_result.pointer, data, num_pixels * sizeof(u16));
  m_texture_stream_buffer.Unmap(D3D11Device::GetD3DContext(), num_pixels * sizeof(u16));

  const VRAMWriteUBOData uniforms =
    GetVRAMWriteUBOData(x, y, width, height, map_result.index_aligned, set_mask, check_mask);

  // the viewport should already be set to the full vram, so just adjust the scissor
  const Common::Rectangle<u32> scaled_bounds = bounds * m_resolution_scale;
  g_host_display->SetScissor(scaled_bounds.left, scaled_bounds.top, scaled_bounds.GetWidth(),
                             scaled_bounds.GetHeight());
  g_host_display->SetPipeline(m_vram_write_pipelines[BoolToUInt8(check_mask && !m_pgxp_depth_buffer)].get());
  g_host_display->PushUniformBuffer(&uniforms, sizeof(uniforms));
  D3D11Device::GetD3DContext()->PSSetShaderResources(0, 1, m_texture_stream_buffer_srv_r16ui.GetAddressOf());
  g_host_display->Draw(3, 0);

  RestoreGraphicsAPIState();
}

std::unique_ptr<GPU> GPU::CreateHardwareD3D11Renderer()
{
  if (!Host::AcquireHostDisplay(RenderAPI::D3D11))
  {
    Log_ErrorPrintf("Host render API is incompatible");
    return nullptr;
  }

  std::unique_ptr<GPU_HW_D3D11> gpu(std::make_unique<GPU_HW_D3D11>());
  if (!gpu->Initialize())
    return nullptr;

  return gpu;
}
