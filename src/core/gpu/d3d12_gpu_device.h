// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "common/timer.h"
#include "common/window_info.h"
#include "common/windows_headers.h"
#include "d3d12/descriptor_heap_manager.h"
#include "d3d12/staging_texture.h"
#include "d3d12/stream_buffer.h"
#include "d3d12/texture.h"
#include "gpu_device.h"
#include "postprocessing_chain.h"
#include <d3d12.h>
#include <dxgi.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <wrl/client.h>

class D3D12GPUDevice final : public GPUDevice
{
public:
  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

  D3D12GPUDevice();
  ~D3D12GPUDevice();

  RenderAPI GetRenderAPI() const override;

  bool HasSurface() const override;

  bool CreateDevice(const WindowInfo& wi, bool vsync) override;
  bool SetupDevice() override;

  bool MakeCurrent() override;
  bool DoneCurrent() override;

  bool ChangeWindow(const WindowInfo& new_wi) override;
  void ResizeWindow(s32 new_window_width, s32 new_window_height) override;
  bool SupportsFullscreen() const override;
  bool IsFullscreen() override;
  bool SetFullscreen(bool fullscreen, u32 width, u32 height, float refresh_rate) override;
  AdapterAndModeList GetAdapterAndModeList() override;
  void DestroySurface() override;

  bool SetPostProcessingChain(const std::string_view& config) override;

  std::unique_ptr<GPUTexture> CreateTexture(u32 width, u32 height, u32 layers, u32 levels, u32 samples,
                                            GPUTexture::Type type, GPUTexture::Format format, const void* data,
                                            u32 data_stride, bool dynamic = false) override;
  bool DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                       u32 out_data_stride) override;
  bool SupportsTextureFormat(GPUTexture::Format format) const override;

  bool GetHostRefreshRate(float* refresh_rate) override;

  void SetVSync(bool enabled) override;

  bool Render(bool skip_present) override;

  bool SetGPUTimingEnabled(bool enabled) override;
  float GetAndResetAccumulatedGPUTime() override;

  static AdapterAndModeList StaticGetAdapterAndModeList();

protected:
  static AdapterAndModeList GetAdapterAndModeList(IDXGIFactory* dxgi_factory);

  virtual bool CreateResources() override;
  virtual void DestroyResources() override;

  bool CreateSwapChain(const DXGI_MODE_DESC* fullscreen_mode);
  bool CreateSwapChainRTV();
  void DestroySwapChainRTVs();

  void RenderDisplay(ID3D12GraphicsCommandList* cmdlist, D3D12::Texture* swap_chain_buf);
  void RenderSoftwareCursor(ID3D12GraphicsCommandList* cmdlist);
  void RenderImGui(ID3D12GraphicsCommandList* cmdlist);

  void RenderDisplay(ID3D12GraphicsCommandList* cmdlist, s32 left, s32 top, s32 width, s32 height,
                     D3D12::Texture* texture, s32 texture_view_x, s32 texture_view_y, s32 texture_view_width,
                     s32 texture_view_height, bool linear_filter);
  void RenderSoftwareCursor(ID3D12GraphicsCommandList* cmdlist, s32 left, s32 top, s32 width, s32 height,
                            GPUTexture* texture_handle);

  ComPtr<IDXGIFactory> m_dxgi_factory;
  ComPtr<IDXGISwapChain> m_swap_chain;
  std::vector<D3D12::Texture> m_swap_chain_buffers;
  u32 m_current_swap_chain_buffer = 0;

  ComPtr<ID3D12RootSignature> m_display_root_signature;
  ComPtr<ID3D12PipelineState> m_display_pipeline;
  ComPtr<ID3D12PipelineState> m_software_cursor_pipeline;
  D3D12::DescriptorHandle m_point_sampler;
  D3D12::DescriptorHandle m_linear_sampler;
  D3D12::DescriptorHandle m_border_sampler;

  D3D12::Texture m_display_pixels_texture;
  D3D12::StagingTexture m_readback_staging_texture;

  bool m_allow_tearing_supported = false;
  bool m_using_allow_tearing = false;
};
