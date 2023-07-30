// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "common/timer.h"
#include "common/window_info.h"
#include "common/windows_headers.h"
#include "d3d11/stream_buffer.h"
#include "d3d11_texture.h"
#include "gpu_device.h"
#include "postprocessing_chain.h"
#include <d3d11.h>
#include <dxgi.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <wrl/client.h>

class D3D11Device final : public GPUDevice
{
public:
  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

  ALWAYS_INLINE static D3D11Device& GetInstance() { return *static_cast<D3D11Device*>(g_host_display.get()); }
  ALWAYS_INLINE static ID3D11Device* GetD3DDevice() { return GetInstance().m_device.Get(); }
  ALWAYS_INLINE static ID3D11DeviceContext* GetD3DContext() { return GetInstance().m_context.Get(); }

  D3D11Device();
  ~D3D11Device();

  RenderAPI GetRenderAPI() const override;
  void* GetDevice() const override;
  void* GetContext() const override;

  bool HasDevice() const override;
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
                                            GPUTexture::Type type, GPUTexture::Format format,
                                            const void* data = nullptr, u32 data_stride = 0,
                                            bool dynamic = false) override;
  bool DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                       u32 out_data_stride) override;
  bool SupportsTextureFormat(GPUTexture::Format format) const override;
  void CopyTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                         u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height) override;
  void ResolveTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                            u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height) override;

  bool GetHostRefreshRate(float* refresh_rate) override;

  bool SetGPUTimingEnabled(bool enabled) override;
  float GetAndResetAccumulatedGPUTime() override;

  void SetVSync(bool enabled) override;

  bool Render(bool skip_present) override;
  bool RenderScreenshot(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect, std::vector<u32>* out_pixels,
                        u32* out_stride, GPUTexture::Format* out_format) override;

  static AdapterAndModeList StaticGetAdapterAndModeList();

protected:
  static constexpr u32 DISPLAY_UNIFORM_BUFFER_SIZE = 64;
  static constexpr u32 IMGUI_VERTEX_BUFFER_SIZE = 4 * 1024 * 1024;
  static constexpr u32 IMGUI_INDEX_BUFFER_SIZE = 2 * 1024 * 1024;
  static constexpr u8 NUM_TIMESTAMP_QUERIES = 3;

  static AdapterAndModeList GetAdapterAndModeList(IDXGIFactory* dxgi_factory);

  bool CheckStagingBufferSize(u32 width, u32 height, DXGI_FORMAT format);
  void DestroyStagingBuffer();

  bool CreateResources() override;
  void DestroyResources() override;

  bool CreateImGuiResources();
  void DestroyImGuiResources();

  bool CreateSwapChain(const DXGI_MODE_DESC* fullscreen_mode);
  bool CreateSwapChainRTV();

  void RenderDisplay();
  void RenderSoftwareCursor();
  void RenderImGui();

  void RenderDisplay(s32 left, s32 top, s32 width, s32 height, D3D11Texture* texture, s32 texture_view_x,
                     s32 texture_view_y, s32 texture_view_width, s32 texture_view_height, bool linear_filter);
  void RenderSoftwareCursor(s32 left, s32 top, s32 width, s32 height, GPUTexture* texture_handle);

  struct PostProcessingStage
  {
    ComPtr<ID3D11VertexShader> vertex_shader;
    ComPtr<ID3D11PixelShader> pixel_shader;
    D3D11Texture output_texture;
    u32 uniforms_size;
  };

  bool CheckPostProcessingRenderTargets(u32 target_width, u32 target_height);
  void ApplyPostProcessingChain(ID3D11RenderTargetView* final_target, s32 final_left, s32 final_top, s32 final_width,
                                s32 final_height, D3D11Texture* texture, s32 texture_view_x, s32 texture_view_y,
                                s32 texture_view_width, s32 texture_view_height, u32 target_width, u32 target_height);

  bool CreateTimestampQueries();
  void DestroyTimestampQueries();
  void PopTimestampQuery();
  void KickTimestampQuery();

  ComPtr<ID3D11Device> m_device;
  ComPtr<ID3D11DeviceContext> m_context;

  ComPtr<IDXGIFactory> m_dxgi_factory;
  ComPtr<IDXGISwapChain> m_swap_chain;
  ComPtr<ID3D11RenderTargetView> m_swap_chain_rtv;

  ComPtr<ID3D11RasterizerState> m_display_rasterizer_state;
  ComPtr<ID3D11DepthStencilState> m_display_depth_stencil_state;
  ComPtr<ID3D11BlendState> m_display_blend_state;
  ComPtr<ID3D11BlendState> m_software_cursor_blend_state;
  ComPtr<ID3D11VertexShader> m_display_vertex_shader;
  ComPtr<ID3D11PixelShader> m_display_pixel_shader;
  ComPtr<ID3D11PixelShader> m_display_alpha_pixel_shader;
  ComPtr<ID3D11SamplerState> m_point_sampler;
  ComPtr<ID3D11SamplerState> m_linear_sampler;
  ComPtr<ID3D11SamplerState> m_border_sampler;

  D3D11::StreamBuffer m_display_uniform_buffer;
  ComPtr<ID3D11Texture2D> m_readback_staging_texture;
  DXGI_FORMAT m_readback_staging_texture_format = DXGI_FORMAT_UNKNOWN;
  u32 m_readback_staging_texture_width = 0;
  u32 m_readback_staging_texture_height = 0;

  bool m_allow_tearing_supported = false;
  bool m_using_flip_model_swap_chain = true;
  bool m_using_allow_tearing = false;

  D3D11Texture m_imgui_texture;
  D3D11::StreamBuffer m_imgui_vertex_buffer;
  D3D11::StreamBuffer m_imgui_index_buffer;
  ComPtr<ID3D11InputLayout> m_imgui_input_layout;
  ComPtr<ID3D11VertexShader> m_imgui_vertex_shader;
  ComPtr<ID3D11PixelShader> m_imgui_pixel_shader;
  ComPtr<ID3D11BlendState> m_imgui_blend_state;

  FrontendCommon::PostProcessingChain m_post_processing_chain;
  D3D11Texture m_post_processing_input_texture;
  std::vector<PostProcessingStage> m_post_processing_stages;
  Common::Timer m_post_processing_timer;

  std::array<std::array<ComPtr<ID3D11Query>, 3>, NUM_TIMESTAMP_QUERIES> m_timestamp_queries = {};
  u8 m_read_timestamp_query = 0;
  u8 m_write_timestamp_query = 0;
  u8 m_waiting_timestamp_queries = 0;
  bool m_timestamp_query_started = false;
  float m_accumulated_gpu_time = 0.0f;
};
