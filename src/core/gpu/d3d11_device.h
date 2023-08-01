// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
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
#include <d3d11_1.h>
#include <dxgi.h>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <wrl/client.h>

class D3D11Device;

class D3D11Framebuffer final : public GPUFramebuffer
{
  friend D3D11Device;

  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

public:
  ~D3D11Framebuffer() override;

  ALWAYS_INLINE u32 GetNumRTVs() const { return m_rtv ? 1 : 0; }
  ALWAYS_INLINE ID3D11RenderTargetView* GetRTV() const { return m_rtv.Get(); }
  ALWAYS_INLINE ID3D11RenderTargetView* const* GetRTVArray() const { return m_rtv.GetAddressOf(); }
  ALWAYS_INLINE ID3D11DepthStencilView* GetDSV() const { return m_dsv.Get(); }

  void SetDebugName(const std::string_view& name) override;
  void CommitClear(ID3D11DeviceContext* context);

private:
  D3D11Framebuffer(GPUTexture* rt, GPUTexture* ds, u32 width, u32 height, ComPtr<ID3D11RenderTargetView> rtv,
                   ComPtr<ID3D11DepthStencilView> dsv);

  ComPtr<ID3D11RenderTargetView> m_rtv;
  ComPtr<ID3D11DepthStencilView> m_dsv;
};

class D3D11Sampler final : public GPUSampler
{
  friend D3D11Device;

  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

public:
  ~D3D11Sampler() override;

  ALWAYS_INLINE ID3D11SamplerState* GetSamplerState() const { return m_ss.Get(); }
  ALWAYS_INLINE ID3D11SamplerState* const* GetSamplerStateArray() const { return m_ss.GetAddressOf(); }

  void SetDebugName(const std::string_view& name) override;

private:
  D3D11Sampler(ComPtr<ID3D11SamplerState> ss);

  ComPtr<ID3D11SamplerState> m_ss;
};

class D3D11Shader final : public GPUShader
{
  friend D3D11Device;

public:
  ~D3D11Shader() override;

  ID3D11VertexShader* GetVertexShader() const;
  ID3D11PixelShader* GetPixelShader() const;
  ID3D11ComputeShader* GetComputeShader() const;

  ALWAYS_INLINE const std::vector<u8>& GetBytecode() const { return m_bytecode; }

  void SetDebugName(const std::string_view& name) override;

private:
  D3D11Shader(GPUShaderStage stage, Microsoft::WRL::ComPtr<ID3D11DeviceChild> shader, std::vector<u8> bytecode);

  Microsoft::WRL::ComPtr<ID3D11DeviceChild> m_shader;
  std::vector<u8> m_bytecode; // only for VS
};

class D3D11Pipeline final : public GPUPipeline
{
  friend D3D11Device;

  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

public:
  ~D3D11Pipeline() override;

  void SetDebugName(const std::string_view& name) override;

  ALWAYS_INLINE ID3D11RasterizerState* GetRasterizerState() const { return m_rs.Get(); }
  ALWAYS_INLINE ID3D11DepthStencilState* GetDepthStencilState() const { return m_ds.Get(); }
  ALWAYS_INLINE ID3D11BlendState* GetBlendState() const { return m_bs.Get(); }
  ALWAYS_INLINE ID3D11InputLayout* GetInputLayout() const { return m_il.Get(); }
  ALWAYS_INLINE ID3D11VertexShader* GetVertexShader() const { return m_vs.Get(); }
  ALWAYS_INLINE ID3D11PixelShader* GetPixelShader() const { return m_ps.Get(); }
  ALWAYS_INLINE D3D11_PRIMITIVE_TOPOLOGY GetPrimitiveTopology() const { return m_topology; }

  void Bind(ID3D11DeviceContext* context);

private:
  D3D11Pipeline(ComPtr<ID3D11RasterizerState> rs, ComPtr<ID3D11DepthStencilState> ds, ComPtr<ID3D11BlendState> bs,
                ComPtr<ID3D11InputLayout> il, ComPtr<ID3D11VertexShader> vs, ComPtr<ID3D11PixelShader> ps,
                D3D11_PRIMITIVE_TOPOLOGY topology);

  ComPtr<ID3D11RasterizerState> m_rs;
  ComPtr<ID3D11DepthStencilState> m_ds;
  ComPtr<ID3D11BlendState> m_bs;
  ComPtr<ID3D11InputLayout> m_il;
  ComPtr<ID3D11VertexShader> m_vs;
  ComPtr<ID3D11PixelShader> m_ps;
  D3D11_PRIMITIVE_TOPOLOGY m_topology;
};

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

  std::string GetShaderCacheBaseName(const std::string_view& type, bool debug) const override;

  std::unique_ptr<GPUTexture> CreateTexture(u32 width, u32 height, u32 layers, u32 levels, u32 samples,
                                            GPUTexture::Type type, GPUTexture::Format format,
                                            const void* data = nullptr, u32 data_stride = 0,
                                            bool dynamic = false) override;
  std::unique_ptr<GPUSampler> CreateSampler(const GPUSampler::Config& config) override;

  bool DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                       u32 out_data_stride) override;
  bool SupportsTextureFormat(GPUTexture::Format format) const override;
  void CopyTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                         u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height) override;
  void ResolveTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                            u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height) override;

  std::unique_ptr<GPUFramebuffer> CreateFramebuffer(GPUTexture* rt = nullptr, u32 rt_layer = 0, u32 rt_level = 0,
                                                    GPUTexture* ds = nullptr, u32 ds_layer = 0,
                                                    u32 ds_level = 0) override;

  std::unique_ptr<GPUShader> CreateShaderFromBinary(GPUShaderStage stage, gsl::span<const u8> data) override;
  std::unique_ptr<GPUShader> CreateShaderFromSource(GPUShaderStage stage, const std::string_view& source,
                                                    std::vector<u8>* out_binary = nullptr) override;
  std::unique_ptr<GPUPipeline> CreatePipeline(const GPUPipeline::GraphicsConfig& config) override;

  void PushDebugGroup(const char* fmt, ...) override;
  void PopDebugGroup() override;
  void InsertDebugMessage(const char* fmt, ...) override;

  void MapVertexBuffer(u32 vertex_size, u32 vertex_count, void** map_ptr, u32* map_space,
                       u32* map_base_vertex) override;
  void UnmapVertexBuffer(u32 vertex_size, u32 vertex_count) override;
  void MapIndexBuffer(u32 index_count, DrawIndex** map_ptr, u32* map_space, u32* map_base_index) override;
  void UnmapIndexBuffer(u32 used_index_count) override;
  void PushUniformBuffer(const void* data, u32 data_size) override;
  void SetFramebuffer(GPUFramebuffer* fb) override;
  void SetPipeline(GPUPipeline* pipeline) override;
  void SetTextureSampler(u32 slot, GPUTexture* texture, GPUSampler* sampler) override;
  void SetViewport(s32 x, s32 y, s32 width, s32 height) override;
  void SetScissor(s32 x, s32 y, s32 width, s32 height) override;
  void Draw(u32 vertex_count, u32 base_vertex) override;
  void DrawIndexed(u32 index_count, u32 base_index, u32 base_vertex) override;

  bool GetHostRefreshRate(float* refresh_rate) override;

  bool SetGPUTimingEnabled(bool enabled) override;
  float GetAndResetAccumulatedGPUTime() override;

  void SetVSync(bool enabled) override;

  bool Render(bool skip_present) override;

  void UnbindFramebuffer(D3D11Framebuffer* fb);
  void UnbindPipeline(D3D11Pipeline* pl);
  void UnbindTexture(D3D11Texture* tex);

  static AdapterAndModeList StaticGetAdapterAndModeList();

private:
  using RasterizationStateMap = std::unordered_map<u8, ComPtr<ID3D11RasterizerState>>;
  using DepthStateMap = std::unordered_map<u8, ComPtr<ID3D11DepthStencilState>>;
  using BlendStateMap = std::unordered_map<u64, ComPtr<ID3D11BlendState>>;
  using InputLayoutMap =
    std::unordered_map<GPUPipeline::InputLayout, ComPtr<ID3D11InputLayout>, GPUPipeline::InputLayoutHash>;

  static constexpr u32 PUSH_UNIFORM_BUFFER_SIZE = 64;
  static constexpr u32 VERTEX_BUFFER_SIZE = 8 * 1024 * 1024;
  static constexpr u32 INDEX_BUFFER_SIZE = 4 * 1024 * 1024;
  static constexpr u8 NUM_TIMESTAMP_QUERIES = 3;

  static AdapterAndModeList GetAdapterAndModeList(IDXGIFactory* dxgi_factory);

  void CommitClear(GPUTexture* t);
  void PreDrawCheck();

  bool CheckStagingBufferSize(u32 width, u32 height, DXGI_FORMAT format);
  void DestroyStagingBuffer();

  bool CreateSwapChain(const DXGI_MODE_DESC* fullscreen_mode);
  bool CreateSwapChainRTV();

  bool CreateBuffers();
  void DestroyBuffers();

  ComPtr<ID3D11RasterizerState> GetRasterizationState(const GPUPipeline::RasterizationState& rs);
  ComPtr<ID3D11DepthStencilState> GetDepthState(const GPUPipeline::DepthState& ds);
  ComPtr<ID3D11BlendState> GetBlendState(const GPUPipeline::BlendState& bs);
  ComPtr<ID3D11InputLayout> GetInputLayout(const GPUPipeline::InputLayout& il, const D3D11Shader* vs);

  bool CreateTimestampQueries();
  void DestroyTimestampQueries();
  void PopTimestampQuery();
  void KickTimestampQuery();

  ComPtr<ID3D11Device> m_device;
  ComPtr<ID3D11DeviceContext> m_context;
  ComPtr<ID3DUserDefinedAnnotation> m_annotation;

  ComPtr<IDXGIFactory> m_dxgi_factory;
  ComPtr<IDXGISwapChain> m_swap_chain;
  ComPtr<ID3D11RenderTargetView> m_swap_chain_rtv;

  RasterizationStateMap m_rasterization_states;
  DepthStateMap m_depth_states;
  BlendStateMap m_blend_states;
  InputLayoutMap m_input_layouts;

  ComPtr<ID3D11Texture2D> m_readback_staging_texture;
  DXGI_FORMAT m_readback_staging_texture_format = DXGI_FORMAT_UNKNOWN;
  u32 m_readback_staging_texture_width = 0;
  u32 m_readback_staging_texture_height = 0;

  bool m_allow_tearing_supported = false;
  bool m_using_flip_model_swap_chain = true;
  bool m_using_allow_tearing = false;

  D3D11::StreamBuffer m_vertex_buffer;
  D3D11::StreamBuffer m_index_buffer;
  D3D11::StreamBuffer m_push_uniform_buffer;

  D3D11Framebuffer* m_current_framebuffer = nullptr;
  D3D11Pipeline* m_current_pipeline = nullptr;

  std::array<std::array<ComPtr<ID3D11Query>, 3>, NUM_TIMESTAMP_QUERIES> m_timestamp_queries = {};
  u8 m_read_timestamp_query = 0;
  u8 m_write_timestamp_query = 0;
  u8 m_waiting_timestamp_queries = 0;
  bool m_timestamp_query_started = false;
  float m_accumulated_gpu_time = 0.0f;
};
