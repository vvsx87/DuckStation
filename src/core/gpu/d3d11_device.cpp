// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "d3d11_device.h"
#include "../host_settings.h"
#include "../settings.h"
#include "../shader_cache_version.h"

#include "common/assert.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/path.h"
#include "common/string_util.h"

#include "imgui.h"

#include "fmt/format.h"

#include <array>
#include <d3dcompiler.h>
#include <dxgi1_5.h>

Log_SetChannel(D3D11Device);

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

static constexpr std::array<float, 4> s_clear_color = {};
static unsigned s_next_bad_shader_id = 1;

static void SetD3DDebugObjectName(ID3D11DeviceChild* obj, const std::string_view& name)
{
  // WKPDID_D3DDebugObjectName
  static constexpr GUID guid = {0x429b8c22, 0x9188, 0x4b0c, 0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00};
  const std::wstring wname = StringUtil::UTF8StringToWideString(name);
  obj->SetPrivateData(guid, static_cast<UINT>(wname.length()) * 2u, wname.c_str());
}

D3D11Device::D3D11Device() = default;

D3D11Device::~D3D11Device()
{
  // TODO: Make virtual Destroy() method instead due to order of shit..
  DestroyStagingBuffer();
  DestroyResources();
  DestroyBuffers();
  DestroySurface();
  m_context.Reset();
  m_device.Reset();
}

RenderAPI D3D11Device::GetRenderAPI() const
{
  return RenderAPI::D3D11;
}

void* D3D11Device::GetDevice() const
{
  return m_device.Get();
}

void* D3D11Device::GetContext() const
{
  return m_context.Get();
}

bool D3D11Device::HasDevice() const
{
  return static_cast<bool>(m_device);
}

bool D3D11Device::HasSurface() const
{
  return static_cast<bool>(m_swap_chain);
}

std::unique_ptr<GPUTexture> D3D11Device::CreateTexture(u32 width, u32 height, u32 layers, u32 levels, u32 samples,
                                                       GPUTexture::Type type, GPUTexture::Format format,
                                                       const void* data, u32 data_stride, bool dynamic /* = false */)
{
  std::unique_ptr<D3D11Texture> tex = std::make_unique<D3D11Texture>();
  if (!tex->Create(m_device.Get(), width, height, layers, levels, samples, type, format, data, data_stride, dynamic))
    tex.reset();

  return tex;
}

bool D3D11Device::DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                                  u32 out_data_stride)
{
  const D3D11Texture* tex = static_cast<const D3D11Texture*>(texture);
  if (!CheckStagingBufferSize(width, height, tex->GetDXGIFormat()))
    return false;

  const CD3D11_BOX box(static_cast<LONG>(x), static_cast<LONG>(y), 0, static_cast<LONG>(x + width),
                       static_cast<LONG>(y + height), 1);
  m_context->CopySubresourceRegion(m_readback_staging_texture.Get(), 0, 0, 0, 0, tex->GetD3DTexture(), 0, &box);

  D3D11_MAPPED_SUBRESOURCE sr;
  HRESULT hr = m_context->Map(m_readback_staging_texture.Get(), 0, D3D11_MAP_READ, 0, &sr);
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Map() failed with HRESULT %08X", hr);
    return false;
  }

  const u32 copy_size = tex->GetPixelSize() * width;
  StringUtil::StrideMemCpy(out_data, out_data_stride, sr.pData, sr.RowPitch, copy_size, height);
  m_context->Unmap(m_readback_staging_texture.Get(), 0);
  return true;
}

bool D3D11Device::CheckStagingBufferSize(u32 width, u32 height, DXGI_FORMAT format)
{
  if (m_readback_staging_texture_width >= width && m_readback_staging_texture_width >= height &&
      m_readback_staging_texture_format == format)
    return true;

  DestroyStagingBuffer();

  CD3D11_TEXTURE2D_DESC desc(format, width, height, 1, 1, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);
  HRESULT hr = m_device->CreateTexture2D(&desc, nullptr, m_readback_staging_texture.ReleaseAndGetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("CreateTexture2D() failed with HRESULT %08X", hr);
    return false;
  }

  return true;
}

void D3D11Device::DestroyStagingBuffer()
{
  m_readback_staging_texture.Reset();
  m_readback_staging_texture_width = 0;
  m_readback_staging_texture_height = 0;
  m_readback_staging_texture_format = DXGI_FORMAT_UNKNOWN;
}

bool D3D11Device::SupportsTextureFormat(GPUTexture::Format format) const
{
  const DXGI_FORMAT dfmt = D3D11Texture::GetDXGIFormat(format);
  if (dfmt == DXGI_FORMAT_UNKNOWN)
    return false;

  UINT support = 0;
  const UINT required = D3D11_FORMAT_SUPPORT_TEXTURE2D | D3D11_FORMAT_SUPPORT_SHADER_SAMPLE;
  return (SUCCEEDED(m_device->CheckFormatSupport(dfmt, &support)) && ((support & required) == required));
}

void D3D11Device::CopyTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level,
                                    GPUTexture* src, u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width,
                                    u32 height)
{
  DebugAssert(src_level < src->GetLevels() && src_layer < src->GetLayers());
  DebugAssert((src_x + width) <= src->GetMipWidth(src_level));
  DebugAssert((src_y + height) <= src->GetMipWidth(src_level));
  DebugAssert(dst_level < dst->GetLevels() && dst_layer < dst->GetLayers());
  DebugAssert((dst_x + width) <= dst->GetMipWidth(dst_level));
  DebugAssert((dst_y + height) <= dst->GetMipWidth(dst_level));

  D3D11Texture* dst11 = static_cast<D3D11Texture*>(dst);
  D3D11Texture* src11 = static_cast<D3D11Texture*>(src);

  if (dst11->IsRenderTargetOrDepthStencil())
  {
    if (src11->GetState() == GPUTexture::State::Cleared)
    {
      if (src11->GetWidth() == dst11->GetWidth() && src11->GetHeight() == dst11->GetHeight())
      {
        // pass clear through
        dst11->m_state = src11->m_state;
        dst11->m_clear_value = src11->m_clear_value;
        return;
      }
    }
    else if (dst_x == 0 && dst_y == 0 && width == dst11->GetMipWidth(dst_level) &&
             height == dst11->GetMipHeight(dst_level))
    {
      // TODO: 11.1 discard
      dst11->SetState(GPUTexture::State::Dirty);
    }

    dst11->CommitClear(m_context.Get());
  }

  src11->CommitClear(m_context.Get());

  const CD3D11_BOX src_box(static_cast<LONG>(src_x), static_cast<LONG>(src_y), 0, static_cast<LONG>(src_x + width),
                           static_cast<LONG>(src_y + height), 1);
  m_context->CopySubresourceRegion(dst11->GetD3DTexture(), D3D11CalcSubresource(dst_level, dst_layer, dst->GetLevels()),
                                   dst_x, dst_y, 0, src11->GetD3DTexture(),
                                   D3D11CalcSubresource(src_level, src_layer, src->GetLevels()), &src_box);
}

void D3D11Device::ResolveTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level,
                                       GPUTexture* src, u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width,
                                       u32 height)
{
  DebugAssert(src_level < src->GetLevels() && src_layer < src->GetLayers());
  DebugAssert((src_x + width) <= src->GetMipWidth(src_level));
  DebugAssert((src_y + height) <= src->GetMipWidth(src_level));
  DebugAssert(dst_level < dst->GetLevels() && dst_layer < dst->GetLayers());
  DebugAssert((dst_x + width) <= dst->GetMipWidth(dst_level));
  DebugAssert((dst_y + height) <= dst->GetMipWidth(dst_level));
  DebugAssert(!dst->IsMultisampled() && src->IsMultisampled());

  // DX11 can't resolve partial rects.
  Assert(src_x == dst_x && src_y == dst_y);

  D3D11Texture* dst11 = static_cast<D3D11Texture*>(dst);
  D3D11Texture* src11 = static_cast<D3D11Texture*>(src);

  src11->CommitClear(m_context.Get());
  dst11->CommitClear(m_context.Get());

  m_context->ResolveSubresource(dst11->GetD3DTexture(), D3D11CalcSubresource(dst_level, dst_layer, dst->GetLevels()),
                                src11->GetD3DTexture(), D3D11CalcSubresource(src_level, src_layer, src->GetLevels()),
                                dst11->GetDXGIFormat());
}

bool D3D11Device::GetHostRefreshRate(float* refresh_rate)
{
  if (m_swap_chain && IsFullscreen())
  {
    DXGI_SWAP_CHAIN_DESC desc;
    if (SUCCEEDED(m_swap_chain->GetDesc(&desc)) && desc.BufferDesc.RefreshRate.Numerator > 0 &&
        desc.BufferDesc.RefreshRate.Denominator > 0)
    {
      Log_InfoPrintf("using fs rr: %u %u", desc.BufferDesc.RefreshRate.Numerator,
                     desc.BufferDesc.RefreshRate.Denominator);
      *refresh_rate = static_cast<float>(desc.BufferDesc.RefreshRate.Numerator) /
                      static_cast<float>(desc.BufferDesc.RefreshRate.Denominator);
      return true;
    }
  }

  return GPUDevice::GetHostRefreshRate(refresh_rate);
}

void D3D11Device::SetVSync(bool enabled)
{
  m_vsync_enabled = enabled;
}

bool D3D11Device::CreateDevice(const WindowInfo& wi, bool vsync)
{
  UINT create_flags = 0;
  if (g_settings.gpu_use_debug_device)
    create_flags |= D3D11_CREATE_DEVICE_DEBUG;

  ComPtr<IDXGIFactory> temp_dxgi_factory;
  HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(temp_dxgi_factory.GetAddressOf()));
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Failed to create DXGI factory: 0x%08X", hr);
    return false;
  }

  u32 adapter_index;
  if (!g_settings.gpu_adapter.empty())
  {
    AdapterAndModeList adapter_info(GetAdapterAndModeList(temp_dxgi_factory.Get()));
    for (adapter_index = 0; adapter_index < static_cast<u32>(adapter_info.adapter_names.size()); adapter_index++)
    {
      if (g_settings.gpu_adapter == adapter_info.adapter_names[adapter_index])
        break;
    }
    if (adapter_index == static_cast<u32>(adapter_info.adapter_names.size()))
    {
      Log_WarningPrintf("Could not find adapter '%s', using first (%s)", g_settings.gpu_adapter.c_str(),
                        adapter_info.adapter_names[0].c_str());
      adapter_index = 0;
    }
  }
  else
  {
    Log_InfoPrintf("No adapter selected, using first.");
    adapter_index = 0;
  }

  ComPtr<IDXGIAdapter> dxgi_adapter;
  hr = temp_dxgi_factory->EnumAdapters(adapter_index, dxgi_adapter.GetAddressOf());
  if (FAILED(hr))
    Log_WarningPrintf("Failed to enumerate adapter %u, using default", adapter_index);

  static constexpr std::array<D3D_FEATURE_LEVEL, 3> requested_feature_levels = {
    {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0}};

  hr =
    D3D11CreateDevice(dxgi_adapter.Get(), dxgi_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, nullptr,
                      create_flags, requested_feature_levels.data(), static_cast<UINT>(requested_feature_levels.size()),
                      D3D11_SDK_VERSION, m_device.GetAddressOf(), nullptr, m_context.GetAddressOf());

  // we re-grab these later, see below
  dxgi_adapter.Reset();
  temp_dxgi_factory.Reset();

  if (FAILED(hr))
  {
    Log_ErrorPrintf("Failed to create D3D device: 0x%08X", hr);
    return false;
  }

  if (g_settings.gpu_use_debug_device && IsDebuggerPresent())
  {
    ComPtr<ID3D11InfoQueue> info;
    hr = m_device.As(&info);
    if (SUCCEEDED(hr))
    {
      info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, TRUE);
      info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_WARNING, TRUE);
    }
  }

  if (g_settings.gpu_use_debug_device)
    m_context.As(&m_annotation);

  // we need the specific factory for the device, otherwise MakeWindowAssociation() is flaky.
  ComPtr<IDXGIDevice> dxgi_device;
  if (FAILED(m_device.As(&dxgi_device)) || FAILED(dxgi_device->GetParent(IID_PPV_ARGS(dxgi_adapter.GetAddressOf()))) ||
      FAILED(dxgi_adapter->GetParent(IID_PPV_ARGS(m_dxgi_factory.GetAddressOf()))))
  {
    Log_WarningPrint("Failed to get parent adapter/device/factory");
    return false;
  }
  ComPtr<IDXGIDevice1> dxgi_device1;
  if (SUCCEEDED(dxgi_device.As(&dxgi_device1)))
    dxgi_device1->SetMaximumFrameLatency(1);

  DXGI_ADAPTER_DESC adapter_desc;
  if (SUCCEEDED(dxgi_adapter->GetDesc(&adapter_desc)))
  {
    char adapter_name_buffer[128];
    const int name_length =
      WideCharToMultiByte(CP_UTF8, 0, adapter_desc.Description, static_cast<int>(std::wcslen(adapter_desc.Description)),
                          adapter_name_buffer, countof(adapter_name_buffer), 0, nullptr);
    if (name_length >= 0)
    {
      adapter_name_buffer[name_length] = 0;
      Log_InfoPrintf("D3D Adapter: %s", adapter_name_buffer);
    }
  }

  m_allow_tearing_supported = false;
  ComPtr<IDXGIFactory5> dxgi_factory5;
  hr = m_dxgi_factory.As(&dxgi_factory5);
  if (SUCCEEDED(hr))
  {
    BOOL allow_tearing_supported = false;
    hr = dxgi_factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing_supported,
                                            sizeof(allow_tearing_supported));
    if (SUCCEEDED(hr))
      m_allow_tearing_supported = (allow_tearing_supported == TRUE);
  }

  m_window_info = wi;
  m_vsync_enabled = vsync;

  if (m_window_info.type != WindowInfo::Type::Surfaceless && !CreateSwapChain(nullptr))
  {
    m_window_info = {};
    return false;
  }

  return true;
}

bool D3D11Device::SetupDevice()
{
  if (!GPUDevice::SetupDevice())
    return false;

  if (!CreateBuffers() || !CreateResources())
    return false;

  return true;
}

bool D3D11Device::MakeCurrent()
{
  return true;
}

bool D3D11Device::DoneCurrent()
{
  return true;
}

bool D3D11Device::CreateSwapChain(const DXGI_MODE_DESC* fullscreen_mode)
{
  HRESULT hr;

  if (m_window_info.type != WindowInfo::Type::Win32)
    return false;

  m_using_flip_model_swap_chain = fullscreen_mode || !Host::GetBoolSettingValue("Display", "UseBlitSwapChain", false);

  const HWND window_hwnd = reinterpret_cast<HWND>(m_window_info.window_handle);
  RECT client_rc{};
  GetClientRect(window_hwnd, &client_rc);
  const u32 width = static_cast<u32>(client_rc.right - client_rc.left);
  const u32 height = static_cast<u32>(client_rc.bottom - client_rc.top);

  DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
  swap_chain_desc.BufferDesc.Width = width;
  swap_chain_desc.BufferDesc.Height = height;
  swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swap_chain_desc.SampleDesc.Count = 1;
  swap_chain_desc.BufferCount = 2;
  swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swap_chain_desc.OutputWindow = window_hwnd;
  swap_chain_desc.Windowed = TRUE;
  swap_chain_desc.SwapEffect = m_using_flip_model_swap_chain ? DXGI_SWAP_EFFECT_FLIP_DISCARD : DXGI_SWAP_EFFECT_DISCARD;

  m_using_allow_tearing = (m_allow_tearing_supported && m_using_flip_model_swap_chain && !fullscreen_mode);
  if (m_using_allow_tearing)
    swap_chain_desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

  if (fullscreen_mode)
  {
    swap_chain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    swap_chain_desc.Windowed = FALSE;
    swap_chain_desc.BufferDesc = *fullscreen_mode;
  }

  Log_InfoPrintf("Creating a %dx%d %s %s swap chain", swap_chain_desc.BufferDesc.Width,
                 swap_chain_desc.BufferDesc.Height, m_using_flip_model_swap_chain ? "flip-discard" : "discard",
                 swap_chain_desc.Windowed ? "windowed" : "full-screen");

  hr = m_dxgi_factory->CreateSwapChain(m_device.Get(), &swap_chain_desc, m_swap_chain.GetAddressOf());
  if (FAILED(hr) && m_using_flip_model_swap_chain)
  {
    Log_WarningPrintf("Failed to create a flip-discard swap chain, trying discard.");
    swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swap_chain_desc.Flags = 0;
    m_using_flip_model_swap_chain = false;
    m_using_allow_tearing = false;

    hr = m_dxgi_factory->CreateSwapChain(m_device.Get(), &swap_chain_desc, m_swap_chain.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("CreateSwapChain failed: 0x%08X", hr);
      return false;
    }
  }

  ComPtr<IDXGIFactory> dxgi_factory;
  hr = m_swap_chain->GetParent(IID_PPV_ARGS(dxgi_factory.GetAddressOf()));
  if (SUCCEEDED(hr))
  {
    hr = dxgi_factory->MakeWindowAssociation(swap_chain_desc.OutputWindow, DXGI_MWA_NO_WINDOW_CHANGES);
    if (FAILED(hr))
      Log_WarningPrintf("MakeWindowAssociation() to disable ALT+ENTER failed");
  }

  return CreateSwapChainRTV();
}

bool D3D11Device::CreateSwapChainRTV()
{
  ComPtr<ID3D11Texture2D> backbuffer;
  HRESULT hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(backbuffer.GetAddressOf()));
  if (FAILED(hr))
  {
    Log_ErrorPrintf("GetBuffer for RTV failed: 0x%08X", hr);
    return false;
  }

  D3D11_TEXTURE2D_DESC backbuffer_desc;
  backbuffer->GetDesc(&backbuffer_desc);

  CD3D11_RENDER_TARGET_VIEW_DESC rtv_desc(D3D11_RTV_DIMENSION_TEXTURE2D, backbuffer_desc.Format, 0, 0,
                                          backbuffer_desc.ArraySize);
  hr = m_device->CreateRenderTargetView(backbuffer.Get(), &rtv_desc, m_swap_chain_rtv.GetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("CreateRenderTargetView for swap chain failed: 0x%08X", hr);
    return false;
  }

  m_window_info.surface_width = backbuffer_desc.Width;
  m_window_info.surface_height = backbuffer_desc.Height;
  Log_InfoPrintf("Swap chain buffer size: %ux%u", m_window_info.surface_width, m_window_info.surface_height);

  if (m_window_info.type == WindowInfo::Type::Win32)
  {
    BOOL fullscreen = FALSE;
    DXGI_SWAP_CHAIN_DESC desc;
    if (SUCCEEDED(m_swap_chain->GetFullscreenState(&fullscreen, nullptr)) && fullscreen &&
        SUCCEEDED(m_swap_chain->GetDesc(&desc)))
    {
      m_window_info.surface_refresh_rate = static_cast<float>(desc.BufferDesc.RefreshRate.Numerator) /
                                           static_cast<float>(desc.BufferDesc.RefreshRate.Denominator);
    }
    else
    {
      m_window_info.surface_refresh_rate = 0.0f;
    }
  }

  return true;
}

bool D3D11Device::ChangeWindow(const WindowInfo& new_wi)
{
  DestroySurface();

  m_window_info = new_wi;
  return CreateSwapChain(nullptr);
}

void D3D11Device::DestroySurface()
{
  m_window_info.SetSurfaceless();
  if (IsFullscreen())
    SetFullscreen(false, 0, 0, 0.0f);

  m_swap_chain_rtv.Reset();
  m_swap_chain.Reset();
}

std::string D3D11Device::GetShaderCacheBaseName(const std::string_view& type, bool debug) const
{
  std::string_view flname;
  switch (m_device->GetFeatureLevel())
  {
      // clang-format off
  case D3D_FEATURE_LEVEL_10_0: flname = "sm40"; break;
  case D3D_FEATURE_LEVEL_10_1: flname = "sm41"; break;
  case D3D_FEATURE_LEVEL_11_0: default: flname = "sm50"; break;
      // clang-format on
  }

  return fmt::format("d3d_{}_{}{}", type, flname, debug ? "_debug" : "");
}

void D3D11Device::ResizeWindow(s32 new_window_width, s32 new_window_height)
{
  if (!m_swap_chain)
    return;

  m_swap_chain_rtv.Reset();

  HRESULT hr = m_swap_chain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN,
                                           m_using_allow_tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
  if (FAILED(hr))
    Log_ErrorPrintf("ResizeBuffers() failed: 0x%08X", hr);

  if (!CreateSwapChainRTV())
    Panic("Failed to recreate swap chain RTV after resize");
}

bool D3D11Device::SupportsFullscreen() const
{
  return true;
}

bool D3D11Device::IsFullscreen()
{
  BOOL is_fullscreen = FALSE;
  return (m_swap_chain && SUCCEEDED(m_swap_chain->GetFullscreenState(&is_fullscreen, nullptr)) && is_fullscreen);
}

bool D3D11Device::SetFullscreen(bool fullscreen, u32 width, u32 height, float refresh_rate)
{
  if (!m_swap_chain)
    return false;

  BOOL is_fullscreen = FALSE;
  HRESULT hr = m_swap_chain->GetFullscreenState(&is_fullscreen, nullptr);
  if (!fullscreen)
  {
    // leaving fullscreen
    if (is_fullscreen)
      return SUCCEEDED(m_swap_chain->SetFullscreenState(FALSE, nullptr));
    else
      return true;
  }

  IDXGIOutput* output;
  if (FAILED(hr = m_swap_chain->GetContainingOutput(&output)))
    return false;

  DXGI_SWAP_CHAIN_DESC current_desc;
  hr = m_swap_chain->GetDesc(&current_desc);
  if (FAILED(hr))
    return false;

  DXGI_MODE_DESC new_mode = current_desc.BufferDesc;
  new_mode.Width = width;
  new_mode.Height = height;
  new_mode.RefreshRate.Numerator = static_cast<UINT>(std::floor(refresh_rate * 1000.0f));
  new_mode.RefreshRate.Denominator = 1000u;

  DXGI_MODE_DESC closest_mode;
  if (FAILED(hr = output->FindClosestMatchingMode(&new_mode, &closest_mode, nullptr)) ||
      new_mode.Format != current_desc.BufferDesc.Format)
  {
    Log_ErrorPrintf("Failed to find closest matching mode, hr=%08X", hr);
    return false;
  }

  if (new_mode.Width == current_desc.BufferDesc.Width && new_mode.Height == current_desc.BufferDesc.Height &&
      new_mode.RefreshRate.Numerator == current_desc.BufferDesc.RefreshRate.Numerator &&
      new_mode.RefreshRate.Denominator == current_desc.BufferDesc.RefreshRate.Denominator)
  {
    Log_InfoPrintf("Fullscreen mode already set");
    return true;
  }

  m_swap_chain_rtv.Reset();
  m_swap_chain.Reset();

  if (!CreateSwapChain(&closest_mode))
  {
    Log_ErrorPrintf("Failed to create a fullscreen swap chain");
    if (!CreateSwapChain(nullptr))
      Panic("Failed to recreate windowed swap chain");

    return false;
  }

  return true;
}

bool D3D11Device::CreateBuffers()
{
  if (!m_vertex_buffer.Create(m_device.Get(), D3D11_BIND_VERTEX_BUFFER, VERTEX_BUFFER_SIZE) ||
      !m_index_buffer.Create(m_device.Get(), D3D11_BIND_INDEX_BUFFER, INDEX_BUFFER_SIZE) ||
      !m_uniform_buffer.Create(m_device.Get(), D3D11_BIND_CONSTANT_BUFFER, MAX_UNIFORM_BUFFER_SIZE))
  {
    Log_ErrorPrintf("Failed to create vertex/index/uniform buffers.");
    return false;
  }

  return true;
}

void D3D11Device::DestroyBuffers()
{
  m_uniform_buffer.Release();
  m_vertex_buffer.Release();
  m_index_buffer.Release();
}

bool D3D11Device::Render(bool skip_present)
{
  if (skip_present || !m_swap_chain)
  {
    ImGui::Render();
    return false;
  }

  // When using vsync, the time here seems to include the time for the buffer to become available.
  // This blows our our GPU usage number considerably, so read the timestamp before the final blit
  // in this configuration. It does reduce accuracy a little, but better than seeing 100% all of
  // the time, when it's more like a couple of percent.
  if (m_vsync_enabled && m_gpu_timing_enabled)
    PopTimestampQuery();

  m_context->ClearRenderTargetView(m_swap_chain_rtv.Get(), s_clear_color.data());
  m_context->OMSetRenderTargets(1, m_swap_chain_rtv.GetAddressOf(), nullptr);
  m_current_framebuffer = nullptr;

  RenderDisplay();

  RenderImGui();

  RenderSoftwareCursor();

  if (!m_vsync_enabled && m_gpu_timing_enabled)
    PopTimestampQuery();

  if (!m_vsync_enabled && m_using_allow_tearing)
    m_swap_chain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
  else
    m_swap_chain->Present(BoolToUInt32(m_vsync_enabled), 0);

  if (m_gpu_timing_enabled)
    KickTimestampQuery();
  
  return true;
}

GPUDevice::AdapterAndModeList D3D11Device::StaticGetAdapterAndModeList()
{
  ComPtr<IDXGIFactory> dxgi_factory;
  HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(dxgi_factory.GetAddressOf()));
  if (FAILED(hr))
    return {};

  return GetAdapterAndModeList(dxgi_factory.Get());
}

GPUDevice::AdapterAndModeList D3D11Device::GetAdapterAndModeList(IDXGIFactory* dxgi_factory)
{
  AdapterAndModeList adapter_info;
  ComPtr<IDXGIAdapter> current_adapter;
  while (SUCCEEDED(dxgi_factory->EnumAdapters(static_cast<UINT>(adapter_info.adapter_names.size()),
                                              current_adapter.ReleaseAndGetAddressOf())))
  {
    DXGI_ADAPTER_DESC adapter_desc;
    std::string adapter_name;
    if (SUCCEEDED(current_adapter->GetDesc(&adapter_desc)))
    {
      char adapter_name_buffer[128];
      const int name_length = WideCharToMultiByte(CP_UTF8, 0, adapter_desc.Description,
                                                  static_cast<int>(std::wcslen(adapter_desc.Description)),
                                                  adapter_name_buffer, countof(adapter_name_buffer), 0, nullptr);
      if (name_length >= 0)
        adapter_name.assign(adapter_name_buffer, static_cast<size_t>(name_length));
      else
        adapter_name.assign("(Unknown)");
    }
    else
    {
      adapter_name.assign("(Unknown)");
    }

    if (adapter_info.fullscreen_modes.empty())
    {
      ComPtr<IDXGIOutput> output;
      if (SUCCEEDED(current_adapter->EnumOutputs(0, &output)))
      {
        UINT num_modes = 0;
        if (SUCCEEDED(output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &num_modes, nullptr)))
        {
          std::vector<DXGI_MODE_DESC> modes(num_modes);
          if (SUCCEEDED(output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &num_modes, modes.data())))
          {
            for (const DXGI_MODE_DESC& mode : modes)
            {
              adapter_info.fullscreen_modes.push_back(GetFullscreenModeString(
                mode.Width, mode.Height,
                static_cast<float>(mode.RefreshRate.Numerator) / static_cast<float>(mode.RefreshRate.Denominator)));
            }
          }
        }
      }
    }

    // handle duplicate adapter names
    if (std::any_of(adapter_info.adapter_names.begin(), adapter_info.adapter_names.end(),
                    [&adapter_name](const std::string& other) { return (adapter_name == other); }))
    {
      std::string original_adapter_name = std::move(adapter_name);

      u32 current_extra = 2;
      do
      {
        adapter_name = StringUtil::StdStringFromFormat("%s (%u)", original_adapter_name.c_str(), current_extra);
        current_extra++;
      } while (std::any_of(adapter_info.adapter_names.begin(), adapter_info.adapter_names.end(),
                           [&adapter_name](const std::string& other) { return (adapter_name == other); }));
    }

    adapter_info.adapter_names.push_back(std::move(adapter_name));
  }

  return adapter_info;
}

GPUDevice::AdapterAndModeList D3D11Device::GetAdapterAndModeList()
{
  return GetAdapterAndModeList(m_dxgi_factory.Get());
}

bool D3D11Device::CreateTimestampQueries()
{
  for (u32 i = 0; i < NUM_TIMESTAMP_QUERIES; i++)
  {
    for (u32 j = 0; j < 3; j++)
    {
      const CD3D11_QUERY_DESC qdesc((j == 0) ? D3D11_QUERY_TIMESTAMP_DISJOINT : D3D11_QUERY_TIMESTAMP);
      const HRESULT hr = m_device->CreateQuery(&qdesc, m_timestamp_queries[i][j].ReleaseAndGetAddressOf());
      if (FAILED(hr))
      {
        m_timestamp_queries = {};
        return false;
      }
    }
  }

  KickTimestampQuery();
  return true;
}

void D3D11Device::DestroyTimestampQueries()
{
  if (!m_timestamp_queries[0][0])
    return;

  if (m_timestamp_query_started)
    m_context->End(m_timestamp_queries[m_write_timestamp_query][1].Get());

  m_timestamp_queries = {};
  m_read_timestamp_query = 0;
  m_write_timestamp_query = 0;
  m_waiting_timestamp_queries = 0;
  m_timestamp_query_started = 0;
}

void D3D11Device::PopTimestampQuery()
{
  while (m_waiting_timestamp_queries > 0)
  {
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint;
    const HRESULT disjoint_hr = m_context->GetData(m_timestamp_queries[m_read_timestamp_query][0].Get(), &disjoint,
                                                   sizeof(disjoint), D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (disjoint_hr != S_OK)
      break;

    if (disjoint.Disjoint)
    {
      Log_VerbosePrintf("GPU timing disjoint, resetting.");
      m_read_timestamp_query = 0;
      m_write_timestamp_query = 0;
      m_waiting_timestamp_queries = 0;
      m_timestamp_query_started = 0;
    }
    else
    {
      u64 start = 0, end = 0;
      const HRESULT start_hr = m_context->GetData(m_timestamp_queries[m_read_timestamp_query][1].Get(), &start,
                                                  sizeof(start), D3D11_ASYNC_GETDATA_DONOTFLUSH);
      const HRESULT end_hr = m_context->GetData(m_timestamp_queries[m_read_timestamp_query][2].Get(), &end, sizeof(end),
                                                D3D11_ASYNC_GETDATA_DONOTFLUSH);
      if (start_hr == S_OK && end_hr == S_OK)
      {
        const float delta =
          static_cast<float>(static_cast<double>(end - start) / (static_cast<double>(disjoint.Frequency) / 1000.0));
        m_accumulated_gpu_time += delta;
        m_read_timestamp_query = (m_read_timestamp_query + 1) % NUM_TIMESTAMP_QUERIES;
        m_waiting_timestamp_queries--;
      }
    }
  }

  if (m_timestamp_query_started)
  {
    m_context->End(m_timestamp_queries[m_write_timestamp_query][2].Get());
    m_context->End(m_timestamp_queries[m_write_timestamp_query][0].Get());
    m_write_timestamp_query = (m_write_timestamp_query + 1) % NUM_TIMESTAMP_QUERIES;
    m_timestamp_query_started = false;
    m_waiting_timestamp_queries++;
  }
}

void D3D11Device::KickTimestampQuery()
{
  if (m_timestamp_query_started || !m_timestamp_queries[0][0] || m_waiting_timestamp_queries == NUM_TIMESTAMP_QUERIES)
    return;

  m_context->Begin(m_timestamp_queries[m_write_timestamp_query][0].Get());
  m_context->End(m_timestamp_queries[m_write_timestamp_query][1].Get());
  m_timestamp_query_started = true;
}

bool D3D11Device::SetGPUTimingEnabled(bool enabled)
{
  if (m_gpu_timing_enabled == enabled)
    return true;

  m_gpu_timing_enabled = enabled;
  if (m_gpu_timing_enabled)
  {
    if (!CreateTimestampQueries())
      return false;

    KickTimestampQuery();
    return true;
  }
  else
  {
    DestroyTimestampQueries();
    return true;
  }
}

float D3D11Device::GetAndResetAccumulatedGPUTime()
{
  const float value = m_accumulated_gpu_time;
  m_accumulated_gpu_time = 0.0f;
  return value;
}

D3D11Framebuffer::D3D11Framebuffer(GPUTexture* rt, GPUTexture* ds, u32 width, u32 height,
                                   ComPtr<ID3D11RenderTargetView> rtv, ComPtr<ID3D11DepthStencilView> dsv)
  : GPUFramebuffer(rt, ds, width, height), m_rtv(std::move(rtv)), m_dsv(std::move(dsv))
{
}

D3D11Framebuffer::~D3D11Framebuffer() = default;

void D3D11Framebuffer::SetDebugName(const std::string_view& name)
{
  if (m_rtv)
    SetD3DDebugObjectName(m_rtv.Get(), name);
  if (m_dsv)
    SetD3DDebugObjectName(m_dsv.Get(), name);
}

void D3D11Framebuffer::CommitClear(ID3D11DeviceContext* context)
{
  if (UNLIKELY(m_rt && m_rt->GetState() != GPUTexture::State::Dirty))
  {
    if (m_rt->GetState() == GPUTexture::State::Invalidated)
      ; // m_context->DiscardView(m_rtv.Get());
    else
      context->ClearRenderTargetView(m_rtv.Get(), m_rt->GetUNormClearColor().data());

    m_rt->SetState(GPUTexture::State::Dirty);
  }

  if (UNLIKELY(m_ds && m_ds->GetState() != GPUTexture::State::Dirty))
  {
    if (m_ds->GetState() == GPUTexture::State::Invalidated)
      ; // m_context->DiscardView(m_dsv.Get());
    else
      context->ClearDepthStencilView(m_dsv.Get(), D3D11_CLEAR_DEPTH, m_ds->GetClearDepth(), 0);

    m_ds->SetState(GPUTexture::State::Dirty);
  }
}

std::unique_ptr<GPUFramebuffer> D3D11Device::CreateFramebuffer(GPUTexture* rt, u32 rt_layer, u32 rt_level,
                                                               GPUTexture* ds, u32 ds_layer, u32 ds_level)
{
  ComPtr<ID3D11RenderTargetView> rtv;
  ComPtr<ID3D11DepthStencilView> dsv;
  HRESULT hr;

  Assert(rt || ds);
  Assert(!rt || (rt_layer < rt->GetLayers() && rt_level < rt->GetLevels()));
  Assert(!ds || (ds_layer < ds->GetLevels() && ds_level < ds->GetLevels()));
  Assert(!rt || !ds ||
         (rt->GetMipWidth(rt_level) == ds->GetMipWidth(ds_level) &&
          rt->GetMipHeight(rt_level) == ds->GetMipHeight(ds_level)));

  if (rt)
  {
    D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
    rtv_desc.Format = static_cast<D3D11Texture*>(rt)->GetDXGIFormat();
    if (rt->IsMultisampled())
    {
      Assert(rt_level == 0);
      if (rt->GetLayers() > 1)
      {
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
        rtv_desc.Texture2DMSArray.ArraySize = rt->GetLayers();
        rtv_desc.Texture2DMSArray.FirstArraySlice = rt_layer;
      }
      else
      {
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
      }
    }
    else
    {
      if (rt->GetLayers() > 1)
      {
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        rtv_desc.Texture2DArray.ArraySize = rt->GetLayers();
        rtv_desc.Texture2DArray.FirstArraySlice = rt_layer;
        rtv_desc.Texture2DArray.MipSlice = rt_level;
      }
      else
      {
        rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        rtv_desc.Texture2D.MipSlice = rt_level;
      }
    }

    if (FAILED(hr = m_device->CreateRenderTargetView(static_cast<D3D11Texture*>(rt)->GetD3DTexture(), &rtv_desc,
                                                     rtv.GetAddressOf())))
    {
      Log_ErrorPrintf("CreateRenderTargetView() failed: %08X", hr);
      return {};
    }
  }

  if (ds)
  {
    D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
    dsv_desc.Format = static_cast<D3D11Texture*>(ds)->GetDXGIFormat();
    if (ds->IsMultisampled())
    {
      Assert(rt_level == 0);
      if (ds->GetLayers() > 1)
      {
        dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
        dsv_desc.Texture2DMSArray.ArraySize = ds->GetLayers();
        dsv_desc.Texture2DMSArray.FirstArraySlice = rt_layer;
      }
      else
      {
        dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
      }
    }
    else
    {
      if (ds->GetLayers() > 1)
      {
        dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        dsv_desc.Texture2DArray.ArraySize = ds->GetLayers();
        dsv_desc.Texture2DArray.FirstArraySlice = rt_layer;
        dsv_desc.Texture2DArray.MipSlice = rt_level;
      }
      else
      {
        dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        dsv_desc.Texture2D.MipSlice = rt_level;
      }
    }

    if (FAILED(hr = m_device->CreateDepthStencilView(static_cast<D3D11Texture*>(ds)->GetD3DTexture(), &dsv_desc,
                                                     dsv.GetAddressOf())))
    {
      Log_ErrorPrintf("CreateDepthStencilView() failed: %08X", hr);
      return {};
    }
  }

  return std::unique_ptr<GPUFramebuffer>(
    new D3D11Framebuffer(rt, ds, rt ? rt->GetMipWidth(rt_level) : ds->GetMipWidth(ds_level),
                         rt ? rt->GetMipHeight(rt_level) : ds->GetMipHeight(ds_level), std::move(rtv), std::move(dsv)));
}

D3D11Sampler::D3D11Sampler(ComPtr<ID3D11SamplerState> ss) : m_ss(std::move(ss))
{
}

D3D11Sampler::~D3D11Sampler() = default;

void D3D11Sampler::SetDebugName(const std::string_view& name)
{
  SetD3DDebugObjectName(m_ss.Get(), name);
}

std::unique_ptr<GPUSampler> D3D11Device::CreateSampler(const GPUSampler::Config& config)
{
  static constexpr std::array<D3D11_TEXTURE_ADDRESS_MODE, static_cast<u8>(GPUSampler::AddressMode::MaxCount)> ta = {{
    D3D11_TEXTURE_ADDRESS_WRAP,   // Repeat
    D3D11_TEXTURE_ADDRESS_CLAMP,  // ClampToEdge
    D3D11_TEXTURE_ADDRESS_BORDER, // ClampToBorder
  }};

  D3D11_SAMPLER_DESC desc = {};
  desc.AddressU = ta[static_cast<u8>(config.address_u.GetValue())];
  desc.AddressV = ta[static_cast<u8>(config.address_v.GetValue())];
  desc.AddressW = ta[static_cast<u8>(config.address_w.GetValue())];
  desc.BorderColor[0] = static_cast<float>(config.border_color & 0xFF) / 255.0f;
  desc.BorderColor[1] = static_cast<float>((config.border_color >> 8) & 0xFF) / 255.0f;
  desc.BorderColor[2] = static_cast<float>((config.border_color >> 16) & 0xFF) / 255.0f;
  desc.BorderColor[3] = static_cast<float>((config.border_color >> 24) & 0xFF) / 255.0f;
  desc.MinLOD = static_cast<float>(config.min_lod);
  desc.MaxLOD = static_cast<float>(config.max_lod);

  if (config.anisotropy > 0)
  {
    desc.Filter = D3D11_FILTER_ANISOTROPIC;
    desc.MaxAnisotropy = config.anisotropy;
  }
  else
  {
    static constexpr u8 filter_count = static_cast<u8>(GPUSampler::Filter::MaxCount);
    static constexpr D3D11_FILTER filters[filter_count][filter_count][filter_count] = {
      {
        {D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT},
        {D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT, D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT},
      },
      {
        {D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR, D3D11_FILTER_MIN_POINT_MAG_MIP_LINEAR},
        {D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR, D3D11_FILTER_MIN_MAG_MIP_LINEAR},
      }};

    desc.Filter = filters[static_cast<u8>(config.mip_filter.GetValue())][static_cast<u8>(config.min_filter.GetValue())]
                         [static_cast<u8>(config.mag_filter.GetValue())];
    desc.MaxAnisotropy = 1;
  }

  // TODO: Pool?
  ComPtr<ID3D11SamplerState> ss;
  const HRESULT hr = m_device->CreateSamplerState(&desc, ss.GetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("CreateSamplerState() failed: %08X", hr);
    return {};
  }

  return std::unique_ptr<GPUSampler>(new D3D11Sampler(std::move(ss)));
}

D3D11Shader::D3D11Shader(GPUShaderStage stage, Microsoft::WRL::ComPtr<ID3D11DeviceChild> shader,
                         std::vector<u8> bytecode)
  : GPUShader(stage), m_shader(std::move(shader)), m_bytecode(std::move(bytecode))
{
}

D3D11Shader::~D3D11Shader() = default;

ID3D11VertexShader* D3D11Shader::GetVertexShader() const
{
  DebugAssert(m_stage == GPUShaderStage::Vertex);
  return static_cast<ID3D11VertexShader*>(m_shader.Get());
}

ID3D11PixelShader* D3D11Shader::GetPixelShader() const
{
  DebugAssert(m_stage == GPUShaderStage::Fragment);
  return static_cast<ID3D11PixelShader*>(m_shader.Get());
}

ID3D11ComputeShader* D3D11Shader::GetComputeShader() const
{
  DebugAssert(m_stage == GPUShaderStage::Compute);
  return static_cast<ID3D11ComputeShader*>(m_shader.Get());
}

void D3D11Shader::SetDebugName(const std::string_view& name)
{
  SetD3DDebugObjectName(m_shader.Get(), name);
}

std::unique_ptr<GPUShader> D3D11Device::CreateShaderFromBinary(GPUShaderStage stage, gsl::span<const u8> data)
{
  ComPtr<ID3D11DeviceChild> shader;
  std::vector<u8> bytecode;
  HRESULT hr;
  switch (stage)
  {
    case GPUShaderStage::Vertex:
      hr = m_device->CreateVertexShader(data.data(), data.size(), nullptr,
                                        reinterpret_cast<ID3D11VertexShader**>(shader.GetAddressOf()));
      bytecode.resize(data.size());
      std::memcpy(bytecode.data(), data.data(), data.size());
      break;

    case GPUShaderStage::Fragment:
      hr = m_device->CreatePixelShader(data.data(), data.size(), nullptr,
                                       reinterpret_cast<ID3D11PixelShader**>(shader.GetAddressOf()));
      break;

    case GPUShaderStage::Compute:
      hr = m_device->CreateComputeShader(data.data(), data.size(), nullptr,
                                         reinterpret_cast<ID3D11ComputeShader**>(shader.GetAddressOf()));
      break;

    default:
      UnreachableCode();
      break;
  }

  if (!shader)
    return {};

  return std::unique_ptr<GPUShader>(new D3D11Shader(stage, std::move(shader), std::move(bytecode)));
}

std::unique_ptr<GPUShader> D3D11Device::CreateShaderFromSource(GPUShaderStage stage, const std::string_view& source,
                                                               std::vector<u8>* out_binary /* = nullptr */)
{
  // TODO: This shouldn't be dependent on build type.
#ifdef _DEBUG
  constexpr bool debug = true;
#else
  constexpr bool debug = false;
#endif

  const char* target;
  switch (m_device->GetFeatureLevel())
  {
    case D3D_FEATURE_LEVEL_10_0:
    {
      static constexpr std::array<const char*, 4> targets = {{"vs_4_0", "ps_4_0", "cs_4_0"}};
      target = targets[static_cast<int>(stage)];
    }
    break;

    case D3D_FEATURE_LEVEL_10_1:
    {
      static constexpr std::array<const char*, 4> targets = {{"vs_4_1", "ps_4_1", "cs_4_1"}};
      target = targets[static_cast<int>(stage)];
    }
    break;

    case D3D_FEATURE_LEVEL_11_0:
    {
      static constexpr std::array<const char*, 4> targets = {{"vs_5_0", "ps_5_0", "cs_5_0"}};
      target = targets[static_cast<int>(stage)];
    }
    break;

    case D3D_FEATURE_LEVEL_11_1:
    default:
    {
      static constexpr std::array<const char*, 4> targets = {{"vs_5_1", "ps_5_1", "cs_5_1"}};
      target = targets[static_cast<int>(stage)];
    }
    break;
  }

  static constexpr UINT flags_non_debug = D3DCOMPILE_OPTIMIZATION_LEVEL3;
  static constexpr UINT flags_debug = D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_DEBUG;

  ComPtr<ID3DBlob> blob;
  ComPtr<ID3DBlob> error_blob;
  const HRESULT hr =
    D3DCompile(source.data(), source.size(), "0", nullptr, nullptr, "main", target,
               debug ? flags_debug : flags_non_debug, 0, blob.GetAddressOf(), error_blob.GetAddressOf());

  std::string error_string;
  if (error_blob)
  {
    error_string.append(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize());
    error_blob.Reset();
  }

  if (FAILED(hr))
  {
    Log_ErrorPrintf("Failed to compile '%s':\n%s", target, error_string.c_str());

    auto fp = FileSystem::OpenManagedCFile(
      Path::Combine(EmuFolders::DataRoot, fmt::format("bad_shader_{}.txt", s_next_bad_shader_id++)).c_str(), "wb");
    if (fp)
    {
      std::fwrite(source.data(), source.size(), 1, fp.get());
      std::fprintf(fp.get(), "\n\nCompile as %s failed: %08X\n", target, hr);
      std::fwrite(error_string.c_str(), error_string.size(), 1, fp.get());
    }

    return {};
  }

  if (!error_string.empty())
    Log_WarningPrintf("'%s' compiled with warnings:\n%s", target, error_string.c_str());

  if (out_binary)
  {
    const size_t size = blob->GetBufferSize();
    out_binary->resize(size);
    std::memcpy(out_binary->data(), blob->GetBufferPointer(), size);
  }

  return CreateShaderFromBinary(
    stage, gsl::span<const u8>(static_cast<const u8*>(blob->GetBufferPointer()), blob->GetBufferSize()));
}

D3D11Pipeline::D3D11Pipeline(ComPtr<ID3D11RasterizerState> rs, ComPtr<ID3D11DepthStencilState> ds,
                             ComPtr<ID3D11BlendState> bs, ComPtr<ID3D11InputLayout> il, ComPtr<ID3D11VertexShader> vs,
                             ComPtr<ID3D11PixelShader> ps, D3D11_PRIMITIVE_TOPOLOGY topology)
  : m_rs(std::move(rs)), m_ds(std::move(ds)), m_bs(std::move(bs)), m_il(std::move(il)), m_vs(std::move(vs)),
    m_ps(std::move(ps)), m_topology(topology)
{
}

D3D11Pipeline::~D3D11Pipeline() = default;

void D3D11Pipeline::SetDebugName(const std::string_view& name)
{
  // can't label this directly
}

void D3D11Pipeline::Bind(ID3D11DeviceContext* context)
{
  // TODO: constant blend factor
  context->IASetInputLayout(GetInputLayout());
  context->IASetPrimitiveTopology(GetPrimitiveTopology());
  context->RSSetState(GetRasterizerState());
  context->OMSetDepthStencilState(GetDepthStencilState(), 0);
  context->OMSetBlendState(GetBlendState(), nullptr, 0xFFFFFFFFu);
  context->VSSetShader(GetVertexShader(), nullptr, 0);
  context->PSSetShader(GetPixelShader(), nullptr, 0);
}

D3D11Device::ComPtr<ID3D11RasterizerState> D3D11Device::GetRasterizationState(const GPUPipeline::RasterizationState& rs)
{
  ComPtr<ID3D11RasterizerState> drs;

  const auto it = m_rasterization_states.find(rs.key);
  if (it != m_rasterization_states.end())
  {
    drs = it->second;
    return drs;
  }

  static constexpr std::array<D3D11_CULL_MODE, static_cast<u32>(GPUPipeline::CullMode::MaxCount)> cull_mapping = {{
    D3D11_CULL_NONE,  // None
    D3D11_CULL_FRONT, // Front
    D3D11_CULL_BACK,  // Back
  }};

  D3D11_RASTERIZER_DESC desc = {};
  desc.FillMode = D3D11_FILL_SOLID;
  desc.CullMode = cull_mapping[static_cast<u8>(rs.cull_mode.GetValue())];
  desc.ScissorEnable = TRUE;
  // desc.MultisampleEnable ???

  HRESULT hr = m_device->CreateRasterizerState(&desc, drs.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create depth state with %08X", hr);

  m_rasterization_states.emplace(rs.key, drs);
  return drs;
}

D3D11Device::ComPtr<ID3D11DepthStencilState> D3D11Device::GetDepthState(const GPUPipeline::DepthState& ds)
{
  ComPtr<ID3D11DepthStencilState> dds;

  const auto it = m_depth_states.find(ds.key);
  if (it != m_depth_states.end())
  {
    dds = it->second;
    return dds;
  }

  static constexpr std::array<D3D11_COMPARISON_FUNC, static_cast<u32>(GPUPipeline::DepthFunc::MaxCount)> func_mapping =
    {{
      D3D11_COMPARISON_NEVER,         // Never
      D3D11_COMPARISON_ALWAYS,        // Always
      D3D11_COMPARISON_LESS,          // Less
      D3D11_COMPARISON_LESS_EQUAL,    // LessEqual
      D3D11_COMPARISON_GREATER,       // Greater
      D3D11_COMPARISON_GREATER_EQUAL, // GreaterEqual
      D3D11_COMPARISON_EQUAL,         // Equal
    }};

  D3D11_DEPTH_STENCIL_DESC desc = {};
  desc.DepthEnable = ds.depth_test != GPUPipeline::DepthFunc::Never;
  desc.DepthFunc = func_mapping[static_cast<u8>(ds.depth_test.GetValue())];
  desc.DepthWriteMask = ds.depth_write ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;

  HRESULT hr = m_device->CreateDepthStencilState(&desc, dds.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create depth state with %08X", hr);

  m_depth_states.emplace(ds.key, dds);
  return dds;
}

D3D11Device::ComPtr<ID3D11BlendState> D3D11Device::GetBlendState(const GPUPipeline::BlendState& bs)
{
  ComPtr<ID3D11BlendState> dbs;

  const auto it = m_blend_states.find(bs.key);
  if (it != m_blend_states.end())
  {
    dbs = it->second;
    return dbs;
  }

  static constexpr std::array<D3D11_BLEND, static_cast<u32>(GPUPipeline::BlendFunc::MaxCount)> blend_mapping = {{
    D3D11_BLEND_ZERO,             // Zero
    D3D11_BLEND_ONE,              // One
    D3D11_BLEND_SRC_COLOR,        // SrcColor
    D3D11_BLEND_INV_SRC_COLOR,    // InvSrcColor
    D3D11_BLEND_DEST_COLOR,       // DstColor
    D3D11_BLEND_INV_DEST_COLOR,   // InvDstColor
    D3D11_BLEND_SRC_ALPHA,        // SrcAlpha
    D3D11_BLEND_INV_SRC_ALPHA,    // InvSrcAlpha
    D3D11_BLEND_SRC1_ALPHA,       // SrcAlpha1
    D3D11_BLEND_INV_SRC1_ALPHA,   // InvSrcAlpha1
    D3D11_BLEND_DEST_ALPHA,       // DstAlpha
    D3D11_BLEND_INV_DEST_ALPHA,   // InvDstAlpha
    D3D11_BLEND_BLEND_FACTOR,     // ConstantColor
    D3D11_BLEND_INV_BLEND_FACTOR, // InvConstantColor
  }};

  static constexpr std::array<D3D11_BLEND_OP, static_cast<u32>(GPUPipeline::BlendOp::MaxCount)> op_mapping = {{
    D3D11_BLEND_OP_ADD,          // Add
    D3D11_BLEND_OP_SUBTRACT,     // Subtract
    D3D11_BLEND_OP_REV_SUBTRACT, // ReverseSubtract
    D3D11_BLEND_OP_MIN,          // Min
    D3D11_BLEND_OP_MAX,          // Max
  }};

  D3D11_BLEND_DESC blend_desc = {};
  D3D11_RENDER_TARGET_BLEND_DESC& tgt_desc = blend_desc.RenderTarget[0];
  tgt_desc.BlendEnable = bs.enable;
  tgt_desc.RenderTargetWriteMask = bs.write_mask;
  if (bs.enable)
  {
    tgt_desc.SrcBlend = blend_mapping[static_cast<u8>(bs.src_blend.GetValue())];
    tgt_desc.DestBlend = blend_mapping[static_cast<u8>(bs.dst_blend.GetValue())];
    tgt_desc.BlendOp = op_mapping[static_cast<u8>(bs.blend_op.GetValue())];
    tgt_desc.SrcBlendAlpha = blend_mapping[static_cast<u8>(bs.src_alpha_blend.GetValue())];
    tgt_desc.DestBlendAlpha = blend_mapping[static_cast<u8>(bs.dst_alpha_blend.GetValue())];
    tgt_desc.BlendOpAlpha = op_mapping[static_cast<u8>(bs.alpha_blend_op.GetValue())];
  }

  HRESULT hr = m_device->CreateBlendState(&blend_desc, dbs.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create blend state with %08X", hr);

  m_blend_states.emplace(bs.key, dbs);
  return dbs;
}

D3D11Device::ComPtr<ID3D11InputLayout> D3D11Device::GetInputLayout(const GPUPipeline::InputLayout& il,
                                                                   const D3D11Shader* vs)
{
  ComPtr<ID3D11InputLayout> dil;
  const auto it = m_input_layouts.find(il);
  if (it != m_input_layouts.end())
  {
    dil = it->second;
    return dil;
  }

#if 0
  static constexpr std::array<const char*, static_cast<u32>(GPUPipeline::VertexAttribute::MaxAttributes)> semantics = {
    {"ATTR0", "ATTR1", "ATTR2", "ATTR3", "ATTR4", "ATTR5", "ATTR6", "ATTR7", "ATTR8", "ATTR9", "ATTR10", "ATTR11",
     "ATTR12", "ATTR13", "ATTR14", "ATTR15"}};
#endif

  static constexpr u32 MAX_COMPONENTS = 4;
  static constexpr const DXGI_FORMAT
    format_mapping[static_cast<u8>(GPUPipeline::VertexAttribute::Type::MaxCount)][MAX_COMPONENTS] = {
      {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
       DXGI_FORMAT_R32G32B32A32_FLOAT},                                                                       // Float
      {DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UINT},           // UInt8
      {DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_SINT},           // SInt8
      {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UNORM},        // UNorm8
      {DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_UINT},    // UInt16
      {DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_SINT},    // SInt16
      {DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_UNORM}, // UNorm16
      {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32G32B32A32_UINT},    // UInt32
      {DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32G32B32A32_SINT},    // SInt32
    };

  D3D11_INPUT_ELEMENT_DESC* elems =
    static_cast<D3D11_INPUT_ELEMENT_DESC*>(alloca(sizeof(D3D11_INPUT_ELEMENT_DESC) * il.vertex_attributes.size()));
  for (size_t i = 0; i < il.vertex_attributes.size(); i++)
  {
    const GPUPipeline::VertexAttribute& va = il.vertex_attributes[i];
    Assert(va.components > 0 && va.components <= MAX_COMPONENTS);

    D3D11_INPUT_ELEMENT_DESC& elem = elems[i];
    elem.SemanticName = "ATTR";
    elem.SemanticIndex = va.index;
    elem.Format = format_mapping[static_cast<u8>(va.type.GetValue())][va.components - 1];
    elem.InputSlot = 0;
    elem.AlignedByteOffset = va.offset;
    elem.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    elem.InstanceDataStepRate = 0;
  }

  HRESULT hr = m_device->CreateInputLayout(elems, static_cast<UINT>(il.vertex_attributes.size()),
                                           vs->GetBytecode().data(), vs->GetBytecode().size(), dil.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create input layout with %08X", hr);

  m_input_layouts.emplace(il, dil);
  return dil;
}

std::unique_ptr<GPUPipeline> D3D11Device::CreatePipeline(const GPUPipeline::GraphicsConfig& config)
{
  ComPtr<ID3D11RasterizerState> rs = GetRasterizationState(config.rasterization);
  ComPtr<ID3D11DepthStencilState> ds = GetDepthState(config.depth);
  ComPtr<ID3D11BlendState> bs = GetBlendState(config.blend);
  if (!rs || !ds || !bs)
    return {};

  ComPtr<ID3D11InputLayout> il;
  if (!config.input_layout.vertex_attributes.empty())
  {
    il = GetInputLayout(config.input_layout, static_cast<const D3D11Shader*>(config.vertex_shader));
    if (!il)
      return {};
  }

  static constexpr std::array<D3D11_PRIMITIVE_TOPOLOGY, static_cast<u32>(GPUPipeline::Primitive::MaxCount)> primitives =
    {{
      D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,     // Points
      D3D11_PRIMITIVE_TOPOLOGY_LINELIST,      // Lines
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,  // Triangles
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, // TriangleStrips
    }};

  return std::unique_ptr<GPUPipeline>(
    new D3D11Pipeline(std::move(rs), std::move(ds), std::move(bs), std::move(il),
                      static_cast<const D3D11Shader*>(config.vertex_shader)->GetVertexShader(),
                      static_cast<const D3D11Shader*>(config.pixel_shader)->GetPixelShader(),
                      primitives[static_cast<u8>(config.primitive)]));
}

void D3D11Device::PushDebugGroup(const char* fmt, ...)
{
  if (!m_annotation)
    return;

  std::va_list ap;
  va_start(ap, fmt);
  std::string str(StringUtil::StdStringFromFormatV(fmt, ap));
  va_end(ap);

  m_annotation->BeginEvent(StringUtil::UTF8StringToWideString(str).c_str());
}

void D3D11Device::PopDebugGroup()
{
  if (!m_annotation)
    return;

  m_annotation->EndEvent();
}

void D3D11Device::InsertDebugMessage(const char* fmt, ...)
{
  if (!m_annotation)
    return;

  std::va_list ap;
  va_start(ap, fmt);
  std::string str(StringUtil::StdStringFromFormatV(fmt, ap));
  va_end(ap);

  m_annotation->SetMarker(StringUtil::UTF8StringToWideString(str).c_str());
}

void D3D11Device::MapVertexBuffer(u32 vertex_size, u32 vertex_count, void** map_ptr, u32* map_space,
                                  u32* map_base_vertex)
{
  const auto res = m_vertex_buffer.Map(m_context.Get(), vertex_size, vertex_size * vertex_count);
  *map_ptr = res.pointer;
  *map_space = res.space_aligned;
  *map_base_vertex = res.index_aligned;
}

void D3D11Device::UnmapVertexBuffer(u32 vertex_size, u32 vertex_count)
{
  m_vertex_buffer.Unmap(m_context.Get(), vertex_size * vertex_count);

  // TODO: cache - should come from pipeline
  const UINT offset = 0;
  m_context->IASetVertexBuffers(0, 1, m_vertex_buffer.GetD3DBufferArray(), &vertex_size, &offset);
}

void D3D11Device::MapIndexBuffer(u32 index_count, DrawIndex** map_ptr, u32* map_space, u32* map_base_index)
{
  const auto res = m_index_buffer.Map(m_context.Get(), sizeof(DrawIndex), sizeof(DrawIndex) * index_count);
  *map_ptr = static_cast<DrawIndex*>(res.pointer);
  *map_space = res.space_aligned;
  *map_base_index = res.index_aligned;
}

void D3D11Device::UnmapIndexBuffer(u32 used_index_count)
{
  m_index_buffer.Unmap(m_context.Get(), sizeof(DrawIndex) * used_index_count);
  m_context->IASetIndexBuffer(m_index_buffer.GetD3DBuffer(), DXGI_FORMAT_R16_UINT, 0);
}

void D3D11Device::PushUniformBuffer(const void* data, u32 data_size)
{
  Assert(data_size <= MAX_UNIFORM_BUFFER_SIZE);

  const auto res = m_uniform_buffer.Map(m_context.Get(), MAX_UNIFORM_BUFFER_SIZE, MAX_UNIFORM_BUFFER_SIZE);
  std::memcpy(res.pointer, data, data_size);
  m_uniform_buffer.Unmap(m_context.Get(), data_size);

  m_context->VSSetConstantBuffers(0, 1, m_uniform_buffer.GetD3DBufferArray());
  m_context->PSSetConstantBuffers(0, 1, m_uniform_buffer.GetD3DBufferArray());
}

void* D3D11Device::MapUniformBuffer(u32 size)
{
  Assert(size <= MAX_UNIFORM_BUFFER_SIZE);

  const auto res = m_uniform_buffer.Map(m_context.Get(), MAX_UNIFORM_BUFFER_SIZE, MAX_UNIFORM_BUFFER_SIZE);
  return res.pointer;
}

void D3D11Device::UnmapUniformBuffer(u32 size)
{
  m_uniform_buffer.Unmap(m_context.Get(), size);
  m_context->VSSetConstantBuffers(0, 1, m_uniform_buffer.GetD3DBufferArray());
  m_context->PSSetConstantBuffers(0, 1, m_uniform_buffer.GetD3DBufferArray());
}

void D3D11Device::SetFramebuffer(GPUFramebuffer* fb)
{
  if (m_current_framebuffer == fb)
    return;

  m_current_framebuffer = static_cast<D3D11Framebuffer*>(fb);
  if (!m_current_framebuffer)
  {
    m_context->OMSetRenderTargets(0, nullptr, nullptr);
    return;
  }

  m_context->OMSetRenderTargets(m_current_framebuffer->GetNumRTVs(), m_current_framebuffer->GetRTVArray(),
                                m_current_framebuffer->GetDSV());
}

void D3D11Device::UnbindFramebuffer(D3D11Framebuffer* fb)
{
  if (m_current_framebuffer != fb)
    return;

  m_current_framebuffer = nullptr;
  m_context->OMSetRenderTargets(0, nullptr, nullptr);
}

void D3D11Device::SetPipeline(GPUPipeline* pipeline)
{
  D3D11Pipeline* PL = static_cast<D3D11Pipeline*>(pipeline);

  // TODO: cache
  PL->Bind(m_context.Get());
}

void D3D11Device::UnbindPipeline(D3D11Pipeline* pl)
{
  if (m_current_pipeline != pl)
    return;

  m_current_pipeline = nullptr;
}

void D3D11Device::SetTextureSampler(u32 slot, GPUTexture* texture, GPUSampler* sampler)
{
  // TODO: cache when old rt == tex
  D3D11Texture* T = static_cast<D3D11Texture*>(texture);
  D3D11Sampler* S = static_cast<D3D11Sampler*>(sampler);
  m_context->PSSetShaderResources(0, 1, T->GetD3DSRVArray());
  m_context->PSSetSamplers(0, 1, S->GetSamplerStateArray());
}

void D3D11Device::UnbindTexture(D3D11Texture* tex)
{
  // TODO
}

void D3D11Device::SetViewport(s32 x, s32 y, s32 width, s32 height)
{
  const CD3D11_VIEWPORT vp(static_cast<float>(x), static_cast<float>(y), static_cast<float>(width),
                           static_cast<float>(height), 0.0f, 1.0f);
  m_context->RSSetViewports(1, &vp);
}

void D3D11Device::SetScissor(s32 x, s32 y, s32 width, s32 height)
{
  const CD3D11_RECT rc(x, y, x + width, y + height);
  m_context->RSSetScissorRects(1, &rc);
}

void D3D11Device::PreDrawCheck()
{
  if (m_current_framebuffer)
    m_current_framebuffer->CommitClear(m_context.Get());
}

void D3D11Device::Draw(u32 vertex_count, u32 base_vertex)
{
  PreDrawCheck();
  m_context->Draw(vertex_count, base_vertex);
}

void D3D11Device::DrawIndexed(u32 index_count, u32 base_index, u32 base_vertex)
{
  PreDrawCheck();
  m_context->DrawIndexed(index_count, base_index, base_vertex);
}

#if 0
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
  FrontendCommon::PostProcessingChain m_post_processing_chain;
  D3D11Texture m_post_processing_input_texture;
  std::vector<PostProcessingStage> m_post_processing_stages;
  Common::Timer m_post_processing_timer;

  bool D3D11Device::SetPostProcessingChain(const std::string_view& config)
{
  if (config.empty())
  {
    m_post_processing_input_texture.Destroy();
    m_post_processing_stages.clear();
    m_post_processing_chain.ClearStages();
    return true;
  }

  if (!m_post_processing_chain.CreateFromString(config))
    return false;

  m_post_processing_stages.clear();

  D3D11::ShaderCache shader_cache;
  shader_cache.Open(EmuFolders::Cache, m_device->GetFeatureLevel(), SHADER_CACHE_VERSION,
                    g_settings.gpu_use_debug_device);

  FrontendCommon::PostProcessingShaderGen shadergen(RenderAPI::D3D11, true);
  u32 max_ubo_size = 0;

  for (u32 i = 0; i < m_post_processing_chain.GetStageCount(); i++)
  {
    const FrontendCommon::PostProcessingShader& shader = m_post_processing_chain.GetShaderStage(i);
    const std::string vs = shadergen.GeneratePostProcessingVertexShader(shader);
    const std::string ps = shadergen.GeneratePostProcessingFragmentShader(shader);

    PostProcessingStage stage;
    stage.uniforms_size = shader.GetUniformsSize();
    stage.vertex_shader = shader_cache.GetVertexShader(m_device.Get(), vs);
    stage.pixel_shader = shader_cache.GetPixelShader(m_device.Get(), ps);
    if (!stage.vertex_shader || !stage.pixel_shader)
    {
      Log_ErrorPrintf("Failed to compile one or more post-processing shaders, disabling.");
      m_post_processing_stages.clear();
      m_post_processing_chain.ClearStages();
      return false;
    }

    max_ubo_size = std::max(max_ubo_size, stage.uniforms_size);
    m_post_processing_stages.push_back(std::move(stage));
  }

  if (m_push_uniform_buffer.GetSize() < max_ubo_size &&
      !m_push_uniform_buffer.Create(m_device.Get(), D3D11_BIND_CONSTANT_BUFFER, max_ubo_size))
  {
    Log_ErrorPrintf("Failed to allocate %u byte constant buffer for postprocessing", max_ubo_size);
    m_post_processing_stages.clear();
    m_post_processing_chain.ClearStages();
    return false;
  }

  m_post_processing_timer.Reset();
  return true;
}

bool D3D11Device::CheckPostProcessingRenderTargets(u32 target_width, u32 target_height)
{
  DebugAssert(!m_post_processing_stages.empty());

  const GPUTexture::Type type = GPUTexture::Type::RenderTarget;
  const GPUTexture::Format format = GPUTexture::Format::RGBA8;

  if (m_post_processing_input_texture.GetWidth() != target_width ||
      m_post_processing_input_texture.GetHeight() != target_height)
  {
    if (!m_post_processing_input_texture.Create(m_device.Get(), target_width, target_height, 1, 1, 1, type, format))
      return false;
  }

  const u32 target_count = (static_cast<u32>(m_post_processing_stages.size()) - 1);
  for (u32 i = 0; i < target_count; i++)
  {
    PostProcessingStage& pps = m_post_processing_stages[i];
    if (pps.output_texture.GetWidth() != target_width || pps.output_texture.GetHeight() != target_height)
    {
      if (!pps.output_texture.Create(m_device.Get(), target_width, target_height, 1, 1, 1, type, format))
        return false;
    }
  }

  return true;
}

void D3D11Device::ApplyPostProcessingChain(ID3D11RenderTargetView* final_target, s32 final_left, s32 final_top,
                                           s32 final_width, s32 final_height, D3D11Texture* texture, s32 texture_view_x,
                                           s32 texture_view_y, s32 texture_view_width, s32 texture_view_height,
                                           u32 target_width, u32 target_height)
{
  if (!CheckPostProcessingRenderTargets(target_width, target_height))
  {
    RenderDisplay(final_left, final_top, final_width, final_height, texture, texture_view_x, texture_view_y,
                  texture_view_width, texture_view_height, IsUsingLinearFiltering());
    return;
  }

  // downsample/upsample - use same viewport for remainder
  m_context->ClearRenderTargetView(m_post_processing_input_texture.GetD3DRTV(), s_clear_color.data());
  m_context->OMSetRenderTargets(1, m_post_processing_input_texture.GetD3DRTVArray(), nullptr);
  RenderDisplay(final_left, final_top, final_width, final_height, texture, texture_view_x, texture_view_y,
                texture_view_width, texture_view_height, IsUsingLinearFiltering());

  const s32 orig_texture_width = texture_view_width;
  const s32 orig_texture_height = texture_view_height;
  texture = &m_post_processing_input_texture;
  texture_view_x = final_left;
  texture_view_y = final_top;
  texture_view_width = final_width;
  texture_view_height = final_height;

  const u32 final_stage = static_cast<u32>(m_post_processing_stages.size()) - 1u;
  for (u32 i = 0; i < static_cast<u32>(m_post_processing_stages.size()); i++)
  {
    PostProcessingStage& pps = m_post_processing_stages[i];
    ID3D11RenderTargetView* rtv = (i == final_stage) ? final_target : pps.output_texture.GetD3DRTV();
    m_context->ClearRenderTargetView(rtv, s_clear_color.data());
    m_context->OMSetRenderTargets(1, &rtv, nullptr);

    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_context->VSSetShader(pps.vertex_shader.Get(), nullptr, 0);
    m_context->PSSetShader(pps.pixel_shader.Get(), nullptr, 0);
    m_context->PSSetShaderResources(0, 1, texture->GetD3DSRVArray());
    m_context->PSSetSamplers(0, 1, m_border_sampler.GetAddressOf());

    const auto map = m_push_uniform_buffer.Map(m_context.Get(), m_push_uniform_buffer.GetSize(), pps.uniforms_size);
    m_post_processing_chain.GetShaderStage(i).FillUniformBuffer(
      map.pointer, texture->GetWidth(), texture->GetHeight(), texture_view_x, texture_view_y, texture_view_width,
      texture_view_height, GetWindowWidth(), GetWindowHeight(), orig_texture_width, orig_texture_height,
      static_cast<float>(m_post_processing_timer.GetTimeSeconds()));
    m_push_uniform_buffer.Unmap(m_context.Get(), pps.uniforms_size);
    m_context->VSSetConstantBuffers(0, 1, m_push_uniform_buffer.GetD3DBufferArray());
    m_context->PSSetConstantBuffers(0, 1, m_push_uniform_buffer.GetD3DBufferArray());

    m_context->Draw(3, 0);

    if (i != final_stage)
      texture = &pps.output_texture;
  }

  ID3D11ShaderResourceView* null_srv = nullptr;
  m_context->PSSetShaderResources(0, 1, &null_srv);
}
void D3D11Device::RenderDisplay()
{
  const auto [left, top, width, height] = CalculateDrawRect(GetWindowWidth(), GetWindowHeight());

  if (HasDisplayTexture() && !m_post_processing_chain.IsEmpty())
  {
    ApplyPostProcessingChain(m_swap_chain_rtv.Get(), left, top, width, height,
                             static_cast<D3D11Texture*>(m_display_texture), m_display_texture_view_x,
                             m_display_texture_view_y, m_display_texture_view_width, m_display_texture_view_height,
                             GetWindowWidth(), GetWindowHeight());
    return;
  }

  m_context->ClearRenderTargetView(m_swap_chain_rtv.Get(), s_clear_color.data());
  m_context->OMSetRenderTargets(1, m_swap_chain_rtv.GetAddressOf(), nullptr);

  if (!HasDisplayTexture())
    return;

  RenderDisplay(left, top, width, height, static_cast<D3D11Texture*>(m_display_texture), m_display_texture_view_x,
                m_display_texture_view_y, m_display_texture_view_width, m_display_texture_view_height,
                IsUsingLinearFiltering());
}
bool D3D11Device::RenderScreenshot(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect,
                                   std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format)
{
  static constexpr GPUTexture::Format hdformat = GPUTexture::Format::RGBA8;

  D3D11Texture render_texture;
  if (!render_texture.Create(m_device.Get(), width, height, 1, 1, 1, GPUTexture::Type::RenderTarget, hdformat))
    return false;

  static constexpr std::array<float, 4> clear_color = {};
  m_context->ClearRenderTargetView(render_texture.GetD3DRTV(), clear_color.data());
  m_context->OMSetRenderTargets(1, render_texture.GetD3DRTVArray(), nullptr);

  if (HasDisplayTexture())
  {
    if (!m_post_processing_chain.IsEmpty())
    {
      ApplyPostProcessingChain(render_texture.GetD3DRTV(), draw_rect.left, draw_rect.top, draw_rect.GetWidth(),
                               draw_rect.GetHeight(), static_cast<D3D11Texture*>(m_display_texture),
                               m_display_texture_view_x, m_display_texture_view_y, m_display_texture_view_width,
                               m_display_texture_view_height, width, height);
    }
    else
    {
      RenderDisplay(draw_rect.left, draw_rect.top, draw_rect.GetWidth(), draw_rect.GetHeight(),
                    static_cast<D3D11Texture*>(m_display_texture), m_display_texture_view_x, m_display_texture_view_y,
                    m_display_texture_view_width, m_display_texture_view_height, IsUsingLinearFiltering());
    }
  }

  m_context->OMSetRenderTargets(0, nullptr, nullptr);

  const u32 stride = GPUTexture::GetPixelSize(hdformat) * width;
  out_pixels->resize(width * height);
  if (!DownloadTexture(&render_texture, 0, 0, width, height, out_pixels->data(), stride))
    return false;

  *out_stride = stride;
  *out_format = hdformat;
  return true;
}

#endif