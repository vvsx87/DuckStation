// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "d3d11_device.h"
#include "../host_settings.h"
#include "../shader_cache_version.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/path.h"
#include "common/rectangle.h"
#include "common/string_util.h"

#include "fmt/format.h"

#include <array>
#include <d3dcompiler.h>
#include <dxgi1_5.h>

Log_SetChannel(D3D11Device);

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

static constexpr std::array<DXGI_FORMAT, static_cast<u32>(GPUTexture::Format::Count)> s_dxgi_mapping = {
  {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B5G6R5_UNORM,
   DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_D16_UNORM}};

static constexpr std::array<float, 4> s_clear_color = {};

static unsigned s_next_bad_shader_id = 1;

static void SetD3DDebugObjectName(ID3D11DeviceChild* obj, const std::string_view& name)
{
#ifdef _DEBUG
  // WKPDID_D3DDebugObjectName
  static constexpr GUID guid = {0x429b8c22, 0x9188, 0x4b0c, 0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00};

  UINT existing_data_size;
  HRESULT hr = obj->GetPrivateData(guid, &existing_data_size, nullptr);
  if (SUCCEEDED(hr) && existing_data_size > 0)
    return;

  obj->SetPrivateData(guid, static_cast<UINT>(name.length()), name.data());
#endif
}

// TODO: FIXME
namespace Host {
extern bool IsFullscreen();
extern void SetFullscreen(bool enabled);
} // namespace Host

D3D11StreamBuffer::D3D11StreamBuffer() : m_size(0), m_position(0)
{
}

D3D11StreamBuffer::D3D11StreamBuffer(ComPtr<ID3D11Buffer> buffer) : m_buffer(std::move(buffer)), m_position(0)
{
  D3D11_BUFFER_DESC desc;
  m_buffer->GetDesc(&desc);
  m_size = desc.ByteWidth;
}

D3D11StreamBuffer::~D3D11StreamBuffer()
{
  Release();
}

bool D3D11StreamBuffer::Create(ID3D11Device* device, D3D11_BIND_FLAG bind_flags, u32 size)
{
  CD3D11_BUFFER_DESC desc(size, bind_flags, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE, 0, 0);
  ComPtr<ID3D11Buffer> buffer;
  HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Creating buffer failed: 0x%08X", hr);
    return false;
  }

  m_buffer = std::move(buffer);
  m_size = size;
  m_position = 0;

  D3D11_FEATURE_DATA_D3D11_OPTIONS options = {};
  hr = device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof(options));
  if (SUCCEEDED(hr))
  {
    if (bind_flags & D3D11_BIND_CONSTANT_BUFFER)
      m_use_map_no_overwrite = options.MapNoOverwriteOnDynamicConstantBuffer;
    else if (bind_flags & D3D11_BIND_SHADER_RESOURCE)
      m_use_map_no_overwrite = options.MapNoOverwriteOnDynamicBufferSRV;
    else
      m_use_map_no_overwrite = true;

    if (!m_use_map_no_overwrite)
    {
      Log_WarningPrintf("Unable to use MAP_NO_OVERWRITE on buffer with bind flag %u, this may affect performance. "
                        "Update your driver/operating system.",
                        static_cast<unsigned>(bind_flags));
    }
  }
  else
  {
    Log_WarningPrintf("ID3D11Device::CheckFeatureSupport() failed: 0x%08X", hr);
    m_use_map_no_overwrite = false;
  }

  return true;
}

void D3D11StreamBuffer::Adopt(ComPtr<ID3D11Buffer> buffer)
{
  m_buffer = std::move(buffer);

  D3D11_BUFFER_DESC desc;
  m_buffer->GetDesc(&desc);
  m_size = desc.ByteWidth;
  m_position = 0;
}

void D3D11StreamBuffer::Release()
{
  m_buffer.Reset();
}

D3D11StreamBuffer::MappingResult D3D11StreamBuffer::Map(ID3D11DeviceContext1* context, u32 alignment, u32 min_size)
{
  m_position = Common::AlignUp(m_position, alignment);
  if ((m_position + min_size) >= m_size || !m_use_map_no_overwrite)
  {
    // wrap around
    m_position = 0;
  }

  D3D11_MAPPED_SUBRESOURCE sr;
  const D3D11_MAP map_type = (m_position == 0) ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_WRITE_NO_OVERWRITE;
  const HRESULT hr = context->Map(m_buffer.Get(), 0, map_type, 0, &sr);
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Map failed: 0x%08X (alignment %u, minsize %u, size %u, position %u, map type %u)", hr, alignment,
                    min_size, m_size, m_position, static_cast<u32>(map_type));
    Panic("Map failed");
    return {};
  }

  return MappingResult{static_cast<char*>(sr.pData) + m_position, m_position, m_position / alignment,
                       (m_size - m_position) / alignment};
}

void D3D11StreamBuffer::Unmap(ID3D11DeviceContext1* context, u32 used_size)
{
  context->Unmap(m_buffer.Get(), 0);
  m_position += used_size;
}

D3D11Device::D3D11Device() = default;

D3D11Device::~D3D11Device()
{
  // Should all be torn down by now.
  Assert(!m_device);
}

RenderAPI D3D11Device::GetRenderAPI() const
{
  return RenderAPI::D3D11;
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
  DebugAssert((src_y + height) <= src->GetMipHeight(src_level));
  DebugAssert(dst_level < dst->GetLevels() && dst_layer < dst->GetLayers());
  DebugAssert((dst_x + width) <= dst->GetMipWidth(dst_level));
  DebugAssert((dst_y + height) <= dst->GetMipHeight(dst_level));

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
  DebugAssert((src_y + height) <= src->GetMipHeight(src_level));
  DebugAssert(dst_level < dst->GetLevels() && dst_layer < dst->GetLayers());
  DebugAssert((dst_x + width) <= dst->GetMipWidth(dst_level));
  DebugAssert((dst_y + height) <= dst->GetMipHeight(dst_level));
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

void D3D11Device::ClearRenderTarget(GPUTexture* t, u32 c)
{
  GPUDevice::ClearRenderTarget(t, c);
  if (m_current_framebuffer && m_current_framebuffer->GetRT() == t)
    static_cast<D3D11Texture*>(t)->CommitClear(m_context.Get());
}

void D3D11Device::ClearDepth(GPUTexture* t, float d)
{
  GPUDevice::ClearDepth(t, d);
  if (m_current_framebuffer && m_current_framebuffer->GetDS() == t)
    static_cast<D3D11Texture*>(t)->CommitClear(m_context.Get());
}

void D3D11Device::InvalidateRenderTarget(GPUTexture* t)
{
  GPUDevice::InvalidateRenderTarget(t);
  if (m_current_framebuffer && (m_current_framebuffer->GetRT() == t || m_current_framebuffer->GetDS() == t))
    static_cast<D3D11Texture*>(t)->CommitClear(m_context.Get());
}

bool D3D11Device::GetHostRefreshRate(float* refresh_rate)
{
  if (m_swap_chain && m_is_exclusive_fullscreen)
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

bool D3D11Device::CreateDevice(const std::string_view& adapter)
{
  UINT create_flags = 0;
  if (m_debug_device)
    create_flags |= D3D11_CREATE_DEVICE_DEBUG;

  ComPtr<IDXGIFactory> temp_dxgi_factory;
  HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(temp_dxgi_factory.GetAddressOf()));
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Failed to create DXGI factory: 0x%08X", hr);
    return false;
  }

  u32 adapter_index;
  if (!adapter.empty())
  {
    AdapterAndModeList adapter_info(GetAdapterAndModeList(temp_dxgi_factory.Get()));
    for (adapter_index = 0; adapter_index < static_cast<u32>(adapter_info.adapter_names.size()); adapter_index++)
    {
      if (adapter == adapter_info.adapter_names[adapter_index])
        break;
    }
    if (adapter_index == static_cast<u32>(adapter_info.adapter_names.size()))
    {
      // TODO: Log_Fmt
      Log_WarningPrintf(
        fmt::format("Could not find adapter '{}', using first ({})", adapter, adapter_info.adapter_names[0]).c_str());
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

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  hr =
    D3D11CreateDevice(dxgi_adapter.Get(), dxgi_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, nullptr,
                      create_flags, requested_feature_levels.data(), static_cast<UINT>(requested_feature_levels.size()),
                      D3D11_SDK_VERSION, device.GetAddressOf(), nullptr, context.GetAddressOf());
  // we re-grab these later, see below
  dxgi_adapter.Reset();
  temp_dxgi_factory.Reset();

  if (FAILED(hr))
  {
    Log_ErrorPrintf("Failed to create D3D device: 0x%08X", hr);
    return false;
  }
  else if (FAILED(hr = device.As(&m_device)) || FAILED(hr = context.As(&m_context)))
  {
    Log_ErrorPrintf("Failed to get D3D11.1 device: 0x%08X", hr);
    return false;
  }

  if (m_debug_device && IsDebuggerPresent())
  {
    ComPtr<ID3D11InfoQueue> info;
    hr = m_device.As(&info);
    if (SUCCEEDED(hr))
    {
      info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, TRUE);
      info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_WARNING, TRUE);
    }
  }

#ifdef _DEBUG
  if (m_debug_device)
    m_context.As(&m_annotation);
#endif

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

  SetFeatures();

  if (m_window_info.type != WindowInfo::Type::Surfaceless && !CreateSwapChain())
    return false;

  if (!CreateBuffers())
    return false;

  return true;
}

void D3D11Device::DestroyDevice()
{
  DestroyStagingBuffer();
  DestroyBuffers();
  m_context.Reset();
  m_device.Reset();
}

void D3D11Device::SetFeatures()
{
  const D3D_FEATURE_LEVEL feature_level = m_device->GetFeatureLevel();

  m_max_texture_size = D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  m_max_multisamples = 1;
  for (u32 multisamples = 2; multisamples < D3D11_MAX_MULTISAMPLE_SAMPLE_COUNT; multisamples++)
  {
    UINT num_quality_levels;
    if (SUCCEEDED(
          m_device->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, multisamples, &num_quality_levels)) &&
        num_quality_levels > 0)
    {
      m_max_multisamples = multisamples;
    }
  }

  m_features.dual_source_blend = true;
  m_features.per_sample_shading = (feature_level >= D3D_FEATURE_LEVEL_10_1);
  m_features.noperspective_interpolation = true;
  m_features.supports_texture_buffers = true;
  m_features.texture_buffers_emulated_with_ssbo = false;

  m_allow_tearing_supported = false;
  ComPtr<IDXGIFactory5> dxgi_factory5;
  HRESULT hr = m_dxgi_factory.As(&dxgi_factory5);
  if (SUCCEEDED(hr))
  {
    BOOL allow_tearing_supported = false;
    hr = dxgi_factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing_supported,
                                            sizeof(allow_tearing_supported));
    if (SUCCEEDED(hr))
      m_allow_tearing_supported = (allow_tearing_supported == TRUE);
  }
}

bool D3D11Device::GetRequestedExclusiveFullscreenModeDesc(IDXGIFactory5* factory, const RECT& window_rect, u32 width,
                                                          u32 height, float refresh_rate, DXGI_FORMAT format,
                                                          DXGI_MODE_DESC* fullscreen_mode, IDXGIOutput** output)
{
  // We need to find which monitor the window is located on.
  const Common::Rectangle<s32> client_rc_vec(window_rect.left, window_rect.top, window_rect.right, window_rect.bottom);

  // The window might be on a different adapter to which we are rendering.. so we have to enumerate them all.
  HRESULT hr;
  ComPtr<IDXGIOutput> first_output, intersecting_output;

  for (u32 adapter_index = 0; !intersecting_output; adapter_index++)
  {
    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapters1(adapter_index, adapter.GetAddressOf());
    if (hr == DXGI_ERROR_NOT_FOUND)
      break;
    else if (FAILED(hr))
      continue;

    for (u32 output_index = 0;; output_index++)
    {
      ComPtr<IDXGIOutput> this_output;
      DXGI_OUTPUT_DESC output_desc;
      hr = adapter->EnumOutputs(output_index, this_output.GetAddressOf());
      if (hr == DXGI_ERROR_NOT_FOUND)
        break;
      else if (FAILED(hr) || FAILED(this_output->GetDesc(&output_desc)))
        continue;

      const Common::Rectangle<s32> output_rc(output_desc.DesktopCoordinates.left, output_desc.DesktopCoordinates.top,
                                             output_desc.DesktopCoordinates.right,
                                             output_desc.DesktopCoordinates.bottom);
      if (!client_rc_vec.Intersects(output_rc))
      {
        intersecting_output = std::move(this_output);
        break;
      }

      // Fallback to the first monitor.
      if (!first_output)
        first_output = std::move(this_output);
    }
  }

  if (!intersecting_output)
  {
    if (!first_output)
    {
      Log_ErrorPrintf("No DXGI output found. Can't use exclusive fullscreen.");
      return false;
    }

    Log_WarningPrint("No DXGI output found for window, using first.");
    intersecting_output = std::move(first_output);
  }

  DXGI_MODE_DESC request_mode = {};
  request_mode.Width = width;
  request_mode.Height = height;
  request_mode.Format = format;
  request_mode.RefreshRate.Numerator = static_cast<UINT>(std::floor(refresh_rate * 1000.0f));
  request_mode.RefreshRate.Denominator = 1000u;

  if (FAILED(hr = intersecting_output->FindClosestMatchingMode(&request_mode, fullscreen_mode, nullptr)) ||
      request_mode.Format != format)
  {
    Log_ErrorPrintf("Failed to find closest matching mode, hr=%08X", hr);
    return false;
  }

  *output = intersecting_output.Get();
  intersecting_output->AddRef();
  return true;
}

bool D3D11Device::CreateSwapChain()
{
  constexpr DXGI_FORMAT swap_chain_format = DXGI_FORMAT_R8G8B8A8_UNORM;

  if (m_window_info.type != WindowInfo::Type::Win32)
    return false;

  const HWND window_hwnd = reinterpret_cast<HWND>(m_window_info.window_handle);
  RECT client_rc{};
  GetClientRect(window_hwnd, &client_rc);

  DXGI_MODE_DESC fullscreen_mode = {};
  ComPtr<IDXGIOutput> fullscreen_output;
  if (Host::IsFullscreen())
  {
    u32 fullscreen_width, fullscreen_height;
    float fullscreen_refresh_rate;
    m_is_exclusive_fullscreen =
      GetRequestedExclusiveFullscreenMode(&fullscreen_width, &fullscreen_height, &fullscreen_refresh_rate) &&
      GetRequestedExclusiveFullscreenModeDesc(m_dxgi_factory.Get(), client_rc, fullscreen_width, fullscreen_height,
                                              fullscreen_refresh_rate, swap_chain_format, &fullscreen_mode,
                                              fullscreen_output.GetAddressOf());
  }
  else
  {
    m_is_exclusive_fullscreen = false;
  }

  m_using_flip_model_swap_chain =
    !Host::GetBoolSettingValue("Display", "UseBlitSwapChain", false) || m_is_exclusive_fullscreen;

  DXGI_SWAP_CHAIN_DESC1 swap_chain_desc = {};
  swap_chain_desc.Width = static_cast<u32>(client_rc.right - client_rc.left);
  swap_chain_desc.Height = static_cast<u32>(client_rc.bottom - client_rc.top);
  swap_chain_desc.Format = swap_chain_format;
  swap_chain_desc.SampleDesc.Count = 1;
  swap_chain_desc.BufferCount = 3;
  swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swap_chain_desc.SwapEffect = m_using_flip_model_swap_chain ? DXGI_SWAP_EFFECT_FLIP_DISCARD : DXGI_SWAP_EFFECT_DISCARD;

  m_using_allow_tearing = (m_allow_tearing_supported && m_using_flip_model_swap_chain && !m_is_exclusive_fullscreen);
  if (m_using_allow_tearing)
    swap_chain_desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

  HRESULT hr = S_OK;

  if (m_is_exclusive_fullscreen)
  {
    DXGI_SWAP_CHAIN_DESC1 fs_sd_desc = swap_chain_desc;
    DXGI_SWAP_CHAIN_FULLSCREEN_DESC fs_desc = {};

    fs_sd_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    fs_sd_desc.Width = fullscreen_mode.Width;
    fs_sd_desc.Height = fullscreen_mode.Height;
    fs_desc.RefreshRate = fullscreen_mode.RefreshRate;
    fs_desc.ScanlineOrdering = fullscreen_mode.ScanlineOrdering;
    fs_desc.Scaling = fullscreen_mode.Scaling;
    fs_desc.Windowed = FALSE;

    Log_VerbosePrintf("Creating a %dx%d exclusive fullscreen swap chain", fs_sd_desc.Width, fs_sd_desc.Height);
    hr = m_dxgi_factory->CreateSwapChainForHwnd(m_device.Get(), window_hwnd, &fs_sd_desc, &fs_desc,
                                                fullscreen_output.Get(), m_swap_chain.ReleaseAndGetAddressOf());
    if (FAILED(hr))
    {
      Log_WarningPrintf("Failed to create fullscreen swap chain, trying windowed.");
      m_is_exclusive_fullscreen = false;
      m_using_allow_tearing = m_allow_tearing_supported && m_using_flip_model_swap_chain;
    }
  }

  if (!m_is_exclusive_fullscreen)
  {
    Log_VerbosePrintf("Creating a %dx%d %s windowed swap chain", swap_chain_desc.Width, swap_chain_desc.Height,
                      m_using_flip_model_swap_chain ? "flip-discard" : "discard");
    hr = m_dxgi_factory->CreateSwapChainForHwnd(m_device.Get(), window_hwnd, &swap_chain_desc, nullptr, nullptr,
                                                m_swap_chain.ReleaseAndGetAddressOf());
  }

  if (FAILED(hr) && m_using_flip_model_swap_chain)
  {
    Log_WarningPrintf("Failed to create a flip-discard swap chain, trying discard.");
    swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swap_chain_desc.Flags = 0;
    m_using_flip_model_swap_chain = false;
    m_using_allow_tearing = false;

    hr = m_dxgi_factory->CreateSwapChainForHwnd(m_device.Get(), window_hwnd, &swap_chain_desc, nullptr, nullptr,
                                                m_swap_chain.ReleaseAndGetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("CreateSwapChainForHwnd failed: 0x%08X", hr);
      return false;
    }
  }

  hr = m_dxgi_factory->MakeWindowAssociation(window_hwnd, DXGI_MWA_NO_WINDOW_CHANGES);
  if (FAILED(hr))
    Log_WarningPrintf("MakeWindowAssociation() to disable ALT+ENTER failed");

  if (!CreateSwapChainRTV())
  {
    DestroySwapChain();
    return false;
  }

  // Render a frame as soon as possible to clear out whatever was previously being displayed.
  m_context->ClearRenderTargetView(m_swap_chain_rtv.Get(), s_clear_color.data());
  m_swap_chain->Present(0, m_using_allow_tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
  return true;
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
  hr = m_device->CreateRenderTargetView(backbuffer.Get(), &rtv_desc, m_swap_chain_rtv.ReleaseAndGetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("CreateRenderTargetView for swap chain failed: 0x%08X", hr);
    m_swap_chain_rtv.Reset();
    return false;
  }

  m_window_info.surface_width = backbuffer_desc.Width;
  m_window_info.surface_height = backbuffer_desc.Height;
  Log_VerbosePrintf("Swap chain buffer size: %ux%u", m_window_info.surface_width, m_window_info.surface_height);

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

void D3D11Device::DestroySwapChain()
{
  if (!m_swap_chain)
    return;

  m_swap_chain_rtv.Reset();

  // switch out of fullscreen before destroying
  BOOL is_fullscreen;
  if (SUCCEEDED(m_swap_chain->GetFullscreenState(&is_fullscreen, nullptr)) && is_fullscreen)
    m_swap_chain->SetFullscreenState(FALSE, nullptr);

  m_swap_chain.Reset();
  m_is_exclusive_fullscreen = false;
}

bool D3D11Device::UpdateWindow()
{
  DestroySwapChain();

  if (!AcquireWindow(false))
    return false;

  if (m_window_info.type != WindowInfo::Type::Surfaceless && !CreateSwapChain())
  {
    Log_ErrorPrintf("Failed to create swap chain on updated window");
    return false;
  }

  return true;
}

void D3D11Device::DestroySurface()
{
  DestroySwapChain();
}

std::string D3D11Device::GetShaderCacheBaseName(const std::string_view& type) const
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

  return fmt::format("d3d_{}_{}{}", type, flname, m_debug_device ? "_debug" : "");
}

void D3D11Device::ResizeWindow(s32 new_window_width, s32 new_window_height, float new_window_scale)
{
  if (!m_swap_chain || m_is_exclusive_fullscreen)
    return;

  m_window_info.surface_scale = new_window_scale;

  if (m_window_info.surface_width == static_cast<u32>(new_window_width) &&
      m_window_info.surface_height == static_cast<u32>(new_window_height))
  {
    return;
  }

  m_swap_chain_rtv.Reset();

  HRESULT hr = m_swap_chain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN,
                                           m_using_allow_tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
  if (FAILED(hr))
    Log_ErrorPrintf("ResizeBuffers() failed: 0x%08X", hr);

  if (!CreateSwapChainRTV())
    Panic("Failed to recreate swap chain RTV after resize");
}

bool D3D11Device::SupportsExclusiveFullscreen() const
{
  return true;
}

bool D3D11Device::CreateBuffers()
{
  if (!m_vertex_buffer.Create(m_device.Get(), D3D11_BIND_VERTEX_BUFFER, VERTEX_BUFFER_SIZE) ||
      !m_index_buffer.Create(m_device.Get(), D3D11_BIND_INDEX_BUFFER, INDEX_BUFFER_SIZE) ||
      !m_uniform_buffer.Create(m_device.Get(), D3D11_BIND_CONSTANT_BUFFER, UNIFORM_BUFFER_SIZE))
  {
    Log_ErrorPrintf("Failed to create vertex/index/uniform buffers.");
    return false;
  }

  // Index buffer never changes :)
  m_context->IASetIndexBuffer(m_index_buffer.GetD3DBuffer(), DXGI_FORMAT_R16_UINT, 0);
  return true;
}

void D3D11Device::DestroyBuffers()
{
  m_uniform_buffer.Release();
  m_vertex_buffer.Release();
  m_index_buffer.Release();
}

bool D3D11Device::BeginPresent(bool skip_present)
{
  if (skip_present || !m_swap_chain)
    return false;

  // Check if we lost exclusive fullscreen. If so, notify the host, so it can switch to windowed mode.
  // This might get called repeatedly if it takes a while to switch back, that's the host's problem.
  BOOL is_fullscreen;
  if (m_is_exclusive_fullscreen &&
      (FAILED(m_swap_chain->GetFullscreenState(&is_fullscreen, nullptr)) || !is_fullscreen))
  {
    Host::SetFullscreen(false);
    return false;
  }

  // When using vsync, the time here seems to include the time for the buffer to become available.
  // This blows our our GPU usage number considerably, so read the timestamp before the final blit
  // in this configuration. It does reduce accuracy a little, but better than seeing 100% all of
  // the time, when it's more like a couple of percent.
  if (m_vsync_enabled && m_gpu_timing_enabled)
    PopTimestampQuery();

  static constexpr float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  m_context->ClearRenderTargetView(m_swap_chain_rtv.Get(), clear_color);
  m_context->OMSetRenderTargets(1, m_swap_chain_rtv.GetAddressOf(), nullptr);
  m_current_framebuffer = nullptr;
  return true;
}

void D3D11Device::EndPresent()
{
  DebugAssert(!m_current_framebuffer);

  if (!m_vsync_enabled && m_gpu_timing_enabled)
    PopTimestampQuery();

  if (!m_vsync_enabled && m_using_allow_tearing)
    m_swap_chain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
  else
    m_swap_chain->Present(BoolToUInt32(m_vsync_enabled), 0);

  if (m_gpu_timing_enabled)
    KickTimestampQuery();
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

D3D11Framebuffer::~D3D11Framebuffer()
{
  D3D11Device::GetInstance().UnbindFramebuffer(this);
}

void D3D11Framebuffer::SetDebugName(const std::string_view& name)
{
  if (m_rtv)
    SetD3DDebugObjectName(m_rtv.Get(), fmt::format("{} RTV", name));
  if (m_dsv)
    SetD3DDebugObjectName(m_dsv.Get(), fmt::format("{} DSV", name));
}

void D3D11Framebuffer::CommitClear(ID3D11DeviceContext1* context)
{
  if (UNLIKELY(m_rt && m_rt->GetState() != GPUTexture::State::Dirty))
  {
    if (m_rt->GetState() == GPUTexture::State::Invalidated)
      context->DiscardView(m_rtv.Get());
    else
      context->ClearRenderTargetView(m_rtv.Get(), m_rt->GetUNormClearColor().data());

    m_rt->SetState(GPUTexture::State::Dirty);
  }

  if (UNLIKELY(m_ds && m_ds->GetState() != GPUTexture::State::Dirty))
  {
    if (m_ds->GetState() == GPUTexture::State::Invalidated)
      context->DiscardView(m_dsv.Get());
    else
      context->ClearDepthStencilView(m_dsv.Get(), D3D11_CLEAR_DEPTH, m_ds->GetClearDepth(), 0);

    m_ds->SetState(GPUTexture::State::Dirty);
  }
}

std::unique_ptr<GPUFramebuffer> D3D11Device::CreateFramebuffer(GPUTexture* rt_or_ds, GPUTexture* ds)
{
  DebugAssert((rt_or_ds || ds) && (!rt_or_ds || rt_or_ds->IsRenderTarget() || (rt_or_ds->IsDepthStencil() && !ds)));
  D3D11Texture* RT = static_cast<D3D11Texture*>((rt_or_ds && rt_or_ds->IsDepthStencil()) ? nullptr : rt_or_ds);
  D3D11Texture* DS = static_cast<D3D11Texture*>((rt_or_ds && rt_or_ds->IsDepthStencil()) ? rt_or_ds : ds);

  ComPtr<ID3D11RenderTargetView> rtv;
  if (RT)
  {
    rtv = RT->GetD3DRTV();
    Assert(rtv);
  }

  ComPtr<ID3D11DepthStencilView> dsv;
  if (DS)
  {
    dsv = DS->GetD3DDSV();
    Assert(dsv);
  }

  return std::unique_ptr<GPUFramebuffer>(new D3D11Framebuffer(RT, DS, RT ? RT->GetWidth() : DS->GetWidth(),
                                                              RT ? RT->GetHeight() : DS->GetHeight(), std::move(rtv),
                                                              std::move(dsv)));
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
               m_debug_device ? flags_debug : flags_non_debug, 0, blob.GetAddressOf(), error_blob.GetAddressOf());

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
      GetShaderDumpPath(fmt::format("bad_shader_{}.txt", s_next_bad_shader_id++)).c_str(), "wb");
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
                             ComPtr<ID3D11PixelShader> ps, D3D11_PRIMITIVE_TOPOLOGY topology, u32 vertex_stride,
                             u32 blend_factor)
  : m_rs(std::move(rs)), m_ds(std::move(ds)), m_bs(std::move(bs)), m_il(std::move(il)), m_vs(std::move(vs)),
    m_ps(std::move(ps)), m_topology(topology), m_vertex_stride(vertex_stride), m_blend_factor(blend_factor),
    m_blend_factor_float(GPUDevice::RGBA8ToFloat(blend_factor))
{
}

D3D11Pipeline::~D3D11Pipeline()
{
  D3D11Device::GetInstance().UnbindPipeline(this);
}

void D3D11Pipeline::SetDebugName(const std::string_view& name)
{
  // can't label this directly
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
  u32 vertex_stride = 0;
  if (!config.input_layout.vertex_attributes.empty())
  {
    il = GetInputLayout(config.input_layout, static_cast<const D3D11Shader*>(config.vertex_shader));
    vertex_stride = config.input_layout.vertex_stride;
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
                      static_cast<const D3D11Shader*>(config.fragment_shader)->GetPixelShader(),
                      primitives[static_cast<u8>(config.primitive)], vertex_stride, config.blend.constant));
}

D3D11Texture::D3D11Texture() = default;

D3D11Texture::D3D11Texture(ComPtr<ID3D11Texture2D> texture, ComPtr<ID3D11ShaderResourceView> srv,
                           ComPtr<ID3D11View> rtv)
  : m_texture(std::move(texture)), m_srv(std::move(srv)), m_rtv_dsv(std::move(rtv))
{
  const D3D11_TEXTURE2D_DESC desc = GetDesc();
  m_width = static_cast<u16>(desc.Width);
  m_height = static_cast<u16>(desc.Height);
  m_layers = static_cast<u8>(desc.ArraySize);
  m_levels = static_cast<u8>(desc.MipLevels);
  m_samples = static_cast<u8>(desc.SampleDesc.Count);
  m_format = LookupBaseFormat(desc.Format);
  m_dynamic = (desc.Usage == D3D11_USAGE_DYNAMIC);
}

D3D11Texture::~D3D11Texture()
{
  Destroy();
}

DXGI_FORMAT D3D11Texture::GetDXGIFormat(Format format)
{
  return s_dxgi_mapping[static_cast<u8>(format)];
}

GPUTexture::Format D3D11Texture::LookupBaseFormat(DXGI_FORMAT dformat)
{
  for (u32 i = 0; i < static_cast<u32>(s_dxgi_mapping.size()); i++)
  {
    if (s_dxgi_mapping[i] == dformat)
      return static_cast<Format>(i);
  }
  return GPUTexture::Format::Unknown;
}

D3D11_TEXTURE2D_DESC D3D11Texture::GetDesc() const
{
  D3D11_TEXTURE2D_DESC desc;
  m_texture->GetDesc(&desc);
  return desc;
}

void D3D11Texture::CommitClear(ID3D11DeviceContext1* context)
{
  if (m_state == GPUTexture::State::Dirty)
    return;

  if (IsDepthStencil())
  {
    if (m_state == GPUTexture::State::Invalidated)
      context->DiscardView(GetD3DDSV());
    else
      context->ClearDepthStencilView(GetD3DDSV(), D3D11_CLEAR_DEPTH, GetClearDepth(), 0);
  }
  else if (IsRenderTarget())
  {
    if (m_state == GPUTexture::State::Invalidated)
      context->DiscardView(GetD3DRTV());
    else
      context->ClearRenderTargetView(GetD3DRTV(), GetUNormClearColor().data());
  }

  m_state = GPUTexture::State::Dirty;
}

bool D3D11Texture::IsValid() const
{
  return static_cast<bool>(m_texture);
}

bool D3D11Texture::Update(u32 x, u32 y, u32 width, u32 height, const void* data, u32 pitch, u32 layer /*= 0*/,
                          u32 level /*= 0*/)
{
  if (m_dynamic)
  {
    void* map;
    u32 map_stride;
    if (!Map(&map, &map_stride, x, y, width, height, layer, level))
      return false;

    StringUtil::StrideMemCpy(map, map_stride, data, pitch, GetPixelSize() * width, height);
    Unmap();
    return true;
  }

  const CD3D11_BOX box(static_cast<LONG>(x), static_cast<LONG>(y), 0, static_cast<LONG>(x + width),
                       static_cast<LONG>(y + height), 1);
  const u32 srnum = D3D11CalcSubresource(level, layer, m_levels);

  ID3D11DeviceContext1* context = D3D11Device::GetD3DContext();
  CommitClear(context);
  context->UpdateSubresource(m_texture.Get(), srnum, &box, data, pitch, 0);
  m_state = GPUTexture::State::Dirty;
  return true;
}

bool D3D11Texture::Map(void** map, u32* map_stride, u32 x, u32 y, u32 width, u32 height, u32 layer /*= 0*/,
                       u32 level /*= 0*/)
{
  if (!m_dynamic || (x + width) > GetMipWidth(level) || (y + height) > GetMipHeight(level) || layer > m_layers ||
      level > m_levels)
  {
    return false;
  }

  const bool discard = (width == m_width && height == m_height);
  const u32 srnum = D3D11CalcSubresource(level, layer, m_levels);

  ID3D11DeviceContext1* context = D3D11Device::GetD3DContext();
  CommitClear(context);

  D3D11_MAPPED_SUBRESOURCE sr;
  HRESULT hr = context->Map(m_texture.Get(), srnum, discard ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_READ_WRITE, 0, &sr);
  if (FAILED(hr))
  {
    Log_ErrorPrintf("Map pixels texture failed: %08X", hr);
    return false;
  }

  *map = static_cast<u8*>(sr.pData) + (y * sr.RowPitch) + (x * GetPixelSize());
  *map_stride = sr.RowPitch;
  m_mapped_subresource = srnum;
  m_state = GPUTexture::State::Dirty;
  return true;
}

void D3D11Texture::Unmap()
{
  D3D11Device::GetD3DContext()->Unmap(m_texture.Get(), m_mapped_subresource);
  m_mapped_subresource = 0;
}

void D3D11Texture::SetDebugName(const std::string_view& name)
{
  SetD3DDebugObjectName(m_texture.Get(), name);
}

bool D3D11Texture::Create(ID3D11Device* device, u32 width, u32 height, u32 layers, u32 levels, u32 samples, Type type,
                          Format format, const void* initial_data /* = nullptr */, u32 initial_data_stride /* = 0 */,
                          bool dynamic /* = false */)
{
  if (!ValidateConfig(width, height, layers, layers, samples, type, format))
    return false;

  u32 bind_flags = 0;
  switch (type)
  {
    case Type::RenderTarget:
      bind_flags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      break;
    case Type::DepthStencil:
      bind_flags = D3D11_BIND_DEPTH_STENCIL; // | D3D11_BIND_SHADER_RESOURCE;
      break;
    case Type::Texture:
      bind_flags = D3D11_BIND_SHADER_RESOURCE;
      break;
    case Type::RWTexture:
      bind_flags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
      break;
    default:
      break;
  }

  CD3D11_TEXTURE2D_DESC desc(GetDXGIFormat(format), width, height, layers, levels, bind_flags,
                             dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT, dynamic ? D3D11_CPU_ACCESS_WRITE : 0,
                             samples, 0, 0);

  D3D11_SUBRESOURCE_DATA srd;
  srd.pSysMem = initial_data;
  srd.SysMemPitch = initial_data_stride;
  srd.SysMemSlicePitch = initial_data_stride * height;

  ComPtr<ID3D11Texture2D> texture;
  const HRESULT tex_hr = device->CreateTexture2D(&desc, initial_data ? &srd : nullptr, texture.GetAddressOf());
  if (FAILED(tex_hr))
  {
    Log_ErrorPrintf(
      "Create texture failed: 0x%08X (%ux%u levels:%u samples:%u format:%u bind_flags:%X initial_data:%p)", tex_hr,
      width, height, levels, samples, static_cast<unsigned>(format), bind_flags, initial_data);
    return false;
  }

  ComPtr<ID3D11ShaderResourceView> srv;
  if (bind_flags & D3D11_BIND_SHADER_RESOURCE)
  {
    const D3D11_SRV_DIMENSION srv_dimension =
      (desc.SampleDesc.Count > 1) ?
        D3D11_SRV_DIMENSION_TEXTURE2DMS :
        (desc.ArraySize > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DARRAY : D3D11_SRV_DIMENSION_TEXTURE2D);
    const CD3D11_SHADER_RESOURCE_VIEW_DESC srv_desc(srv_dimension, desc.Format, 0, desc.MipLevels, 0, desc.ArraySize);
    const HRESULT hr = device->CreateShaderResourceView(texture.Get(), &srv_desc, srv.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create SRV for texture failed: 0x%08X", hr);
      return false;
    }
  }

  ComPtr<ID3D11View> rtv_dsv;
  if (bind_flags & D3D11_BIND_RENDER_TARGET)
  {
    const D3D11_RTV_DIMENSION rtv_dimension =
      (desc.SampleDesc.Count > 1) ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
    const CD3D11_RENDER_TARGET_VIEW_DESC rtv_desc(rtv_dimension, desc.Format, 0, 0, desc.ArraySize);
    ComPtr<ID3D11RenderTargetView> rtv;
    const HRESULT hr = device->CreateRenderTargetView(texture.Get(), &rtv_desc, rtv.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create RTV for texture failed: 0x%08X", hr);
      return false;
    }

    rtv_dsv = std::move(rtv);
  }
  else if (bind_flags & D3D11_BIND_DEPTH_STENCIL)
  {
    const D3D11_DSV_DIMENSION dsv_dimension =
      (desc.SampleDesc.Count > 1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
    const CD3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc(dsv_dimension, desc.Format, 0, 0, desc.ArraySize);
    ComPtr<ID3D11DepthStencilView> dsv;
    const HRESULT hr = device->CreateDepthStencilView(texture.Get(), &dsv_desc, dsv.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create DSV for texture failed: 0x%08X", hr);
      return false;
    }

    rtv_dsv = std::move(dsv);
  }

  m_texture = std::move(texture);
  m_srv = std::move(srv);
  m_rtv_dsv = std::move(rtv_dsv);
  m_width = static_cast<u16>(width);
  m_height = static_cast<u16>(height);
  m_layers = static_cast<u8>(layers);
  m_levels = static_cast<u8>(levels);
  m_samples = static_cast<u8>(samples);
  m_type = type;
  m_format = format;
  m_dynamic = dynamic;
  return true;
}

bool D3D11Texture::Adopt(ID3D11Device* device, ComPtr<ID3D11Texture2D> texture)
{
  D3D11_TEXTURE2D_DESC desc;
  texture->GetDesc(&desc);

  ComPtr<ID3D11ShaderResourceView> srv;
  if (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
  {
    const D3D11_SRV_DIMENSION srv_dimension =
      (desc.SampleDesc.Count > 1) ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
    const CD3D11_SHADER_RESOURCE_VIEW_DESC srv_desc(srv_dimension, desc.Format, 0, desc.MipLevels, 0, desc.ArraySize);
    const HRESULT hr = device->CreateShaderResourceView(texture.Get(), &srv_desc, srv.ReleaseAndGetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create SRV for adopted texture failed: 0x%08X", hr);
      return false;
    }
  }

  ComPtr<ID3D11View> rtv_dsv;
  if (desc.BindFlags & D3D11_BIND_RENDER_TARGET)
  {
    const D3D11_RTV_DIMENSION rtv_dimension =
      (desc.SampleDesc.Count > 1) ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
    const CD3D11_RENDER_TARGET_VIEW_DESC rtv_desc(rtv_dimension, desc.Format, 0, 0, desc.ArraySize);
    ComPtr<ID3D11RenderTargetView> rtv;
    const HRESULT hr = device->CreateRenderTargetView(texture.Get(), &rtv_desc, rtv.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create RTV for adopted texture failed: 0x%08X", hr);
      return false;
    }

    rtv_dsv = std::move(rtv);
  }
  else if (desc.BindFlags & D3D11_BIND_DEPTH_STENCIL)
  {
    const D3D11_DSV_DIMENSION dsv_dimension =
      (desc.SampleDesc.Count > 1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
    const CD3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc(dsv_dimension, desc.Format, 0, 0, desc.ArraySize);
    ComPtr<ID3D11DepthStencilView> dsv;
    const HRESULT hr = device->CreateDepthStencilView(texture.Get(), &dsv_desc, dsv.GetAddressOf());
    if (FAILED(hr))
    {
      Log_ErrorPrintf("Create DSV for adopted texture failed: 0x%08X", hr);
      return false;
    }

    rtv_dsv = std::move(dsv);
  }

  m_texture = std::move(texture);
  m_srv = std::move(srv);
  m_rtv_dsv = std::move(rtv_dsv);
  m_width = static_cast<u16>(desc.Width);
  m_height = static_cast<u16>(desc.Height);
  m_layers = static_cast<u8>(desc.ArraySize);
  m_levels = static_cast<u8>(desc.MipLevels);
  m_samples = static_cast<u8>(desc.SampleDesc.Count);
  m_dynamic = (desc.Usage == D3D11_USAGE_DYNAMIC);
  m_state = GPUTexture::State::Dirty;
  return true;
}

void D3D11Texture::Destroy()
{
  D3D11Device::GetInstance().UnbindTexture(this);
  m_rtv_dsv.Reset();
  m_srv.Reset();
  m_texture.Reset();
  m_dynamic = false;
  ClearBaseProperties();
}

D3D11TextureBuffer::D3D11TextureBuffer(Format format, u32 size_in_elements) : GPUTextureBuffer(format, size_in_elements)
{
}

D3D11TextureBuffer::~D3D11TextureBuffer() = default;

bool D3D11TextureBuffer::CreateBuffer(ID3D11Device* device)
{
  if (!m_buffer.Create(device, D3D11_BIND_SHADER_RESOURCE, GetSizeInBytes()))
    return false;

  static constexpr std::array<DXGI_FORMAT, static_cast<u32>(Format::MaxCount)> dxgi_formats = {{
    DXGI_FORMAT_R16_UINT,
  }};

  CD3D11_SHADER_RESOURCE_VIEW_DESC srv_desc(m_buffer.GetD3DBuffer(), dxgi_formats[static_cast<u32>(m_format)], 0,
                                            m_size_in_elements);
  const HRESULT hr = device->CreateShaderResourceView(m_buffer.GetD3DBuffer(), &srv_desc, m_srv.GetAddressOf());
  if (FAILED(hr))
  {
    Log_ErrorPrintf("CreateShaderResourceView() failed: %08X", hr);
    return false;
  }

  return true;
}

void* D3D11TextureBuffer::Map(u32 required_elements)
{
  const u32 esize = GetElementSize(m_format);
  const auto res = m_buffer.Map(D3D11Device::GetD3DContext(), esize, esize * required_elements);
  m_current_position = res.index_aligned;
  return res.pointer;
}

void D3D11TextureBuffer::Unmap(u32 used_elements)
{
  m_buffer.Unmap(D3D11Device::GetD3DContext(), used_elements * GetElementSize(m_format));
}

std::unique_ptr<GPUTextureBuffer> D3D11Device::CreateTextureBuffer(GPUTextureBuffer::Format format,
                                                                   u32 size_in_elements)
{
  std::unique_ptr<D3D11TextureBuffer> tb = std::make_unique<D3D11TextureBuffer>(format, size_in_elements);
  if (!tb->CreateBuffer(m_device.Get()))
    tb.reset();

  return tb;
}

void D3D11Device::PushDebugGroup(const char* fmt, ...)
{
#ifdef _DEBUG
  if (!m_annotation)
    return;

  std::va_list ap;
  va_start(ap, fmt);
  std::string str(StringUtil::StdStringFromFormatV(fmt, ap));
  va_end(ap);

  m_annotation->BeginEvent(StringUtil::UTF8StringToWideString(str).c_str());
#endif
}

void D3D11Device::PopDebugGroup()
{
#ifdef _DEBUG
  if (!m_annotation)
    return;

  m_annotation->EndEvent();
#endif
}

void D3D11Device::InsertDebugMessage(const char* fmt, ...)
{
#ifdef _DEBUG
  if (!m_annotation)
    return;

  std::va_list ap;
  va_start(ap, fmt);
  std::string str(StringUtil::StdStringFromFormatV(fmt, ap));
  va_end(ap);

  m_annotation->SetMarker(StringUtil::UTF8StringToWideString(str).c_str());
#endif
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
}

void D3D11Device::PushUniformBuffer(const void* data, u32 data_size)
{
  const u32 used_space = Common::AlignUpPow2(data_size, UNIFORM_BUFFER_ALIGNMENT);
  const auto res = m_uniform_buffer.Map(m_context.Get(), UNIFORM_BUFFER_ALIGNMENT, used_space);
  std::memcpy(res.pointer, data, data_size);
  m_uniform_buffer.Unmap(m_context.Get(), data_size);

  const UINT first_constant = (res.index_aligned * UNIFORM_BUFFER_ALIGNMENT) / 16u;
  const UINT num_constants = (used_space * UNIFORM_BUFFER_ALIGNMENT) / 16u;
  m_context->VSSetConstantBuffers1(0, 1, m_uniform_buffer.GetD3DBufferArray(), &first_constant, &num_constants);
  m_context->PSSetConstantBuffers1(0, 1, m_uniform_buffer.GetD3DBufferArray(), &first_constant, &num_constants);
}

void* D3D11Device::MapUniformBuffer(u32 size)
{
  const u32 used_space = Common::AlignUpPow2(size, UNIFORM_BUFFER_ALIGNMENT);
  const auto res = m_uniform_buffer.Map(m_context.Get(), UNIFORM_BUFFER_ALIGNMENT, used_space);
  return res.pointer;
}

void D3D11Device::UnmapUniformBuffer(u32 size)
{
  const u32 used_space = Common::AlignUpPow2(size, UNIFORM_BUFFER_ALIGNMENT);
  const UINT first_constant = m_uniform_buffer.GetPosition() / 16u;
  const UINT num_constants = used_space / 16u;

  m_uniform_buffer.Unmap(m_context.Get(), used_space);
  m_context->VSSetConstantBuffers1(0, 1, m_uniform_buffer.GetD3DBufferArray(), &first_constant, &num_constants);
  m_context->PSSetConstantBuffers1(0, 1, m_uniform_buffer.GetD3DBufferArray(), &first_constant, &num_constants);
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

  // Make sure textures aren't bound.
  if (D3D11Texture* rt = static_cast<D3D11Texture*>(fb->GetRT()); rt)
  {
    const ID3D11ShaderResourceView* srv = rt->GetD3DSRV();
    for (u32 i = 0; i < MAX_TEXTURE_SAMPLERS; i++)
    {
      if (m_current_textures[i] == srv)
      {
        m_current_textures[i] = nullptr;
        m_context->PSSetShaderResources(i, 1, &m_current_textures[i]);
      }
    }
  }
  if (D3D11Texture* ds = static_cast<D3D11Texture*>(fb->GetDS()); ds)
  {
    const ID3D11ShaderResourceView* srv = ds->GetD3DSRV();
    for (u32 i = 0; i < MAX_TEXTURE_SAMPLERS; i++)
    {
      if (m_current_textures[i] == srv)
      {
        m_current_textures[i] = nullptr;
        m_context->PSSetShaderResources(i, 1, &m_current_textures[i]);
      }
    }
  }

  m_current_framebuffer->CommitClear(m_context.Get());
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
  if (m_current_pipeline == pipeline)
    return;

  D3D11Pipeline* const PL = static_cast<D3D11Pipeline*>(pipeline);
  m_current_pipeline = PL;

  if (ID3D11InputLayout* il = PL->GetInputLayout(); m_current_input_layout != il)
  {
    m_current_input_layout = il;
    m_context->IASetInputLayout(il);
  }

  if (const u32 vertex_stride = PL->GetVertexStride(); m_current_vertex_stride != vertex_stride)
  {
    const UINT offset = 0;
    m_current_vertex_stride = PL->GetVertexStride();
    m_context->IASetVertexBuffers(0, 1, m_vertex_buffer.GetD3DBufferArray(), &m_current_vertex_stride, &offset);
  }

  if (D3D_PRIMITIVE_TOPOLOGY topology = PL->GetPrimitiveTopology(); m_current_primitive_topology != topology)
  {
    m_current_primitive_topology = topology;
    m_context->IASetPrimitiveTopology(topology);
  }

  if (ID3D11VertexShader* vs = PL->GetVertexShader(); m_current_vertex_shader != vs)
  {
    m_current_vertex_shader = vs;
    m_context->VSSetShader(vs, nullptr, 0);
  }

  if (ID3D11PixelShader* ps = PL->GetPixelShader(); m_current_pixel_shader != ps)
  {
    m_current_pixel_shader = ps;
    m_context->PSSetShader(ps, nullptr, 0);
  }

  if (ID3D11RasterizerState* rs = PL->GetRasterizerState(); m_current_rasterizer_state != rs)
  {
    m_current_rasterizer_state = rs;
    m_context->RSSetState(rs);
  }

  if (ID3D11DepthStencilState* ds = PL->GetDepthStencilState(); m_current_depth_state != ds)
  {
    m_current_depth_state = ds;
    m_context->OMSetDepthStencilState(ds, 0);
  }

  if (ID3D11BlendState* bs = PL->GetBlendState();
      m_current_blend_state != bs || m_current_blend_factor != PL->GetBlendFactor())
  {
    m_current_blend_state = bs;
    m_current_blend_factor = PL->GetBlendFactor();
    m_context->OMSetBlendState(bs, RGBA8ToFloat(m_current_blend_factor).data(), 0xFFFFFFFFu);
  }
}

void D3D11Device::UnbindPipeline(D3D11Pipeline* pl)
{
  if (m_current_pipeline != pl)
    return;

  // Let the runtime deal with the dead objects...
  m_current_pipeline = nullptr;
}

void D3D11Device::SetTextureSampler(u32 slot, GPUTexture* texture, GPUSampler* sampler)
{
  ID3D11ShaderResourceView* T = texture ? static_cast<D3D11Texture*>(texture)->GetD3DSRV() : nullptr;
  ID3D11SamplerState* S = sampler ? static_cast<D3D11Sampler*>(sampler)->GetSamplerState() : nullptr;

  // Runtime will null these if we don't...
  DebugAssert(!m_current_framebuffer || !texture ||
              (m_current_framebuffer->GetRT() != texture && m_current_framebuffer->GetDS() != texture));

  if (m_current_textures[slot] != T)
  {
    m_current_textures[slot] = T;
    m_context->PSSetShaderResources(slot, 1, &T);
  }
  if (m_current_samplers[slot] != S)
  {
    m_current_samplers[slot] = S;
    m_context->PSSetSamplers(slot, 1, &S);
  }
}

void D3D11Device::SetTextureBuffer(u32 slot, GPUTextureBuffer* buffer)
{
  ID3D11ShaderResourceView* B = buffer ? static_cast<D3D11TextureBuffer*>(buffer)->GetSRV() : nullptr;
  if (m_current_textures[slot] != B)
  {
    m_current_textures[slot] = B;
    m_context->PSSetShaderResources(slot, 1, &B);
  }
}

void D3D11Device::UnbindTexture(D3D11Texture* tex)
{
  if (const ID3D11ShaderResourceView* srv = tex->GetD3DSRV(); srv)
  {
    for (u32 i = 0; i < MAX_TEXTURE_SAMPLERS; i++)
    {
      if (m_current_textures[i] == srv)
      {
        m_current_textures[i] = nullptr;
        m_context->PSSetShaderResources(i, 1, &m_current_textures[i]);
      }
    }
  }

  if (m_current_framebuffer && m_current_framebuffer->GetRT() == tex)
    SetFramebuffer(nullptr);
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

void D3D11Device::Draw(u32 vertex_count, u32 base_vertex)
{
  m_context->Draw(vertex_count, base_vertex);
}

void D3D11Device::DrawIndexed(u32 index_count, u32 base_index, u32 base_vertex)
{
  m_context->DrawIndexed(index_count, base_index, base_vertex);
}
