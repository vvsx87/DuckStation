// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_hw_backend.h"
#include "cpu_core.h"
#include "cpu_pgxp.h"
#include "gpu_hw_shadergen.h"
#include "gpu_sw_backend.h"
#include "host.h"
#include "settings.h"
#include "system.h"

#include "util/imgui_manager.h"
#include "util/state_wrapper.h"

#include "common/align.h"
#include "common/assert.h"
#include "common/log.h"
#include "common/scoped_guard.h"
#include "common/string_util.h"

#include "IconsFontAwesome5.h"
#include "imgui.h"

#include <cmath>
#include <sstream>
#include <tuple>

Log_SetChannel(GPUHWBackend);

// TODO FIXME: CMR2.0 state 1 has semitransparent strip

// TODO: instead of full state restore, only restore what changed

static constexpr GPUTexture::Format VRAM_RT_FORMAT = GPUTexture::Format::RGBA8;
static constexpr GPUTexture::Format VRAM_DS_FORMAT = GPUTexture::Format::D16;

#ifdef _DEBUG
static u32 s_draw_number = 0;
#endif

template<typename T>
ALWAYS_INLINE static constexpr std::tuple<T, T> MinMax(T v1, T v2)
{
  if (v1 > v2)
    return std::tie(v2, v1);
  else
    return std::tie(v1, v2);
}

ALWAYS_INLINE static u32 GetMaxResolutionScale()
{
  return g_gpu_device->GetMaxTextureSize() / VRAM_WIDTH;
}

ALWAYS_INLINE_RELEASE static u32 GetBoxDownsampleScale(u32 resolution_scale)
{
  u32 scale = std::min<u32>(resolution_scale, g_gpu_settings.gpu_downsample_scale);
  while ((resolution_scale % scale) != 0)
    scale--;
  return scale;
}

ALWAYS_INLINE static bool ShouldClampUVs()
{
  // We only need UV limits if PGXP is enabled, or texture filtering is enabled.
  return g_gpu_settings.gpu_pgxp_enable || g_gpu_settings.gpu_texture_filter != GPUTextureFilter::Nearest;
}

ALWAYS_INLINE static bool ShouldDisableColorPerspective()
{
  return g_gpu_settings.gpu_pgxp_enable && g_gpu_settings.gpu_pgxp_texture_correction &&
         !g_gpu_settings.gpu_pgxp_color_correction;
}

/// Returns true if the specified texture filtering mode requires dual-source blending.
ALWAYS_INLINE static bool IsBlendedTextureFiltering(GPUTextureFilter filter)
{
  return (filter == GPUTextureFilter::Bilinear || filter == GPUTextureFilter::JINC2 || filter == GPUTextureFilter::xBR);
}

/// Computes the area affected by a VRAM transfer, including wrap-around of X.
static Common::Rectangle<u32> GetVRAMTransferBounds(u32 x, u32 y, u32 width, u32 height)
{
  Common::Rectangle<u32> out_rc = Common::Rectangle<u32>::FromExtents(x % VRAM_WIDTH, y % VRAM_HEIGHT, width, height);
  if (out_rc.right > VRAM_WIDTH)
  {
    out_rc.left = 0;
    out_rc.right = VRAM_WIDTH;
  }
  if (out_rc.bottom > VRAM_HEIGHT)
  {
    out_rc.top = 0;
    out_rc.bottom = VRAM_HEIGHT;
  }
  return out_rc;
}

namespace {
class ShaderCompileProgressTracker
{
public:
  ShaderCompileProgressTracker(std::string title, u32 total)
    : m_title(std::move(title)), m_min_time(Common::Timer::ConvertSecondsToValue(1.0)),
      m_update_interval(Common::Timer::ConvertSecondsToValue(0.1)), m_start_time(Common::Timer::GetCurrentValue()),
      m_last_update_time(0), m_progress(0), m_total(total)
  {
  }
  ~ShaderCompileProgressTracker() = default;

  void Increment(u32 progress = 1)
  {
    m_progress += progress;

    const u64 tv = Common::Timer::GetCurrentValue();
    if ((tv - m_start_time) >= m_min_time && (tv - m_last_update_time) >= m_update_interval)
    {
      Host::DisplayLoadingScreen(m_title.c_str(), 0, static_cast<int>(m_total), static_cast<int>(m_progress));
      m_last_update_time = tv;
    }
  }

private:
  std::string m_title;
  u64 m_min_time;
  u64 m_update_interval;
  u64 m_start_time;
  u64 m_last_update_time;
  u32 m_progress;
  u32 m_total;
};
} // namespace

GPUHWBackend::GPUHWBackend() : GPUBackend()
{
#ifdef _DEBUG
  s_draw_number = 0;
#endif
}

GPUHWBackend::~GPUHWBackend()
{
  // TODO FIXME
  // if (m_sw_renderer)
  //{
  // m_sw_renderer->Shutdown();
  // m_sw_renderer.reset();
  //}
}

ALWAYS_INLINE void GPUHWBackend::BatchVertex::Set(float x_, float y_, float z_, float w_, u32 color_, u32 texpage_,
                                                  u16 packed_texcoord, u32 uv_limits_)
{
  Set(x_, y_, z_, w_, color_, texpage_, packed_texcoord & 0xFF, (packed_texcoord >> 8), uv_limits_);
}

ALWAYS_INLINE void GPUHWBackend::BatchVertex::Set(float x_, float y_, float z_, float w_, u32 color_, u32 texpage_,
                                                  u16 u_, u16 v_, u32 uv_limits_)
{
  x = x_;
  y = y_;
  z = z_;
  w = w_;
  color = color_;
  texpage = texpage_;
  u = u_;
  v = v_;
  uv_limits = uv_limits_;
}

ALWAYS_INLINE u32 GPUHWBackend::BatchVertex::PackUVLimits(u32 min_u, u32 max_u, u32 min_v, u32 max_v)
{
  return min_u | (min_v << 8) | (max_u << 16) | (max_v << 24);
}

ALWAYS_INLINE void GPUHWBackend::BatchVertex::SetUVLimits(u32 min_u, u32 max_u, u32 min_v, u32 max_v)
{
  uv_limits = PackUVLimits(min_u, max_u, min_v, max_v);
}

ALWAYS_INLINE void GPUHWBackend::AddVertex(const BatchVertex& v)
{
  std::memcpy(m_batch_current_vertex_ptr, &v, sizeof(BatchVertex));
  m_batch_current_vertex_ptr++;
}

template<typename... Args>
ALWAYS_INLINE void GPUHWBackend::AddNewVertex(Args&&... args)
{
  m_batch_current_vertex_ptr->Set(std::forward<Args>(args)...);
  m_batch_current_vertex_ptr++;
}

bool GPUHWBackend::Initialize()
{
  if (!GPUBackend::Initialize())
    return false;

  const GPUDevice::Features features = g_gpu_device->GetFeatures();

  m_resolution_scale = CalculateResolutionScale();
  m_multisamples = std::min(g_gpu_settings.gpu_multisamples, g_gpu_device->GetMaxMultisamples());
  m_supports_dual_source_blend = features.dual_source_blend;
  m_supports_framebuffer_fetch = features.framebuffer_fetch;
  m_per_sample_shading = g_gpu_settings.gpu_per_sample_shading && features.per_sample_shading;
  m_true_color = g_gpu_settings.gpu_true_color;
  m_scaled_dithering = g_gpu_settings.gpu_scaled_dithering;
  m_texture_filtering = g_gpu_settings.gpu_texture_filter;
  m_clamp_uvs = ShouldClampUVs();
  m_compute_uv_range = m_clamp_uvs;
  m_chroma_smoothing = g_gpu_settings.gpu_24bit_chroma_smoothing;
  m_downsample_mode = GetDownsampleMode(m_resolution_scale);
  m_wireframe_mode = g_gpu_settings.gpu_wireframe_mode;
  m_disable_color_perspective = features.noperspective_interpolation && ShouldDisableColorPerspective();

  CheckSettings();

  UpdateSoftwareRenderer(false);

  PrintSettingsToLog();

  if (!CompilePipelines())
  {
    Log_ErrorPrint("Failed to compile pipelines");
    return false;
  }

  if (!CreateBuffers())
  {
    Log_ErrorPrint("Failed to create framebuffer");
    return false;
  }

  RestoreDeviceContext();
  return true;
}

void GPUHWBackend::Shutdown()
{
  DestroyBuffers();
  DestroyPipelines();
}

void GPUHWBackend::Reset(bool clear_vram)
{
  GPUBackend::Reset(clear_vram);

  m_batch_current_vertex_ptr = m_batch_start_vertex_ptr;
  m_last_texture_window_bits = 0;
  m_texpage_bits = INVALID_TEXPAGE_BITS;
  m_texpage_dirty = false;
  m_compute_uv_range = m_clamp_uvs;

  // if (m_sw_renderer)
  // m_sw_renderer->Reset(clear_vram);

  m_batch = {};
  m_batch_ubo_data = {};
  m_batch_ubo_dirty = true;
  m_current_depth = 1;

  if (clear_vram)
    ClearFramebuffer();
}

bool GPUHWBackend::DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display)
{
  if (!GPUBackend::DoState(sw, host_texture, update_display))
    return false;

  if (host_texture)
  {
    GPUTexture* tex = *host_texture;
    if (sw.IsReading())
    {
      if (tex->GetWidth() != m_vram_texture->GetWidth() || tex->GetHeight() != m_vram_texture->GetHeight() ||
          tex->GetSamples() != m_vram_texture->GetSamples())
      {
        return false;
      }

      g_gpu_device->CopyTextureRegion(m_vram_texture.get(), 0, 0, 0, 0, tex, 0, 0, 0, 0, tex->GetWidth(),
                                      tex->GetHeight());
    }
    else
    {
      if (!tex || tex->GetWidth() != m_vram_texture->GetWidth() || tex->GetHeight() != m_vram_texture->GetHeight() ||
          tex->GetSamples() != m_vram_texture->GetSamples())
      {
        delete tex;

        tex =
          g_gpu_device
            ->FetchTexture(m_vram_texture->GetWidth(), m_vram_texture->GetHeight(), 1, 1, m_vram_texture->GetSamples(),
                           GPUTexture::Type::RenderTarget, GPUTexture::Format::RGBA8, nullptr, 0)
            .release();
        *host_texture = tex;
        if (!tex)
          return false;
      }

      g_gpu_device->CopyTextureRegion(tex, 0, 0, 0, 0, m_vram_texture.get(), 0, 0, 0, 0, tex->GetWidth(),
                                      tex->GetHeight());
    }
  }

  // invalidate the whole VRAM read texture when loading state
  if (sw.IsReading())
  {
    m_batch_current_vertex_ptr = m_batch_start_vertex_ptr;
    SetFullVRAMDirtyRectangle();
    ResetBatchVertexDepth();
  }

  return true;
}

void GPUHWBackend::RestoreDeviceContext()
{
  g_gpu_device->SetTextureSampler(0, m_vram_read_texture.get(), g_gpu_device->GetNearestSampler());
  g_gpu_device->SetRenderTarget(m_vram_texture.get(), m_vram_depth_texture.get());
  g_gpu_device->SetViewport(0, 0, m_vram_texture->GetWidth(), m_vram_texture->GetHeight());
  SetScissor();
  m_batch_ubo_dirty = true;
}

void GPUHWBackend::UpdateSettings(const Settings& old_settings)
{
  GPUBackend::UpdateSettings(old_settings);

  const GPUDevice::Features features = g_gpu_device->GetFeatures();

  const u32 resolution_scale = CalculateResolutionScale();
  const u32 multisamples = std::min(g_gpu_settings.gpu_multisamples, g_gpu_device->GetMaxMultisamples());
  const bool per_sample_shading = g_gpu_settings.gpu_per_sample_shading && features.noperspective_interpolation;
  const GPUDownsampleMode downsample_mode = GetDownsampleMode(resolution_scale);
  const GPUWireframeMode wireframe_mode =
    features.geometry_shaders ? g_gpu_settings.gpu_wireframe_mode : GPUWireframeMode::Disabled;
  const bool clamp_uvs = ShouldClampUVs();
  const bool disable_color_perspective = features.noperspective_interpolation && ShouldDisableColorPerspective();

  // TODO: Use old_settings
  const bool framebuffer_changed =
    (m_resolution_scale != resolution_scale || m_multisamples != multisamples || m_downsample_mode != downsample_mode ||
     (m_downsample_mode == GPUDownsampleMode::Box &&
      g_gpu_settings.gpu_downsample_scale != old_settings.gpu_downsample_scale));
  const bool shaders_changed =
    (m_resolution_scale != resolution_scale || m_multisamples != multisamples ||
     m_true_color != g_gpu_settings.gpu_true_color || m_per_sample_shading != per_sample_shading ||
     m_scaled_dithering != g_gpu_settings.gpu_scaled_dithering ||
     m_texture_filtering != g_gpu_settings.gpu_texture_filter || m_clamp_uvs != clamp_uvs ||
     m_chroma_smoothing != g_gpu_settings.gpu_24bit_chroma_smoothing || m_downsample_mode != downsample_mode ||
     (m_downsample_mode == GPUDownsampleMode::Box &&
      g_gpu_settings.gpu_downsample_scale != old_settings.gpu_downsample_scale) ||
     m_wireframe_mode != wireframe_mode || m_pgxp_depth_buffer != g_gpu_settings.UsingPGXPDepthBuffer() ||
     m_disable_color_perspective != disable_color_perspective);

  if (m_resolution_scale != resolution_scale)
  {
    Host::AddIconOSDMessage(
      "ResolutionScaleChanged", ICON_FA_PAINT_BRUSH,
      fmt::format(TRANSLATE_FS("GPU_HW", "Resolution scale set to {0}x (display {1}x{2}, VRAM {3}x{4})"),
                  resolution_scale, m_display_width * resolution_scale, m_display_height * resolution_scale,
                  VRAM_WIDTH * resolution_scale, VRAM_HEIGHT * resolution_scale),
      Host::OSD_INFO_DURATION);
  }

  if (m_multisamples != multisamples || m_per_sample_shading != per_sample_shading)
  {
    if (per_sample_shading)
    {
      Host::AddIconOSDMessage(
        "MultisamplingChanged", ICON_FA_PAINT_BRUSH,
        fmt::format(TRANSLATE_FS("GPU_HW", "Multisample anti-aliasing set to {}x (SSAA)."), multisamples),
        Host::OSD_INFO_DURATION);
    }
    else
    {
      Host::AddIconOSDMessage(
        "MultisamplingChanged", ICON_FA_PAINT_BRUSH,
        fmt::format(TRANSLATE_FS("GPU_HW", "Multisample anti-aliasing set to {}x."), multisamples),
        Host::OSD_INFO_DURATION);
    }
  }

  // Back up VRAM if we're recreating the framebuffer.
  if (framebuffer_changed)
  {
    RestoreDeviceContext();
    ReadVRAM(0, 0, VRAM_WIDTH, VRAM_HEIGHT);
    DestroyBuffers();
  }

  m_resolution_scale = resolution_scale;
  m_multisamples = multisamples;
  m_per_sample_shading = per_sample_shading;
  m_true_color = g_gpu_settings.gpu_true_color;
  m_scaled_dithering = g_gpu_settings.gpu_scaled_dithering;
  m_texture_filtering = g_gpu_settings.gpu_texture_filter;
  m_clamp_uvs = clamp_uvs;
  m_compute_uv_range = m_clamp_uvs;
  m_chroma_smoothing = g_gpu_settings.gpu_24bit_chroma_smoothing;
  m_downsample_mode = downsample_mode;
  m_wireframe_mode = wireframe_mode;
  m_disable_color_perspective = disable_color_perspective;

  CheckSettings();

  if (m_pgxp_depth_buffer != g_gpu_settings.UsingPGXPDepthBuffer())
  {
    m_pgxp_depth_buffer = g_gpu_settings.UsingPGXPDepthBuffer();
    m_batch.use_depth_buffer = false;
    if (m_pgxp_depth_buffer)
      ClearDepthBuffer();
  }

  UpdateSoftwareRenderer(true);

  PrintSettingsToLog();

  if (shaders_changed)
  {
    DestroyPipelines();
    if (!CompilePipelines())
      Panic("Failed to recompile pipelnes.");
  }

  if (framebuffer_changed)
  {
    // TODO: weird vram loss when rapidly changing resolutions
    if (!CreateBuffers())
      Panic("Failed to recreate buffers.");

    RestoreDeviceContext();
    UpdateVRAM(0, 0, VRAM_WIDTH, VRAM_HEIGHT, GPU::m_vram_ptr, {});
    UpdateDepthBufferFromMaskBit();
  }
}

void GPUHWBackend::CheckSettings()
{
  const GPUDevice::Features features = g_gpu_device->GetFeatures();

  if (m_multisamples != g_gpu_settings.gpu_multisamples)
  {
    Host::AddIconOSDMessage("MSAAUnsupported", ICON_FA_EXCLAMATION_TRIANGLE,
                            fmt::format(TRANSLATE_FS("GPU_HW", "{}x MSAA is not supported, using {}x instead."),
                                        g_gpu_settings.gpu_multisamples, m_multisamples),
                            Host::OSD_CRITICAL_ERROR_DURATION);
  }
  else
  {
    Host::RemoveKeyedOSDMessage("MSAAUnsupported");
  }

  if (!m_per_sample_shading && g_gpu_settings.gpu_per_sample_shading)
  {
    Host::AddIconOSDMessage("SSAAUnsupported", ICON_FA_EXCLAMATION_TRIANGLE,
                            TRANSLATE_STR("GPU_HW", "SSAA is not supported, using MSAA instead."),
                            Host::OSD_ERROR_DURATION);
  }
  if (!features.dual_source_blend && !features.framebuffer_fetch && IsBlendedTextureFiltering(m_texture_filtering))
  {
    Host::AddIconOSDMessage(
      "TextureFilterUnsupported", ICON_FA_EXCLAMATION_TRIANGLE,
      fmt::format(TRANSLATE_FS("GPU_HW", "Texture filter '{}' is not supported with the current renderer."),
                  Settings::GetTextureFilterDisplayName(m_texture_filtering), Host::OSD_ERROR_DURATION));
    m_texture_filtering = GPUTextureFilter::Nearest;
  }

  if (!features.noperspective_interpolation && !ShouldDisableColorPerspective())
    Log_WarningPrint("Disable color perspective not supported, but should be used.");

  if (!features.geometry_shaders && m_wireframe_mode != GPUWireframeMode::Disabled)
  {
    Host::AddIconOSDMessage(
      "GeometryShadersUnsupported", ICON_FA_EXCLAMATION_TRIANGLE,
      TRANSLATE("GPU_HW", "Geometry shaders are not supported by your GPU, and are required for wireframe rendering."),
      Host::OSD_CRITICAL_ERROR_DURATION);
    m_wireframe_mode = GPUWireframeMode::Disabled;
  }

  if (m_downsample_mode == GPUDownsampleMode::Box)
  {
    const u32 resolution_scale = CalculateResolutionScale();
    const u32 box_downscale = GetBoxDownsampleScale(resolution_scale);
    if (box_downscale != g_gpu_settings.gpu_downsample_scale || box_downscale == resolution_scale)
    {
      Host::AddIconOSDMessage(
        "BoxDownsampleUnsupported", ICON_FA_PAINT_BRUSH,
        fmt::format(TRANSLATE_FS(
                      "GPU_HW", "Resolution scale {0}x is not divisible by downsample scale {1}x, using {2}x instead."),
                    resolution_scale, g_gpu_settings.gpu_downsample_scale, box_downscale),
        Host::OSD_WARNING_DURATION);
    }
    else
    {
      Host::RemoveKeyedOSDMessage("BoxDownsampleUnsupported");
    }

    if (box_downscale == g_gpu_settings.gpu_resolution_scale)
      m_downsample_mode = GPUDownsampleMode::Disabled;
  }

  m_pgxp_depth_buffer = g_gpu_settings.UsingPGXPDepthBuffer();
}

u32 GPUHWBackend::CalculateResolutionScale() const
{
  const u32 max_resolution_scale = GetMaxResolutionScale();

  u32 scale;
  if (g_gpu_settings.gpu_resolution_scale != 0)
  {
    scale = std::clamp<u32>(g_gpu_settings.gpu_resolution_scale, 1, max_resolution_scale);
  }
  else
  {
    // Auto scaling. When the system is starting and all borders crop is enabled, the registers are zero, and
    // display_height therefore is also zero. Use the default size from the region in this case.
    // TODO: Check these values, this was CRTC state
    const s32 height = (m_display_height != 0) ? static_cast<s32>(m_display_height) :
                                                 ((g_gpu_settings.region == ConsoleRegion::PAL) ?
                                                    (GPU::PAL_VERTICAL_ACTIVE_END - GPU::PAL_VERTICAL_ACTIVE_START) :
                                                    (GPU::NTSC_VERTICAL_ACTIVE_END - GPU::NTSC_VERTICAL_ACTIVE_START));
    const s32 preferred_scale =
      static_cast<s32>(std::ceil(static_cast<float>(g_gpu_device->GetWindowHeight()) / height));
    Log_VerboseFmt("Height = {}, preferred scale = {}", height, preferred_scale);

    scale = static_cast<u32>(std::clamp<s32>(preferred_scale, 1, max_resolution_scale));
  }

  if (g_gpu_settings.gpu_downsample_mode == GPUDownsampleMode::Adaptive && scale > 1 && !Common::IsPow2(scale))
  {
    const u32 new_scale = Common::PreviousPow2(scale);
    Log_WarningFmt("Resolution scale {}x not supported for adaptive downsampling, using {}x", scale, new_scale);

    if (g_gpu_settings.gpu_resolution_scale != 0)
    {
      Host::AddIconOSDMessage(
        "ResolutionNotPow2", ICON_FA_PAINT_BRUSH,
        fmt::format(
          TRANSLATE_FS("GPU_HW", "Resolution scale {0}x not supported for adaptive downsampling, using {1}x."), scale,
          new_scale),
        Host::OSD_WARNING_DURATION);
    }

    scale = new_scale;
  }

  return scale;
}

// void GPU_HW::UpdateResolutionScale()
// {
//   GPU::UpdateResolutionScale();
//
//   if (CalculateResolutionScale() != m_resolution_scale)
//     UpdateSettings(g_gpu_settings);
// }

GPUDownsampleMode GPUHWBackend::GetDownsampleMode(u32 resolution_scale) const
{
  return (resolution_scale == 1) ? GPUDownsampleMode::Disabled : g_gpu_settings.gpu_downsample_mode;
}

bool GPUHWBackend::IsUsingMultisampling() const
{
  return m_multisamples > 1;
}

bool GPUHWBackend::IsUsingDownsampling(const GPUBackendUpdateDisplayCommand* cmd) const
{
  return (m_downsample_mode != GPUDownsampleMode::Disabled && !cmd->display_24bit);
}

void GPUHWBackend::SetFullVRAMDirtyRectangle()
{
  m_vram_dirty_draw_rect.Set(0, 0, VRAM_WIDTH, VRAM_HEIGHT);
  m_texpage_bits = INVALID_TEXPAGE_BITS;
  m_texpage_dirty = false;
  m_compute_uv_range = m_clamp_uvs;
}

void GPUHWBackend::ClearVRAMDirtyRectangle()
{
  m_vram_dirty_draw_rect.SetInvalid();
  m_vram_dirty_write_rect.SetInvalid();
}

// std::tuple<u32, u32> GPU_HW::GetEffectiveDisplayResolution(bool scaled /* = true */)
// {
//   const u32 scale = scaled ? m_resolution_scale : 1u;
//   return std::make_tuple(m_crtc_state.display_vram_width * scale, m_crtc_state.display_vram_height * scale);
// }
//
// std::tuple<u32, u32> GPU_HW::GetFullDisplayResolution(bool scaled /* = true */)
// {
//   const u32 scale = scaled ? m_resolution_scale : 1u;
//   return std::make_tuple(m_crtc_state.display_width * scale, m_crtc_state.display_height * scale);
// }

void GPUHWBackend::PrintSettingsToLog()
{
  Log_InfoFmt("Resolution Scale: {} ({}x{}), maximum {}", m_resolution_scale, VRAM_WIDTH * m_resolution_scale,
              VRAM_HEIGHT * m_resolution_scale, GetMaxResolutionScale());
  Log_InfoFmt("Multisampling: {}x{}", m_multisamples, m_per_sample_shading ? " (per sample shading)" : "");
  Log_InfoFmt("Dithering: {}{}", m_true_color ? "Disabled" : "Enabled",
              (!m_true_color && m_scaled_dithering) ? " (Scaled)" : "");
  Log_InfoFmt("Texture Filtering: {}", Settings::GetTextureFilterDisplayName(m_texture_filtering));
  Log_InfoFmt("Dual-source blending: {}", m_supports_dual_source_blend ? "Supported" : "Not supported");
  Log_InfoFmt("Clamping UVs: {}", m_clamp_uvs ? "YES" : "NO");
  Log_InfoFmt("Depth buffer: {}", m_pgxp_depth_buffer ? "YES" : "NO");
  Log_InfoFmt("Downsampling: {}", Settings::GetDownsampleModeDisplayName(m_downsample_mode));
  Log_InfoFmt("Wireframe rendering: {}", Settings::GetGPUWireframeModeDisplayName(m_wireframe_mode));
  // Log_InfoFmt("Using software renderer for readbacks: {}", m_sw_renderer ? "YES" : "NO");
}

bool GPUHWBackend::CreateBuffers()
{
  DestroyBuffers();

  // scale vram size to internal resolution
  const u32 texture_width = VRAM_WIDTH * m_resolution_scale;
  const u32 texture_height = VRAM_HEIGHT * m_resolution_scale;
  const u8 samples = static_cast<u8>(m_multisamples);

  // Needed for Metal resolve.
  const GPUTexture::Type read_texture_type = (g_gpu_device->GetRenderAPI() == RenderAPI::Metal && m_multisamples > 1) ?
                                               GPUTexture::Type::RWTexture :
                                               GPUTexture::Type::Texture;

  if (!(m_vram_texture = g_gpu_device->FetchTexture(texture_width, texture_height, 1, 1, samples,
                                                    GPUTexture::Type::RenderTarget, VRAM_RT_FORMAT)) ||
      !(m_vram_depth_texture = g_gpu_device->FetchTexture(texture_width, texture_height, 1, 1, samples,
                                                          GPUTexture::Type::DepthStencil, VRAM_DS_FORMAT)) ||
      !(m_vram_read_texture =
          g_gpu_device->FetchTexture(texture_width, texture_height, 1, 1, 1, read_texture_type, VRAM_RT_FORMAT)) ||
      !(m_vram_readback_texture = g_gpu_device->FetchTexture(VRAM_WIDTH / 2, VRAM_HEIGHT, 1, 1, 1,
                                                             GPUTexture::Type::RenderTarget, VRAM_RT_FORMAT)))
  {
    return false;
  }

  GL_OBJECT_NAME(m_vram_texture, "VRAM Texture");
  GL_OBJECT_NAME(m_vram_depth_texture, "VRAM Depth Texture");
  GL_OBJECT_NAME(m_vram_read_texture, "VRAM Read Texture");
  GL_OBJECT_NAME(m_vram_readback_texture, "VRAM Readback Texture");

  if (g_gpu_device->GetFeatures().supports_texture_buffers)
  {
    if (!(m_vram_upload_buffer =
            g_gpu_device->CreateTextureBuffer(GPUTextureBuffer::Format::R16UI, GPUDevice::MIN_TEXEL_BUFFER_ELEMENTS)))
    {
      return false;
    }

    GL_OBJECT_NAME(m_vram_upload_buffer, "VRAM Upload Buffer");
  }

  Log_InfoFmt("Created HW framebuffer of {}x{}", texture_width, texture_height);

  if (m_downsample_mode == GPUDownsampleMode::Adaptive)
    m_downsample_scale_or_levels = GetAdaptiveDownsamplingMipLevels();
  else if (m_downsample_mode == GPUDownsampleMode::Box)
    m_downsample_scale_or_levels = m_resolution_scale / GetBoxDownsampleScale(m_resolution_scale);

  g_gpu_device->SetRenderTarget(m_vram_texture.get(), m_vram_depth_texture.get());
  SetFullVRAMDirtyRectangle();
  return true;
}

void GPUHWBackend::ClearFramebuffer()
{
  g_gpu_device->ClearRenderTarget(m_vram_texture.get(), 0);
  g_gpu_device->ClearDepth(m_vram_depth_texture.get(), m_pgxp_depth_buffer ? 1.0f : 0.0f);
  ClearVRAMDirtyRectangle();

  if (m_display_private_texture)
    g_gpu_device->ClearRenderTarget(m_display_private_texture.get(), 0);

  m_last_depth_z = 1.0f;
}

void GPUHWBackend::DestroyBuffers()
{
  ClearDisplayTexture();

  m_vram_upload_buffer.reset();
  g_gpu_device->RecycleTexture(std::move(m_downsample_texture));
  g_gpu_device->RecycleTexture(std::move(m_vram_read_texture));
  g_gpu_device->RecycleTexture(std::move(m_vram_depth_texture));
  g_gpu_device->RecycleTexture(std::move(m_vram_texture));
  g_gpu_device->RecycleTexture(std::move(m_vram_readback_texture));
  g_gpu_device->RecycleTexture(std::move(m_display_private_texture));
}

bool GPUHWBackend::CompilePipelines()
{
  const GPUDevice::Features features = g_gpu_device->GetFeatures();
  GPU_HW_ShaderGen shadergen(g_gpu_device->GetRenderAPI(), m_resolution_scale, m_multisamples, m_per_sample_shading,
                             m_true_color, m_scaled_dithering, m_texture_filtering, m_clamp_uvs, m_pgxp_depth_buffer,
                             m_disable_color_perspective, m_supports_dual_source_blend, m_supports_framebuffer_fetch);

  ShaderCompileProgressTracker progress("Compiling Pipelines", 2 + (4 * 5 * 9 * 2 * 2) + (3 * 4 * 5 * 9 * 2 * 2) + 1 +
                                                                 2 + (2 * 2) + 2 + 1 + 1 + (2 * 3) + 1);

  // vertex shaders - [textured]
  // fragment shaders - [render_mode][texture_mode][dithering][interlacing]
  static constexpr auto destroy_shader = [](std::unique_ptr<GPUShader>& s) { s.reset(); };
  DimensionalArray<std::unique_ptr<GPUShader>, 2> batch_vertex_shaders{};
  DimensionalArray<std::unique_ptr<GPUShader>, 2, 2, 9, 5, 4> batch_fragment_shaders{};
  ScopedGuard batch_shader_guard([&batch_vertex_shaders, &batch_fragment_shaders]() {
    batch_vertex_shaders.enumerate(destroy_shader);
    batch_fragment_shaders.enumerate(destroy_shader);
  });

  for (u8 textured = 0; textured < 2; textured++)
  {
    const std::string vs = shadergen.GenerateBatchVertexShader(ConvertToBoolUnchecked(textured));
    if (!(batch_vertex_shaders[textured] = g_gpu_device->CreateShader(GPUShaderStage::Vertex, vs)))
      return false;

    progress.Increment();
  }

  for (u8 render_mode = 0; render_mode < 4; render_mode++)
  {
    for (u8 transparency_mode = 0; transparency_mode < 5; transparency_mode++)
    {
      if (m_supports_framebuffer_fetch)
      {
        // Don't need multipass shaders.
        if (render_mode != static_cast<u8>(BatchRenderMode::TransparencyDisabled) &&
            render_mode != static_cast<u8>(BatchRenderMode::TransparentAndOpaque))
        {
          progress.Increment(2 * 2 * 9);
          continue;
        }
      }
      else
      {
        // Can't generate shader blending.
        if (transparency_mode != static_cast<u8>(GPUTransparencyMode::Disabled))
        {
          progress.Increment(2 * 2 * 9);
          continue;
        }
      }

      for (u8 texture_mode = 0; texture_mode < 9; texture_mode++)
      {
        for (u8 dithering = 0; dithering < 2; dithering++)
        {
          for (u8 interlacing = 0; interlacing < 2; interlacing++)
          {
            const std::string fs = shadergen.GenerateBatchFragmentShader(
              static_cast<BatchRenderMode>(render_mode), static_cast<GPUTransparencyMode>(transparency_mode),
              static_cast<GPUTextureMode>(texture_mode), ConvertToBoolUnchecked(dithering),
              ConvertToBoolUnchecked(interlacing));

            if (!(batch_fragment_shaders[render_mode][transparency_mode][texture_mode][dithering][interlacing] =
                    g_gpu_device->CreateShader(GPUShaderStage::Fragment, fs)))
            {
              return false;
            }

            progress.Increment();
          }
        }
      }
    }
  }

  static constexpr GPUPipeline::VertexAttribute vertex_attributes[] = {
    GPUPipeline::VertexAttribute::Make(0, GPUPipeline::VertexAttribute::Semantic::Position, 0,
                                       GPUPipeline::VertexAttribute::Type::Float, 4, offsetof(BatchVertex, x)),
    GPUPipeline::VertexAttribute::Make(1, GPUPipeline::VertexAttribute::Semantic::Color, 0,
                                       GPUPipeline::VertexAttribute::Type::UNorm8, 4, offsetof(BatchVertex, color)),
    GPUPipeline::VertexAttribute::Make(2, GPUPipeline::VertexAttribute::Semantic::TexCoord, 0,
                                       GPUPipeline::VertexAttribute::Type::UInt32, 1, offsetof(BatchVertex, u)),
    GPUPipeline::VertexAttribute::Make(3, GPUPipeline::VertexAttribute::Semantic::TexCoord, 1,
                                       GPUPipeline::VertexAttribute::Type::UInt32, 1, offsetof(BatchVertex, texpage)),
    GPUPipeline::VertexAttribute::Make(4, GPUPipeline::VertexAttribute::Semantic::TexCoord, 2,
                                       GPUPipeline::VertexAttribute::Type::UNorm8, 4, offsetof(BatchVertex, uv_limits)),
  };
  static constexpr u32 NUM_BATCH_VERTEX_ATTRIBUTES = 2;
  static constexpr u32 NUM_BATCH_TEXTURED_VERTEX_ATTRIBUTES = 4;
  static constexpr u32 NUM_BATCH_TEXTURED_LIMITS_VERTEX_ATTRIBUTES = 5;

  GPUPipeline::GraphicsConfig plconfig = {};
  plconfig.layout = GPUPipeline::Layout::SingleTextureAndUBO;
  plconfig.input_layout.vertex_stride = sizeof(BatchVertex);
  plconfig.rasterization = GPUPipeline::RasterizationState::GetNoCullState();
  plconfig.primitive = GPUPipeline::Primitive::Triangles;
  plconfig.SetTargetFormats(VRAM_RT_FORMAT, VRAM_DS_FORMAT);
  plconfig.samples = m_multisamples;
  plconfig.per_sample_shading = m_per_sample_shading;
  plconfig.geometry_shader = nullptr;

  // [depth_test][render_mode][texture_mode][transparency_mode][dithering][interlacing]
  for (u8 depth_test = 0; depth_test < 3; depth_test++)
  {
    for (u8 render_mode = 0; render_mode < 4; render_mode++)
    {
      if (m_supports_framebuffer_fetch)
      {
        // Don't need multipass shaders.
        if (render_mode != static_cast<u8>(BatchRenderMode::TransparencyDisabled) &&
            render_mode != static_cast<u8>(BatchRenderMode::TransparentAndOpaque))
        {
          progress.Increment(2 * 2 * 9 * 5);
          continue;
        }
      }

      for (u8 transparency_mode = 0; transparency_mode < 5; transparency_mode++)
      {
        for (u8 texture_mode = 0; texture_mode < 9; texture_mode++)
        {
          for (u8 dithering = 0; dithering < 2; dithering++)
          {
            for (u8 interlacing = 0; interlacing < 2; interlacing++)
            {
              static constexpr std::array<GPUPipeline::DepthFunc, 3> depth_test_values = {
                GPUPipeline::DepthFunc::Always, GPUPipeline::DepthFunc::GreaterEqual,
                GPUPipeline::DepthFunc::LessEqual};
              const bool textured = (static_cast<GPUTextureMode>(texture_mode) != GPUTextureMode::Disabled);
              const bool use_shader_blending =
                (textured && NeedsShaderBlending(static_cast<GPUTransparencyMode>(transparency_mode)));

              plconfig.input_layout.vertex_attributes =
                textured ?
                  (m_clamp_uvs ? std::span<const GPUPipeline::VertexAttribute>(
                                   vertex_attributes, NUM_BATCH_TEXTURED_LIMITS_VERTEX_ATTRIBUTES) :
                                 std::span<const GPUPipeline::VertexAttribute>(vertex_attributes,
                                                                               NUM_BATCH_TEXTURED_VERTEX_ATTRIBUTES)) :
                  std::span<const GPUPipeline::VertexAttribute>(vertex_attributes, NUM_BATCH_VERTEX_ATTRIBUTES);

              plconfig.vertex_shader = batch_vertex_shaders[BoolToUInt8(textured)].get();
              plconfig.fragment_shader =
                batch_fragment_shaders[render_mode]
                                      [use_shader_blending ? transparency_mode :
                                                             static_cast<u8>(GPUTransparencyMode::Disabled)]
                                      [texture_mode][dithering][interlacing]
                                        .get();

              plconfig.depth.depth_test = depth_test_values[depth_test];
              plconfig.depth.depth_write = !m_pgxp_depth_buffer || depth_test != 0;
              plconfig.blend = GPUPipeline::BlendState::GetNoBlendingState();

              if (!use_shader_blending &&
                  ((static_cast<GPUTransparencyMode>(transparency_mode) != GPUTransparencyMode::Disabled &&
                    (static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::TransparencyDisabled &&
                     static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::OnlyOpaque)) ||
                   (textured && IsBlendedTextureFiltering(m_texture_filtering))))
              {
                plconfig.blend.enable = true;
                plconfig.blend.src_alpha_blend = GPUPipeline::BlendFunc::One;
                plconfig.blend.dst_alpha_blend = GPUPipeline::BlendFunc::Zero;
                plconfig.blend.alpha_blend_op = GPUPipeline::BlendOp::Add;

                if (m_supports_dual_source_blend)
                {
                  plconfig.blend.src_blend = GPUPipeline::BlendFunc::One;
                  plconfig.blend.dst_blend = GPUPipeline::BlendFunc::SrcAlpha1;
                  plconfig.blend.blend_op =
                    (static_cast<GPUTransparencyMode>(transparency_mode) ==
                       GPUTransparencyMode::BackgroundMinusForeground &&
                     static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::TransparencyDisabled &&
                     static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::OnlyOpaque) ?
                      GPUPipeline::BlendOp::ReverseSubtract :
                      GPUPipeline::BlendOp::Add;
                }
                else
                {
                  // TODO: This isn't entirely accurate, 127.5 versus 128.
                  // But if we use fbfetch on Mali, it doesn't matter.
                  plconfig.blend.src_blend = GPUPipeline::BlendFunc::One;
                  plconfig.blend.dst_blend = GPUPipeline::BlendFunc::One;
                  if (static_cast<GPUTransparencyMode>(transparency_mode) ==
                      GPUTransparencyMode::HalfBackgroundPlusHalfForeground)
                  {
                    plconfig.blend.dst_blend = GPUPipeline::BlendFunc::ConstantColor;
                    plconfig.blend.dst_alpha_blend = GPUPipeline::BlendFunc::ConstantColor;
                    plconfig.blend.constant = 0x00808080u;
                  }

                  plconfig.blend.blend_op =
                    (static_cast<GPUTransparencyMode>(transparency_mode) ==
                       GPUTransparencyMode::BackgroundMinusForeground &&
                     static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::TransparencyDisabled &&
                     static_cast<BatchRenderMode>(render_mode) != BatchRenderMode::OnlyOpaque) ?
                      GPUPipeline::BlendOp::ReverseSubtract :
                      GPUPipeline::BlendOp::Add;
                }
              }

              if (!(m_batch_pipelines[depth_test][render_mode][texture_mode][transparency_mode][dithering]
                                     [interlacing] = g_gpu_device->CreatePipeline(plconfig)))
              {
                return false;
              }

              progress.Increment();
            }
          }
        }
      }
    }
  }

  if (m_wireframe_mode != GPUWireframeMode::Disabled)
  {
    std::unique_ptr<GPUShader> gs =
      g_gpu_device->CreateShader(GPUShaderStage::Geometry, shadergen.GenerateWireframeGeometryShader());
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateWireframeFragmentShader());
    if (!gs || !fs)
      return false;

    GL_OBJECT_NAME(gs, "Batch Wireframe Geometry Shader");
    GL_OBJECT_NAME(fs, "Batch Wireframe Fragment Shader");

    plconfig.input_layout.vertex_attributes =
      std::span<const GPUPipeline::VertexAttribute>(vertex_attributes, NUM_BATCH_VERTEX_ATTRIBUTES);
    plconfig.blend = (m_wireframe_mode == GPUWireframeMode::OverlayWireframe) ?
                       GPUPipeline::BlendState::GetAlphaBlendingState() :
                       GPUPipeline::BlendState::GetNoBlendingState();
    plconfig.blend.write_mask = 0x7;
    plconfig.depth = GPUPipeline::DepthState::GetNoTestsState();
    plconfig.vertex_shader = batch_vertex_shaders[0].get();
    plconfig.geometry_shader = gs.get();
    plconfig.fragment_shader = fs.get();

    if (!(m_wireframe_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;

    GL_OBJECT_NAME(m_wireframe_pipeline, "Batch Wireframe Pipeline");

    plconfig.vertex_shader = nullptr;
    plconfig.geometry_shader = nullptr;
    plconfig.fragment_shader = nullptr;
  }

  batch_shader_guard.Run();

  std::unique_ptr<GPUShader> fullscreen_quad_vertex_shader =
    g_gpu_device->CreateShader(GPUShaderStage::Vertex, shadergen.GenerateScreenQuadVertexShader());
  if (!fullscreen_quad_vertex_shader)
    return false;

  progress.Increment();

  // common state
  plconfig.input_layout.vertex_attributes = {};
  plconfig.input_layout.vertex_stride = 0;
  plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
  plconfig.per_sample_shading = false;
  plconfig.blend = GPUPipeline::BlendState::GetNoBlendingState();
  plconfig.vertex_shader = fullscreen_quad_vertex_shader.get();

  // VRAM fill
  for (u8 wrapped = 0; wrapped < 2; wrapped++)
  {
    for (u8 interlaced = 0; interlaced < 2; interlaced++)
    {
      std::unique_ptr<GPUShader> fs = g_gpu_device->CreateShader(
        GPUShaderStage::Fragment,
        shadergen.GenerateVRAMFillFragmentShader(ConvertToBoolUnchecked(wrapped), ConvertToBoolUnchecked(interlaced)));
      if (!fs)
        return false;

      plconfig.fragment_shader = fs.get();
      plconfig.depth = GPUPipeline::DepthState::GetAlwaysWriteState();

      if (!(m_vram_fill_pipelines[wrapped][interlaced] = g_gpu_device->CreatePipeline(plconfig)))
        return false;

      progress.Increment();
    }
  }

  // VRAM copy
  {
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateVRAMCopyFragmentShader());
    if (!fs)
      return false;

    plconfig.fragment_shader = fs.get();
    for (u8 depth_test = 0; depth_test < 2; depth_test++)
    {
      plconfig.depth.depth_write = true;
      plconfig.depth.depth_test =
        (depth_test != 0) ? GPUPipeline::DepthFunc::GreaterEqual : GPUPipeline::DepthFunc::Always;

      if (!(m_vram_copy_pipelines[depth_test] = g_gpu_device->CreatePipeline(plconfig)))
        return false;

      GL_OBJECT_NAME_FMT(m_vram_copy_pipelines[depth_test], "VRAM Write Pipeline, depth={}", depth_test);

      progress.Increment();
    }
  }

  // VRAM write
  {
    const bool use_buffer = features.supports_texture_buffers;
    const bool use_ssbo = features.texture_buffers_emulated_with_ssbo;
    std::unique_ptr<GPUShader> fs = g_gpu_device->CreateShader(
      GPUShaderStage::Fragment, shadergen.GenerateVRAMWriteFragmentShader(use_buffer, use_ssbo));
    if (!fs)
      return false;

    plconfig.layout =
      use_buffer ? GPUPipeline::Layout::SingleTextureBufferAndPushConstants : GPUPipeline::Layout::SingleTextureAndUBO;
    plconfig.fragment_shader = fs.get();
    for (u8 depth_test = 0; depth_test < 2; depth_test++)
    {
      plconfig.depth.depth_write = true;
      plconfig.depth.depth_test =
        (depth_test != 0) ? GPUPipeline::DepthFunc::GreaterEqual : GPUPipeline::DepthFunc::Always;

      if (!(m_vram_write_pipelines[depth_test] = g_gpu_device->CreatePipeline(plconfig)))
        return false;

      GL_OBJECT_NAME_FMT(m_vram_write_pipelines[depth_test], "VRAM Write Pipeline, depth={}", depth_test);

      progress.Increment();
    }
  }

  plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;

  // VRAM update depth
  {
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateVRAMUpdateDepthFragmentShader());
    if (!fs)
      return false;

    plconfig.fragment_shader = fs.get();
    plconfig.SetTargetFormats(GPUTexture::Format::Unknown, VRAM_DS_FORMAT);
    plconfig.depth = GPUPipeline::DepthState::GetAlwaysWriteState();
    plconfig.blend.write_mask = 0;

    if (!(m_vram_update_depth_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;

    GL_OBJECT_NAME(m_vram_update_depth_pipeline, "VRAM Update Depth Pipeline");

    progress.Increment();
  }

  plconfig.SetTargetFormats(VRAM_RT_FORMAT);
  plconfig.depth = GPUPipeline::DepthState::GetNoTestsState();
  plconfig.blend = GPUPipeline::BlendState::GetNoBlendingState();
  plconfig.samples = 1;
  plconfig.per_sample_shading = false;

  // VRAM read
  {
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateVRAMReadFragmentShader());
    if (!fs)
      return false;

    plconfig.fragment_shader = fs.get();

    if (!(m_vram_readback_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;

    GL_OBJECT_NAME(m_vram_readback_pipeline, "VRAM Read Pipeline");
    progress.Increment();
  }

  // Display
  {
    for (u8 depth_24 = 0; depth_24 < 2; depth_24++)
    {
      for (u8 interlace_mode = 0; interlace_mode < 3; interlace_mode++)
      {
        std::unique_ptr<GPUShader> fs = g_gpu_device->CreateShader(
          GPUShaderStage::Fragment,
          shadergen.GenerateDisplayFragmentShader(
            ConvertToBoolUnchecked(depth_24), static_cast<InterlacedRenderMode>(interlace_mode), m_chroma_smoothing));
        if (!fs)
          return false;

        plconfig.fragment_shader = fs.get();

        if (!(m_display_pipelines[depth_24][interlace_mode] = g_gpu_device->CreatePipeline(plconfig)))
          return false;

        progress.Increment();
      }
    }
  }

  {
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateCopyFragmentShader());
    if (!fs)
      return false;

    plconfig.fragment_shader = fs.get();
    if (!(m_copy_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
  }

  if (m_downsample_mode == GPUDownsampleMode::Adaptive)
  {
    std::unique_ptr<GPUShader> vs =
      g_gpu_device->CreateShader(GPUShaderStage::Vertex, shadergen.GenerateAdaptiveDownsampleVertexShader());
    std::unique_ptr<GPUShader> fs =
      g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateAdaptiveDownsampleMipFragmentShader(true));
    if (!vs || !fs)
      return false;
    GL_OBJECT_NAME(fs, "Downsample Vertex Shader");
    GL_OBJECT_NAME(fs, "Downsample First Pass Fragment Shader");
    plconfig.vertex_shader = vs.get();
    plconfig.fragment_shader = fs.get();
    if (!(m_downsample_first_pass_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
    GL_OBJECT_NAME(m_downsample_first_pass_pipeline, "Downsample First Pass Pipeline");

    fs = g_gpu_device->CreateShader(GPUShaderStage::Fragment,
                                    shadergen.GenerateAdaptiveDownsampleMipFragmentShader(false));
    if (!fs)
      return false;
    GL_OBJECT_NAME(fs, "Downsample Mid Pass Fragment Shader");
    plconfig.fragment_shader = fs.get();
    if (!(m_downsample_mid_pass_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
    GL_OBJECT_NAME(m_downsample_mid_pass_pipeline, "Downsample Mid Pass Pipeline");

    fs = g_gpu_device->CreateShader(GPUShaderStage::Fragment, shadergen.GenerateAdaptiveDownsampleBlurFragmentShader());
    if (!fs)
      return false;
    GL_OBJECT_NAME(fs, "Downsample Blur Pass Fragment Shader");
    plconfig.fragment_shader = fs.get();
    plconfig.SetTargetFormats(GPUTexture::Format::R8);
    if (!(m_downsample_blur_pass_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
    GL_OBJECT_NAME(m_downsample_blur_pass_pipeline, "Downsample Blur Pass Pipeline");

    fs = g_gpu_device->CreateShader(GPUShaderStage::Fragment,
                                    shadergen.GenerateAdaptiveDownsampleCompositeFragmentShader());
    if (!fs)
      return false;
    GL_OBJECT_NAME(fs, "Downsample Composite Pass Fragment Shader");
    plconfig.layout = GPUPipeline::Layout::MultiTextureAndPushConstants;
    plconfig.fragment_shader = fs.get();
    plconfig.SetTargetFormats(VRAM_RT_FORMAT);
    if (!(m_downsample_composite_pass_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;
    GL_OBJECT_NAME(m_downsample_composite_pass_pipeline, "Downsample Blur Pass Pipeline");

    GPUSampler::Config config = GPUSampler::GetLinearConfig();
    config.min_lod = 0;
    config.max_lod = GPUSampler::Config::LOD_MAX;
    if (!(m_downsample_lod_sampler = g_gpu_device->CreateSampler(config)))
      return false;
    GL_OBJECT_NAME(m_downsample_lod_sampler, "Downsample LOD Sampler");
    config.mip_filter = GPUSampler::Filter::Linear;
    if (!(m_downsample_composite_sampler = g_gpu_device->CreateSampler(config)))
      return false;
    GL_OBJECT_NAME(m_downsample_composite_sampler, "Downsample Trilinear Sampler");
  }
  else if (m_downsample_mode == GPUDownsampleMode::Box)
  {
    std::unique_ptr<GPUShader> fs = g_gpu_device->CreateShader(
      GPUShaderStage::Fragment, shadergen.GenerateBoxSampleDownsampleFragmentShader(
                                  m_resolution_scale / GetBoxDownsampleScale(m_resolution_scale)));
    if (!fs)
      return false;

    GL_OBJECT_NAME(fs, "Downsample First Pass Fragment Shader");
    plconfig.fragment_shader = fs.get();

    if (!(m_downsample_first_pass_pipeline = g_gpu_device->CreatePipeline(plconfig)))
      return false;

    GL_OBJECT_NAME(m_downsample_first_pass_pipeline, "Downsample First Pass Pipeline");
  }

  progress.Increment();

#undef UPDATE_PROGRESS

  return true;
}

void GPUHWBackend::DestroyPipelines()
{
  static constexpr auto destroy = [](std::unique_ptr<GPUPipeline>& p) { p.reset(); };

  m_wireframe_pipeline.reset();

  m_batch_pipelines.enumerate(destroy);

  m_vram_fill_pipelines.enumerate(destroy);

  for (std::unique_ptr<GPUPipeline>& p : m_vram_write_pipelines)
    destroy(p);

  for (std::unique_ptr<GPUPipeline>& p : m_vram_copy_pipelines)
    destroy(p);

  destroy(m_vram_readback_pipeline);
  destroy(m_vram_update_depth_pipeline);

  destroy(m_downsample_first_pass_pipeline);
  destroy(m_downsample_mid_pass_pipeline);
  destroy(m_downsample_blur_pass_pipeline);
  destroy(m_downsample_composite_pass_pipeline);
  m_downsample_composite_sampler.reset();

  m_copy_pipeline.reset();

  m_display_pipelines.enumerate(destroy);
}

GPUHWBackend::BatchRenderMode GPUHWBackend::BatchConfig::GetRenderMode() const
{
  return transparency_mode == GPUTransparencyMode::Disabled ? BatchRenderMode::TransparencyDisabled :
                                                              BatchRenderMode::TransparentAndOpaque;
}

void GPUHWBackend::UpdateVRAMReadTexture(bool drawn, bool written)
{
  GL_SCOPE("UpdateVRAMReadTexture()");

  const auto update = [this](Common::Rectangle<u32>& rect, u8 dbit) {
    if (m_texpage_dirty & dbit)
    {
      m_texpage_dirty &= ~dbit;
      if (!m_texpage_dirty)
        GL_INS_FMT("{} texpage is no longer dirty", (dbit & TEXPAGE_DIRTY_DRAWN_RECT) ? "DRAW" : "WRITE");
    }

    const auto scaled_rect = rect * m_resolution_scale;
    if (m_vram_texture->IsMultisampled())
    {
      if (g_gpu_device->GetFeatures().partial_msaa_resolve)
      {
        g_gpu_device->ResolveTextureRegion(m_vram_read_texture.get(), scaled_rect.left, scaled_rect.top, 0, 0,
                                           m_vram_texture.get(), scaled_rect.left, scaled_rect.top,
                                           scaled_rect.GetWidth(), scaled_rect.GetHeight());
      }
      else
      {
        g_gpu_device->ResolveTextureRegion(m_vram_read_texture.get(), 0, 0, 0, 0, m_vram_texture.get(), 0, 0,
                                           m_vram_texture->GetWidth(), m_vram_texture->GetHeight());
      }
    }
    else
    {
      g_gpu_device->CopyTextureRegion(m_vram_read_texture.get(), scaled_rect.left, scaled_rect.top, 0, 0,
                                      m_vram_texture.get(), scaled_rect.left, scaled_rect.top, 0, 0,
                                      scaled_rect.GetWidth(), scaled_rect.GetHeight());
    }

    m_renderer_stats.num_vram_read_texture_updates++;
    rect.SetInvalid();
  };

  if (drawn)
  {
    DebugAssert(m_vram_dirty_draw_rect.Valid());
    GL_INS_FMT("Updating draw rect {},{} => {},{} ({}x{})", m_vram_dirty_draw_rect.left, m_vram_dirty_draw_rect.right,
               m_vram_dirty_draw_rect.top, m_vram_dirty_draw_rect.bottom, m_vram_dirty_draw_rect.GetWidth(),
               m_vram_dirty_draw_rect.GetHeight());

    u8 dbits = TEXPAGE_DIRTY_DRAWN_RECT;
    if (written && m_vram_dirty_draw_rect.Intersects(m_vram_dirty_write_rect))
    {
      DebugAssert(m_vram_dirty_write_rect.Valid());
      GL_INS_FMT("Including write rect {},{} => {},{} ({}x{})", m_vram_dirty_write_rect.left,
                 m_vram_dirty_write_rect.right, m_vram_dirty_write_rect.top, m_vram_dirty_write_rect.bottom,
                 m_vram_dirty_write_rect.GetWidth(), m_vram_dirty_write_rect.GetHeight());
      m_vram_dirty_draw_rect.Include(m_vram_dirty_write_rect);
      m_vram_dirty_write_rect.SetInvalid();
      dbits = TEXPAGE_DIRTY_DRAWN_RECT | TEXPAGE_DIRTY_WRITTEN_RECT;
      written = false;
    }

    update(m_vram_dirty_draw_rect, dbits);
  }
  if (written)
  {
    GL_INS_FMT("Updating write rect {},{} => {},{} ({}x{})", m_vram_dirty_write_rect.left,
               m_vram_dirty_write_rect.right, m_vram_dirty_write_rect.top, m_vram_dirty_write_rect.bottom,
               m_vram_dirty_write_rect.GetWidth(), m_vram_dirty_write_rect.GetHeight());
    update(m_vram_dirty_write_rect, TEXPAGE_DIRTY_WRITTEN_RECT);
  }
}

void GPUHWBackend::UpdateDepthBufferFromMaskBit()
{
  if (m_pgxp_depth_buffer)
    return;

  // Viewport should already be set full, only need to fudge the scissor.
  g_gpu_device->SetScissor(0, 0, m_vram_texture->GetWidth(), m_vram_texture->GetHeight());
  g_gpu_device->InvalidateRenderTarget(m_vram_depth_texture.get());
  g_gpu_device->SetRenderTargets(nullptr, 0, m_vram_depth_texture.get());
  g_gpu_device->SetPipeline(m_vram_update_depth_pipeline.get());
  g_gpu_device->SetTextureSampler(0, m_vram_texture.get(), g_gpu_device->GetNearestSampler());
  g_gpu_device->Draw(3, 0);

  // Restore.
  g_gpu_device->SetTextureSampler(0, m_vram_read_texture.get(), g_gpu_device->GetNearestSampler());
  g_gpu_device->SetRenderTarget(m_vram_texture.get(), m_vram_depth_texture.get());
  SetScissor();
}

void GPUHWBackend::ClearDepthBuffer()
{
  DebugAssert(m_pgxp_depth_buffer);

  g_gpu_device->ClearDepth(m_vram_depth_texture.get(), 1.0f);
  m_last_depth_z = 1.0f;
}

void GPUHWBackend::SetScissor()
{
  const s32 left = m_drawing_area.left * m_resolution_scale;
  const s32 right = std::max<u32>((m_drawing_area.right + 1) * m_resolution_scale, left + 1);
  const s32 top = m_drawing_area.top * m_resolution_scale;
  const s32 bottom = std::max<u32>((m_drawing_area.bottom + 1) * m_resolution_scale, top + 1);

  g_gpu_device->SetScissor(left, top, right - left, bottom - top);
}

void GPUHWBackend::MapBatchVertexPointer(u32 required_vertices)
{
  DebugAssert(!m_batch_start_vertex_ptr);

  void* map;
  u32 space;
  g_gpu_device->MapVertexBuffer(sizeof(BatchVertex), required_vertices, &map, &space, &m_batch_base_vertex);

  m_batch_start_vertex_ptr = static_cast<BatchVertex*>(map);
  m_batch_current_vertex_ptr = m_batch_start_vertex_ptr;
  m_batch_end_vertex_ptr = m_batch_start_vertex_ptr + space;
}

void GPUHWBackend::UnmapBatchVertexPointer(u32 used_vertices)
{
  DebugAssert(m_batch_start_vertex_ptr);
  g_gpu_device->UnmapVertexBuffer(sizeof(BatchVertex), used_vertices);
  m_batch_start_vertex_ptr = nullptr;
  m_batch_end_vertex_ptr = nullptr;
  m_batch_current_vertex_ptr = nullptr;
}

void GPUHWBackend::DrawBatchVertices(BatchRenderMode render_mode, u32 num_vertices, u32 base_vertex)
{
  // [depth_test][render_mode][texture_mode][transparency_mode][dithering][interlacing]
  const u8 depth_test = m_batch.use_depth_buffer ? static_cast<u8>(2) : BoolToUInt8(m_batch.check_mask_before_draw);
  g_gpu_device->SetPipeline(
    m_batch_pipelines[depth_test][static_cast<u8>(render_mode)][static_cast<u8>(m_batch.texture_mode)][static_cast<u8>(
      m_batch.transparency_mode)][BoolToUInt8(m_batch.dithering)][BoolToUInt8(m_batch.interlacing)]
      .get());
  g_gpu_device->Draw(num_vertices, base_vertex);
}

void GPUHWBackend::ClearDisplay()
{
  ClearDisplayTexture();

  if (m_display_private_texture)
    g_gpu_device->ClearRenderTarget(m_display_private_texture.get(), 0xFF000000u);
}

void GPUHWBackend::HandleFlippedQuadTextureCoordinates(BatchVertex* vertices)
{
  // Taken from beetle-psx gpu_polygon.cpp
  // For X/Y flipped 2D sprites, PSX games rely on a very specific rasterization behavior. If U or V is decreasing in X
  // or Y, and we use the provided U/V as is, we will sample the wrong texel as interpolation covers an entire pixel,
  // while PSX samples its interpolation essentially in the top-left corner and splats that interpolant across the
  // entire pixel. While we could emulate this reasonably well in native resolution by shifting our vertex coords by
  // 0.5, this breaks in upscaling scenarios, because we have several samples per native sample and we need NN rules to
  // hit the same UV every time. One approach here is to use interpolate at offset or similar tricks to generalize the
  // PSX interpolation patterns, but the problem is that vertices sharing an edge will no longer see the same UV (due to
  // different plane derivatives), we end up sampling outside the intended boundary and artifacts are inevitable, so the
  // only case where we can apply this fixup is for "sprites" or similar which should not share edges, which leads to
  // this unfortunate code below.

  // It might be faster to do more direct checking here, but the code below handles primitives in any order and
  // orientation, and is far more SIMD-friendly if needed.
  const float abx = vertices[1].x - vertices[0].x;
  const float aby = vertices[1].y - vertices[0].y;
  const float bcx = vertices[2].x - vertices[1].x;
  const float bcy = vertices[2].y - vertices[1].y;
  const float cax = vertices[0].x - vertices[2].x;
  const float cay = vertices[0].y - vertices[2].y;

  // Compute static derivatives, just assume W is uniform across the primitive and that the plane equation remains the
  // same across the quad. (which it is, there is no Z.. yet).
  const float dudx = -aby * static_cast<float>(vertices[2].u) - bcy * static_cast<float>(vertices[0].u) -
                     cay * static_cast<float>(vertices[1].u);
  const float dvdx = -aby * static_cast<float>(vertices[2].v) - bcy * static_cast<float>(vertices[0].v) -
                     cay * static_cast<float>(vertices[1].v);
  const float dudy = +abx * static_cast<float>(vertices[2].u) + bcx * static_cast<float>(vertices[0].u) +
                     cax * static_cast<float>(vertices[1].u);
  const float dvdy = +abx * static_cast<float>(vertices[2].v) + bcx * static_cast<float>(vertices[0].v) +
                     cax * static_cast<float>(vertices[1].v);
  const float area = bcx * cay - bcy * cax;

  // Detect and reject any triangles with 0 size texture area
  const s32 texArea = (vertices[1].u - vertices[0].u) * (vertices[2].v - vertices[0].v) -
                      (vertices[2].u - vertices[0].u) * (vertices[1].v - vertices[0].v);

  // Leverage PGXP to further avoid 3D polygons that just happen to align this way after projection
  const bool is_3d = (vertices[0].w != vertices[1].w || vertices[0].w != vertices[2].w);

  // Shouldn't matter as degenerate primitives will be culled anyways.
  if (area == 0.0f || texArea == 0 || is_3d)
    return;

  // Use floats here as it'll be faster than integer divides.
  const float rcp_area = 1.0f / area;
  const float dudx_area = dudx * rcp_area;
  const float dudy_area = dudy * rcp_area;
  const float dvdx_area = dvdx * rcp_area;
  const float dvdy_area = dvdy * rcp_area;
  const bool neg_dudx = dudx_area < 0.0f;
  const bool neg_dudy = dudy_area < 0.0f;
  const bool neg_dvdx = dvdx_area < 0.0f;
  const bool neg_dvdy = dvdy_area < 0.0f;
  const bool zero_dudx = dudx_area == 0.0f;
  const bool zero_dudy = dudy_area == 0.0f;
  const bool zero_dvdx = dvdx_area == 0.0f;
  const bool zero_dvdy = dvdy_area == 0.0f;

  // If we have negative dU or dV in any direction, increment the U or V to work properly with nearest-neighbor in
  // this impl. If we don't have 1:1 pixel correspondence, this creates a slight "shift" in the sprite, but we
  // guarantee that we don't sample garbage at least. Overall, this is kinda hacky because there can be legitimate,
  // rare cases where 3D meshes hit this scenario, and a single texel offset can pop in, but this is way better than
  // having borked 2D overall.
  //
  // TODO: If perf becomes an issue, we can probably SIMD the 8 comparisons above,
  // create an 8-bit code, and use a LUT to get the offsets.
  // Case 1: U is decreasing in X, but no change in Y.
  // Case 2: U is decreasing in Y, but no change in X.
  // Case 3: V is decreasing in X, but no change in Y.
  // Case 4: V is decreasing in Y, but no change in X.
  if ((neg_dudx && zero_dudy) || (neg_dudy && zero_dudx))
  {
    vertices[0].u++;
    vertices[1].u++;
    vertices[2].u++;
    vertices[3].u++;
  }

  if ((neg_dvdx && zero_dvdy) || (neg_dvdy && zero_dvdx))
  {
    vertices[0].v++;
    vertices[1].v++;
    vertices[2].v++;
    vertices[3].v++;
  }
}

void GPUHWBackend::ComputePolygonUVLimits(const GPUBackendDrawCommand* cmd, BatchVertex* vertices, u32 num_vertices)
{
  u32 min_u = vertices[0].u, max_u = vertices[0].u, min_v = vertices[0].v, max_v = vertices[0].v;
  for (u32 i = 1; i < num_vertices; i++)
  {
    min_u = std::min<u32>(min_u, vertices[i].u);
    max_u = std::max<u32>(max_u, vertices[i].u);
    min_v = std::min<u32>(min_v, vertices[i].v);
    max_v = std::max<u32>(max_v, vertices[i].v);
  }

  if (min_u != max_u)
    max_u--;
  if (min_v != max_v)
    max_v--;

  if (m_texpage_dirty)
    CheckForTexPageOverlap(cmd, num_vertices, min_u, min_v, max_u, max_v);

  for (u32 i = 0; i < num_vertices; i++)
    vertices[i].SetUVLimits(min_u, max_u, min_v, max_v);
}

void GPUHWBackend::SetBatchDepthBuffer(bool enabled, u32 num_vertices)
{
  if (m_batch.use_depth_buffer == enabled)
    return;

  if (GetBatchVertexCount() > 0)
  {
    FlushRender();
    EnsureVertexBufferSpace(num_vertices);
  }

  m_batch.use_depth_buffer = enabled;
}

void GPUHWBackend::CheckForDepthClear(const BatchVertex* vertices, u32 num_vertices)
{
  DebugAssert(num_vertices == 3 || num_vertices == 4);
  float average_z;
  if (num_vertices == 3)
    average_z = std::min((vertices[0].w + vertices[1].w + vertices[2].w) / 3.0f, 1.0f);
  else
    average_z = std::min((vertices[0].w + vertices[1].w + vertices[2].w + vertices[3].w) / 4.0f, 1.0f);

  if ((average_z - m_last_depth_z) >= g_gpu_settings.gpu_pgxp_depth_clear_threshold)
  {
    if (GetBatchVertexCount() > 0)
    {
      FlushRender();
      EnsureVertexBufferSpace(num_vertices);
    }

    ClearDepthBuffer();
  }

  m_last_depth_z = average_z;
}

u32 GPUHWBackend::GetAdaptiveDownsamplingMipLevels() const
{
  u32 levels = 0;
  u32 current_width = VRAM_WIDTH * m_resolution_scale;
  while (current_width >= VRAM_WIDTH)
  {
    levels++;
    current_width /= 2;
  }

  return levels;
}

void GPUHWBackend::DrawLine(const GPUBackendDrawLineCommand* cmd)
{
  const u32 required_vertices = cmd->num_vertices * 6;
  PrepareDraw(cmd, required_vertices);
  SetBatchDepthBuffer(false, required_vertices);

  const u32 num_vertices = cmd->num_vertices;
  const float depth = GetCurrentNormalizedVertexDepth();
  DebugAssert(GetBatchVertexSpace() >= (num_vertices * 6));

  for (u32 i = 1; i < num_vertices; i++)
  {
    const GPUBackendDrawLineCommand::Vertex& start = cmd->vertices[i - 1];
    const GPUBackendDrawLineCommand::Vertex& end = cmd->vertices[i];

    const u32 clip_left =
      static_cast<u32>(std::clamp<s32>(std::min(start.x, end.x), m_drawing_area.left, m_drawing_area.right));
    const u32 clip_right =
      static_cast<u32>(std::clamp<s32>(std::max(start.x, end.x), m_drawing_area.left, m_drawing_area.right)) + 1u;
    const u32 clip_top =
      static_cast<u32>(std::clamp<s32>(std::min(start.y, end.y), m_drawing_area.top, m_drawing_area.bottom));
    const u32 clip_bottom =
      static_cast<u32>(std::clamp<s32>(std::max(start.y, end.y), m_drawing_area.top, m_drawing_area.bottom)) + 1u;

    m_vram_dirty_draw_rect.Include(clip_left, clip_right, clip_top, clip_bottom);

    // TODO: Should we do a PGXP lookup here? Most lines are 2D.
    DrawLine(static_cast<float>(start.x), static_cast<float>(start.y), start.color, static_cast<float>(end.x),
             static_cast<float>(end.y), end.color, depth);
  }

  // m_sw_renderer->PushCommand(cmd);
}

void GPUHWBackend::DrawLine(float x0, float y0, u32 col0, float x1, float y1, u32 col1, float depth)
{
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  std::array<BatchVertex, 4> output;
  if (dx == 0.0f && dy == 0.0f)
  {
    // Degenerate, render a point.
    output[0].Set(x0, y0, depth, 1.0f, col0, 0, 0, 0);
    output[1].Set(x0 + 1.0f, y0, depth, 1.0f, col0, 0, 0, 0);
    output[2].Set(x1, y1 + 1.0f, depth, 1.0f, col0, 0, 0, 0);
    output[3].Set(x1 + 1.0f, y1 + 1.0f, depth, 1.0f, col0, 0, 0, 0);
  }
  else
  {
    const float abs_dx = std::fabs(dx);
    const float abs_dy = std::fabs(dy);
    float fill_dx, fill_dy;
    float dxdk, dydk;
    float pad_x0 = 0.0f;
    float pad_x1 = 0.0f;
    float pad_y0 = 0.0f;
    float pad_y1 = 0.0f;

    // Check for vertical or horizontal major lines.
    // When expanding to a rect, do so in the appropriate direction.
    // FIXME: This scheme seems to kinda work, but it seems very hard to find a method
    // that looks perfect on every game.
    // Vagrant Story speech bubbles are a very good test case here!
    if (abs_dx > abs_dy)
    {
      fill_dx = 0.0f;
      fill_dy = 1.0f;
      dxdk = 1.0f;
      dydk = dy / abs_dx;

      if (dx > 0.0f)
      {
        // Right
        pad_x1 = 1.0f;
        pad_y1 = dydk;
      }
      else
      {
        // Left
        pad_x0 = 1.0f;
        pad_y0 = -dydk;
      }
    }
    else
    {
      fill_dx = 1.0f;
      fill_dy = 0.0f;
      dydk = 1.0f;
      dxdk = dx / abs_dy;

      if (dy > 0.0f)
      {
        // Down
        pad_y1 = 1.0f;
        pad_x1 = dxdk;
      }
      else
      {
        // Up
        pad_y0 = 1.0f;
        pad_x0 = -dxdk;
      }
    }

    const float ox0 = x0 + pad_x0;
    const float oy0 = y0 + pad_y0;
    const float ox1 = x1 + pad_x1;
    const float oy1 = y1 + pad_y1;

    output[0].Set(ox0, oy0, depth, 1.0f, col0, 0, 0, 0);
    output[1].Set(ox0 + fill_dx, oy0 + fill_dy, depth, 1.0f, col0, 0, 0, 0);
    output[2].Set(ox1, oy1, depth, 1.0f, col1, 0, 0, 0);
    output[3].Set(ox1 + fill_dx, oy1 + fill_dy, depth, 1.0f, col1, 0, 0, 0);
  }

  AddVertex(output[0]);
  AddVertex(output[1]);
  AddVertex(output[2]);
  AddVertex(output[3]);
  AddVertex(output[2]);
  AddVertex(output[1]);
}

void GPUHWBackend::DrawSprite(const GPUBackendDrawSpriteCommand* cmd)
{
  // we can split the rectangle up into potentially 8 quads
  PrepareDraw(cmd, MAX_VERTICES_FOR_SPRITE);
  SetBatchDepthBuffer(false, MAX_VERTICES_FOR_SPRITE);
  DebugAssert(GetBatchVertexSpace() >= MAX_VERTICES_FOR_SPRITE);

  const u32 orig_tex_left = ZeroExtend32(Truncate8(cmd->texcoord));
  const u32 orig_tex_top = ZeroExtend32(cmd->texcoord) >> 8;
  const s32 width = static_cast<s32>(ZeroExtend32(cmd->width));
  const s32 height = static_cast<s32>(ZeroExtend32(cmd->height));

  if (cmd->rc.texture_enable && m_texpage_dirty)
  {
    const bool wrap_x = ((orig_tex_left + width) >= TEXTURE_PAGE_WIDTH);
    const bool wrap_y = ((orig_tex_top + height) >= TEXTURE_PAGE_HEIGHT);
    CheckForTexPageOverlap(cmd, MAX_VERTICES_FOR_SPRITE, wrap_x ? 0 : orig_tex_left, wrap_y ? 0 : orig_tex_top,
                           wrap_x ? (TEXTURE_PAGE_WIDTH - 1) : (orig_tex_left + static_cast<u32>(width) - 1),
                           wrap_y ? (TEXTURE_PAGE_HEIGHT - 1) : (orig_tex_top + static_cast<u32>(height) - 1));
  }

  const s32 pos_x = cmd->x;
  const s32 pos_y = cmd->y;
  const u32 texpage = m_texpage_bits;
  const u32 color = cmd->color;
  const float depth = GetCurrentNormalizedVertexDepth();

  // Split the rectangle into multiple quads if it's greater than 256x256, as the texture page should repeat.
  u32 tex_top = orig_tex_top;
  for (s32 y_offset = 0; y_offset < height;)
  {
    const s32 quad_height = std::min<s32>(height - y_offset, TEXTURE_PAGE_WIDTH - tex_top);
    const float quad_start_y = static_cast<float>(pos_y + y_offset);
    const float quad_end_y = quad_start_y + static_cast<float>(quad_height);
    const u32 tex_bottom = tex_top + static_cast<u32>(quad_height);

    u32 tex_left = orig_tex_left;
    for (s32 x_offset = 0; x_offset < width;)
    {
      const s32 quad_width = std::min<s32>(width - x_offset, TEXTURE_PAGE_HEIGHT - tex_left);
      const float quad_start_x = static_cast<float>(pos_x + x_offset);
      const float quad_end_x = quad_start_x + static_cast<float>(quad_width);
      const u32 tex_right = tex_left + static_cast<u32>(quad_width);
      const u32 uv_limits = BatchVertex::PackUVLimits(tex_left, tex_right - 1, tex_top, tex_bottom - 1);

      AddNewVertex(quad_start_x, quad_start_y, depth, 1.0f, color, texpage, Truncate16(tex_left), Truncate16(tex_top),
                   uv_limits);
      AddNewVertex(quad_end_x, quad_start_y, depth, 1.0f, color, texpage, Truncate16(tex_right), Truncate16(tex_top),
                   uv_limits);
      AddNewVertex(quad_start_x, quad_end_y, depth, 1.0f, color, texpage, Truncate16(tex_left), Truncate16(tex_bottom),
                   uv_limits);

      AddNewVertex(quad_start_x, quad_end_y, depth, 1.0f, color, texpage, Truncate16(tex_left), Truncate16(tex_bottom),
                   uv_limits);
      AddNewVertex(quad_end_x, quad_start_y, depth, 1.0f, color, texpage, Truncate16(tex_right), Truncate16(tex_top),
                   uv_limits);
      AddNewVertex(quad_end_x, quad_end_y, depth, 1.0f, color, texpage, Truncate16(tex_right), Truncate16(tex_bottom),
                   uv_limits);

      x_offset += quad_width;
      tex_left = 0;
    }

    y_offset += quad_height;
    tex_top = 0;
  }

  const u32 clip_left = static_cast<u32>(std::clamp<s32>(pos_x, m_drawing_area.left, m_drawing_area.right));
  const u32 clip_right =
    static_cast<u32>(std::clamp<s32>(pos_x + width, m_drawing_area.left, m_drawing_area.right)) + 1u;
  const u32 clip_top = static_cast<u32>(std::clamp<s32>(pos_y, m_drawing_area.top, m_drawing_area.bottom));
  const u32 clip_bottom =
    static_cast<u32>(std::clamp<s32>(pos_y + height, m_drawing_area.top, m_drawing_area.bottom)) + 1u;

  m_vram_dirty_draw_rect.Include(clip_left, clip_right, clip_top, clip_bottom);

  //   if (m_sw_renderer)
  //   {
  //     GPUBackendDrawRectangleCommand* cmd = m_sw_renderer->NewDrawRectangleCommand();
  //     FillDrawCommand(cmd, rc);
  //     cmd->color = color;
  //     cmd->x = pos_x;
  //     cmd->y = pos_y;
  //     cmd->width = static_cast<u16>(width);
  //     cmd->height = static_cast<u16>(height);
  //     cmd->texcoord = (static_cast<u16>(texcoord_y) << 8) | static_cast<u16>(texcoord_x);
  //     m_sw_renderer->PushCommand(cmd);
  //   }
}

void GPUHWBackend::DrawPolygon(const GPUBackendDrawPolygonCommand* cmd)
{
  const u32 required_vertices = cmd->rc.quad_polygon ? 6 : 3;
  PrepareDraw(cmd, required_vertices);
  SetBatchDepthBuffer(false, required_vertices);

  // TODO: This could write directly to the mapped GPU pointer. But watch out for the reads below.
  const float depth = GetCurrentNormalizedVertexDepth();
  const u32 num_vertices = cmd->num_vertices;
  const u32 texpage = m_texpage_bits;
  std::array<BatchVertex, 4> vertices;
  s32 min_x = std::numeric_limits<s32>::max(), min_y = std::numeric_limits<s32>::max();
  s32 max_x = std::numeric_limits<s32>::min(), max_y = std::numeric_limits<s32>::min();
  for (u32 i = 0; i < num_vertices; i++)
  {
    const GPUBackendDrawPolygonCommand::Vertex& vert = cmd->vertices[i];
    vertices[i].Set(static_cast<float>(vert.x), static_cast<float>(vert.y), depth, 1.0f, vert.color, texpage,
                    vert.texcoord, 0xFFFF0000u);
    min_x = std::min(min_x, vert.x);
    max_x = std::max(max_x, vert.x);
    min_y = std::min(min_y, vert.y);
    max_y = std::max(max_y, vert.y);
  }

  if (cmd->rc.quad_polygon && m_resolution_scale > 1)
    HandleFlippedQuadTextureCoordinates(vertices.data());

  if (m_compute_uv_range && cmd->rc.texture_enable)
    ComputePolygonUVLimits(cmd, vertices.data(), num_vertices);

  const u32 clip_left = static_cast<u32>(std::clamp<s32>(min_x, m_drawing_area.left, m_drawing_area.right));
  const u32 clip_right = static_cast<u32>(std::clamp<s32>(max_x, m_drawing_area.left, m_drawing_area.right)) + 1u;
  const u32 clip_top = static_cast<u32>(std::clamp<s32>(min_y, m_drawing_area.top, m_drawing_area.bottom));
  const u32 clip_bottom = static_cast<u32>(std::clamp<s32>(max_y, m_drawing_area.top, m_drawing_area.bottom)) + 1u;

  m_vram_dirty_draw_rect.Include(clip_left, clip_right, clip_top, clip_bottom);
  std::memcpy(m_batch_current_vertex_ptr, vertices.data(), sizeof(BatchVertex) * 3);
  m_batch_current_vertex_ptr += 3;

  if (cmd->rc.quad_polygon)
  {
    AddVertex(vertices[2]);
    AddVertex(vertices[1]);
    AddVertex(vertices[3]);
  }

  // m_sw_renderer->PushCommand(cmd);
}

void GPUHWBackend::DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd)
{
  const u32 required_vertices = cmd->rc.quad_polygon ? 6 : 3;
  PrepareDraw(cmd, required_vertices);

  // TODO: This could write directly to the mapped GPU pointer. But watch out for the reads below.
  const float depth = GetCurrentNormalizedVertexDepth();
  const u32 num_vertices = cmd->num_vertices;
  const u32 texpage = m_texpage_bits;
  std::array<BatchVertex, 4> vertices;
  s32 min_x = std::numeric_limits<s32>::max(), min_y = std::numeric_limits<s32>::max();
  s32 max_x = std::numeric_limits<s32>::min(), max_y = std::numeric_limits<s32>::min();
  for (u32 i = 0; i < num_vertices; i++)
  {
    const GPUBackendDrawPrecisePolygonCommand::Vertex& vert = cmd->vertices[i];
    vertices[i].Set(vert.x, vert.y, depth, vert.w, vert.color, texpage, vert.texcoord, 0xFFFF0000u);
    min_x = std::min(min_x, vert.native_x);
    max_x = std::max(max_x, vert.native_x);
    min_y = std::min(min_y, vert.native_y);
    max_y = std::max(max_y, vert.native_y);
  }

  const bool use_depth = (m_pgxp_depth_buffer && cmd->valid_w && !cmd->rc.transparency_enable);
  SetBatchDepthBuffer(use_depth, required_vertices);
  if (use_depth)
    CheckForDepthClear(vertices.data(), num_vertices);

  if (cmd->rc.quad_polygon && m_resolution_scale > 1)
    HandleFlippedQuadTextureCoordinates(vertices.data());

  if (m_compute_uv_range && cmd->rc.texture_enable)
    ComputePolygonUVLimits(cmd, vertices.data(), num_vertices);

  const u32 clip_left = static_cast<u32>(std::clamp<s32>(min_x, m_drawing_area.left, m_drawing_area.right));
  const u32 clip_right = static_cast<u32>(std::clamp<s32>(max_x, m_drawing_area.left, m_drawing_area.right)) + 1u;
  const u32 clip_top = static_cast<u32>(std::clamp<s32>(min_y, m_drawing_area.top, m_drawing_area.bottom));
  const u32 clip_bottom = static_cast<u32>(std::clamp<s32>(max_y, m_drawing_area.top, m_drawing_area.bottom)) + 1u;

  m_vram_dirty_draw_rect.Include(clip_left, clip_right, clip_top, clip_bottom);
  std::memcpy(m_batch_current_vertex_ptr, vertices.data(), sizeof(BatchVertex) * 3);
  m_batch_current_vertex_ptr += 3;

  if (cmd->rc.quad_polygon)
  {
    AddVertex(vertices[2]);
    AddVertex(vertices[1]);
    AddVertex(vertices[3]);
  }

  // m_sw_renderer->PushCommand(cmd);
}

bool GPUHWBackend::BlitVRAMReplacementTexture(const TextureReplacementTexture* tex, u32 dst_x, u32 dst_y, u32 width,
                                              u32 height)
{
  if (!m_vram_replacement_texture || m_vram_replacement_texture->GetWidth() < tex->GetWidth() ||
      m_vram_replacement_texture->GetHeight() < tex->GetHeight())
  {
    g_gpu_device->RecycleTexture(std::move(m_vram_replacement_texture));

    if (!(m_vram_replacement_texture =
            g_gpu_device->FetchTexture(tex->GetWidth(), tex->GetHeight(), 1, 1, 1, GPUTexture::Type::DynamicTexture,
                                       GPUTexture::Format::RGBA8, tex->GetPixels(), tex->GetPitch())))
    {
      return false;
    }
  }
  else
  {
    if (!m_vram_replacement_texture->Update(0, 0, width, height, tex->GetPixels(), tex->GetPitch()))
    {
      Log_ErrorFmt("Update {}x{} texture failed.", width, height);
      return false;
    }
  }

  g_gpu_device->SetRenderTarget(m_vram_texture.get(), m_vram_depth_texture.get()); // TODO: needed?
  g_gpu_device->SetTextureSampler(0, m_vram_replacement_texture.get(), g_gpu_device->GetLinearSampler());
  g_gpu_device->SetPipeline(m_copy_pipeline.get());
  g_gpu_device->SetViewportAndScissor(dst_x, dst_y, width, height);
  g_gpu_device->Draw(3, 0);

  RestoreDeviceContext();
  return true;
}

void GPUHWBackend::IncludeVRAMDirtyRectangle(Common::Rectangle<u32>&rect, const Common::Rectangle<u32>&new_rect)
{
  rect.Include(new_rect);

  // the vram area can include the texture page, but the game can leave it as-is. in this case, set it as dirty so the
  // shadow texture is updated
  //   if (!m_draw_mode.IsTexturePageChanged() &&
  //       (m_draw_mode.mode_reg.GetTexturePageRectangle().Intersects(new_rect) ||
  //        (m_draw_mode.mode_reg.IsUsingPalette() &&
  //         m_draw_mode.palette_reg.GetRectangle(m_draw_mode.mode_reg.texture_mode).Intersects(new_rect))))
  //   {
  //     m_draw_mode.SetTexturePageChanged();
  //   }

  // TODO: This is a nuclear option, we only need to invalidate it if it overlaps.
  m_texpage_bits = INVALID_TEXPAGE_BITS;
}

ALWAYS_INLINE_RELEASE void GPUHWBackend::CheckForTexPageOverlap(const GPUBackendDrawCommand* cmd, u32 num_vertices,
                                                                u32 min_u, u32 min_v, u32 max_u, u32 max_v)
{
  static constexpr std::array<std::array<u8, 2>, 4> uv_shifts_adds = {{{2, 3}, {1, 1}, {0, 0}, {0, 0}}};

  const u32 xoffs = (m_texpage_bits & 0xFu) * 64u;
  const u32 yoffs = ((m_texpage_bits >> 4) & 1u) * 256u;
  const u32 xshift = uv_shifts_adds[(m_texpage_bits >> 7) & 3][0];
  const u32 xadd = uv_shifts_adds[(m_texpage_bits >> 7) & 3][1];

  const u32 vram_min_u = (((min_u & cmd->window.and_x) | cmd->window.or_x) >> xshift) + xoffs;
  const u32 vram_max_u = ((((max_u & cmd->window.and_x) | cmd->window.or_x) + xadd) >> xshift) + xoffs;
  const u32 vram_min_v = ((min_v & cmd->window.and_y) | cmd->window.or_y) + yoffs;
  const u32 vram_max_v = ((max_v & cmd->window.and_y) | cmd->window.or_y) + yoffs;

  // Log_InfoFmt("{}: {},{} => {},{}", s_draw_number, vram_min_u, vram_min_v, vram_max_u, vram_max_v);

  if (vram_min_u < m_current_uv_range.left || vram_min_v < m_current_uv_range.top ||
      vram_max_u >= m_current_uv_range.right || vram_max_v >= m_current_uv_range.bottom)
  {
    m_current_uv_range.Include(vram_min_u, vram_max_u, vram_min_v, vram_max_v);

    bool update_drawn = false, update_written = false;
    if (m_texpage_dirty & TEXPAGE_DIRTY_DRAWN_RECT)
    {
      DebugAssert(m_vram_dirty_draw_rect.Valid());
      update_drawn = m_current_uv_range.Intersects(m_vram_dirty_draw_rect);
      if (update_drawn)
      {
        GL_INS_FMT("Updating VRAM cache due to UV {{{},{} => {},{}}} intersection with dirty DRAW {{{},{} => {},{}}}",
                   m_current_uv_range.left, m_current_uv_range.top, m_current_uv_range.right, m_current_uv_range.bottom,
                   m_vram_dirty_draw_rect.left, m_vram_dirty_draw_rect.top, m_vram_dirty_draw_rect.right,
                   m_vram_dirty_draw_rect.bottom);
      }
    }
    if (m_texpage_dirty & TEXPAGE_DIRTY_WRITTEN_RECT)
    {
      DebugAssert(m_vram_dirty_write_rect.Valid());
      update_written = m_current_uv_range.Intersects(m_vram_dirty_write_rect);
      if (update_written)
      {
        GL_INS_FMT("Updating VRAM cache due to UV {{{},{} => {},{}}} intersection with dirty WRITE {{{},{} => {},{}}}",
                   m_current_uv_range.left, m_current_uv_range.top, m_current_uv_range.right, m_current_uv_range.bottom,
                   m_vram_dirty_write_rect.left, m_vram_dirty_write_rect.top, m_vram_dirty_write_rect.right,
                   m_vram_dirty_write_rect.bottom);
      }
    }

    if (update_drawn || update_written)
    {
      if (GetBatchVertexCount() > 0)
      {
        FlushRender();
        EnsureVertexBufferSpace(num_vertices);
      }

      UpdateVRAMReadTexture(update_drawn, update_written);
    }
  }
}

ALWAYS_INLINE bool GPUHWBackend::IsFlushed() const
{
  return m_batch_current_vertex_ptr == m_batch_start_vertex_ptr;
}

GPUHWBackend::InterlacedRenderMode
GPUHWBackend::GetInterlacedRenderMode(const GPUBackendUpdateDisplayCommand* cmd) const
{
  if (cmd->interlaced_display_enabled)
  {
    return cmd->interlaced_display_interleaved ? InterlacedRenderMode::InterleavedFields :
                                                 InterlacedRenderMode::SeparateFields;
  }
  else
  {
    return InterlacedRenderMode::None;
  }
}

ALWAYS_INLINE_RELEASE bool GPUHWBackend::NeedsTwoPassRendering() const
{
  // We need two-pass rendering when using BG-FG blending and texturing, as the transparency can be enabled
  // on a per-pixel basis, and the opaque pixels shouldn't be blended at all.

  return (m_batch.texture_mode != GPUTextureMode::Disabled && !m_supports_framebuffer_fetch &&
          (m_batch.transparency_mode == GPUTransparencyMode::BackgroundMinusForeground ||
           (!m_supports_dual_source_blend && m_batch.transparency_mode != GPUTransparencyMode::Disabled)));
}

ALWAYS_INLINE_RELEASE bool GPUHWBackend::NeedsShaderBlending(GPUTransparencyMode transparency) const
{
  return (m_supports_framebuffer_fetch &&
          (transparency == GPUTransparencyMode::BackgroundMinusForeground ||
           (!m_supports_dual_source_blend &&
            (transparency != GPUTransparencyMode::Disabled || IsBlendedTextureFiltering(m_texture_filtering)))));
}

ALWAYS_INLINE u32 GPUHWBackend::GetBatchVertexSpace() const
{
  return static_cast<u32>(m_batch_end_vertex_ptr - m_batch_current_vertex_ptr);
}

ALWAYS_INLINE u32 GPUHWBackend::GetBatchVertexCount() const
{
  return static_cast<u32>(m_batch_current_vertex_ptr - m_batch_start_vertex_ptr);
}

void GPUHWBackend::EnsureVertexBufferSpace(u32 required_vertices)
{
  // can we fit these vertices in the current depth buffer range?
  if ((m_current_depth + required_vertices) > MAX_BATCH_VERTEX_COUNTER_IDS)
  {
    // implies FlushRender()
    ResetBatchVertexDepth();
  }

  if (m_batch_current_vertex_ptr)
  {
    if (GetBatchVertexSpace() >= required_vertices)
      return;

    FlushRender();
  }

  MapBatchVertexPointer(required_vertices);
}

void GPUHWBackend::ResetBatchVertexDepth()
{
  if (m_pgxp_depth_buffer)
    return;

  Log_PerfPrint("Resetting batch vertex depth");
  FlushRender();
  UpdateDepthBufferFromMaskBit();

  m_current_depth = 1;
}

ALWAYS_INLINE float GPUHWBackend::GetCurrentNormalizedVertexDepth() const
{
  return 1.0f - (static_cast<float>(m_current_depth) / 65535.0f);
}

void GPUHWBackend::UpdateSoftwareRenderer(bool copy_vram_from_hw)
{
  const bool current_enabled = false; // (m_sw_renderer != nullptr);
  const bool new_enabled = g_gpu_settings.gpu_use_software_renderer_for_readbacks;
  if (current_enabled == new_enabled)
    return;

#if 0
  m_vram_ptr = m_vram_shadow.data();

  if (!new_enabled)
  {
    if (m_sw_renderer)
      m_sw_renderer->Shutdown();
    m_sw_renderer.reset();
    return;
  }

  std::unique_ptr<GPU_SW_Backend> sw_renderer = std::make_unique<GPU_SW_Backend>();
  if (!sw_renderer->Initialize(true))
    return;

  // We need to fill in the SW renderer's VRAM with the current state for hot toggles.
  if (copy_vram_from_hw)
  {
    FlushRender();
    ReadVRAM(0, 0, VRAM_WIDTH, VRAM_HEIGHT);
    std::memcpy(sw_renderer->GetVRAM(), m_vram_ptr, sizeof(u16) * VRAM_WIDTH * VRAM_HEIGHT);

    // Sync the drawing area.
    GPUBackendSetDrawingAreaCommand* cmd = sw_renderer->NewSetDrawingAreaCommand();
    cmd->new_area = m_drawing_area;
    sw_renderer->PushCommand(cmd);
  }

  m_sw_renderer = std::move(sw_renderer);
  m_vram_ptr = m_sw_renderer->GetVRAM();
#else
  Panic("FIXME");
#endif
}

void GPUHWBackend::FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color, GPUBackendCommandParameters params)
{
  GL_SCOPE_FMT("FillVRAM({},{} => {},{} ({}x{}) with 0x{:08X}", x, y, x + width, y + height, width, height, color);

  //   if (m_sw_renderer)
  //   {
  //     GPUBackendFillVRAMCommand* cmd = m_sw_renderer->NewFillVRAMCommand();
  //     FillBackendCommandParameters(cmd);
  //     cmd->x = static_cast<u16>(x);
  //     cmd->y = static_cast<u16>(y);
  //     cmd->width = static_cast<u16>(width);
  //     cmd->height = static_cast<u16>(height);
  //     cmd->color = color;
  //     m_sw_renderer->PushCommand(cmd);
  //   }

  GL_INS_FMT("Dirty draw area before: {},{} => {},{} ({}x{})", m_vram_dirty_draw_rect.left, m_vram_dirty_draw_rect.top,
             m_vram_dirty_draw_rect.right, m_vram_dirty_draw_rect.bottom, m_vram_dirty_draw_rect.GetWidth(),
             m_vram_dirty_draw_rect.GetHeight());

  IncludeVRAMDirtyRectangle(
    m_vram_dirty_draw_rect,
    Common::Rectangle<u32>::FromExtents(x, y, width, height).Clamped(0, 0, VRAM_WIDTH, VRAM_HEIGHT));

  GL_INS_FMT("Dirty draw area after: {},{} => {},{} ({}x{})", m_vram_dirty_draw_rect.left, m_vram_dirty_draw_rect.top,
             m_vram_dirty_draw_rect.right, m_vram_dirty_draw_rect.bottom, m_vram_dirty_draw_rect.GetWidth(),
             m_vram_dirty_draw_rect.GetHeight());

  const bool is_oversized = (((x + width) > VRAM_WIDTH || (y + height) > VRAM_HEIGHT));
  g_gpu_device->SetPipeline(
    m_vram_fill_pipelines[BoolToUInt8(is_oversized)][BoolToUInt8(params.interlaced_rendering)].get());

  const Common::Rectangle<u32> bounds(GetVRAMTransferBounds(x, y, width, height));
  g_gpu_device->SetViewportAndScissor(bounds.left * m_resolution_scale, bounds.top * m_resolution_scale,
                                      bounds.GetWidth() * m_resolution_scale, bounds.GetHeight() * m_resolution_scale);

  struct VRAMFillUBOData
  {
    u32 u_dst_x;
    u32 u_dst_y;
    u32 u_end_x;
    u32 u_end_y;
    std::array<float, 4> u_fill_color;
    u32 u_interlaced_displayed_field;
  };
  VRAMFillUBOData uniforms;
  uniforms.u_dst_x = (x % VRAM_WIDTH) * m_resolution_scale;
  uniforms.u_dst_y = (y % VRAM_HEIGHT) * m_resolution_scale;
  uniforms.u_end_x = ((x + width) % VRAM_WIDTH) * m_resolution_scale;
  uniforms.u_end_y = ((y + height) % VRAM_HEIGHT) * m_resolution_scale;
  // drop precision unless true colour is enabled
  uniforms.u_fill_color =
    GPUDevice::RGBA8ToFloat(m_true_color ? color : VRAMRGBA5551ToRGBA8888(VRAMRGBA8888ToRGBA5551(color)));
  uniforms.u_interlaced_displayed_field = params.active_line_lsb;
  g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
  g_gpu_device->Draw(3, 0);

  RestoreDeviceContext();
}

void GPUHWBackend::ReadVRAM(u32 x, u32 y, u32 width, u32 height)
{
  GL_PUSH_FMT("ReadVRAM({},{} => {},{} ({}x{})", x, y, x + width, y + height, width, height);

  //   if (m_sw_renderer)
  //   {
  //     m_sw_renderer->Sync(false);
  //     GL_POP();
  //     return;
  //   }

  // Get bounds with wrap-around handled.
  const Common::Rectangle<u32> copy_rect = GetVRAMTransferBounds(x, y, width, height);
  const u32 encoded_width = (copy_rect.GetWidth() + 1) / 2;
  const u32 encoded_height = copy_rect.GetHeight();

  // Encode the 24-bit texture as 16-bit.
  const u32 uniforms[4] = {copy_rect.left, copy_rect.top, copy_rect.GetWidth(), copy_rect.GetHeight()};
  g_gpu_device->SetRenderTarget(m_vram_readback_texture.get());
  g_gpu_device->SetPipeline(m_vram_readback_pipeline.get());
  g_gpu_device->SetTextureSampler(0, m_vram_texture.get(), g_gpu_device->GetNearestSampler());
  g_gpu_device->SetViewportAndScissor(0, 0, encoded_width, encoded_height);
  g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));
  g_gpu_device->Draw(3, 0);
  m_vram_readback_texture->MakeReadyForSampling();
  GL_POP();

  // Stage the readback and copy it into our shadow buffer.
  g_gpu_device->DownloadTexture(m_vram_readback_texture.get(), 0, 0, encoded_width, encoded_height,
                                reinterpret_cast<u32*>(&GPU::m_vram_ptr[copy_rect.top * VRAM_WIDTH + copy_rect.left]),
                                VRAM_WIDTH * sizeof(u16));

  RestoreDeviceContext();
}

void GPUHWBackend::UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, GPUBackendCommandParameters params)
{
  GL_SCOPE_FMT("UpdateVRAM({},{} => {},{} ({}x{})", x, y, x + width, y + height, width, height);

  //   if (m_sw_renderer)
  //   {
  //     const u32 num_words = width * height;
  //     GPUBackendUpdateVRAMCommand* cmd = m_sw_renderer->NewUpdateVRAMCommand(num_words);
  //     FillBackendCommandParameters(cmd);
  //     cmd->params.set_mask_while_drawing = set_mask;
  //     cmd->params.check_mask_before_draw = check_mask;
  //     cmd->x = static_cast<u16>(x);
  //     cmd->y = static_cast<u16>(y);
  //     cmd->width = static_cast<u16>(width);
  //     cmd->height = static_cast<u16>(height);
  //     std::memcpy(cmd->data, data, sizeof(u16) * num_words);
  //     m_sw_renderer->PushCommand(cmd);
  //   }

  const Common::Rectangle<u32> bounds = GetVRAMTransferBounds(x, y, width, height);
  DebugAssert(bounds.right <= VRAM_WIDTH && bounds.bottom <= VRAM_HEIGHT);
  IncludeVRAMDirtyRectangle(m_vram_dirty_write_rect, bounds);

  if (params.check_mask_before_draw)
  {
    // set new vertex counter since we want this to take into consideration previous masked pixels
    m_current_depth++;
  }
  else
  {
    const TextureReplacementTexture* rtex = g_texture_replacements.GetVRAMWriteReplacement(width, height, data);
    if (rtex && BlitVRAMReplacementTexture(rtex, x * m_resolution_scale, y * m_resolution_scale,
                                           width * m_resolution_scale, height * m_resolution_scale))
    {
      return;
    }
  }

  std::unique_ptr<GPUTexture> upload_texture;
  u32 map_index;

  if (!g_gpu_device->GetFeatures().supports_texture_buffers)
  {
    map_index = 0;
    upload_texture = g_gpu_device->FetchTexture(width, height, 1, 1, 1, GPUTexture::Type::Texture,
                                                GPUTexture::Format::R16U, data, width * sizeof(u16));
    if (!upload_texture)
    {
      Log_ErrorFmt("Failed to get {}x{} upload texture. Things are gonna break.", width, height);
      return;
    }
  }
  else
  {
    const u32 num_pixels = width * height;
    void* map = m_vram_upload_buffer->Map(num_pixels);
    map_index = m_vram_upload_buffer->GetCurrentPosition();
    std::memcpy(map, data, num_pixels * sizeof(u16));
    m_vram_upload_buffer->Unmap(num_pixels);
  }

  struct VRAMWriteUBOData
  {
    u32 u_dst_x;
    u32 u_dst_y;
    u32 u_end_x;
    u32 u_end_y;
    u32 u_width;
    u32 u_height;
    u32 u_buffer_base_offset;
    u32 u_mask_or_bits;
    float u_depth_value;
  };
  const VRAMWriteUBOData uniforms = {(x % VRAM_WIDTH),
                                     (y % VRAM_HEIGHT),
                                     ((x + width) % VRAM_WIDTH),
                                     ((y + height) % VRAM_HEIGHT),
                                     width,
                                     height,
                                     map_index,
                                     (params.set_mask_while_drawing) ? 0x8000u : 0x00,
                                     GetCurrentNormalizedVertexDepth()};

  // the viewport should already be set to the full vram, so just adjust the scissor
  const Common::Rectangle<u32> scaled_bounds = bounds * m_resolution_scale;
  g_gpu_device->SetScissor(scaled_bounds.left, scaled_bounds.top, scaled_bounds.GetWidth(), scaled_bounds.GetHeight());
  g_gpu_device->SetPipeline(
    m_vram_write_pipelines[BoolToUInt8(params.check_mask_before_draw && !m_pgxp_depth_buffer)].get());
  g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
  if (upload_texture)
  {
    g_gpu_device->SetTextureSampler(0, upload_texture.get(), g_gpu_device->GetNearestSampler());
    g_gpu_device->Draw(3, 0);
    g_gpu_device->RecycleTexture(std::move(upload_texture));
  }
  else
  {
    g_gpu_device->SetTextureBuffer(0, m_vram_upload_buffer.get());
    g_gpu_device->Draw(3, 0);
  }

  RestoreDeviceContext();
}

void GPUHWBackend::CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height,
                            GPUBackendCommandParameters params)
{
  GL_SCOPE_FMT("CopyVRAM({}x{} @ {},{} => {},{}", width, height, src_x, src_y, dst_x, dst_y);

  //   if (m_sw_renderer)
  //   {
  //     GPUBackendCopyVRAMCommand* cmd = m_sw_renderer->NewCopyVRAMCommand();
  //     FillBackendCommandParameters(cmd);
  //     cmd->src_x = static_cast<u16>(src_x);
  //     cmd->src_y = static_cast<u16>(src_y);
  //     cmd->dst_x = static_cast<u16>(dst_x);
  //     cmd->dst_y = static_cast<u16>(dst_y);
  //     cmd->width = static_cast<u16>(width);
  //     cmd->height = static_cast<u16>(height);
  //     m_sw_renderer->PushCommand(cmd);
  //   }

  // masking enabled, oversized, or overlapping
  const bool use_shader =
    (params.IsMaskingEnabled() || ((src_x % VRAM_WIDTH) + width) > VRAM_WIDTH ||
     ((src_y % VRAM_HEIGHT) + height) > VRAM_HEIGHT || ((dst_x % VRAM_WIDTH) + width) > VRAM_WIDTH ||
     ((dst_y % VRAM_HEIGHT) + height) > VRAM_HEIGHT);
  const Common::Rectangle<u32> src_bounds = GetVRAMTransferBounds(src_x, src_y, width, height);
  const Common::Rectangle<u32> dst_bounds = GetVRAMTransferBounds(dst_x, dst_y, width, height);
  const bool intersect_with_draw = m_vram_dirty_draw_rect.Intersects(src_bounds);
  const bool intersect_with_write = m_vram_dirty_write_rect.Intersects(src_bounds);

  if (use_shader || IsUsingMultisampling())
  {
    if (intersect_with_draw || intersect_with_write)
      UpdateVRAMReadTexture(intersect_with_draw, intersect_with_write);
    IncludeVRAMDirtyRectangle(m_vram_dirty_draw_rect, dst_bounds);

    struct VRAMCopyUBOData
    {
      u32 u_src_x;
      u32 u_src_y;
      u32 u_dst_x;
      u32 u_dst_y;
      u32 u_end_x;
      u32 u_end_y;
      u32 u_width;
      u32 u_height;
      u32 u_set_mask_bit;
      float u_depth_value;
    };
    const VRAMCopyUBOData uniforms = {(src_x % VRAM_WIDTH) * m_resolution_scale,
                                      (src_y % VRAM_HEIGHT) * m_resolution_scale,
                                      (dst_x % VRAM_WIDTH) * m_resolution_scale,
                                      (dst_y % VRAM_HEIGHT) * m_resolution_scale,
                                      ((dst_x + width) % VRAM_WIDTH) * m_resolution_scale,
                                      ((dst_y + height) % VRAM_HEIGHT) * m_resolution_scale,
                                      width * m_resolution_scale,
                                      height * m_resolution_scale,
                                      params.set_mask_while_drawing ? 1u : 0u,
                                      GetCurrentNormalizedVertexDepth()};

    // VRAM read texture should already be bound.
    const Common::Rectangle<u32> dst_bounds_scaled(dst_bounds * m_resolution_scale);
    g_gpu_device->SetViewportAndScissor(dst_bounds_scaled.left, dst_bounds_scaled.top, dst_bounds_scaled.GetWidth(),
                                        dst_bounds_scaled.GetHeight());
    g_gpu_device->SetPipeline(
      m_vram_copy_pipelines[BoolToUInt8(params.check_mask_before_draw && !m_pgxp_depth_buffer)].get());
    g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
    g_gpu_device->Draw(3, 0);
    RestoreDeviceContext();

    if (params.check_mask_before_draw && !m_pgxp_depth_buffer)
      m_current_depth++;

    return;
  }

  GPUTexture* src_tex = m_vram_texture.get();
  const bool overlaps_with_self = src_bounds.Intersects(dst_bounds);
  if (!g_gpu_device->GetFeatures().texture_copy_to_self || overlaps_with_self)
  {
    src_tex = m_vram_read_texture.get();
    if (intersect_with_draw || intersect_with_write)
      UpdateVRAMReadTexture(intersect_with_draw, intersect_with_write);
  }

  Common::Rectangle<u32>* update_rect;
  if (intersect_with_draw || intersect_with_write)
  {
    update_rect = intersect_with_draw ? &m_vram_dirty_draw_rect : &m_vram_dirty_write_rect;
  }
  else
  {
    const bool use_write =
      (m_vram_dirty_write_rect.Valid() && m_vram_dirty_draw_rect.Valid() &&
       m_vram_dirty_write_rect.GetDistance(dst_bounds) < m_vram_dirty_draw_rect.GetDistance(dst_bounds));
    update_rect = use_write ? &m_vram_dirty_write_rect : &m_vram_dirty_draw_rect;
  }
  IncludeVRAMDirtyRectangle(*update_rect, dst_bounds);

  if (params.check_mask_before_draw)
  {
    // set new vertex counter since we want this to take into consideration previous masked pixels
    m_current_depth++;
  }

  g_gpu_device->CopyTextureRegion(m_vram_texture.get(), dst_x * m_resolution_scale, dst_y * m_resolution_scale, 0, 0,
                                  src_tex, src_x * m_resolution_scale, src_y * m_resolution_scale, 0, 0,
                                  width * m_resolution_scale, height * m_resolution_scale);
  if (src_tex != m_vram_texture.get())
    m_vram_read_texture->MakeReadyForSampling();
}

void GPUHWBackend::PrepareDraw(const GPUBackendDrawCommand* cmd, u32 num_vertices)
{
  GPUTextureMode texture_mode;
  if (cmd->rc.IsTexturingEnabled())
  {
    const u32 texpage_bits =
      (ZeroExtend32(cmd->palette.bits) << 16) | ZeroExtend32(cmd->draw_mode.bits & GPUDrawModeReg::TEXTURE_PAGE_MASK);

    // texture page changed - check that the new page doesn't intersect the drawing area
    if (m_texpage_bits != texpage_bits)
    {
      m_texpage_bits = texpage_bits;

#if 0
      if (m_vram_dirty_rect.Valid())
      {
        GL_INS_FMT("VRAM DIRTY: {},{} => {},{}", m_vram_dirty_rect.left, m_vram_dirty_rect.top, m_vram_dirty_rect.right,
                   m_vram_dirty_rect.bottom);

        auto tpr = m_draw_mode.mode_reg.GetTexturePageRectangle();
        GL_INS_FMT("PAGE RECT: {},{} => {},{}", tpr.left, tpr.top, tpr.right, tpr.bottom);
        if (m_draw_mode.mode_reg.IsUsingPalette())
        {
          tpr = m_draw_mode.GetTexturePaletteRectangle();
          GL_INS_FMT("PALETTE RECT: {},{} => {},{}", tpr.left, tpr.top, tpr.right, tpr.bottom);
        }
      }
#endif

      if (cmd->draw_mode.IsUsingPalette())
      {
        const Common::Rectangle<u32> palette_rect =
          cmd->palette.GetRectangle(cmd->draw_mode.texture_mode);
        const bool update_drawn = palette_rect.Intersects(m_vram_dirty_draw_rect);
        const bool update_written = palette_rect.Intersects(m_vram_dirty_write_rect);
        if (update_drawn || update_written)
        {
          GL_INS("Palette in VRAM dirty area, flushing cache");
          if (!IsFlushed())
            FlushRender();

          UpdateVRAMReadTexture(update_drawn, update_written);
        }
      }

      const Common::Rectangle<u32> page_rect = cmd->draw_mode.GetTexturePageRectangle();
      u8 new_texpage_dirty = m_vram_dirty_draw_rect.Intersects(page_rect) ? TEXPAGE_DIRTY_DRAWN_RECT : 0;
      new_texpage_dirty |= m_vram_dirty_write_rect.Intersects(page_rect) ? TEXPAGE_DIRTY_WRITTEN_RECT : 0;

      if (new_texpage_dirty != 0)
      {
        GL_INS("Texpage is in dirty area, checking UV ranges");
        m_texpage_dirty = new_texpage_dirty;
        m_compute_uv_range = true;
        m_current_uv_range.SetInvalid();
      }
      else
      {
        m_compute_uv_range = m_clamp_uvs;
        if (m_texpage_dirty)
          GL_INS("Texpage is no longer dirty");
        m_texpage_dirty = 0;
      }
    }

    texture_mode = cmd->draw_mode.texture_mode;
    if (cmd->rc.raw_texture_enable)
    {
      texture_mode =
        static_cast<GPUTextureMode>(static_cast<u8>(texture_mode) | static_cast<u8>(GPUTextureMode::RawTextureBit));
    }
  }
  else
  {
    texture_mode = GPUTextureMode::Disabled;
  }

  // has any state changed which requires a new batch?
  // Reverse blending breaks with mixed transparent and opaque pixels, so we have to do one draw per polygon.
  // If we have fbfetch, we don't need to draw it in two passes. Test case: Suikoden 2 shadows.
  // TODO: make this suck less.. somehow. probably arrange the relevant bits in a comparable pattern
  const GPUTransparencyMode transparency_mode =
    cmd->rc.transparency_enable ? cmd->draw_mode.transparency_mode : GPUTransparencyMode::Disabled;
  const bool dithering_enable = (!m_true_color && cmd->rc.IsDitheringEnabled()) ? cmd->draw_mode.dither_enable : false;
  if (texture_mode != m_batch.texture_mode || transparency_mode != m_batch.transparency_mode ||
      (transparency_mode == GPUTransparencyMode::BackgroundMinusForeground && !m_supports_framebuffer_fetch) ||
      dithering_enable != m_batch.dithering || m_batch.check_mask_before_draw != cmd->params.check_mask_before_draw ||
      m_batch.set_mask_while_drawing != cmd->params.set_mask_while_drawing ||
      m_batch.interlacing != cmd->params.interlaced_rendering || cmd->window.bits != m_last_texture_window_bits ||
      m_drawing_area_changed)
  {
    FlushRender();
  }

  EnsureVertexBufferSpace(num_vertices);

  if (GetBatchVertexCount() == 0)
  {
    // transparency mode change
    if (transparency_mode != GPUTransparencyMode::Disabled &&
        (texture_mode == GPUTextureMode::Disabled || !NeedsShaderBlending(transparency_mode)))
    {
      static constexpr float transparent_alpha[4][2] = {{0.5f, 0.5f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {0.25f, 1.0f}};

      const float src_alpha_factor = transparent_alpha[static_cast<u32>(transparency_mode)][0];
      const float dst_alpha_factor = transparent_alpha[static_cast<u32>(transparency_mode)][1];
      m_batch_ubo_dirty |= (m_batch_ubo_data.u_src_alpha_factor != src_alpha_factor ||
                            m_batch_ubo_data.u_dst_alpha_factor != dst_alpha_factor);
      m_batch_ubo_data.u_src_alpha_factor = src_alpha_factor;
      m_batch_ubo_data.u_dst_alpha_factor = dst_alpha_factor;
    }

    const bool check_mask_before_draw = cmd->params.check_mask_before_draw;
    const bool set_mask_while_drawing = cmd->params.set_mask_while_drawing;
    if (m_batch.check_mask_before_draw != check_mask_before_draw ||
        m_batch.set_mask_while_drawing != set_mask_while_drawing)
    {
      m_batch.check_mask_before_draw = check_mask_before_draw;
      m_batch.set_mask_while_drawing = set_mask_while_drawing;
      m_batch_ubo_dirty |= (m_batch_ubo_data.u_set_mask_while_drawing != BoolToUInt32(set_mask_while_drawing));
      m_batch_ubo_data.u_set_mask_while_drawing = BoolToUInt32(set_mask_while_drawing);
    }

    m_batch.interlacing = cmd->params.interlaced_rendering;
    if (m_batch.interlacing)
    {
      const u32 displayed_field = cmd->params.active_line_lsb;
      m_batch_ubo_dirty |= (m_batch_ubo_data.u_interlaced_displayed_field != displayed_field);
      m_batch_ubo_data.u_interlaced_displayed_field = displayed_field;
    }

    // update state
    m_batch.texture_mode = texture_mode;
    m_batch.transparency_mode = transparency_mode;
    m_batch.dithering = dithering_enable;

    if (cmd->window.bits != m_last_texture_window_bits)
    {
      m_last_texture_window_bits = cmd->window.bits;

      // TODO: This can be bitmath instead
      m_batch_ubo_data.u_texture_window_and[0] = ZeroExtend32(cmd->window.and_x);
      m_batch_ubo_data.u_texture_window_and[1] = ZeroExtend32(cmd->window.and_y);
      m_batch_ubo_data.u_texture_window_or[0] = ZeroExtend32(cmd->window.or_x);
      m_batch_ubo_data.u_texture_window_or[1] = ZeroExtend32(cmd->window.or_y);
      m_batch_ubo_dirty = true;
    }

    if (m_drawing_area_changed)
    {
      m_drawing_area_changed = false;
      SetScissor();

      if (m_pgxp_depth_buffer && m_last_depth_z < 1.0f)
        ClearDepthBuffer();

      //       if (m_sw_renderer)
      //       {
      //         GPUBackendSetDrawingAreaCommand* cmd = m_sw_renderer->NewSetDrawingAreaCommand();
      //         cmd->new_area = m_drawing_area;
      //         m_sw_renderer->PushCommand(cmd);
      //       }
    }
  }

  if (cmd->params.check_mask_before_draw)
    m_current_depth++;
}

void GPUHWBackend::FlushRender()
{
  if (!m_batch_current_vertex_ptr)
    return;

  const u32 vertex_count = GetBatchVertexCount();
  UnmapBatchVertexPointer(vertex_count);

  if (vertex_count == 0)
    return;

#ifdef _DEBUG
  GL_SCOPE_FMT("Hardware Draw {}", ++s_draw_number);
#endif

  GL_INS_FMT("Dirty draw area: {},{} => {},{} ({}x{})", m_vram_dirty_draw_rect.left, m_vram_dirty_draw_rect.top,
             m_vram_dirty_draw_rect.right, m_vram_dirty_draw_rect.bottom, m_vram_dirty_draw_rect.GetWidth(),
             m_vram_dirty_draw_rect.GetHeight());

  if (m_batch_ubo_dirty)
  {
    g_gpu_device->UploadUniformBuffer(&m_batch_ubo_data, sizeof(m_batch_ubo_data));
    m_renderer_stats.num_uniform_buffer_updates++;
    m_batch_ubo_dirty = false;
  }

  if (m_wireframe_mode != GPUWireframeMode::OnlyWireframe)
  {
    if (NeedsTwoPassRendering())
    {
      m_renderer_stats.num_batches += 2;
      DrawBatchVertices(BatchRenderMode::OnlyOpaque, vertex_count, m_batch_base_vertex);
      DrawBatchVertices(BatchRenderMode::OnlyTransparent, vertex_count, m_batch_base_vertex);
    }
    else
    {
      m_renderer_stats.num_batches++;
      DrawBatchVertices(m_batch.GetRenderMode(), vertex_count, m_batch_base_vertex);
    }
  }

  if (m_wireframe_mode != GPUWireframeMode::Disabled)
  {
    m_renderer_stats.num_batches++;
    g_gpu_device->SetPipeline(m_wireframe_pipeline.get());
    g_gpu_device->Draw(vertex_count, m_batch_base_vertex);
  }
}

void GPUHWBackend::DrawingAreaChanged()
{
  m_drawing_area_changed = true;
}

void GPUHWBackend::UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd)
{
  FlushRender();

  if (g_gpu_settings.debugging.show_vram)
  {
    if (IsUsingMultisampling())
    {
      UpdateVRAMReadTexture(true, true);
      SetDisplayTexture(m_vram_read_texture.get(), 0, 0, m_vram_read_texture->GetWidth(),
                        m_vram_read_texture->GetHeight());
    }
    else
    {
      SetDisplayTexture(m_vram_texture.get(), 0, 0, m_vram_texture->GetWidth(), m_vram_texture->GetHeight());
    }

    SetDisplayParameters(VRAM_WIDTH, VRAM_HEIGHT, 0, 0, VRAM_WIDTH, VRAM_HEIGHT,
                         static_cast<float>(VRAM_WIDTH) / static_cast<float>(VRAM_HEIGHT));
  }
  else
  {
    // TODO: use a dynamically sized texture
    SetDisplayParameters(cmd->display_width, cmd->display_height, cmd->display_origin_left, cmd->display_origin_top,
                         cmd->display_vram_width, cmd->display_vram_height, cmd->display_aspect_ratio);

    const u32 resolution_scale = cmd->display_24bit ? 1 : m_resolution_scale;
    const u32 vram_offset_x = cmd->display_vram_left;
    const u32 vram_offset_y = cmd->display_vram_top;
    const u32 scaled_vram_offset_x = vram_offset_x * resolution_scale;
    const u32 scaled_vram_offset_y = vram_offset_y * resolution_scale;
    const u32 display_width = cmd->display_vram_width;
    const u32 display_height = cmd->display_vram_height;
    const u32 scaled_display_width = display_width * resolution_scale;
    const u32 scaled_display_height = display_height * resolution_scale;
    const InterlacedRenderMode interlaced = GetInterlacedRenderMode(cmd);

    if (cmd->display_disabled)
    {
      ClearDisplayTexture();
    }
    else if (!cmd->display_24bit && interlaced == InterlacedRenderMode::None && !IsUsingMultisampling() &&
             (scaled_vram_offset_x + scaled_display_width) <= m_vram_texture->GetWidth() &&
             (scaled_vram_offset_y + scaled_display_height) <= m_vram_texture->GetHeight())
    {

      if (IsUsingDownsampling(cmd))
      {
        DownsampleFramebuffer(m_vram_texture.get(), scaled_vram_offset_x, scaled_vram_offset_y, scaled_display_width,
                              scaled_display_height);
      }
      else
      {
        SetDisplayTexture(m_vram_texture.get(), scaled_vram_offset_x, scaled_vram_offset_y, scaled_display_width,
                          scaled_display_height);
      }
    }
    else
    {
      if (!m_display_private_texture || m_display_private_texture->GetWidth() != scaled_display_width ||
          m_display_private_texture->GetHeight() != scaled_display_height)
      {
        g_gpu_device->RecycleTexture(std::move(m_display_private_texture));
        if (!(m_display_private_texture = g_gpu_device->FetchTexture(
                scaled_display_width, scaled_display_height, 1, 1, 1, GPUTexture::Type::RenderTarget, VRAM_RT_FORMAT)))
        {
          Log_ErrorFmt("Failed to create {}x{} display texture", scaled_display_width, scaled_display_height);
          ClearDisplayTexture();
          return;
        }

        GL_OBJECT_NAME(m_display_private_texture, "Display Texture");
      }

      // TODO: discard vs load for interlaced
      if (interlaced == InterlacedRenderMode::None)
        g_gpu_device->InvalidateRenderTarget(m_display_private_texture.get());

      g_gpu_device->SetRenderTarget(m_display_private_texture.get());
      g_gpu_device->SetPipeline(
        m_display_pipelines[BoolToUInt8(cmd->display_24bit)][static_cast<u8>(interlaced)].get());
      g_gpu_device->SetTextureSampler(0, m_vram_texture.get(), g_gpu_device->GetNearestSampler());

      const u32 reinterpret_field_offset =
        (interlaced != InterlacedRenderMode::None) ? cmd->interlaced_display_field : 0;
      const u32 reinterpret_start_x = cmd->X * resolution_scale;
      const u32 reinterpret_crop_left = (cmd->display_vram_left - cmd->X) * resolution_scale;
      const u32 uniforms[4] = {reinterpret_start_x, scaled_vram_offset_y + reinterpret_field_offset,
                               reinterpret_crop_left, reinterpret_field_offset};
      g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));

      g_gpu_device->SetViewportAndScissor(0, 0, scaled_display_width, scaled_display_height);
      g_gpu_device->Draw(3, 0);

      if (IsUsingDownsampling(cmd))
        DownsampleFramebuffer(m_display_private_texture.get(), 0, 0, scaled_display_width, scaled_display_height);
      else
        SetDisplayTexture(m_display_private_texture.get(), 0, 0, scaled_display_width, scaled_display_height);

      RestoreDeviceContext();
    }
  }
}

void GPUHWBackend::DownsampleFramebuffer(GPUTexture* source, u32 left, u32 top, u32 width, u32 height)
{
  if (m_downsample_mode == GPUDownsampleMode::Adaptive)
    DownsampleFramebufferAdaptive(source, left, top, width, height);
  else
    DownsampleFramebufferBoxFilter(source, left, top, width, height);
}

void GPUHWBackend::DownsampleFramebufferAdaptive(GPUTexture* source, u32 left, u32 top, u32 width, u32 height)
{
  GL_PUSH_FMT("DownsampleFramebufferAdaptive ({},{} => {},{})", left, top, left + width, left + height);

  struct SmoothingUBOData
  {
    float min_uv[2];
    float max_uv[2];
    float rcp_size[2];
    float lod;
  };

  if (!m_downsample_texture || m_downsample_texture->GetWidth() != width || m_downsample_texture->GetHeight() != height)
  {
    g_gpu_device->RecycleTexture(std::move(m_downsample_texture));
    m_downsample_texture =
      g_gpu_device->FetchTexture(width, height, 1, 1, 1, GPUTexture::Type::RenderTarget, VRAM_RT_FORMAT);
  }
  std::unique_ptr<GPUTexture, GPUDevice::PooledTextureDeleter> level_texture = g_gpu_device->FetchAutoRecycleTexture(
    width, height, 1, m_downsample_scale_or_levels, 1, GPUTexture::Type::Texture, VRAM_RT_FORMAT);
  std::unique_ptr<GPUTexture, GPUDevice::PooledTextureDeleter> weight_texture =
    g_gpu_device->FetchAutoRecycleTexture(std::max(width >> (m_downsample_scale_or_levels - 1), 1u),
                                          std::max(height >> (m_downsample_scale_or_levels - 1), 1u), 1, 1, 1,
                                          GPUTexture::Type::RenderTarget, GPUTexture::Format::R8);
  if (!m_downsample_texture || !level_texture || !weight_texture)
  {
    Log_ErrorFmt("Failed to create {}x{} RTs for adaptive downsampling", width, height);
    SetDisplayTexture(source, left, top, width, height);
    return;
  }

  g_gpu_device->CopyTextureRegion(level_texture.get(), 0, 0, 0, 0, source, left, top, 0, 0, width, height);
  g_gpu_device->SetTextureSampler(0, level_texture.get(), m_downsample_lod_sampler.get());

  SmoothingUBOData uniforms;

  // create mip chain
  for (u32 level = 1; level < m_downsample_scale_or_levels; level++)
  {
    GL_SCOPE_FMT("Create miplevel {}", level);

    const u32 level_width = width >> level;
    const u32 level_height = height >> level;
    const float rcp_width = 1.0f / static_cast<float>(level_texture->GetMipWidth(level));
    const float rcp_height = 1.0f / static_cast<float>(level_texture->GetMipHeight(level));
    uniforms.min_uv[0] = 0.0f;
    uniforms.min_uv[1] = 0.0f;
    uniforms.max_uv[0] = static_cast<float>(level_width) * rcp_width;
    uniforms.max_uv[1] = static_cast<float>(level_height) * rcp_height;
    uniforms.rcp_size[0] = rcp_width;
    uniforms.rcp_size[1] = rcp_height;
    uniforms.lod = static_cast<float>(level - 1);

    g_gpu_device->InvalidateRenderTarget(m_downsample_texture.get());
    g_gpu_device->SetRenderTarget(m_downsample_texture.get());
    g_gpu_device->SetViewportAndScissor(0, 0, level_width, level_height);
    g_gpu_device->SetPipeline((level == 1) ? m_downsample_first_pass_pipeline.get() :
                                             m_downsample_mid_pass_pipeline.get());
    g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
    g_gpu_device->Draw(3, 0);
    g_gpu_device->CopyTextureRegion(level_texture.get(), 0, 0, 0, level, m_downsample_texture.get(), 0, 0, 0, 0,
                                    level_width, level_height);
  }

  // blur pass at lowest level
  {
    GL_SCOPE("Blur");

    const u32 last_level = m_downsample_scale_or_levels - 1;
    const u32 last_width = level_texture->GetMipWidth(last_level);
    const u32 last_height = level_texture->GetMipHeight(last_level);
    const float rcp_width = 1.0f / static_cast<float>(m_downsample_texture->GetWidth());
    const float rcp_height = 1.0f / static_cast<float>(m_downsample_texture->GetHeight());
    uniforms.min_uv[0] = 0.0f;
    uniforms.min_uv[1] = 0.0f;
    uniforms.max_uv[0] = static_cast<float>(last_width) * rcp_width;
    uniforms.max_uv[1] = static_cast<float>(last_height) * rcp_height;
    uniforms.rcp_size[0] = rcp_width;
    uniforms.rcp_size[1] = rcp_height;
    uniforms.lod = 0.0f;

    m_downsample_texture->MakeReadyForSampling();
    g_gpu_device->InvalidateRenderTarget(weight_texture.get());
    g_gpu_device->SetRenderTarget(weight_texture.get());
    g_gpu_device->SetTextureSampler(0, m_downsample_texture.get(), g_gpu_device->GetNearestSampler());
    g_gpu_device->SetViewportAndScissor(0, 0, last_width, last_height);
    g_gpu_device->SetPipeline(m_downsample_blur_pass_pipeline.get());
    g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
    g_gpu_device->Draw(3, 0);
    weight_texture->MakeReadyForSampling();
  }

  // composite downsampled and upsampled images together
  {
    GL_SCOPE("Composite");

    uniforms.min_uv[0] = 0.0f;
    uniforms.min_uv[1] = 0.0f;
    uniforms.max_uv[0] = 1.0f;
    uniforms.max_uv[1] = 1.0f;

    g_gpu_device->InvalidateRenderTarget(m_downsample_texture.get());
    g_gpu_device->SetRenderTarget(m_downsample_texture.get());
    g_gpu_device->SetTextureSampler(0, level_texture.get(), m_downsample_composite_sampler.get());
    g_gpu_device->SetTextureSampler(1, weight_texture.get(), m_downsample_lod_sampler.get());
    g_gpu_device->SetViewportAndScissor(0, 0, width, height);
    g_gpu_device->SetPipeline(m_downsample_composite_pass_pipeline.get());
    g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));
    g_gpu_device->Draw(3, 0);
    m_downsample_texture->MakeReadyForSampling();
  }

  GL_POP();

  RestoreDeviceContext();

  SetDisplayTexture(m_downsample_texture.get(), 0, 0, width, height);
}

void GPUHWBackend::DownsampleFramebufferBoxFilter(GPUTexture* source, u32 left, u32 top, u32 width, u32 height)
{
  GL_SCOPE_FMT("DownsampleFramebufferBoxFilter({},{} => {},{} ({}x{})", left, top, left + width, top + height, width,
               height);

  const u32 ds_width = width / m_downsample_scale_or_levels;
  const u32 ds_height = height / m_downsample_scale_or_levels;

  if (!m_downsample_texture || m_downsample_texture->GetWidth() != ds_width ||
      m_downsample_texture->GetHeight() != ds_height)
  {
    g_gpu_device->RecycleTexture(std::move(m_downsample_texture));
    m_downsample_texture =
      g_gpu_device->FetchTexture(ds_width, ds_height, 1, 1, 1, GPUTexture::Type::RenderTarget, VRAM_RT_FORMAT);
  }
  if (!m_downsample_texture)
  {
    Log_ErrorFmt("Failed to create {}x{} RT for box downsampling", width, height);
    SetDisplayTexture(source, left, top, width, height);
    return;
  }

  source->MakeReadyForSampling();

  const u32 uniforms[4] = {left, top, 0u, 0u};
  g_gpu_device->PushUniformBuffer(uniforms, sizeof(uniforms));

  g_gpu_device->InvalidateRenderTarget(m_downsample_texture.get());
  g_gpu_device->SetRenderTarget(m_downsample_texture.get());
  g_gpu_device->SetPipeline(m_downsample_first_pass_pipeline.get());
  g_gpu_device->SetTextureSampler(0, source, g_gpu_device->GetNearestSampler());
  g_gpu_device->SetViewportAndScissor(0, 0, ds_width, ds_height);
  g_gpu_device->Draw(3, 0);

  RestoreDeviceContext();

  SetDisplayTexture(m_downsample_texture.get(), 0, 0, ds_width, ds_height);
}

#if 0
void GPU_HW::DrawRendererStats(bool is_idle_frame)
{
  if (!is_idle_frame)
  {
    m_last_renderer_stats = m_renderer_stats;
    m_renderer_stats = {};
  }

  if (ImGui::CollapsingHeader("Renderer Statistics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    static const ImVec4 active_color{1.0f, 1.0f, 1.0f, 1.0f};
    static const ImVec4 inactive_color{0.4f, 0.4f, 0.4f, 1.0f};
    const auto& stats = m_last_renderer_stats;

    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, 200.0f * Host::GetOSDScale());

    ImGui::TextUnformatted("Resolution Scale:");
    ImGui::NextColumn();
    ImGui::Text("%u (VRAM %ux%u)", m_resolution_scale, VRAM_WIDTH * m_resolution_scale,
                VRAM_HEIGHT * m_resolution_scale);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Effective Display Resolution:");
    ImGui::NextColumn();
    ImGui::Text("%ux%u", m_crtc_state.display_vram_width * m_resolution_scale,
                m_crtc_state.display_vram_height * m_resolution_scale);
    ImGui::NextColumn();

    ImGui::TextUnformatted("True Color:");
    ImGui::NextColumn();
    ImGui::TextColored(m_true_color ? active_color : inactive_color, m_true_color ? "Enabled" : "Disabled");
    ImGui::NextColumn();

    ImGui::TextUnformatted("Scaled Dithering:");
    ImGui::NextColumn();
    ImGui::TextColored(m_scaled_dithering ? active_color : inactive_color, m_scaled_dithering ? "Enabled" : "Disabled");
    ImGui::NextColumn();

    ImGui::TextUnformatted("Texture Filtering:");
    ImGui::NextColumn();
    ImGui::TextColored((m_texture_filtering != GPUTextureFilter::Nearest) ? active_color : inactive_color, "%s",
                       Settings::GetTextureFilterDisplayName(m_texture_filtering));
    ImGui::NextColumn();

    ImGui::TextUnformatted("PGXP:");
    ImGui::NextColumn();
    ImGui::TextColored(g_gpu_settings.gpu_pgxp_enable ? active_color : inactive_color, "Geom");
    ImGui::SameLine();
    ImGui::TextColored((g_gpu_settings.gpu_pgxp_enable && g_gpu_settings.gpu_pgxp_culling) ? active_color : inactive_color,
                       "Cull");
    ImGui::SameLine();
    ImGui::TextColored(
      (g_gpu_settings.gpu_pgxp_enable && g_gpu_settings.gpu_pgxp_texture_correction) ? active_color : inactive_color, "Tex");
    ImGui::SameLine();
    ImGui::TextColored((g_gpu_settings.gpu_pgxp_enable && g_gpu_settings.gpu_pgxp_vertex_cache) ? active_color : inactive_color,
                       "Cache");
    ImGui::NextColumn();

    ImGui::TextUnformatted("Batches Drawn:");
    ImGui::NextColumn();
    ImGui::Text("%u", stats.num_batches);
    ImGui::NextColumn();

    ImGui::TextUnformatted("VRAM Read Texture Updates:");
    ImGui::NextColumn();
    ImGui::Text("%u", stats.num_vram_read_texture_updates);
    ImGui::NextColumn();

    ImGui::TextUnformatted("Uniform Buffer Updates: ");
    ImGui::NextColumn();
    ImGui::Text("%u", stats.num_uniform_buffer_updates);
    ImGui::NextColumn();

    ImGui::Columns(1);
  }
}
#endif

std::unique_ptr<GPUBackend> GPUBackend::CreateHardwareBackend()
{
  std::unique_ptr<GPUHWBackend> gpu(std::make_unique<GPUHWBackend>());
  if (!gpu->Initialize())
    return nullptr;

  return gpu;
}
