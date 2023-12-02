// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_backend.h"
#include "gpu.h"
#include "gpu_shadergen.h"
#include "gpu_thread.h"
#include "host.h"
#include "settings.h"

#include "util/gpu_device.h"
#include "util/postprocessing.h"
#include "util/state_wrapper.h"

#include "common/align.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/string_util.h"
#include "common/timer.h"

#include "stb_image_resize.h"
#include "stb_image_write.h"

#include <thread>

Log_SetChannel(GPUBackend);

static std::atomic<u32> s_queued_frames;
static std::atomic_bool s_waiting_for_gpu_thread;
static Threading::KernelSemaphore s_gpu_thread_wait;

static std::tuple<u32, u32> s_last_display_source_size;

GPUBackend::GPUBackend() = default;

GPUBackend::~GPUBackend() = default;

bool GPUBackend::Initialize()
{
  if (!CompileDisplayPipeline())
  {
    Host::ReportErrorAsync("Error", "Failed to compile base GPU pipelines.");
    return false;
  }

  return true;
}

void GPUBackend::Reset(bool clear_vram)
{
  DrawingAreaChanged(Common::Rectangle<u32>(0, 0, 0, 0));
}

bool GPUBackend::DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display)
{
  Panic("TODO FIXME");
}

void GPUBackend::UpdateSettings(const Settings& old_settings)
{
  if (g_gpu_settings.display_scaling != old_settings.display_scaling)
  {
    if (!CompileDisplayPipeline())
      Panic("Failed to compile display pipeline on settings change.");
  }
}

void GPUBackend::Shutdown()
{
}

void GPUBackend::RestoreDeviceContext()
{
}

GPUBackendResetCommand* GPUBackend::NewResetCommand()
{
  return static_cast<GPUBackendResetCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::Reset, sizeof(GPUBackendResetCommand)));
}

GPUThreadCommand* GPUBackend::NewClearDisplayCommand()
{
  return static_cast<GPUThreadCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ClearDisplay, sizeof(GPUThreadCommand)));
}

GPUBackendUpdateDisplayCommand* GPUBackend::NewUpdateDisplayCommand()
{
  return static_cast<GPUBackendUpdateDisplayCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::UpdateDisplay, sizeof(GPUBackendUpdateDisplayCommand)));
}

GPUBackendReadVRAMCommand* GPUBackend::NewReadVRAMCommand()
{
  return static_cast<GPUBackendReadVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::ReadVRAM, sizeof(GPUBackendReadVRAMCommand)));
}

GPUBackendFillVRAMCommand* GPUBackend::NewFillVRAMCommand()
{
  return static_cast<GPUBackendFillVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::FillVRAM, sizeof(GPUBackendFillVRAMCommand)));
}

GPUBackendUpdateVRAMCommand* GPUBackend::NewUpdateVRAMCommand(u32 num_words)
{
  const u32 size = sizeof(GPUBackendUpdateVRAMCommand) + (num_words * sizeof(u16));
  GPUBackendUpdateVRAMCommand* cmd =
    static_cast<GPUBackendUpdateVRAMCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::UpdateVRAM, size));
  return cmd;
}

GPUBackendCopyVRAMCommand* GPUBackend::NewCopyVRAMCommand()
{
  return static_cast<GPUBackendCopyVRAMCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::CopyVRAM, sizeof(GPUBackendCopyVRAMCommand)));
}

GPUBackendSetDrawingAreaCommand* GPUBackend::NewSetDrawingAreaCommand()
{
  return static_cast<GPUBackendSetDrawingAreaCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::SetDrawingArea, sizeof(GPUBackendSetDrawingAreaCommand)));
}

GPUBackendDrawPolygonCommand* GPUBackend::NewDrawPolygonCommand(u32 num_vertices)
{
  const u32 size = sizeof(GPUBackendDrawPolygonCommand) + (num_vertices * sizeof(GPUBackendDrawPolygonCommand::Vertex));
  GPUBackendDrawPolygonCommand* cmd =
    static_cast<GPUBackendDrawPolygonCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::DrawPolygon, size));
  cmd->num_vertices = Truncate8(num_vertices);
  return cmd;
}

GPUBackendDrawPrecisePolygonCommand* GPUBackend::NewDrawPrecisePolygonCommand(u32 num_vertices)
{
  const u32 size =
    sizeof(GPUBackendDrawPrecisePolygonCommand) + (num_vertices * sizeof(GPUBackendDrawPrecisePolygonCommand::Vertex));
  GPUBackendDrawPrecisePolygonCommand* cmd = static_cast<GPUBackendDrawPrecisePolygonCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::DrawPrecisePolygon, size));
  cmd->num_vertices = Truncate8(num_vertices);
  return cmd;
}

GPUBackendDrawSpriteCommand* GPUBackend::NewDrawRectangleCommand()
{
  return static_cast<GPUBackendDrawSpriteCommand*>(
    GPUThread::AllocateCommand(GPUBackendCommandType::DrawRectangle, sizeof(GPUBackendDrawSpriteCommand)));
}

GPUBackendDrawLineCommand* GPUBackend::NewDrawLineCommand(u32 num_vertices)
{
  const u32 size = sizeof(GPUBackendDrawLineCommand) + (num_vertices * sizeof(GPUBackendDrawLineCommand::Vertex));
  GPUBackendDrawLineCommand* cmd =
    static_cast<GPUBackendDrawLineCommand*>(GPUThread::AllocateCommand(GPUBackendCommandType::DrawLine, size));
  cmd->num_vertices = Truncate16(num_vertices);
  return cmd;
}

void GPUBackend::PushCommand(GPUThreadCommand* cmd)
{
  GPUThread::PushCommand(cmd);
}

void GPUBackend::PushCommandAndWakeThread(GPUThreadCommand* cmd)
{
  GPUThread::PushCommandAndWakeThread(cmd);
}

void GPUBackend::PushCommandAndSync(GPUThreadCommand* cmd, bool spin)
{
  GPUThread::PushCommandAndSync(cmd, spin);
}

bool GPUBackend::IsUsingHardwareBackend()
{
  return (GPUThread::GetRequestedRenderer().value_or(GPURenderer::Software) != GPURenderer::Software);
}

bool GPUBackend::BeginQueueFrame()
{
  const u32 queued_frames = s_queued_frames.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (queued_frames < g_settings.gpu_max_queued_frames)
    return false;

  Log_DevFmt("<-- {} queued frames, {} max, blocking CPU thread", queued_frames, g_settings.gpu_max_queued_frames);
  s_waiting_for_gpu_thread.store(true, std::memory_order_release);
  return true;
}

void GPUBackend::WaitForOneQueuedFrame()
{
  s_gpu_thread_wait.Wait();
}

bool GPUBackend::RenderScreenshotToBuffer(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect, bool postfx,
                                          std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format)
{
  bool result;

  GPUThreadRenderScreenshotToBufferCommand* cmd =
    static_cast<GPUThreadRenderScreenshotToBufferCommand*>(GPUThread::AllocateCommand(
      GPUBackendCommandType::RenderScreenshotToBuffer, sizeof(GPUThreadRenderScreenshotToBufferCommand)));
  cmd->width = width;
  cmd->height = height;
  cmd->draw_rect = draw_rect;
  cmd->postfx = postfx;
  cmd->out_pixels = out_pixels;
  cmd->out_stride = out_stride;
  cmd->out_format = out_format;
  cmd->out_result = &result;
  PushCommandAndSync(cmd, false);

  return result;
}

std::tuple<u32, u32> GPUBackend::GetLastDisplaySourceSize()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  return s_last_display_source_size;
}

void GPUBackend::HandleCommand(const GPUThreadCommand* cmd)
{
  switch (cmd->type)
  {
    case GPUBackendCommandType::Reset:
    {
      const GPUBackendResetCommand* ccmd = static_cast<const GPUBackendResetCommand*>(cmd);
      Reset(ccmd->clear_vram);
    }
    break;

    case GPUBackendCommandType::ClearDisplay:
    {
      ClearDisplay();
    }
    break;

    case GPUBackendCommandType::UpdateDisplay:
    {
      const GPUBackendUpdateDisplayCommand* ccmd = static_cast<const GPUBackendUpdateDisplayCommand*>(cmd);
      UpdateDisplay(ccmd);

      if (ccmd->present_frame)
      {
        GPUThread::Internal::PresentFrame(true, ccmd->present_time);

        s_queued_frames.fetch_sub(1);

        bool expected = true;
        if (s_waiting_for_gpu_thread.compare_exchange_strong(expected, false))
        {
          Log_DevFmt("--> Unblocking CPU thread");
          s_gpu_thread_wait.Post();
        }
      }
    }
    break;

    case GPUBackendCommandType::RenderScreenshotToBuffer:
    {
      HandleRenderScreenshotToBuffer(static_cast<const GPUThreadRenderScreenshotToBufferCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::ReadVRAM:
    {
      const GPUBackendReadVRAMCommand* ccmd = static_cast<const GPUBackendReadVRAMCommand*>(cmd);
      ReadVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height));
    }
    break;

    case GPUBackendCommandType::FillVRAM:
    {
      const GPUBackendFillVRAMCommand* ccmd = static_cast<const GPUBackendFillVRAMCommand*>(cmd);
      FillVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height),
               ccmd->color, ccmd->params);
    }
    break;

    case GPUBackendCommandType::UpdateVRAM:
    {
      const GPUBackendUpdateVRAMCommand* ccmd = static_cast<const GPUBackendUpdateVRAMCommand*>(cmd);
      UpdateVRAM(ZeroExtend32(ccmd->x), ZeroExtend32(ccmd->y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height),
                 ccmd->data, ccmd->params);
    }
    break;

    case GPUBackendCommandType::CopyVRAM:
    {
      const GPUBackendCopyVRAMCommand* ccmd = static_cast<const GPUBackendCopyVRAMCommand*>(cmd);
      CopyVRAM(ZeroExtend32(ccmd->src_x), ZeroExtend32(ccmd->src_y), ZeroExtend32(ccmd->dst_x),
               ZeroExtend32(ccmd->dst_y), ZeroExtend32(ccmd->width), ZeroExtend32(ccmd->height), ccmd->params);
    }
    break;

    case GPUBackendCommandType::SetDrawingArea:
    {
      FlushRender();
      DrawingAreaChanged(static_cast<const GPUBackendSetDrawingAreaCommand*>(cmd)->new_area);
    }
    break;

    case GPUBackendCommandType::DrawPolygon:
    {
      DrawPolygon(static_cast<const GPUBackendDrawPolygonCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::DrawPrecisePolygon:
    {
      DrawPrecisePolygon(static_cast<const GPUBackendDrawPrecisePolygonCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::DrawRectangle:
    {
      DrawSprite(static_cast<const GPUBackendDrawSpriteCommand*>(cmd));
    }
    break;

    case GPUBackendCommandType::DrawLine:
    {
      DrawLine(static_cast<const GPUBackendDrawLineCommand*>(cmd));
    }
    break;

    default:
      Panic("Unhandled command");
      break;
  }
}

void GPUBackend::FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color, GPUBackendCommandParameters params)
{
  const u16 color16 = VRAMRGBA8888ToRGBA5551(color);
  if ((x + width) <= VRAM_WIDTH && !params.interlaced_rendering)
  {
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      std::fill_n(&g_vram[row * VRAM_WIDTH + x], width, color16);
    }
  }
  else if (params.interlaced_rendering)
  {
    // Hardware tests show that fills seem to break on the first two lines when the offset matches the displayed field.
    const u32 active_field = params.active_line_lsb;
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      if ((row & u32(1)) == active_field)
        continue;

      u16* row_ptr = &g_vram[row * VRAM_WIDTH];
      for (u32 xoffs = 0; xoffs < width; xoffs++)
      {
        const u32 col = (x + xoffs) % VRAM_WIDTH;
        row_ptr[col] = color16;
      }
    }
  }
  else
  {
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      const u32 row = (y + yoffs) % VRAM_HEIGHT;
      u16* row_ptr = &g_vram[row * VRAM_WIDTH];
      for (u32 xoffs = 0; xoffs < width; xoffs++)
      {
        const u32 col = (x + xoffs) % VRAM_WIDTH;
        row_ptr[col] = color16;
      }
    }
  }
}

void GPUBackend::UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, GPUBackendCommandParameters params)
{
  // Fast path when the copy is not oversized.
  if ((x + width) <= VRAM_WIDTH && (y + height) <= VRAM_HEIGHT && !params.IsMaskingEnabled())
  {
    const u16* src_ptr = static_cast<const u16*>(data);
    u16* dst_ptr = &g_vram[y * VRAM_WIDTH + x];
    for (u32 yoffs = 0; yoffs < height; yoffs++)
    {
      std::copy_n(src_ptr, width, dst_ptr);
      src_ptr += width;
      dst_ptr += VRAM_WIDTH;
    }
  }
  else
  {
    // Slow path when we need to handle wrap-around.
    // During transfer/render operations, if ((dst_pixel & mask_and) == 0) { pixel = src_pixel | mask_or }
    const u16* src_ptr = static_cast<const u16*>(data);
    const u16 mask_and = params.GetMaskAND();
    const u16 mask_or = params.GetMaskOR();

    for (u32 row = 0; row < height;)
    {
      u16* dst_row_ptr = &g_vram[((y + row++) % VRAM_HEIGHT) * VRAM_WIDTH];
      for (u32 col = 0; col < width;)
      {
        // TODO: Handle unaligned reads...
        u16* pixel_ptr = &dst_row_ptr[(x + col++) % VRAM_WIDTH];
        if (((*pixel_ptr) & mask_and) == 0)
          *pixel_ptr = *(src_ptr++) | mask_or;
      }
    }
  }
}

void GPUBackend::CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height,
                          GPUBackendCommandParameters params)
{
  // Break up oversized copies. This behavior has not been verified on console.
  if ((src_x + width) > VRAM_WIDTH || (dst_x + width) > VRAM_WIDTH)
  {
    u32 remaining_rows = height;
    u32 current_src_y = src_y;
    u32 current_dst_y = dst_y;
    while (remaining_rows > 0)
    {
      const u32 rows_to_copy =
        std::min<u32>(remaining_rows, std::min<u32>(VRAM_HEIGHT - current_src_y, VRAM_HEIGHT - current_dst_y));

      u32 remaining_columns = width;
      u32 current_src_x = src_x;
      u32 current_dst_x = dst_x;
      while (remaining_columns > 0)
      {
        const u32 columns_to_copy =
          std::min<u32>(remaining_columns, std::min<u32>(VRAM_WIDTH - current_src_x, VRAM_WIDTH - current_dst_x));
        CopyVRAM(current_src_x, current_src_y, current_dst_x, current_dst_y, columns_to_copy, rows_to_copy, params);
        current_src_x = (current_src_x + columns_to_copy) % VRAM_WIDTH;
        current_dst_x = (current_dst_x + columns_to_copy) % VRAM_WIDTH;
        remaining_columns -= columns_to_copy;
      }

      current_src_y = (current_src_y + rows_to_copy) % VRAM_HEIGHT;
      current_dst_y = (current_dst_y + rows_to_copy) % VRAM_HEIGHT;
      remaining_rows -= rows_to_copy;
    }

    return;
  }

  // This doesn't have a fast path, but do we really need one? It's not common.
  const u16 mask_and = params.GetMaskAND();
  const u16 mask_or = params.GetMaskOR();

  // Copy in reverse when src_x < dst_x, this is verified on console.
  if (src_x < dst_x || ((src_x + width - 1) % VRAM_WIDTH) < ((dst_x + width - 1) % VRAM_WIDTH))
  {
    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[((src_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];
      u16* dst_row_ptr = &g_vram[((dst_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];

      for (s32 col = static_cast<s32>(width - 1); col >= 0; col--)
      {
        const u16 src_pixel = src_row_ptr[(src_x + static_cast<u32>(col)) % VRAM_WIDTH];
        u16* dst_pixel_ptr = &dst_row_ptr[(dst_x + static_cast<u32>(col)) % VRAM_WIDTH];
        if ((*dst_pixel_ptr & mask_and) == 0)
          *dst_pixel_ptr = src_pixel | mask_or;
      }
    }
  }
  else
  {
    for (u32 row = 0; row < height; row++)
    {
      const u16* src_row_ptr = &g_vram[((src_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];
      u16* dst_row_ptr = &g_vram[((dst_y + row) % VRAM_HEIGHT) * VRAM_WIDTH];

      for (u32 col = 0; col < width; col++)
      {
        const u16 src_pixel = src_row_ptr[(src_x + col) % VRAM_WIDTH];
        u16* dst_pixel_ptr = &dst_row_ptr[(dst_x + col) % VRAM_WIDTH];
        if ((*dst_pixel_ptr & mask_and) == 0)
          *dst_pixel_ptr = src_pixel | mask_or;
      }
    }
  }
}

bool GPUBackend::CompileDisplayPipeline()
{
  GPUShaderGen shadergen(g_gpu_device->GetRenderAPI(), g_gpu_device->GetFeatures().dual_source_blend,
                         g_gpu_device->GetFeatures().framebuffer_fetch);

  GPUPipeline::GraphicsConfig plconfig;
  plconfig.layout = GPUPipeline::Layout::SingleTextureAndPushConstants;
  plconfig.input_layout.vertex_stride = 0;
  plconfig.primitive = GPUPipeline::Primitive::Triangles;
  plconfig.rasterization = GPUPipeline::RasterizationState::GetNoCullState();
  plconfig.depth = GPUPipeline::DepthState::GetNoTestsState();
  plconfig.blend = GPUPipeline::BlendState::GetNoBlendingState();
  plconfig.SetTargetFormats(g_gpu_device->HasSurface() ? g_gpu_device->GetWindowFormat() : GPUTexture::Format::RGBA8);
  plconfig.depth_format = GPUTexture::Format::Unknown;
  plconfig.samples = 1;
  plconfig.per_sample_shading = false;

  std::string vs = shadergen.GenerateDisplayVertexShader();
  std::string fs;
  switch (g_gpu_settings.display_scaling)
  {
    case DisplayScalingMode::BilinearSharp:
      fs = shadergen.GenerateDisplaySharpBilinearFragmentShader();
      break;

    case DisplayScalingMode::BilinearSmooth:
      fs = shadergen.GenerateDisplayFragmentShader(true);
      break;

    case DisplayScalingMode::Nearest:
    case DisplayScalingMode::NearestInteger:
    default:
      fs = shadergen.GenerateDisplayFragmentShader(false);
      break;
  }

  std::unique_ptr<GPUShader> vso = g_gpu_device->CreateShader(GPUShaderStage::Vertex, vs);
  std::unique_ptr<GPUShader> fso = g_gpu_device->CreateShader(GPUShaderStage::Fragment, fs);
  if (!vso || !fso)
    return false;
  GL_OBJECT_NAME(vso, "Display Vertex Shader");
  GL_OBJECT_NAME_FMT(fso, "Display Fragment Shader [{}]",
                     Settings::GetDisplayScalingName(g_gpu_settings.display_scaling));

  plconfig.vertex_shader = vso.get();
  plconfig.fragment_shader = fso.get();
  plconfig.geometry_shader = nullptr;
  if (!(m_display_pipeline = g_gpu_device->CreatePipeline(plconfig)))
    return false;
  GL_OBJECT_NAME_FMT(m_display_pipeline, "Display Pipeline [{}]",
                     Settings::GetDisplayScalingName(g_gpu_settings.display_scaling));

  return true;
}

void GPUBackend::ClearDisplayTexture()
{
  m_display_texture = nullptr;
  m_display_texture_view_x = 0;
  m_display_texture_view_y = 0;
  m_display_texture_view_width = 0;
  m_display_texture_view_height = 0;
  s_last_display_source_size = {};
  std::atomic_thread_fence(std::memory_order_release);
}

void GPUBackend::SetDisplayTexture(GPUTexture* texture, s32 view_x, s32 view_y, s32 view_width, s32 view_height)
{
  DebugAssert(texture);
  m_display_texture = texture;
  m_display_texture_view_x = view_x;
  m_display_texture_view_y = view_y;
  m_display_texture_view_width = view_width;
  m_display_texture_view_height = view_height;
  s_last_display_source_size = {static_cast<u32>(view_width), static_cast<u32>(view_height)};
  std::atomic_thread_fence(std::memory_order_release);
}

void GPUBackend::SetDisplayParameters(s32 display_width, s32 display_height, s32 active_left, s32 active_top,
                                      s32 active_width, s32 active_height, float display_aspect_ratio)
{
  m_display_width = display_width;
  m_display_height = display_height;
  m_display_active_left = active_left;
  m_display_active_top = active_top;
  m_display_active_width = active_width;
  m_display_active_height = active_height;
  m_display_aspect_ratio = display_aspect_ratio;
}

bool GPUBackend::PresentDisplay()
{
  if (!HasDisplayTexture())
    return g_gpu_device->BeginPresent(false);

  const Common::Rectangle<s32> draw_rect =
    CalculateDrawRect(g_gpu_device->GetWindowWidth(), g_gpu_device->GetWindowHeight());
  return RenderDisplay(nullptr, draw_rect, true);
}

bool GPUBackend::RenderDisplay(GPUTexture* target, const Common::Rectangle<s32>& draw_rect, bool postfx)
{
  GL_SCOPE_FMT("RenderDisplay: {}x{} at {},{}", draw_rect.left, draw_rect.top, draw_rect.GetWidth(),
               draw_rect.GetHeight());

  if (m_display_texture)
    m_display_texture->MakeReadyForSampling();

  bool texture_filter_linear = false;

  struct Uniforms
  {
    float src_rect[4];
    float src_size[4];
    float clamp_rect[4];
    float params[4];
  } uniforms;
  std::memset(uniforms.params, 0, sizeof(uniforms.params));

  switch (g_gpu_settings.display_scaling)
  {
    case DisplayScalingMode::Nearest:
    case DisplayScalingMode::NearestInteger:
      break;

    case DisplayScalingMode::BilinearSmooth:
      texture_filter_linear = true;
      break;

    case DisplayScalingMode::BilinearSharp:
    {
      texture_filter_linear = true;
      uniforms.params[0] = std::max(
        std::floor(static_cast<float>(draw_rect.GetWidth()) / static_cast<float>(m_display_texture_view_width)), 1.0f);
      uniforms.params[1] = std::max(
        std::floor(static_cast<float>(draw_rect.GetHeight()) / static_cast<float>(m_display_texture_view_height)),
        1.0f);
      uniforms.params[2] = 0.5f - 0.5f / uniforms.params[0];
      uniforms.params[3] = 0.5f - 0.5f / uniforms.params[1];
    }
    break;

    default:
      UnreachableCode();
      break;
  }

  const GPUTexture::Format hdformat = target ? target->GetFormat() : g_gpu_device->GetWindowFormat();
  const u32 target_width = target ? target->GetWidth() : g_gpu_device->GetWindowWidth();
  const u32 target_height = target ? target->GetHeight() : g_gpu_device->GetWindowHeight();
  const bool really_postfx = (postfx && HasDisplayTexture() && PostProcessing::IsActive() &&
                              PostProcessing::CheckTargets(hdformat, target_width, target_height));
  if (really_postfx)
  {
    g_gpu_device->ClearRenderTarget(PostProcessing::GetInputTexture(), 0);
    g_gpu_device->SetRenderTarget(PostProcessing::GetInputTexture());
  }
  else
  {
    if (target)
      g_gpu_device->SetRenderTarget(target);
    else if (!g_gpu_device->BeginPresent(false))
      return false;
  }

  if (!HasDisplayTexture())
    return true;

  g_gpu_device->SetPipeline(m_display_pipeline.get());
  g_gpu_device->SetTextureSampler(
    0, m_display_texture, texture_filter_linear ? g_gpu_device->GetLinearSampler() : g_gpu_device->GetNearestSampler());

  // For bilinear, clamp to 0.5/SIZE-0.5 to avoid bleeding from the adjacent texels in VRAM. This is because
  // 1.0 in UV space is not the bottom-right texel, but a mix of the bottom-right and wrapped/next texel.
  const float rcp_width = 1.0f / static_cast<float>(m_display_texture->GetWidth());
  const float rcp_height = 1.0f / static_cast<float>(m_display_texture->GetHeight());
  uniforms.src_rect[0] = static_cast<float>(m_display_texture_view_x) * rcp_width;
  uniforms.src_rect[1] = static_cast<float>(m_display_texture_view_y) * rcp_height;
  uniforms.src_rect[2] = static_cast<float>(m_display_texture_view_width) * rcp_width;
  uniforms.src_rect[3] = static_cast<float>(m_display_texture_view_height) * rcp_height;
  uniforms.clamp_rect[0] = (static_cast<float>(m_display_texture_view_x) + 0.5f) * rcp_width;
  uniforms.clamp_rect[1] = (static_cast<float>(m_display_texture_view_y) + 0.5f) * rcp_height;
  uniforms.clamp_rect[2] =
    (static_cast<float>(m_display_texture_view_x + m_display_texture_view_width) - 0.5f) * rcp_width;
  uniforms.clamp_rect[3] =
    (static_cast<float>(m_display_texture_view_y + m_display_texture_view_height) - 0.5f) * rcp_height;
  uniforms.src_size[0] = static_cast<float>(m_display_texture->GetWidth());
  uniforms.src_size[1] = static_cast<float>(m_display_texture->GetHeight());
  uniforms.src_size[2] = rcp_width;
  uniforms.src_size[3] = rcp_height;
  g_gpu_device->PushUniformBuffer(&uniforms, sizeof(uniforms));

  g_gpu_device->SetViewportAndScissor(draw_rect.left, draw_rect.top, draw_rect.GetWidth(), draw_rect.GetHeight());
  g_gpu_device->Draw(3, 0);

  if (really_postfx)
  {
    return PostProcessing::Apply(target, draw_rect.left, draw_rect.top, draw_rect.GetWidth(), draw_rect.GetHeight(),
                                 m_display_texture_view_width, m_display_texture_view_height);
  }
  else
  {
    return true;
  }
}

Common::Rectangle<float> GPUBackend::CalculateDrawRect(s32 window_width, s32 window_height, float* out_left_padding,
                                                       float* out_top_padding, float* out_scale, float* out_x_scale,
                                                       bool apply_aspect_ratio /* = true */) const
{
  const float window_ratio = static_cast<float>(window_width) / static_cast<float>(window_height);
  const float x_scale =
    apply_aspect_ratio ?
      (m_display_aspect_ratio / (static_cast<float>(m_display_width) / static_cast<float>(m_display_height))) :
      1.0f;
  const float display_width = g_gpu_settings.display_stretch_vertically ? static_cast<float>(m_display_width) :
                                                                          static_cast<float>(m_display_width) * x_scale;
  const float display_height = g_gpu_settings.display_stretch_vertically ?
                                 static_cast<float>(m_display_height) / x_scale :
                                 static_cast<float>(m_display_height);
  const float active_left = g_gpu_settings.display_stretch_vertically ?
                              static_cast<float>(m_display_active_left) :
                              static_cast<float>(m_display_active_left) * x_scale;
  const float active_top = g_gpu_settings.display_stretch_vertically ?
                             static_cast<float>(m_display_active_top) / x_scale :
                             static_cast<float>(m_display_active_top);
  const float active_width = g_gpu_settings.display_stretch_vertically ?
                               static_cast<float>(m_display_active_width) :
                               static_cast<float>(m_display_active_width) * x_scale;
  const float active_height = g_gpu_settings.display_stretch_vertically ?
                                static_cast<float>(m_display_active_height) / x_scale :
                                static_cast<float>(m_display_active_height);
  if (out_x_scale)
    *out_x_scale = x_scale;

  // now fit it within the window
  float scale;
  if ((display_width / display_height) >= window_ratio)
  {
    // align in middle vertically
    scale = static_cast<float>(window_width) / display_width;
    if (g_gpu_settings.display_scaling == DisplayScalingMode::NearestInteger)
      scale = std::max(std::floor(scale), 1.0f);

    if (out_left_padding)
    {
      if (g_gpu_settings.display_scaling == DisplayScalingMode::NearestInteger)
        *out_left_padding = std::max<float>((static_cast<float>(window_width) - display_width * scale) / 2.0f, 0.0f);
      else
        *out_left_padding = 0.0f;
    }
    if (out_top_padding)
    {
      switch (g_gpu_settings.display_alignment)
      {
        case DisplayAlignment::RightOrBottom:
          *out_top_padding = std::max<float>(static_cast<float>(window_height) - (display_height * scale), 0.0f);
          break;

        case DisplayAlignment::Center:
          *out_top_padding =
            std::max<float>((static_cast<float>(window_height) - (display_height * scale)) / 2.0f, 0.0f);
          break;

        case DisplayAlignment::LeftOrTop:
        default:
          *out_top_padding = 0.0f;
          break;
      }
    }
  }
  else
  {
    // align in middle horizontally
    scale = static_cast<float>(window_height) / display_height;
    if (g_gpu_settings.display_scaling == DisplayScalingMode::NearestInteger)
      scale = std::max(std::floor(scale), 1.0f);

    if (out_left_padding)
    {
      switch (g_gpu_settings.display_alignment)
      {
        case DisplayAlignment::RightOrBottom:
          *out_left_padding = std::max<float>(static_cast<float>(window_width) - (display_width * scale), 0.0f);
          break;

        case DisplayAlignment::Center:
          *out_left_padding =
            std::max<float>((static_cast<float>(window_width) - (display_width * scale)) / 2.0f, 0.0f);
          break;

        case DisplayAlignment::LeftOrTop:
        default:
          *out_left_padding = 0.0f;
          break;
      }
    }

    if (out_top_padding)
    {
      if (g_gpu_settings.display_scaling == DisplayScalingMode::NearestInteger)
        *out_top_padding = std::max<float>((static_cast<float>(window_height) - (display_height * scale)) / 2.0f, 0.0f);
      else
        *out_top_padding = 0.0f;
    }
  }

  if (out_scale)
    *out_scale = scale;

  return Common::Rectangle<float>::FromExtents(active_left * scale, active_top * scale, active_width * scale,
                                               active_height * scale);
}

Common::Rectangle<s32> GPUBackend::CalculateDrawRect(s32 window_width, s32 window_height,
                                                     bool apply_aspect_ratio /* = true */) const
{
  float left_padding, top_padding;
  const Common::Rectangle<float> draw_rc =
    CalculateDrawRect(window_width, window_height, &left_padding, &top_padding, nullptr, nullptr, apply_aspect_ratio);

  // TODO: This should be a float rectangle. But because GL is lame, it only has integer viewports...
  return Common::Rectangle<s32>::FromExtents(
    static_cast<s32>(draw_rc.left + left_padding), static_cast<s32>(draw_rc.top + top_padding),
    static_cast<s32>(draw_rc.GetWidth()), static_cast<s32>(draw_rc.GetHeight()));
}

static bool CompressAndWriteTextureToFile(u32 width, u32 height, std::string filename, FileSystem::ManagedCFilePtr fp,
                                          bool clear_alpha, bool flip_y, u32 resize_width, u32 resize_height,
                                          std::vector<u32> texture_data, u32 texture_data_stride,
                                          GPUTexture::Format texture_format)
{

  const char* extension = std::strrchr(filename.c_str(), '.');
  if (!extension)
  {
    Log_ErrorPrintf("Unable to determine file extension for '%s'", filename.c_str());
    return false;
  }

  if (!GPUTexture::ConvertTextureDataToRGBA8(width, height, texture_data, texture_data_stride, texture_format))
    return false;

  if (clear_alpha)
  {
    for (u32& pixel : texture_data)
      pixel |= 0xFF000000;
  }

  if (flip_y)
    GPUTexture::FlipTextureDataRGBA8(width, height, texture_data, texture_data_stride);

  if (resize_width > 0 && resize_height > 0 && (resize_width != width || resize_height != height))
  {
    std::vector<u32> resized_texture_data(resize_width * resize_height);
    u32 resized_texture_stride = sizeof(u32) * resize_width;
    if (!stbir_resize_uint8(reinterpret_cast<u8*>(texture_data.data()), width, height, texture_data_stride,
                            reinterpret_cast<u8*>(resized_texture_data.data()), resize_width, resize_height,
                            resized_texture_stride, 4))
    {
      Log_ErrorPrintf("Failed to resize texture data from %ux%u to %ux%u", width, height, resize_width, resize_height);
      return false;
    }

    width = resize_width;
    height = resize_height;
    texture_data = std::move(resized_texture_data);
    texture_data_stride = resized_texture_stride;
  }

  const auto write_func = [](void* context, void* data, int size) {
    std::fwrite(data, 1, size, static_cast<std::FILE*>(context));
  };

  bool result = false;
  if (StringUtil::Strcasecmp(extension, ".png") == 0)
  {
    result =
      (stbi_write_png_to_func(write_func, fp.get(), width, height, 4, texture_data.data(), texture_data_stride) != 0);
  }
  else if (StringUtil::Strcasecmp(extension, ".jpg") == 0)
  {
    result = (stbi_write_jpg_to_func(write_func, fp.get(), width, height, 4, texture_data.data(), 95) != 0);
  }
  else if (StringUtil::Strcasecmp(extension, ".tga") == 0)
  {
    result = (stbi_write_tga_to_func(write_func, fp.get(), width, height, 4, texture_data.data()) != 0);
  }
  else if (StringUtil::Strcasecmp(extension, ".bmp") == 0)
  {
    result = (stbi_write_bmp_to_func(write_func, fp.get(), width, height, 4, texture_data.data()) != 0);
  }

  if (!result)
  {
    Log_ErrorPrintf("Unknown extension in filename '%s' or save error: '%s'", filename.c_str(), extension);
    return false;
  }

  return true;
}

bool GPUBackend::WriteDisplayTextureToFile(std::string filename, bool full_resolution /* = true */,
                                           bool apply_aspect_ratio /* = true */, bool compress_on_thread /* = false */)
{
  if (!m_display_texture)
    return false;

  s32 resize_width = 0;
  s32 resize_height = std::abs(m_display_texture_view_height);
  if (apply_aspect_ratio)
  {
    const float ss_width_scale = static_cast<float>(m_display_active_width) / static_cast<float>(m_display_width);
    const float ss_height_scale = static_cast<float>(m_display_active_height) / static_cast<float>(m_display_height);
    const float ss_aspect_ratio = m_display_aspect_ratio * ss_width_scale / ss_height_scale;
    resize_width = g_gpu_settings.display_stretch_vertically ?
                     m_display_texture_view_width :
                     static_cast<s32>(static_cast<float>(resize_height) * ss_aspect_ratio);
    resize_height = g_gpu_settings.display_stretch_vertically ?
                      static_cast<s32>(static_cast<float>(resize_height) /
                                       (m_display_aspect_ratio /
                                        (static_cast<float>(m_display_width) / static_cast<float>(m_display_height)))) :
                      resize_height;
  }
  else
  {
    resize_width = m_display_texture_view_width;
  }

  if (!full_resolution)
  {
    const s32 resolution_scale = std::abs(m_display_texture_view_height) / m_display_active_height;
    resize_height /= resolution_scale;
    resize_width /= resolution_scale;
  }

  if (resize_width <= 0 || resize_height <= 0)
    return false;

  const u32 read_x = static_cast<u32>(m_display_texture_view_x);
  const u32 read_y = static_cast<u32>(m_display_texture_view_y);
  const u32 read_width = static_cast<u32>(m_display_texture_view_width);
  const u32 read_height = static_cast<u32>(m_display_texture_view_height);

  std::vector<u32> texture_data(read_width * read_height);
  const u32 texture_data_stride =
    Common::AlignUpPow2(GPUTexture::GetPixelSize(m_display_texture->GetFormat()) * read_width, 4);
  if (!g_gpu_device->DownloadTexture(m_display_texture, read_x, read_y, read_width, read_height, texture_data.data(),
                                     texture_data_stride))
  {
    Log_ErrorPrintf("Texture download failed");
    RestoreDeviceContext();
    return false;
  }

  RestoreDeviceContext();

  auto fp = FileSystem::OpenManagedCFile(filename.c_str(), "wb");
  if (!fp)
  {
    Log_ErrorPrintf("Can't open file '%s': errno %d", filename.c_str(), errno);
    return false;
  }

  constexpr bool clear_alpha = true;
  const bool flip_y = g_gpu_device->UsesLowerLeftOrigin();

  if (!compress_on_thread)
  {
    return CompressAndWriteTextureToFile(read_width, read_height, std::move(filename), std::move(fp), clear_alpha,
                                         flip_y, resize_width, resize_height, std::move(texture_data),
                                         texture_data_stride, m_display_texture->GetFormat());
  }

  std::thread compress_thread(CompressAndWriteTextureToFile, read_width, read_height, std::move(filename),
                              std::move(fp), clear_alpha, flip_y, resize_width, resize_height, std::move(texture_data),
                              texture_data_stride, m_display_texture->GetFormat());
  compress_thread.detach();
  return true;
}

void GPUBackend::HandleRenderScreenshotToBuffer(const GPUThreadRenderScreenshotToBufferCommand* cmd)
{
  const GPUTexture::Format hdformat =
    g_gpu_device->HasSurface() ? g_gpu_device->GetWindowFormat() : GPUTexture::Format::RGBA8;

  std::unique_ptr<GPUTexture> render_texture =
    g_gpu_device->FetchTexture(cmd->width, cmd->height, 1, 1, 1, GPUTexture::Type::RenderTarget, hdformat);
  if (!render_texture)
  {
    *cmd->out_result = false;
    return;
  }

  g_gpu_device->ClearRenderTarget(render_texture.get(), 0);

  // TODO: this should use copy shader instead.
  RenderDisplay(render_texture.get(), cmd->draw_rect, cmd->postfx);

  const u32 stride = GPUTexture::GetPixelSize(hdformat) * cmd->width;
  cmd->out_pixels->resize(cmd->width * cmd->height);
  if (!g_gpu_device->DownloadTexture(render_texture.get(), 0, 0, cmd->width, cmd->height, cmd->out_pixels->data(),
                                     stride))
  {
    *cmd->out_result = false;
    RestoreDeviceContext();
    return;
  }

  *cmd->out_stride = stride;
  *cmd->out_format = hdformat;
  *cmd->out_result = true;
  RestoreDeviceContext();
}

bool GPUBackend::RenderScreenshotToFile(std::string filename, bool internal_resolution /* = false */,
                                        bool compress_on_thread /* = false */)
{
  u32 width = g_gpu_device->GetWindowWidth();
  u32 height = g_gpu_device->GetWindowHeight();
  Common::Rectangle<s32> draw_rect = CalculateDrawRect(width, height);

  if (internal_resolution && m_display_texture_view_width != 0 && m_display_texture_view_height != 0)
  {
    const u32 draw_width = static_cast<u32>(draw_rect.GetWidth());
    const u32 draw_height = static_cast<u32>(draw_rect.GetHeight());

    // If internal res, scale the computed draw rectangle to the internal res.
    // We re-use the draw rect because it's already been AR corrected.
    const float sar =
      static_cast<float>(m_display_texture_view_width) / static_cast<float>(m_display_texture_view_height);
    const float dar = static_cast<float>(draw_width) / static_cast<float>(draw_height);
    if (sar >= dar)
    {
      // stretch height, preserve width
      const float scale = static_cast<float>(m_display_texture_view_width) / static_cast<float>(draw_width);
      width = m_display_texture_view_width;
      height = static_cast<u32>(std::round(static_cast<float>(draw_height) * scale));
    }
    else
    {
      // stretch width, preserve height
      const float scale = static_cast<float>(m_display_texture_view_height) / static_cast<float>(draw_height);
      width = static_cast<u32>(std::round(static_cast<float>(draw_width) * scale));
      height = m_display_texture_view_height;
    }

    // DX11 won't go past 16K texture size.
    constexpr u32 MAX_TEXTURE_SIZE = 16384;
    if (width > MAX_TEXTURE_SIZE)
    {
      height = static_cast<u32>(static_cast<float>(height) /
                                (static_cast<float>(width) / static_cast<float>(MAX_TEXTURE_SIZE)));
      width = MAX_TEXTURE_SIZE;
    }
    if (height > MAX_TEXTURE_SIZE)
    {
      height = MAX_TEXTURE_SIZE;
      width = static_cast<u32>(static_cast<float>(width) /
                               (static_cast<float>(height) / static_cast<float>(MAX_TEXTURE_SIZE)));
    }

    // Remove padding, it's not part of the framebuffer.
    draw_rect.Set(0, 0, static_cast<s32>(width), static_cast<s32>(height));
  }
  if (width == 0 || height == 0)
    return false;

  std::vector<u32> pixels;
  u32 pixels_stride;
  GPUTexture::Format pixels_format;
  if (!RenderScreenshotToBuffer(width, height, draw_rect, !internal_resolution, &pixels, &pixels_stride,
                                &pixels_format))
  {
    Log_ErrorPrintf("Failed to render %ux%u screenshot", width, height);
    return false;
  }

  auto fp = FileSystem::OpenManagedCFile(filename.c_str(), "wb");
  if (!fp)
  {
    Log_ErrorPrintf("Can't open file '%s': errno %d", filename.c_str(), errno);
    return false;
  }

  if (!compress_on_thread)
  {
    return CompressAndWriteTextureToFile(width, height, std::move(filename), std::move(fp), true,
                                         g_gpu_device->UsesLowerLeftOrigin(), width, height, std::move(pixels),
                                         pixels_stride, pixels_format);
  }

  std::thread compress_thread(CompressAndWriteTextureToFile, width, height, std::move(filename), std::move(fp), true,
                              g_gpu_device->UsesLowerLeftOrigin(), width, height, std::move(pixels), pixels_stride,
                              pixels_format);
  compress_thread.detach();
  return true;
}
