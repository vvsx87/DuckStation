// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "gpu_thread.h"
#include "fullscreen_ui.h"
#include "gpu_backend.h"
#include "host.h"
#include "imgui_overlays.h"
#include "settings.h"
#include "shader_cache_version.h"
#include "system.h"

#include "util/gpu_device.h"
#include "util/imgui_manager.h"
#include "util/state_wrapper.h"

#include "common/align.h"
#include "common/log.h"
#include "common/timer.h"
#include "common/threading.h"

#include "IconsFontAwesome5.h"
#include "imgui.h"

#include <optional>

Log_SetChannel(GPUThread);

namespace GPUThread {
enum : u32
{
  COMMAND_QUEUE_SIZE = 4 * 1024 * 1024,
  THRESHOLD_TO_WAKE_GPU = 256
};

/// Starts the thread, if it hasn't already been started.
/// TODO: Persist thread
static bool Start(std::optional<GPURenderer> api);

static void RunGPULoop();
static u32 GetPendingCommandSize();
static void WakeGPUThread();

static bool CreateDeviceOnThread(RenderAPI api);
static void DestroyDeviceOnThread();

static void CreateGPUBackendOnThread();
static void ChangeGPUBackendOnThread();
static void DestroyGPUBackendOnThread();

static void UpdateSettingsOnThread(const Settings& old_settings);
static void UpdateVSyncOnThread();

static RenderAPI s_render_api = RenderAPI::None;
static std::unique_ptr<GPUBackend> s_gpu_backend;
static std::optional<GPURenderer> s_requested_renderer;
static float s_requested_max_displayed_fps = 0.0f;
static bool s_start_fullscreen_ui = false;
static bool s_requested_vsync = false;

static bool s_last_frame_skipped = false;
static u32 s_presents_since_last_update = 0;
static float s_accumulated_gpu_time = 0.0f;

static Threading::KernelSemaphore m_sync_semaphore;
static Threading::Thread m_gpu_thread;
static std::atomic_bool s_open_flag{false};
static std::atomic_bool s_shutdown_flag{false};
static std::atomic_bool s_run_idle_flag{false};

static std::mutex m_sync_mutex;
static Threading::WorkSema s_work_sema;

static FixedHeapArray<u8, COMMAND_QUEUE_SIZE> m_command_fifo_data;
alignas(64) static std::atomic<u32> m_command_fifo_read_ptr{0};
alignas(64) static std::atomic<u32> m_command_fifo_write_ptr{0};
} // namespace GPUThread

const Threading::ThreadHandle& GPUThread::GetThreadHandle()
{
  return m_gpu_thread;
}

RenderAPI GPUThread::GetRenderAPI()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  return s_render_api;
}

bool GPUThread::IsStarted()
{
  return m_gpu_thread.Joinable();
}

bool GPUThread::WasFullscreenUIRequested()
{
  return s_start_fullscreen_ui;
}

bool GPUThread::Start(std::optional<GPURenderer> renderer)
{
  Assert(!IsStarted());

  Log_VerbosePrint("Starting GPU thread...");

  s_requested_renderer = renderer;
  g_gpu_settings = g_settings;
  s_shutdown_flag.store(false, std::memory_order_release);
  s_run_idle_flag.store(false, std::memory_order_release);
  m_gpu_thread.Start(&GPUThread::RunGPULoop);
  m_sync_semaphore.Wait();

  if (!s_open_flag.load(std::memory_order_acquire))
  {
    Log_ErrorPrint("Failed to create GPU thread.");
    m_gpu_thread.Join();
    return false;
  }

  Log_InfoPrint("GPU thread started.");
  return true;
}

bool GPUThread::StartFullscreenUI()
{
  // NOTE: Racey read.
  if (FullscreenUI::IsInitialized())
    return true;

  if (IsStarted())
  {
    RunOnThread([]() {
      // TODO: Error handling.
      if (!FullscreenUI::Initialize())
        Panic("Failed");
    });

    return true;
  }

  s_start_fullscreen_ui = true;
  if (!Start(std::nullopt))
  {
    s_start_fullscreen_ui = false;
    return false;
  }

  return true;
}

std::optional<GPURenderer> GPUThread::GetRequestedRenderer()
{
  return s_requested_renderer;
}

bool GPUThread::CreateGPUBackend(GPURenderer renderer)
{
  if (IsStarted())
  {
    s_requested_renderer = renderer;
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return true;
  }
  else
  {
    return Start(renderer);
  }
}

bool GPUThread::SwitchGPUBackend(GPURenderer renderer, bool force_recreate_device)
{
  if (!force_recreate_device)
  {
    s_requested_renderer = renderer;
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return true;
  }

  const bool was_running_fsui = s_start_fullscreen_ui;
  Shutdown();
  s_requested_renderer = renderer;
  s_start_fullscreen_ui = was_running_fsui;
  if (!Start(renderer))
  {
    s_requested_renderer.reset();
    s_start_fullscreen_ui = false;
    return false;
  }

  return true;
}

void GPUThread::DestroyGPUBackend()
{
  if (!IsStarted())
    return;

  if (s_start_fullscreen_ui)
  {
    Log_VerboseFmt("Keeping GPU thread open for fullscreen UI");
    s_requested_renderer.reset();
    std::atomic_thread_fence(std::memory_order_release);
    PushCommandAndSync(AllocateCommand(GPUBackendCommandType::ChangeBackend, sizeof(GPUThreadCommand)), false);
    return;
  }

  Shutdown();
}

void GPUThread::Shutdown()
{
  if (!IsStarted())
    return;

  s_shutdown_flag.store(true, std::memory_order_release);
  s_start_fullscreen_ui = false;
  s_requested_renderer.reset();

  WakeGPUThread();
  m_gpu_thread.Join();
  Log_InfoPrint("GPU thread stopped.");
}

GPUThreadCommand* GPUThread::AllocateCommand(GPUBackendCommandType command, u32 size)
{
  // Ensure size is a multiple of 4 so we don't end up with an unaligned command.
  size = Common::AlignUpPow2(size, 4);

  for (;;)
  {
    u32 read_ptr = m_command_fifo_read_ptr.load();
    u32 write_ptr = m_command_fifo_write_ptr.load();
    if (read_ptr > write_ptr)
    {
      u32 available_size = read_ptr - write_ptr;
      while (available_size < (size + sizeof(GPUBackendCommandType)))
      {
        WakeGPUThread();
        read_ptr = m_command_fifo_read_ptr.load();
        available_size = (read_ptr > write_ptr) ? (read_ptr - write_ptr) : (COMMAND_QUEUE_SIZE - write_ptr);
      }
    }
    else
    {
      const u32 available_size = COMMAND_QUEUE_SIZE - write_ptr;
      if ((size + sizeof(GPUBackendCommand)) > available_size)
      {
        // allocate a dummy command to wrap the buffer around
        GPUBackendCommand* dummy_cmd = reinterpret_cast<GPUBackendCommand*>(&m_command_fifo_data[write_ptr]);
        dummy_cmd->type = GPUBackendCommandType::Wraparound;
        dummy_cmd->size = available_size;
        dummy_cmd->params.bits = 0;
        m_command_fifo_write_ptr.store(0);
        continue;
      }
    }

    GPUThreadCommand* cmd = reinterpret_cast<GPUThreadCommand*>(&m_command_fifo_data[write_ptr]);
    cmd->type = command;
    cmd->size = size;
    return cmd;
  }
}

u32 GPUThread::GetPendingCommandSize()
{
  const u32 read_ptr = m_command_fifo_read_ptr.load();
  const u32 write_ptr = m_command_fifo_write_ptr.load();
  return (write_ptr >= read_ptr) ? (write_ptr - read_ptr) : (COMMAND_QUEUE_SIZE - read_ptr + write_ptr);
}

void GPUThread::PushCommand(GPUThreadCommand* cmd)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  if (GetPendingCommandSize() >= THRESHOLD_TO_WAKE_GPU)
    WakeGPUThread();
}

void GPUThread::PushCommandAndWakeThread(GPUThreadCommand* cmd)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  WakeGPUThread();
}

void GPUThread::PushCommandAndSync(GPUThreadCommand* cmd, bool spin)
{
  const u32 new_write_ptr = m_command_fifo_write_ptr.fetch_add(cmd->size) + cmd->size;
  DebugAssert(new_write_ptr <= COMMAND_QUEUE_SIZE);
  UNREFERENCED_VARIABLE(new_write_ptr);
  WakeGPUThread();

  if (spin)
    s_work_sema.WaitForEmptyWithSpin();
  else
    s_work_sema.WaitForEmpty();
}

void GPUThread::WakeGPUThread()
{
  s_work_sema.NotifyOfWork();
}

void GPUThread::RunGPULoop()
{
  Threading::SetNameOfCurrentThread("GPUThread");

  if (!CreateDeviceOnThread(
        Settings::GetRenderAPIForRenderer(s_requested_renderer.value_or(g_gpu_settings.gpu_renderer))))
  {
    Host::ReleaseRenderWindow();
    s_open_flag.store(false, std::memory_order_release);
    m_sync_semaphore.Post();
    return;
  }

  CreateGPUBackendOnThread();

  s_open_flag.store(true, std::memory_order_release);
  m_sync_semaphore.Post();

  for (;;)
  {
    u32 write_ptr = m_command_fifo_write_ptr.load(std::memory_order_acquire);
    u32 read_ptr = m_command_fifo_read_ptr.load(std::memory_order_relaxed);
    if (read_ptr == write_ptr)
    {
      if (s_shutdown_flag.load(std::memory_order_relaxed))
      {
        break;
      }
      else if (s_run_idle_flag.load(std::memory_order_relaxed))
      {
        if (!s_work_sema.CheckForWork())
        {
          Internal::PresentFrame(false, 0);
          if (!g_gpu_device->IsVsyncEnabled())
            g_gpu_device->ThrottlePresentation();

          continue;
        }

        // we should have something to do, since we got woken...
      }
      else
      {
        s_work_sema.WaitForWork();
        continue;
      }
    }

    if (write_ptr < read_ptr)
      write_ptr = COMMAND_QUEUE_SIZE;

    while (read_ptr < write_ptr)
    {
      GPUThreadCommand* cmd = reinterpret_cast<GPUThreadCommand*>(&m_command_fifo_data[read_ptr]);
      read_ptr += cmd->size;

      switch (cmd->type)
      {
        case GPUBackendCommandType::Wraparound:
        {
          DebugAssert(read_ptr == COMMAND_QUEUE_SIZE);
          write_ptr = m_command_fifo_write_ptr.load();
          read_ptr = 0;
        }
        break;

        case GPUBackendCommandType::AsyncCall:
        {
          GPUThreadAsyncCallCommand* acmd = static_cast<GPUThreadAsyncCallCommand*>(cmd);
          acmd->func();
          acmd->~GPUThreadAsyncCallCommand();
        }
        break;

        case GPUBackendCommandType::ChangeBackend:
        {
          ChangeGPUBackendOnThread();
        }
        break;

        case GPUBackendCommandType::UpdateVSync:
        {
          UpdateVSyncOnThread();
        }
        break;

        default:
        {
          DebugAssert(s_gpu_backend);
          s_gpu_backend->HandleCommand(cmd);
        }
        break;
      }
    }

    m_command_fifo_read_ptr.store(read_ptr, std::memory_order_release);
  }

  DestroyGPUBackendOnThread();
  DestroyDeviceOnThread();
  Host::ReleaseRenderWindow();
}

bool GPUThread::CreateDeviceOnThread(RenderAPI api)
{
  DebugAssert(!g_gpu_device);

  Log_VerboseFmt("Trying to create a {} GPU device...", GPUDevice::RenderAPIToString(api));
  g_gpu_device = GPUDevice::CreateDeviceForAPI(api);

  u32 disabled_features = 0;
  if (g_gpu_settings.gpu_disable_dual_source_blend)
    disabled_features |= GPUDevice::FEATURE_MASK_DUAL_SOURCE_BLEND;
  if (g_gpu_settings.gpu_disable_framebuffer_fetch)
    disabled_features |= GPUDevice::FEATURE_MASK_FRAMEBUFFER_FETCH;

  if (!g_gpu_device ||
      !g_gpu_device->Create(
        g_gpu_settings.gpu_adapter,
        g_gpu_settings.gpu_disable_shader_cache ? std::string_view() : std::string_view(EmuFolders::Cache),
        SHADER_CACHE_VERSION, g_gpu_settings.gpu_use_debug_device, s_requested_vsync,
        g_gpu_settings.gpu_threaded_presentation, static_cast<GPUDevice::FeatureMask>(disabled_features)))
  {
    Log_ErrorPrintf("Failed to create GPU device.");
    if (g_gpu_device)
      g_gpu_device->Destroy();
    g_gpu_device.reset();

    Host::ReportErrorAsync("Error",
                           fmt::format("Failed to create render device. This may be due to your GPU not supporting the "
                                       "chosen renderer ({}), or because your graphics drivers need to be updated.",
                                       GPUDevice::RenderAPIToString(api)));

    s_render_api = RenderAPI::None;
    std::atomic_thread_fence(std::memory_order_release);
    return false;
  }

  if (!ImGuiManager::Initialize(g_gpu_settings.display_osd_scale / 100.0f, g_gpu_settings.display_show_osd_messages) ||
      (s_start_fullscreen_ui && !FullscreenUI::Initialize()))
  {
    Log_ErrorPrintf("Failed to initialize ImGuiManager.");
    FullscreenUI::Shutdown();
    ImGuiManager::Shutdown();
    g_gpu_device->Destroy();
    g_gpu_device.reset();
    s_render_api = RenderAPI::None;
    std::atomic_thread_fence(std::memory_order_release);
    return false;
  }

  s_accumulated_gpu_time = 0.0f;
  s_presents_since_last_update = 0;
  g_gpu_device->SetGPUTimingEnabled(g_settings.display_show_gpu);
  g_gpu_device->SetDisplayMaxFPS(s_requested_max_displayed_fps);
  Log_VerboseFmt("Vsync is {}, max displayed FPS is {}", s_requested_vsync ? "enabled" : "disabled",
                 s_requested_max_displayed_fps);

  s_render_api = g_gpu_device->GetRenderAPI();
  std::atomic_thread_fence(std::memory_order_release);

  return true;
}

void GPUThread::DestroyDeviceOnThread()
{
  if (!g_gpu_device)
    return;

  ImGuiManager::DestroyOverlayTextures();
  FullscreenUI::Shutdown();
  ImGuiManager::Shutdown();

  Log_VerboseFmt("Destroying {} GPU device...", GPUDevice::RenderAPIToString(g_gpu_device->GetRenderAPI()));
  g_gpu_device->Destroy();
  g_gpu_device.reset();
}

void GPUThread::CreateGPUBackendOnThread()
{
  Assert(!s_gpu_backend);
  if (!s_requested_renderer.has_value())
    return;

  const bool is_hardware = (s_requested_renderer.value() != GPURenderer::Software);

  if (is_hardware)
    s_gpu_backend = GPUBackend::CreateHardwareBackend();
  else
    s_gpu_backend = GPUBackend::CreateSoftwareBackend();

  if (!s_gpu_backend)
  {
    Log_ErrorFmt("Failed to create {} renderer", Settings::GetRendererName(s_requested_renderer.value()));

    if (is_hardware)
    {
      Host::AddIconOSDMessage(
        "GPUBackendCreationFailed", ICON_FA_PAINT_ROLLER,
        fmt::format(TRANSLATE_FS("OSDMessage", "Failed to initialize {} renderer, falling back to software renderer."),
                    Settings::GetRendererName(s_requested_renderer.value())),
        Host::OSD_CRITICAL_ERROR_DURATION);

      s_requested_renderer = GPURenderer::Software;
      s_gpu_backend = GPUBackend::CreateSoftwareBackend();
      if (!s_gpu_backend)
        Panic("Failed to initialize software backend.");
    }
  }
}

ALWAYS_INLINE_RELEASE void GPUThread::ChangeGPUBackendOnThread()
{
  std::atomic_thread_fence(std::memory_order_acquire);
  if (!s_requested_renderer.has_value())
  {
    if (s_gpu_backend)
      DestroyGPUBackendOnThread();

    return;
  }

  if (s_requested_renderer.value() == GPURenderer::Software)
  {
    // Just recreate the backend, software works with everything.
    DestroyGPUBackendOnThread();
    CreateGPUBackendOnThread();
    return;
  }

  DestroyGPUBackendOnThread();

  DebugAssert(g_gpu_device);
  const RenderAPI current_api = s_render_api;
  const RenderAPI expected_api = Settings::GetRenderAPIForRenderer(s_requested_renderer.value());
  if (!GPUDevice::IsSameRenderAPI(current_api, expected_api))
  {
    Log_WarningFmt("Recreating GPU device, expecting {} got {}", GPUDevice::RenderAPIToString(expected_api),
                   GPUDevice::RenderAPIToString(current_api));

    DestroyDeviceOnThread();

    // Things tend to break when you don't recreate the window, after switching APIs.
    Host::ReleaseRenderWindow();

    if (!CreateDeviceOnThread(expected_api))
    {
      Host::AddIconOSDMessage(
        "DeviceSwitchFailed", ICON_FA_PAINT_ROLLER,
        fmt::format(TRANSLATE_FS("OSDMessage", "Failed to create {} GPU device, reverting to {}."),
                    GPUDevice::RenderAPIToString(expected_api), GPUDevice::RenderAPIToString(current_api)),
        Host::OSD_CRITICAL_ERROR_DURATION);

      Host::ReleaseRenderWindow();
      if (!CreateDeviceOnThread(current_api))
        Panic("Failed to switch back to old API after creation failure");
    }
  }

  CreateGPUBackendOnThread();
}

void GPUThread::DestroyGPUBackendOnThread()
{
  if (!s_gpu_backend)
    return;

  Log_VerboseFmt("Shutting down GPU backend...");

  s_gpu_backend->Shutdown();
  s_gpu_backend.reset();
}

void GPUThread::UpdateSettingsOnThread(const Settings& old_settings)
{
  if (g_gpu_settings.display_show_gpu != old_settings.display_show_gpu)
  {
    g_gpu_device->SetGPUTimingEnabled(g_gpu_settings.display_show_gpu);
    s_accumulated_gpu_time = 0.0f;
    s_presents_since_last_update = 0;
  }

  if (s_gpu_backend)
    s_gpu_backend->UpdateSettings(old_settings);
}

void GPUThread::UpdateVSyncOnThread()
{
  std::atomic_thread_fence(std::memory_order_acquire);

  const bool vsync = s_requested_vsync;
  const float max_displayed_fps = s_requested_max_displayed_fps;

  if (g_gpu_device->IsVsyncEnabled() != vsync)
    g_gpu_device->SetVSync(vsync);
  g_gpu_device->SetDisplayMaxFPS(max_displayed_fps);

  Log_VerboseFmt("Vsync is {}, max displayed FPS is {}", vsync ? "enabled" : "disabled", max_displayed_fps);
}

void GPUThread::RunOnThread(AsyncCallType func)
{
  GPUThreadAsyncCallCommand* cmd = static_cast<GPUThreadAsyncCallCommand*>(
    AllocateCommand(GPUBackendCommandType::AsyncCall, sizeof(GPUThreadAsyncCallCommand)));
  new (cmd) GPUThreadAsyncCallCommand;
  cmd->func = std::move(func);
  PushCommandAndWakeThread(cmd);
}

void GPUThread::UpdateSettings()
{
  AssertMsg(IsStarted(), "GPU Thread is running");

  RunOnThread([settings = g_settings]() {
    Log_VerbosePrint("Updating GPU settings on thread...");

    Settings old_settings = std::move(g_gpu_settings);
    g_gpu_settings = std::move(settings);

    UpdateSettingsOnThread(old_settings);
  });
}

void GPUThread::ResizeDisplayWindow(s32 width, s32 height, float scale)
{
  AssertMsg(IsStarted(), "GPU Thread is running");
  RunOnThread([width, height, scale]() {
    if (!g_gpu_device)
      return;

    Log_DevPrintf("Display window resized to %dx%d", width, height);

    g_gpu_device->ResizeWindow(width, height, scale);
    ImGuiManager::WindowResized();

    // If we're paused, re-present the current frame at the new window size.
    if (System::IsValid())
    {
      if (System::IsPaused())
        Internal::PresentFrame(false, 0);

      System::HostDisplayResized();
    }
  });
}

void GPUThread::UpdateDisplayWindow()
{
  AssertMsg(IsStarted(), "MTGS is running");
  RunOnThread([]() {
    if (!g_gpu_device)
      return;

    if (!g_gpu_device->UpdateWindow())
    {
      Host::ReportErrorAsync("Error", "Failed to change window after update. The log may contain more information.");
      return;
    }

    ImGuiManager::WindowResized();

    // If we're paused, re-present the current frame at the new window size.
    if (System::IsValid() && System::IsPaused())
      Internal::PresentFrame(false, 0);
  });
}

void GPUThread::SetVSync(bool enabled, float max_displayed_fps)
{
  Assert(IsStarted());

  if (s_requested_vsync == enabled && s_requested_max_displayed_fps == max_displayed_fps)
    return;

  s_requested_vsync = enabled;
  s_requested_max_displayed_fps = max_displayed_fps;
  std::atomic_thread_fence(std::memory_order_release);
  PushCommandAndWakeThread(AllocateCommand(GPUBackendCommandType::UpdateVSync, sizeof(GPUThreadCommand)));
}

void GPUThread::PresentCurrentFrame()
{
  if (s_run_idle_flag.load(std::memory_order_relaxed))
  {
    // If we're running idle, we're going to re-present anyway.
    return;
  }

  RunOnThread([]() { Internal::PresentFrame(false, 0); });
}

void GPUThread::Internal::PresentFrame(bool allow_skip_present, Common::Timer::Value present_time)
{
  const bool skip_present =
    (allow_skip_present &&
     (g_gpu_device->ShouldSkipDisplayingFrame() ||
      (present_time != 0 && Common::Timer::GetCurrentValue() > present_time && !s_last_frame_skipped)));

  Host::BeginPresentFrame();

  // acquire for IO.MousePos.
  std::atomic_thread_fence(std::memory_order_acquire);

  if (!skip_present)
  {
    FullscreenUI::Render();
    ImGuiManager::RenderTextOverlays();
    ImGuiManager::RenderOSDMessages();

    if (System::GetState() == System::State::Running)
      ImGuiManager::RenderSoftwareCursors();
  }

  // Debug windows are always rendered, otherwise mouse input breaks on skip.
  ImGuiManager::RenderOverlayWindows();
  ImGuiManager::RenderDebugWindows();

  if (s_gpu_backend && !skip_present)
    s_last_frame_skipped = !s_gpu_backend->PresentDisplay();
  else
    s_last_frame_skipped = !g_gpu_device->BeginPresent(skip_present);

  if (!s_last_frame_skipped)
  {
    g_gpu_device->RenderImGui();
    g_gpu_device->EndPresent();

    if (g_gpu_device->IsGPUTimingEnabled())
    {
      std::unique_lock lock(m_sync_mutex);
      s_accumulated_gpu_time += g_gpu_device->GetAndResetAccumulatedGPUTime();
      s_presents_since_last_update++;
    }
  }
  else
  {
    // Still need to kick ImGui or it gets cranky.
    ImGui::Render();
  }

  ImGuiManager::NewFrame();

  if (s_gpu_backend)
    s_gpu_backend->RestoreDeviceContext();
}

void GPUThread::GetAccumulatedGPUTime(float* time, u32* frames)
{
  std::unique_lock lock(m_sync_mutex);
  *time = std::exchange(s_accumulated_gpu_time, 0.0f);
  *frames = std::exchange(s_presents_since_last_update, 0);
}

void GPUThread::SetRunIdle(bool enabled)
{
  // NOTE: Should only be called on the GS thread.
  s_run_idle_flag.store(enabled, std::memory_order_release);
  Log_DevFmt("GPU thread now {} idle", enabled ? "running" : "NOT running");
}
