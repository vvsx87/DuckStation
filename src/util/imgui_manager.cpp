// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "imgui_manager.h"
#include "gpu_device.h"
#include "host.h"
#include "image.h"
#include "imgui_fullscreen.h"
#include "input_manager.h"

#include "common/assert.h"
#include "common/easing.h"
#include "common/error.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/string_util.h"
#include "common/timer.h"

#include "IconsFontAwesome5.h"
#include "fmt/format.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <unordered_map>

Log_SetChannel(ImGuiManager);

namespace ImGuiManager {
namespace {

struct SoftwareCursor
{
  std::string image_path;
  std::unique_ptr<GPUTexture> texture;
  u32 color;
  float scale;
  float extent_x;
  float extent_y;
  std::pair<float, float> pos;
};

struct OSDMessage
{
  std::string key;
  std::string text;
  Common::Timer::Value start_time;
  Common::Timer::Value move_time;
  float duration;
  float target_y;
  float last_y;
};

} // namespace

static void UpdateScale();
static void SetStyle();
static void SetKeyMap();
static bool LoadFontData();
static bool AddImGuiFonts(bool fullscreen_fonts);
static ImFont* AddTextFont(float size);
static ImFont* AddFixedFont(float size);
static bool AddIconFonts(float size);
static void AcquirePendingOSDMessages(Common::Timer::Value current_time);
static void DrawOSDMessages(Common::Timer::Value current_time);
static void CreateSoftwareCursorTextures();
static void UpdateSoftwareCursorTexture(u32 index);
static void DestroySoftwareCursorTextures();
static void DrawSoftwareCursor(const SoftwareCursor& sc, const std::pair<float, float>& pos);
static bool IsMainViewport(const ImGuiViewport* vp);
static void PlatformCreateWindow(ImGuiViewport* vp);
static void PlatformDestroyWindow(ImGuiViewport* vp);
static void PlatformShowWindow(ImGuiViewport* vp);
static void PlatformSetWindowPos(ImGuiViewport* vp, ImVec2 pos);
static ImVec2 PlatformGetWindowPos(ImGuiViewport* vp);
static void PlatformSetWindowSize(ImGuiViewport* vp, ImVec2 size);
static ImVec2 PlatformGetWindowSize(ImGuiViewport* vp);
static void PlatformSetWindowFocus(ImGuiViewport* vp);
static bool PlatformGetWindowFocus(ImGuiViewport* vp);
static bool PlatformGetWindowMinimized(ImGuiViewport* vp);
static void PlatformSetWindowTitle(ImGuiViewport* vp, const char* str);

static float s_global_prescale = 1.0f; // before window scale
static float s_global_scale = 1.0f;

static std::string s_font_path;
static std::vector<ImWchar> s_font_range;

static ImFont* s_standard_font;
static ImFont* s_fixed_font;
static ImFont* s_medium_font;
static ImFont* s_large_font;

static std::vector<u8> s_standard_font_data;
static std::vector<u8> s_fixed_font_data;
static std::vector<u8> s_icon_fa_font_data;
static std::vector<u8> s_icon_pf_font_data;

static float s_window_width;
static float s_window_height;
static Common::Timer s_last_render_time;

// cached copies of WantCaptureKeyboard/Mouse, used to know when to dispatch events
static std::atomic_bool s_imgui_wants_keyboard{false};
static std::atomic_bool s_imgui_wants_mouse{false};

// mapping of host key -> imgui key
static std::unordered_map<u32, ImGuiKey> s_imgui_key_map;

static constexpr float OSD_FADE_IN_TIME = 0.1f;
static constexpr float OSD_FADE_OUT_TIME = 0.4f;

static std::deque<OSDMessage> s_osd_active_messages;
static std::deque<OSDMessage> s_osd_posted_messages;
static std::mutex s_osd_messages_lock;
static bool s_show_osd_messages = true;
static bool s_scale_changed = false;

static std::array<ImGuiManager::SoftwareCursor, InputManager::MAX_SOFTWARE_CURSORS> s_software_cursors = {};

static size_t s_main_viewport_dummy_ptr;
} // namespace ImGuiManager

void ImGuiManager::SetFontPathAndRange(std::string path, std::vector<u16> range)
{
  if (s_font_path == path && s_font_range == range)
    return;

  s_font_path = std::move(path);
  s_font_range = std::move(range);
  s_standard_font_data = {};

  if (ImGui::GetCurrentContext())
  {
    ImGui::EndFrame();

    if (!LoadFontData())
      Panic("Failed to load font data");

    if (!AddImGuiFonts(HasFullscreenFonts()))
      Panic("Failed to create ImGui font text");

    if (!g_gpu_device->UpdateImGuiFontTexture())
      Panic("Failed to recreate font texture after scale+resize");

    NewFrame();
  }
}

void ImGuiManager::SetGlobalScale(float global_scale)
{
  if (s_global_prescale == global_scale)
    return;

  s_global_prescale = global_scale;
  s_scale_changed = true;
}

void ImGuiManager::SetShowOSDMessages(bool enable)
{
  if (s_show_osd_messages == enable)
    return;

  s_show_osd_messages = enable;
  if (!enable)
    Host::ClearOSDMessages();
}

bool ImGuiManager::Initialize(float global_scale, bool show_osd_messages, Error* error)
{
  if (!LoadFontData())
  {
    Error::SetString(error, "Failed to load font data");
    return false;
  }

  s_global_prescale = global_scale;
  s_global_scale = std::max(g_gpu_device->GetWindowScale() * global_scale, 1.0f);
  s_scale_changed = false;
  s_show_osd_messages = show_osd_messages;

  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.BackendFlags |= ImGuiBackendFlags_HasGamepad | ImGuiBackendFlags_RendererHasVtxOffset |
                     ImGuiBackendFlags_PlatformHasViewports | ImGuiBackendFlags_RendererHasViewports;
  io.BackendUsingLegacyKeyArrays = 0;
  io.BackendUsingLegacyNavInputArray = 0;
#ifndef __ANDROID__
  // Android has no keyboard, nor are we using ImGui for any actual user-interactable windows.
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad |
                    ImGuiConfigFlags_NoMouseCursorChange | ImGuiConfigFlags_ViewportsEnable;
#else
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
#endif

  s_window_width = static_cast<float>(g_gpu_device->GetWindowWidth());
  s_window_height = static_cast<float>(g_gpu_device->GetWindowHeight());
  io.DisplayFramebufferScale = ImVec2(1, 1); // We already scale things ourselves, this would double-apply scaling
  io.DisplaySize = ImVec2(s_window_width, s_window_height);

  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    pio.Platform_CreateWindow = PlatformCreateWindow;
    pio.Platform_DestroyWindow = PlatformDestroyWindow;
    pio.Platform_ShowWindow = PlatformShowWindow;
    pio.Platform_SetWindowPos = PlatformSetWindowPos;
    pio.Platform_GetWindowPos = PlatformGetWindowPos;
    pio.Platform_SetWindowSize = PlatformSetWindowSize;
    pio.Platform_GetWindowSize = PlatformGetWindowSize;
    pio.Platform_SetWindowFocus = PlatformSetWindowFocus;
    pio.Platform_GetWindowFocus = PlatformGetWindowFocus;
    pio.Platform_GetWindowMinimized = PlatformGetWindowMinimized;
    pio.Platform_SetWindowTitle = PlatformSetWindowTitle;

    ImGuiPlatformMonitor monitor;
    monitor.MainSize = ImVec2(3840, 2160);
    pio.Monitors.push_back(monitor);

    // TODO: This should point to the host render window.
    pio.Viewports[0]->PlatformHandle = &s_main_viewport_dummy_ptr;
    pio.Viewports[0]->PlatformUserData = &s_main_viewport_dummy_ptr;
  }

  SetKeyMap();
  SetStyle();

  if (!AddImGuiFonts(false) || !g_gpu_device->UpdateImGuiFontTexture())
  {
    Error::SetString(error, "Failed to create ImGui font text");
    ImGui::DestroyContext();
    return false;
  }

  // don't need the font data anymore, save some memory
  ImGui::GetIO().Fonts->ClearTexData();

  NewFrame();

  CreateSoftwareCursorTextures();
  return true;
}

void ImGuiManager::Shutdown()
{
  DestroySoftwareCursorTextures();

  if (ImGui::GetCurrentContext())
    ImGui::DestroyContext();

  s_standard_font = nullptr;
  s_fixed_font = nullptr;
  s_medium_font = nullptr;
  s_large_font = nullptr;
  ImGuiFullscreen::SetFonts(nullptr, nullptr, nullptr);
}

float ImGuiManager::GetWindowWidth()
{
  return s_window_width;
}

float ImGuiManager::GetWindowHeight()
{
  return s_window_height;
}

void ImGuiManager::WindowResized()
{
  const u32 new_width = g_gpu_device ? g_gpu_device->GetWindowWidth() : 0;
  const u32 new_height = g_gpu_device ? g_gpu_device->GetWindowHeight() : 0;

  s_window_width = static_cast<float>(new_width);
  s_window_height = static_cast<float>(new_height);
  ImGui::GetIO().DisplaySize = ImVec2(s_window_width, s_window_height);

  // Scale might have changed as a result of window resize.
  RequestScaleUpdate();
}

void ImGuiManager::RequestScaleUpdate()
{
  // Might need to update the scale.
  s_scale_changed = true;
}

void ImGuiManager::UpdateScale()
{
  const float window_scale = g_gpu_device ? g_gpu_device->GetWindowScale() : 1.0f;
  const float scale = std::max(window_scale * s_global_prescale, 1.0f);

  if ((!HasFullscreenFonts() || !ImGuiFullscreen::UpdateLayoutScale()) && scale == s_global_scale)
    return;

  s_global_scale = scale;

  ImGui::GetStyle() = ImGuiStyle();
  ImGui::GetStyle().WindowMinSize = ImVec2(1.0f, 1.0f);
  SetStyle();
  ImGui::GetStyle().ScaleAllSizes(scale);

  if (!AddImGuiFonts(HasFullscreenFonts()))
    Panic("Failed to create ImGui font text");

  if (!g_gpu_device->UpdateImGuiFontTexture())
    Panic("Failed to recreate font texture after scale+resize");
}

void ImGuiManager::NewFrame()
{
  ImGuiIO& io = ImGui::GetIO();
  io.DeltaTime = static_cast<float>(s_last_render_time.GetTimeSecondsAndReset());

  if (s_scale_changed)
  {
    s_scale_changed = false;
    UpdateScale();
  }

  ImGui::NewFrame();

  // Disable nav input on the implicit (Debug##Default) window. Otherwise we end up requesting keyboard
  // focus when there's nothing there. We use GetCurrentWindowRead() because otherwise it'll make it visible.
  ImGui::GetCurrentWindowRead()->Flags |= ImGuiWindowFlags_NoNavInputs;
  s_imgui_wants_keyboard.store(io.WantCaptureKeyboard, std::memory_order_relaxed);
  s_imgui_wants_mouse.store(io.WantCaptureMouse, std::memory_order_release);
}

void ImGuiManager::SetStyle()
{
  ImGuiStyle& style = ImGui::GetStyle();
  style = ImGuiStyle();
  style.WindowMinSize = ImVec2(1.0f, 1.0f);

  ImVec4* colors = style.Colors;
  colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
  colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
  colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
  colors[ImGuiCol_Border] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);
  colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.12f, 0.14f, 0.65f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
  colors[ImGuiCol_MenuBarBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.18f, 0.22f, 0.25f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.09f, 0.21f, 0.31f, 1.00f);
  colors[ImGuiCol_CheckMark] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.37f, 0.61f, 1.00f, 1.00f);
  colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.33f, 0.38f, 0.46f, 1.00f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.27f, 0.32f, 0.38f, 1.00f);
  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.29f, 0.55f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.33f, 0.38f, 0.46f, 1.00f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.27f, 0.32f, 0.38f, 1.00f);
  colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
  colors[ImGuiCol_SeparatorHovered] = ImVec4(0.33f, 0.38f, 0.46f, 1.00f);
  colors[ImGuiCol_SeparatorActive] = ImVec4(0.27f, 0.32f, 0.38f, 1.00f);
  colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
  colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.33f, 0.38f, 0.46f, 1.00f);
  colors[ImGuiCol_ResizeGripActive] = ImVec4(0.27f, 0.32f, 0.38f, 1.00f);
  colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.38f, 0.46f, 1.00f);
  colors[ImGuiCol_TabActive] = ImVec4(0.27f, 0.32f, 0.38f, 1.00f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
  colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
  colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
  colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
  colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
  colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
  colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
  colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

  style.ScaleAllSizes(s_global_scale);
}

void ImGuiManager::SetKeyMap()
{
  struct KeyMapping
  {
    ImGuiKey index;
    const char* name;
    const char* alt_name;
  };

  static constexpr KeyMapping mapping[] = {{ImGuiKey_LeftArrow, "Left", nullptr},
                                           {ImGuiKey_RightArrow, "Right", nullptr},
                                           {ImGuiKey_UpArrow, "Up", nullptr},
                                           {ImGuiKey_DownArrow, "Down", nullptr},
                                           {ImGuiKey_PageUp, "PageUp", nullptr},
                                           {ImGuiKey_PageDown, "PageDown", nullptr},
                                           {ImGuiKey_Home, "Home", nullptr},
                                           {ImGuiKey_End, "End", nullptr},
                                           {ImGuiKey_Insert, "Insert", nullptr},
                                           {ImGuiKey_Delete, "Delete", nullptr},
                                           {ImGuiKey_Backspace, "Backspace", nullptr},
                                           {ImGuiKey_Space, "Space", nullptr},
                                           {ImGuiKey_Enter, "Return", nullptr},
                                           {ImGuiKey_Escape, "Escape", nullptr},
                                           {ImGuiKey_LeftCtrl, "LeftCtrl", "Ctrl"},
                                           {ImGuiKey_LeftShift, "LeftShift", "Shift"},
                                           {ImGuiKey_LeftAlt, "LeftAlt", "Alt"},
                                           {ImGuiKey_LeftSuper, "LeftSuper", "Super"},
                                           {ImGuiKey_RightCtrl, "RightCtrl", nullptr},
                                           {ImGuiKey_RightShift, "RightShift", nullptr},
                                           {ImGuiKey_RightAlt, "RightAlt", nullptr},
                                           {ImGuiKey_RightSuper, "RightSuper", nullptr},
                                           {ImGuiKey_Menu, "Menu", nullptr},
                                           {ImGuiKey_0, "0", nullptr},
                                           {ImGuiKey_1, "1", nullptr},
                                           {ImGuiKey_2, "2", nullptr},
                                           {ImGuiKey_3, "3", nullptr},
                                           {ImGuiKey_4, "4", nullptr},
                                           {ImGuiKey_5, "5", nullptr},
                                           {ImGuiKey_6, "6", nullptr},
                                           {ImGuiKey_7, "7", nullptr},
                                           {ImGuiKey_8, "8", nullptr},
                                           {ImGuiKey_9, "9", nullptr},
                                           {ImGuiKey_A, "A", nullptr},
                                           {ImGuiKey_B, "B", nullptr},
                                           {ImGuiKey_C, "C", nullptr},
                                           {ImGuiKey_D, "D", nullptr},
                                           {ImGuiKey_E, "E", nullptr},
                                           {ImGuiKey_F, "F", nullptr},
                                           {ImGuiKey_G, "G", nullptr},
                                           {ImGuiKey_H, "H", nullptr},
                                           {ImGuiKey_I, "I", nullptr},
                                           {ImGuiKey_J, "J", nullptr},
                                           {ImGuiKey_K, "K", nullptr},
                                           {ImGuiKey_L, "L", nullptr},
                                           {ImGuiKey_M, "M", nullptr},
                                           {ImGuiKey_N, "N", nullptr},
                                           {ImGuiKey_O, "O", nullptr},
                                           {ImGuiKey_P, "P", nullptr},
                                           {ImGuiKey_Q, "Q", nullptr},
                                           {ImGuiKey_R, "R", nullptr},
                                           {ImGuiKey_S, "S", nullptr},
                                           {ImGuiKey_T, "T", nullptr},
                                           {ImGuiKey_U, "U", nullptr},
                                           {ImGuiKey_V, "V", nullptr},
                                           {ImGuiKey_W, "W", nullptr},
                                           {ImGuiKey_X, "X", nullptr},
                                           {ImGuiKey_Y, "Y", nullptr},
                                           {ImGuiKey_Z, "Z", nullptr},
                                           {ImGuiKey_F1, "F1", nullptr},
                                           {ImGuiKey_F2, "F2", nullptr},
                                           {ImGuiKey_F3, "F3", nullptr},
                                           {ImGuiKey_F4, "F4", nullptr},
                                           {ImGuiKey_F5, "F5", nullptr},
                                           {ImGuiKey_F6, "F6", nullptr},
                                           {ImGuiKey_F7, "F7", nullptr},
                                           {ImGuiKey_F8, "F8", nullptr},
                                           {ImGuiKey_F9, "F9", nullptr},
                                           {ImGuiKey_F10, "F10", nullptr},
                                           {ImGuiKey_F11, "F11", nullptr},
                                           {ImGuiKey_F12, "F12", nullptr},
                                           {ImGuiKey_Apostrophe, "Apostrophe", nullptr},
                                           {ImGuiKey_Comma, "Comma", nullptr},
                                           {ImGuiKey_Minus, "Minus", nullptr},
                                           {ImGuiKey_Period, "Period", nullptr},
                                           {ImGuiKey_Slash, "Slash", nullptr},
                                           {ImGuiKey_Semicolon, "Semicolon", nullptr},
                                           {ImGuiKey_Equal, "Equal", nullptr},
                                           {ImGuiKey_LeftBracket, "BracketLeft", nullptr},
                                           {ImGuiKey_Backslash, "Backslash", nullptr},
                                           {ImGuiKey_RightBracket, "BracketRight", nullptr},
                                           {ImGuiKey_GraveAccent, "QuoteLeft", nullptr},
                                           {ImGuiKey_CapsLock, "CapsLock", nullptr},
                                           {ImGuiKey_ScrollLock, "ScrollLock", nullptr},
                                           {ImGuiKey_NumLock, "NumLock", nullptr},
                                           {ImGuiKey_PrintScreen, "PrintScreen", nullptr},
                                           {ImGuiKey_Pause, "Pause", nullptr},
                                           {ImGuiKey_Keypad0, "Keypad0", nullptr},
                                           {ImGuiKey_Keypad1, "Keypad1", nullptr},
                                           {ImGuiKey_Keypad2, "Keypad2", nullptr},
                                           {ImGuiKey_Keypad3, "Keypad3", nullptr},
                                           {ImGuiKey_Keypad4, "Keypad4", nullptr},
                                           {ImGuiKey_Keypad5, "Keypad5", nullptr},
                                           {ImGuiKey_Keypad6, "Keypad6", nullptr},
                                           {ImGuiKey_Keypad7, "Keypad7", nullptr},
                                           {ImGuiKey_Keypad8, "Keypad8", nullptr},
                                           {ImGuiKey_Keypad9, "Keypad9", nullptr},
                                           {ImGuiKey_KeypadDecimal, "KeypadPeriod", nullptr},
                                           {ImGuiKey_KeypadDivide, "KeypadDivide", nullptr},
                                           {ImGuiKey_KeypadMultiply, "KeypadMultiply", nullptr},
                                           {ImGuiKey_KeypadSubtract, "KeypadMinus", nullptr},
                                           {ImGuiKey_KeypadAdd, "KeypadPlus", nullptr},
                                           {ImGuiKey_KeypadEnter, "KeypadReturn", nullptr},
                                           {ImGuiKey_KeypadEqual, "KeypadEqual", nullptr}};

  s_imgui_key_map.clear();
  for (const KeyMapping& km : mapping)
  {
    std::optional<u32> map(InputManager::ConvertHostKeyboardStringToCode(km.name));
    if (!map.has_value() && km.alt_name)
      map = InputManager::ConvertHostKeyboardStringToCode(km.alt_name);
    if (map.has_value())
      s_imgui_key_map[map.value()] = km.index;
  }
}

bool ImGuiManager::LoadFontData()
{
  if (s_standard_font_data.empty())
  {
    std::optional<std::vector<u8>> font_data = s_font_path.empty() ?
                                                 Host::ReadResourceFile("fonts/Roboto-Regular.ttf", true) :
                                                 FileSystem::ReadBinaryFile(s_font_path.c_str());
    if (!font_data.has_value())
      return false;

    s_standard_font_data = std::move(font_data.value());
  }

  if (s_fixed_font_data.empty())
  {
    std::optional<std::vector<u8>> font_data = Host::ReadResourceFile("fonts/RobotoMono-Medium.ttf", true);
    if (!font_data.has_value())
      return false;

    s_fixed_font_data = std::move(font_data.value());
  }

  if (s_icon_fa_font_data.empty())
  {
    std::optional<std::vector<u8>> font_data = Host::ReadResourceFile("fonts/fa-solid-900.ttf", true);
    if (!font_data.has_value())
      return false;

    s_icon_fa_font_data = std::move(font_data.value());
  }

  if (s_icon_pf_font_data.empty())
  {
    std::optional<std::vector<u8>> font_data = Host::ReadResourceFile("fonts/promptfont.otf", true);
    if (!font_data.has_value())
      return false;

    s_icon_pf_font_data = std::move(font_data.value());
  }

  return true;
}

ImFont* ImGuiManager::AddTextFont(float size)
{
  static const ImWchar default_ranges[] = {
    // Basic Latin + Latin Supplement + Central European diacritics
    0x0020,
    0x017F,

    // Cyrillic + Cyrillic Supplement
    0x0400,
    0x052F,

    // Cyrillic Extended-A
    0x2DE0,
    0x2DFF,

    // Cyrillic Extended-B
    0xA640,
    0xA69F,

    0,
  };

  ImFontConfig cfg;
  cfg.FontDataOwnedByAtlas = false;
  return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(s_standard_font_data.data(),
                                                    static_cast<int>(s_standard_font_data.size()), size, &cfg,
                                                    s_font_range.empty() ? default_ranges : s_font_range.data());
}

ImFont* ImGuiManager::AddFixedFont(float size)
{
  ImFontConfig cfg;
  cfg.FontDataOwnedByAtlas = false;
  return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(s_fixed_font_data.data(),
                                                    static_cast<int>(s_fixed_font_data.size()), size, &cfg, nullptr);
}

bool ImGuiManager::AddIconFonts(float size)
{
  static constexpr ImWchar range_fa[] = {
    0xe086, 0xe086, 0xf002, 0xf002, 0xf005, 0xf005, 0xf007, 0xf007, 0xf00c, 0xf00e, 0xf011, 0xf011, 0xf013, 0xf013,
    0xf017, 0xf017, 0xf019, 0xf019, 0xf01c, 0xf01c, 0xf021, 0xf021, 0xf023, 0xf023, 0xf025, 0xf025, 0xf027, 0xf028,
    0xf02e, 0xf02e, 0xf030, 0xf030, 0xf03a, 0xf03a, 0xf03d, 0xf03d, 0xf049, 0xf04c, 0xf050, 0xf050, 0xf059, 0xf059,
    0xf05e, 0xf05e, 0xf062, 0xf063, 0xf065, 0xf065, 0xf067, 0xf067, 0xf071, 0xf071, 0xf075, 0xf075, 0xf077, 0xf078,
    0xf07b, 0xf07c, 0xf084, 0xf085, 0xf091, 0xf091, 0xf0a0, 0xf0a0, 0xf0ac, 0xf0ad, 0xf0c5, 0xf0c5, 0xf0c7, 0xf0c9,
    0xf0cb, 0xf0cb, 0xf0d0, 0xf0d0, 0xf0dc, 0xf0dc, 0xf0e2, 0xf0e2, 0xf0e7, 0xf0e7, 0xf0eb, 0xf0eb, 0xf0f1, 0xf0f1,
    0xf0f3, 0xf0f3, 0xf0fe, 0xf0fe, 0xf110, 0xf110, 0xf119, 0xf119, 0xf11b, 0xf11c, 0xf140, 0xf140, 0xf144, 0xf144,
    0xf14a, 0xf14a, 0xf15b, 0xf15b, 0xf15d, 0xf15d, 0xf188, 0xf188, 0xf191, 0xf192, 0xf1ab, 0xf1ab, 0xf1dd, 0xf1de,
    0xf1e6, 0xf1e6, 0xf1eb, 0xf1eb, 0xf1f8, 0xf1f8, 0xf1fc, 0xf1fc, 0xf242, 0xf242, 0xf245, 0xf245, 0xf26c, 0xf26c,
    0xf279, 0xf279, 0xf2d0, 0xf2d0, 0xf2db, 0xf2db, 0xf2f2, 0xf2f2, 0xf2f5, 0xf2f5, 0xf3c1, 0xf3c1, 0xf3fd, 0xf3fd,
    0xf410, 0xf410, 0xf466, 0xf466, 0xf500, 0xf500, 0xf51f, 0xf51f, 0xf538, 0xf538, 0xf545, 0xf545, 0xf547, 0xf548,
    0xf552, 0xf552, 0xf57a, 0xf57a, 0xf5a2, 0xf5a2, 0xf5aa, 0xf5aa, 0xf5e7, 0xf5e7, 0xf65d, 0xf65e, 0xf6a9, 0xf6a9,
    0xf6cf, 0xf6cf, 0xf794, 0xf794, 0xf7c2, 0xf7c2, 0xf807, 0xf807, 0xf815, 0xf815, 0xf818, 0xf818, 0xf84c, 0xf84c,
    0xf8cc, 0xf8cc, 0x0,    0x0};
  static constexpr ImWchar range_pf[] = {
    0x2196, 0x2199, 0x219e, 0x21a1, 0x21b0, 0x21b3, 0x21ba, 0x21c3, 0x21c7, 0x21ca, 0x21d0, 0x21d4, 0x21dc, 0x21dd,
    0x21e0, 0x21e3, 0x21ed, 0x21ee, 0x21f7, 0x21f8, 0x21fa, 0x21fb, 0x227a, 0x227d, 0x235e, 0x235e, 0x2360, 0x2361,
    0x2364, 0x2366, 0x23b2, 0x23b4, 0x23f4, 0x23f7, 0x2427, 0x243a, 0x243c, 0x243c, 0x243e, 0x243e, 0x2460, 0x246b,
    0x24f5, 0x24fd, 0x24ff, 0x24ff, 0x278a, 0x278e, 0x27fc, 0x27fc, 0xe001, 0xe001, 0xff21, 0xff3a, 0x0,    0x0};

  {
    ImFontConfig cfg;
    cfg.MergeMode = true;
    cfg.PixelSnapH = true;
    cfg.GlyphMinAdvanceX = size;
    cfg.GlyphMaxAdvanceX = size;
    cfg.FontDataOwnedByAtlas = false;

    if (!ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
          s_icon_fa_font_data.data(), static_cast<int>(s_icon_fa_font_data.size()), size * 0.75f, &cfg, range_fa))
    {
      return false;
    }
  }

  {
    ImFontConfig cfg;
    cfg.MergeMode = true;
    cfg.PixelSnapH = true;
    cfg.GlyphMinAdvanceX = size;
    cfg.GlyphMaxAdvanceX = size;
    cfg.FontDataOwnedByAtlas = false;

    if (!ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
          s_icon_pf_font_data.data(), static_cast<int>(s_icon_pf_font_data.size()), size * 1.2f, &cfg, range_pf))
    {
      return false;
    }
  }

  return true;
}

bool ImGuiManager::AddImGuiFonts(bool fullscreen_fonts)
{
  const float standard_font_size = std::ceil(15.0f * s_global_scale);

  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  s_standard_font = AddTextFont(standard_font_size);
  if (!s_standard_font || !AddIconFonts(standard_font_size))
    return false;

  s_fixed_font = AddFixedFont(standard_font_size);
  if (!s_fixed_font)
    return false;

  if (fullscreen_fonts)
  {
    const float medium_font_size = std::ceil(ImGuiFullscreen::LayoutScale(ImGuiFullscreen::LAYOUT_MEDIUM_FONT_SIZE));
    s_medium_font = AddTextFont(medium_font_size);
    if (!s_medium_font || !AddIconFonts(medium_font_size))
      return false;

    const float large_font_size = std::ceil(ImGuiFullscreen::LayoutScale(ImGuiFullscreen::LAYOUT_LARGE_FONT_SIZE));
    s_large_font = AddTextFont(large_font_size);
    if (!s_large_font || !AddIconFonts(large_font_size))
      return false;
  }
  else
  {
    s_medium_font = nullptr;
    s_large_font = nullptr;
  }

  ImGuiFullscreen::SetFonts(s_standard_font, s_medium_font, s_large_font);

  return io.Fonts->Build();
}

bool ImGuiManager::AddFullscreenFontsIfMissing()
{
  if (HasFullscreenFonts())
    return true;

  // can't do this in the middle of a frame
  ImGui::EndFrame();

  if (!AddImGuiFonts(true))
  {
    Log_ErrorPrint("Failed to lazily allocate fullscreen fonts.");
    AddImGuiFonts(false);
  }

  g_gpu_device->UpdateImGuiFontTexture();
  NewFrame();

  return HasFullscreenFonts();
}

bool ImGuiManager::HasFullscreenFonts()
{
  return (s_medium_font && s_large_font);
}

void Host::AddOSDMessage(std::string message, float duration /*= 2.0f*/)
{
  AddKeyedOSDMessage(std::string(), std::move(message), duration);
}

void Host::AddKeyedOSDMessage(std::string key, std::string message, float duration /* = 2.0f */)
{
  if (!key.empty())
    Log_InfoPrintf("OSD [%s]: %s", key.c_str(), message.c_str());
  else
    Log_InfoPrintf("OSD: %s", message.c_str());

  if (!ImGuiManager::s_show_osd_messages)
    return;

  const Common::Timer::Value current_time = Common::Timer::GetCurrentValue();

  ImGuiManager::OSDMessage msg;
  msg.key = std::move(key);
  msg.text = std::move(message);
  msg.duration = duration;
  msg.start_time = current_time;
  msg.move_time = current_time;
  msg.target_y = -1.0f;
  msg.last_y = -1.0f;

  std::unique_lock<std::mutex> lock(ImGuiManager::s_osd_messages_lock);
  ImGuiManager::s_osd_posted_messages.push_back(std::move(msg));
}

void Host::AddFormattedOSDMessage(float duration, const char* format, ...)
{
  std::va_list ap;
  va_start(ap, format);
  std::string ret = StringUtil::StdStringFromFormatV(format, ap);
  va_end(ap);
  return AddKeyedOSDMessage(std::string(), std::move(ret), duration);
}

void Host::AddIconOSDMessage(std::string key, const char* icon, std::string message, float duration /* = 2.0f */)
{
  return AddKeyedOSDMessage(std::move(key), fmt::format("{}  {}", icon, message), duration);
}

void Host::AddKeyedFormattedOSDMessage(std::string key, float duration, const char* format, ...)
{
  std::va_list ap;
  va_start(ap, format);
  std::string ret = StringUtil::StdStringFromFormatV(format, ap);
  va_end(ap);
  return AddKeyedOSDMessage(std::move(key), std::move(ret), duration);
}

void Host::RemoveKeyedOSDMessage(std::string key)
{
  if (!ImGuiManager::s_show_osd_messages)
    return;

  ImGuiManager::OSDMessage msg = {};
  msg.key = std::move(key);
  msg.duration = 0.0f;

  std::unique_lock<std::mutex> lock(ImGuiManager::s_osd_messages_lock);
  ImGuiManager::s_osd_posted_messages.push_back(std::move(msg));
}

void Host::ClearOSDMessages()
{
  {
    std::unique_lock<std::mutex> lock(ImGuiManager::s_osd_messages_lock);
    ImGuiManager::s_osd_posted_messages.clear();
  }

  ImGuiManager::s_osd_active_messages.clear();
}

void ImGuiManager::AcquirePendingOSDMessages(Common::Timer::Value current_time)
{
  std::atomic_thread_fence(std::memory_order_consume);
  if (s_osd_posted_messages.empty())
    return;

  std::unique_lock lock(s_osd_messages_lock);
  for (;;)
  {
    if (s_osd_posted_messages.empty())
      break;

    OSDMessage& new_msg = s_osd_posted_messages.front();
    std::deque<OSDMessage>::iterator iter;
    if (!new_msg.key.empty() && (iter = std::find_if(s_osd_active_messages.begin(), s_osd_active_messages.end(),
                                                     [&new_msg](const OSDMessage& other) {
                                                       return new_msg.key == other.key;
                                                     })) != s_osd_active_messages.end())
    {
      iter->text = std::move(new_msg.text);
      iter->duration = new_msg.duration;

      // Don't fade it in again
      const float time_passed =
        static_cast<float>(Common::Timer::ConvertValueToSeconds(current_time - iter->start_time));
      iter->start_time = current_time - Common::Timer::ConvertSecondsToValue(std::min(time_passed, OSD_FADE_IN_TIME));
    }
    else
    {
      s_osd_active_messages.push_back(std::move(new_msg));
    }

    s_osd_posted_messages.pop_front();

    static constexpr size_t MAX_ACTIVE_OSD_MESSAGES = 512;
    if (s_osd_active_messages.size() > MAX_ACTIVE_OSD_MESSAGES)
      s_osd_active_messages.pop_front();
  }
}

void ImGuiManager::DrawOSDMessages(Common::Timer::Value current_time)
{
  static constexpr float MOVE_DURATION = 0.5f;

  ImFont* const font = ImGui::GetFont();
  const float scale = s_global_scale;
  const float spacing = std::ceil(5.0f * scale);
  const float margin = std::ceil(10.0f * scale);
  const float padding = std::ceil(8.0f * scale);
  const float rounding = std::ceil(5.0f * scale);
  const float max_width = s_window_width - (margin + padding) * 2.0f;
  float position_x = margin;
  float position_y = margin;

  auto iter = s_osd_active_messages.begin();
  while (iter != s_osd_active_messages.end())
  {
    OSDMessage& msg = *iter;
    const float time_passed = static_cast<float>(Common::Timer::ConvertValueToSeconds(current_time - msg.start_time));
    if (time_passed >= msg.duration)
    {
      iter = s_osd_active_messages.erase(iter);
      continue;
    }

    ++iter;

    u8 opacity;
    if (time_passed < OSD_FADE_IN_TIME)
      opacity = static_cast<u8>((time_passed / OSD_FADE_IN_TIME) * 255.0f);
    else if (time_passed > (msg.duration - OSD_FADE_OUT_TIME))
      opacity = static_cast<u8>(std::min((msg.duration - time_passed) / OSD_FADE_OUT_TIME, 1.0f) * 255.0f);
    else
      opacity = 255;

    const float expected_y = position_y;
    float actual_y = msg.last_y;
    if (msg.target_y != expected_y)
    {
      msg.move_time = current_time;
      msg.target_y = expected_y;
      msg.last_y = (msg.last_y < 0.0f) ? expected_y : msg.last_y;
      actual_y = msg.last_y;
    }
    else if (actual_y != expected_y)
    {
      const float time_since_move =
        static_cast<float>(Common::Timer::ConvertValueToSeconds(current_time - msg.move_time));
      if (time_since_move >= MOVE_DURATION)
      {
        msg.move_time = current_time;
        msg.last_y = msg.target_y;
        actual_y = msg.last_y;
      }
      else
      {
        const float frac = Easing::OutExpo(time_since_move / MOVE_DURATION);
        actual_y = msg.last_y - ((msg.last_y - msg.target_y) * frac);
      }
    }

    if (actual_y >= ImGui::GetIO().DisplaySize.y)
      break;

    const ImVec2 pos(position_x, actual_y);
    const ImVec2 text_size(font->CalcTextSizeA(font->FontSize, max_width, max_width, msg.text.c_str(),
                                               msg.text.c_str() + msg.text.length()));
    const ImVec2 size(text_size.x + padding * 2.0f, text_size.y + padding * 2.0f);
    const ImVec4 text_rect(pos.x + padding, pos.y + padding, pos.x + size.x - padding, pos.y + size.y - padding);

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(0x21, 0x21, 0x21, opacity), rounding);
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(0x48, 0x48, 0x48, opacity), rounding);
    dl->AddText(font, font->FontSize, ImVec2(text_rect.x, text_rect.y), IM_COL32(0xff, 0xff, 0xff, opacity),
                msg.text.c_str(), msg.text.c_str() + msg.text.length(), max_width, &text_rect);
    position_y += size.y + spacing;
  }
}

void ImGuiManager::RenderOSDMessages()
{
  const Common::Timer::Value current_time = Common::Timer::GetCurrentValue();
  AcquirePendingOSDMessages(current_time);
  DrawOSDMessages(current_time);
}

float ImGuiManager::GetGlobalScale()
{
  return s_global_scale;
}

float Host::GetOSDScale()
{
  return ImGuiManager::s_global_scale;
}

ImFont* ImGuiManager::GetStandardFont()
{
  return s_standard_font;
}

ImFont* ImGuiManager::GetFixedFont()
{
  return s_fixed_font;
}

ImFont* ImGuiManager::GetMediumFont()
{
  AddFullscreenFontsIfMissing();
  return s_medium_font;
}

ImFont* ImGuiManager::GetLargeFont()
{
  AddFullscreenFontsIfMissing();
  return s_large_font;
}

bool ImGuiManager::WantsTextInput()
{
  return s_imgui_wants_keyboard.load(std::memory_order_acquire);
}

bool ImGuiManager::WantsMouseInput()
{
  return s_imgui_wants_mouse.load(std::memory_order_acquire);
}

void ImGuiManager::AddTextInput(std::string str)
{
  if (!ImGui::GetCurrentContext())
    return;

  if (!s_imgui_wants_keyboard.load(std::memory_order_acquire))
    return;

  ImGui::GetIO().AddInputCharactersUTF8(str.c_str());
}

void ImGuiManager::UpdateMousePosition(float x, float y)
{
  if (!ImGui::GetCurrentContext())
    return;

  ImGui::GetIO().MousePos = ImVec2(x, y);
  std::atomic_thread_fence(std::memory_order_release);
}

bool ImGuiManager::ProcessPointerButtonEvent(InputBindingKey key, float value)
{
  if (!ImGui::GetCurrentContext() || key.data >= std::size(ImGui::GetIO().MouseDown))
    return false;

  // still update state anyway
  ImGui::GetIO().AddMouseButtonEvent(key.data, value != 0.0f);

  return s_imgui_wants_mouse.load(std::memory_order_acquire);
}

bool ImGuiManager::ProcessPointerAxisEvent(InputBindingKey key, float value)
{
  if (!ImGui::GetCurrentContext() || value == 0.0f || key.data < static_cast<u32>(InputPointerAxis::WheelX))
    return false;

  // still update state anyway
  const bool horizontal = (key.data == static_cast<u32>(InputPointerAxis::WheelX));
  ImGui::GetIO().AddMouseWheelEvent(horizontal ? value : 0.0f, horizontal ? 0.0f : value);

  return s_imgui_wants_mouse.load(std::memory_order_acquire);
}

bool ImGuiManager::ProcessHostKeyEvent(InputBindingKey key, float value)
{
  decltype(s_imgui_key_map)::iterator iter;
  if (!ImGui::GetCurrentContext() || (iter = s_imgui_key_map.find(key.data)) == s_imgui_key_map.end())
    return false;

  // still update state anyway
  ImGui::GetIO().AddKeyEvent(iter->second, value != 0.0);

  return s_imgui_wants_keyboard.load(std::memory_order_acquire);
}

bool ImGuiManager::ProcessGenericInputEvent(GenericInputBinding key, float value)
{
  static constexpr ImGuiKey key_map[] = {
    ImGuiKey_None,             // Unknown,
    ImGuiKey_GamepadDpadUp,    // DPadUp
    ImGuiKey_GamepadDpadRight, // DPadRight
    ImGuiKey_GamepadDpadLeft,  // DPadLeft
    ImGuiKey_GamepadDpadDown,  // DPadDown
    ImGuiKey_None,             // LeftStickUp
    ImGuiKey_None,             // LeftStickRight
    ImGuiKey_None,             // LeftStickDown
    ImGuiKey_None,             // LeftStickLeft
    ImGuiKey_GamepadL3,        // L3
    ImGuiKey_None,             // RightStickUp
    ImGuiKey_None,             // RightStickRight
    ImGuiKey_None,             // RightStickDown
    ImGuiKey_None,             // RightStickLeft
    ImGuiKey_GamepadR3,        // R3
    ImGuiKey_GamepadFaceUp,    // Triangle
    ImGuiKey_GamepadFaceRight, // Circle
    ImGuiKey_GamepadFaceDown,  // Cross
    ImGuiKey_GamepadFaceLeft,  // Square
    ImGuiKey_GamepadBack,      // Select
    ImGuiKey_GamepadStart,     // Start
    ImGuiKey_None,             // System
    ImGuiKey_GamepadL1,        // L1
    ImGuiKey_GamepadL2,        // L2
    ImGuiKey_GamepadR1,        // R1
    ImGuiKey_GamepadL2,        // R2
  };

  if (!ImGui::GetCurrentContext() || !s_imgui_wants_keyboard.load(std::memory_order_acquire))
    return false;

  if (static_cast<u32>(key) >= std::size(key_map) || key_map[static_cast<u32>(key)] == ImGuiKey_None)
    return false;

  ImGui::GetIO().AddKeyAnalogEvent(key_map[static_cast<u32>(key)], (value > 0.0f), value);
  return true;
}

void ImGuiManager::CreateSoftwareCursorTextures()
{
  for (u32 i = 0; i < static_cast<u32>(s_software_cursors.size()); i++)
  {
    if (!s_software_cursors[i].image_path.empty())
      UpdateSoftwareCursorTexture(i);
  }
}

void ImGuiManager::DestroySoftwareCursorTextures()
{
  for (SoftwareCursor& sc : s_software_cursors)
    sc.texture.reset();
}

void ImGuiManager::UpdateSoftwareCursorTexture(u32 index)
{
  SoftwareCursor& sc = s_software_cursors[index];
  if (sc.image_path.empty())
  {
    sc.texture.reset();
    return;
  }

  RGBA8Image image;
  if (!image.LoadFromFile(sc.image_path.c_str()))
  {
    Log_ErrorPrintf("Failed to load software cursor %u image '%s'", index, sc.image_path.c_str());
    return;
  }
  g_gpu_device->RecycleTexture(std::move(sc.texture));
  sc.texture = g_gpu_device->FetchTexture(image.GetWidth(), image.GetHeight(), 1, 1, 1, GPUTexture::Type::Texture,
                                          GPUTexture::Format::RGBA8, image.GetPixels(), image.GetPitch());
  if (!sc.texture)
  {
    Log_ErrorPrintf("Failed to upload %ux%u software cursor %u image '%s'", image.GetWidth(), image.GetHeight(), index,
                    sc.image_path.c_str());
    return;
  }

  sc.extent_x = std::ceil(static_cast<float>(image.GetWidth()) * sc.scale * s_global_scale) / 2.0f;
  sc.extent_y = std::ceil(static_cast<float>(image.GetHeight()) * sc.scale * s_global_scale) / 2.0f;
}

void ImGuiManager::DrawSoftwareCursor(const SoftwareCursor& sc, const std::pair<float, float>& pos)
{
  if (!sc.texture)
    return;

  const ImVec2 min(pos.first - sc.extent_x, pos.second - sc.extent_y);
  const ImVec2 max(pos.first + sc.extent_x, pos.second + sc.extent_y);

  ImDrawList* dl = ImGui::GetForegroundDrawList();

  dl->AddImage(reinterpret_cast<ImTextureID>(sc.texture.get()), min, max, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
               sc.color);
}

void ImGuiManager::RenderSoftwareCursors()
{
  // This one's okay to race, worst that happens is we render the wrong number of cursors for a frame.
  const u32 pointer_count = InputManager::MAX_POINTER_DEVICES;
  for (u32 i = 0; i < pointer_count; i++)
    DrawSoftwareCursor(s_software_cursors[i], InputManager::GetPointerAbsolutePosition(i));

  for (u32 i = InputManager::MAX_POINTER_DEVICES; i < InputManager::MAX_SOFTWARE_CURSORS; i++)
    DrawSoftwareCursor(s_software_cursors[i], s_software_cursors[i].pos);
}

void ImGuiManager::SetSoftwareCursor(u32 index, std::string image_path, float image_scale, u32 multiply_color)
{
  DebugAssert(index < std::size(s_software_cursors));
  SoftwareCursor& sc = s_software_cursors[index];
  sc.color = multiply_color | 0xFF000000;
  if (sc.image_path == image_path && sc.scale == image_scale)
    return;

  const bool is_hiding_or_showing = (image_path.empty() != sc.image_path.empty());
  sc.image_path = std::move(image_path);
  sc.scale = image_scale;
  if (g_gpu_device)
    UpdateSoftwareCursorTexture(index);

  // Hide the system cursor when we activate a software cursor.
  if (is_hiding_or_showing && index == 0)
    InputManager::UpdateHostMouseMode();
}

bool ImGuiManager::HasSoftwareCursor(u32 index)
{
  return (index < s_software_cursors.size() && !s_software_cursors[index].image_path.empty());
}

void ImGuiManager::ClearSoftwareCursor(u32 index)
{
  SetSoftwareCursor(index, std::string(), 0.0f, 0);
}

void ImGuiManager::SetSoftwareCursorPosition(u32 index, float pos_x, float pos_y)
{
  DebugAssert(index >= InputManager::MAX_POINTER_DEVICES);
  SoftwareCursor& sc = s_software_cursors[index];
  sc.pos.first = pos_x;
  sc.pos.second = pos_y;
}

bool ImGuiManager::IsMainViewport(const ImGuiViewport* vp)
{
  return (vp->PlatformHandle == &s_main_viewport_dummy_ptr);
}

void ImGuiManager::PlatformCreateWindow(ImGuiViewport* vp)
{
  Panic("Not implemented");
}

void ImGuiManager::PlatformDestroyWindow(ImGuiViewport* vp)
{
  Panic("Not implemented");
}

void ImGuiManager::PlatformShowWindow(ImGuiViewport* vp)
{
  Panic("Not implemented");
}

void ImGuiManager::PlatformSetWindowPos(ImGuiViewport* vp, ImVec2 pos)
{
  Panic("Not implemented");
}

ImVec2 ImGuiManager::PlatformGetWindowPos(ImGuiViewport* vp)
{
  //Panic("Not implemented");
  return ImVec2(0, 0);
}

void ImGuiManager::PlatformSetWindowSize(ImGuiViewport* vp, ImVec2 size)
{
  Panic("Not implemented");
}

ImVec2 ImGuiManager::PlatformGetWindowSize(ImGuiViewport* vp)
{
  if (IsMainViewport(vp))
    return ImGui::GetIO().DisplaySize;

  Panic("Not implemented");
}

void ImGuiManager::PlatformSetWindowFocus(ImGuiViewport* vp)
{
  Panic("Not implemented");
}

bool ImGuiManager::PlatformGetWindowFocus(ImGuiViewport* vp)
{
  if (IsMainViewport(vp))
    return true;

  Panic("Not implemented");
}

bool ImGuiManager::PlatformGetWindowMinimized(ImGuiViewport* vp)
{
  // Panic("Not implemented");
  return false;
}

void ImGuiManager::PlatformSetWindowTitle(ImGuiViewport* vp, const char* str)
{
  Panic("Not implemented");
}
