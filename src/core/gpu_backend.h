// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_types.h"

#include "util/gpu_texture.h"

#include "common/rectangle.h"

#include <tuple>

class GPUFramebuffer;
class GPUPipeline;

struct Settings;
class StateWrapper;

// DESIGN NOTE: Only static methods should be called on the CPU thread.
// You specifically don't have a global pointer available for this reason.

class GPUBackend
{
public:
  static GPUBackendResetCommand* NewResetCommand();
  static GPUThreadCommand* NewClearDisplayCommand();
  static GPUBackendUpdateDisplayCommand* NewUpdateDisplayCommand();
  static GPUBackendReadVRAMCommand* NewReadVRAMCommand();
  static GPUBackendFillVRAMCommand* NewFillVRAMCommand();
  static GPUBackendUpdateVRAMCommand* NewUpdateVRAMCommand(u32 num_words);
  static GPUBackendCopyVRAMCommand* NewCopyVRAMCommand();
  static GPUBackendSetDrawingAreaCommand* NewSetDrawingAreaCommand();
  static GPUBackendDrawPolygonCommand* NewDrawPolygonCommand(u32 num_vertices);
  static GPUBackendDrawPrecisePolygonCommand* NewDrawPrecisePolygonCommand(u32 num_vertices);
  static GPUBackendDrawSpriteCommand* NewDrawRectangleCommand();
  static GPUBackendDrawLineCommand* NewDrawLineCommand(u32 num_vertices);
  static void PushCommand(GPUThreadCommand* cmd);
  static void PushCommandAndWakeThread(GPUThreadCommand* cmd);
  static void PushCommandAndSync(GPUThreadCommand* cmd, bool spin);

  static bool IsUsingHardwareBackend();

  static std::unique_ptr<GPUBackend> CreateHardwareBackend();
  static std::unique_ptr<GPUBackend> CreateSoftwareBackend();

  static bool BeginQueueFrame();
  static void WaitForOneQueuedFrame();

  static bool RenderScreenshotToBuffer(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect, bool postfx,
                                       std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format);

  static std::tuple<u32, u32> GetLastDisplaySourceSize();

public:
  GPUBackend();
  virtual ~GPUBackend();

  ALWAYS_INLINE const void* GetDisplayTextureHandle() const { return m_display_texture; }
  ALWAYS_INLINE s32 GetDisplayWidth() const { return m_display_width; }
  ALWAYS_INLINE s32 GetDisplayHeight() const { return m_display_height; }
  ALWAYS_INLINE s32 GetDisplayViewWidth() const { return m_display_texture_view_width; }
  ALWAYS_INLINE s32 GetDisplayViewHeight() const { return m_display_texture_view_height; }
  ALWAYS_INLINE float GetDisplayAspectRatio() const { return m_display_aspect_ratio; }
  ALWAYS_INLINE bool HasDisplayTexture() const { return static_cast<bool>(m_display_texture); }

  virtual bool Initialize();
  virtual void Shutdown();

  void HandleCommand(const GPUThreadCommand* cmd);

  /// Draws the current display texture, with any post-processing.
  bool PresentDisplay();

  virtual void UpdateSettings(const Settings& old_settings);

  // Graphics API state reset/restore - call when drawing the UI etc.
  // TODO: replace with "invalidate cached state"
  virtual void RestoreDeviceContext();

protected:
  virtual void Reset(bool clear_vram);
  virtual bool DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display);

  virtual void ReadVRAM(u32 x, u32 y, u32 width, u32 height) = 0;
  virtual void FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color, GPUBackendCommandParameters params);
  virtual void UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, GPUBackendCommandParameters params);
  virtual void CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height,
                        GPUBackendCommandParameters params);

  virtual void DrawPolygon(const GPUBackendDrawPolygonCommand* cmd) = 0;
  virtual void DrawPrecisePolygon(const GPUBackendDrawPrecisePolygonCommand* cmd) = 0;
  virtual void DrawSprite(const GPUBackendDrawSpriteCommand* cmd) = 0;
  virtual void DrawLine(const GPUBackendDrawLineCommand* cmd) = 0;

  virtual void FlushRender() = 0;
  virtual void DrawingAreaChanged() = 0;

  virtual void ClearDisplay() = 0;
  virtual void UpdateDisplay(const GPUBackendUpdateDisplayCommand* cmd) = 0;

  /// TODO: Updates the resolution scale when it's set to automatic.
  // void UpdateResolutionScale();

  /// Helper function for computing the draw rectangle in a larger window.
  Common::Rectangle<s32> CalculateDrawRect(s32 window_width, s32 window_height, bool apply_aspect_ratio = true) const;

  /// Helper function to save current display texture to PNG.
  bool WriteDisplayTextureToFile(std::string filename, bool full_resolution = true, bool apply_aspect_ratio = true,
                                 bool compress_on_thread = false);

  /// Renders the display, optionally with postprocessing to the specified image.
  void HandleRenderScreenshotToBuffer(const GPUThreadRenderScreenshotToBufferCommand* cmd);

  /// Helper function to save screenshot to PNG.
  bool RenderScreenshotToFile(std::string filename, bool internal_resolution = false, bool compress_on_thread = false);

  bool CompileDisplayPipeline();

  void ClearDisplayTexture();
  void SetDisplayTexture(GPUTexture* texture, s32 view_x, s32 view_y, s32 view_width, s32 view_height);
  void SetDisplayParameters(s32 display_width, s32 display_height, s32 active_left, s32 active_top, s32 active_width,
                            s32 active_height, float display_aspect_ratio);

  Common::Rectangle<float> CalculateDrawRect(s32 window_width, s32 window_height, float* out_left_padding,
                                             float* out_top_padding, float* out_scale, float* out_x_scale,
                                             bool apply_aspect_ratio = true) const;

  bool RenderDisplay(GPUTexture* target, const Common::Rectangle<s32>& draw_rect, bool postfx);

  Common::Rectangle<u32> m_drawing_area{};

  s32 m_display_width = 0;
  s32 m_display_height = 0;
  s32 m_display_active_left = 0;
  s32 m_display_active_top = 0;
  s32 m_display_active_width = 0;
  s32 m_display_active_height = 0;
  float m_display_aspect_ratio = 1.0f;

  std::unique_ptr<GPUPipeline> m_display_pipeline;
  GPUTexture* m_display_texture = nullptr;
  s32 m_display_texture_view_x = 0;
  s32 m_display_texture_view_y = 0;
  s32 m_display_texture_view_width = 0;
  s32 m_display_texture_view_height = 0;
};
