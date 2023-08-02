// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "common/timer.h"
#include "common/window_info.h"
#include "gl/context.h"
#include "gl/loader.h"
#include "gl/program.h"
#include "gl/stream_buffer.h"
#include "gl/texture.h"
#include "gpu_device.h"
#include "postprocessing_chain.h"
#include <memory>

class OpenGLGPUDevice final : public GPUDevice
{
public:
  OpenGLGPUDevice();
  ~OpenGLGPUDevice();

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

  void SetVSync(bool enabled) override;

  bool Render(bool skip_present) override;

  bool SetGPUTimingEnabled(bool enabled) override;
  float GetAndResetAccumulatedGPUTime() override;

  ALWAYS_INLINE GL::Context* GetGLContext() const { return m_gl_context.get(); }
  ALWAYS_INLINE bool UsePBOForUploads() const { return m_use_pbo_for_pixels; }
  ALWAYS_INLINE bool UseGLES3DrawPath() const { return m_use_gles2_draw_path; }
  ALWAYS_INLINE std::vector<u8>& GetTextureRepackBuffer() { return m_texture_repack_buffer; }

  GL::StreamBuffer* GetTextureStreamBuffer();

protected:
  static constexpr u8 NUM_TIMESTAMP_QUERIES = 3;

  const char* GetGLSLVersionString() const;
  std::string GetGLSLVersionHeader() const;

  bool CreateResources() override;
  void DestroyResources() override;

  void SetSwapInterval();

  void RenderDisplay();
  void RenderImGui();
  void RenderSoftwareCursor();

  void RenderDisplay(s32 left, s32 bottom, s32 width, s32 height, GL::Texture* texture, s32 texture_view_x,
                     s32 texture_view_y, s32 texture_view_width, s32 texture_view_height, bool linear_filter);
  void RenderSoftwareCursor(s32 left, s32 bottom, s32 width, s32 height, GPUTexture* texture_handle);

  void CreateTimestampQueries();
  void DestroyTimestampQueries();
  void PopTimestampQuery();
  void KickTimestampQuery();

  std::unique_ptr<GL::Context> m_gl_context;

  GL::Program m_display_program;
  GL::Program m_cursor_program;
  GLuint m_display_vao = 0;
  GLuint m_display_nearest_sampler = 0;
  GLuint m_display_linear_sampler = 0;
  GLuint m_display_border_sampler = 0;
  GLuint m_uniform_buffer_alignment = 1;

  std::unique_ptr<GL::StreamBuffer> m_texture_stream_buffer;
  std::vector<u8> m_texture_repack_buffer;
  u32 m_texture_stream_buffer_offset = 0;

  std::array<GLuint, NUM_TIMESTAMP_QUERIES> m_timestamp_queries = {};
  float m_accumulated_gpu_time = 0.0f;
  u8 m_read_timestamp_query = 0;
  u8 m_write_timestamp_query = 0;
  u8 m_waiting_timestamp_queries = 0;
  bool m_timestamp_query_started = false;

  bool m_use_gles2_draw_path = false;
  bool m_use_pbo_for_pixels = false;
};
