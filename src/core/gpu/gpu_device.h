// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_shader_cache.h"
#include "gpu_texture.h"

#include "common/bitfield.h"
#include "common/rectangle.h"
#include "common/types.h"
#include "common/window_info.h"

#include "gsl/span"

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

enum class RenderAPI : u32
{
  None,
  D3D11,
  D3D12,
  Vulkan,
  OpenGL,
  OpenGLES
};

class GPUFramebuffer
{
public:
  GPUFramebuffer(GPUTexture* rt, GPUTexture* ds, u32 width, u32 height);
  virtual ~GPUFramebuffer();

  ALWAYS_INLINE GPUTexture* GetRT() const { return m_rt; }
  ALWAYS_INLINE GPUTexture* GetDS() const { return m_ds; }

  ALWAYS_INLINE u32 GetWidth() const { return m_width; }
  ALWAYS_INLINE u32 GetHeight() const { return m_height; }

  virtual void SetDebugName(const std::string_view& name) = 0;

protected:
  GPUTexture* m_rt;
  GPUTexture* m_ds;
  u32 m_width;
  u32 m_height;
};

class GPUSampler
{
public:
  enum class Filter
  {
    Nearest,
    Linear,

    MaxCount
  };

  enum class AddressMode
  {
    Repeat,
    ClampToEdge,
    ClampToBorder,

    MaxCount
  };

  union Config
  {
    BitField<u64, Filter, 0, 1> min_filter;
    BitField<u64, Filter, 1, 1> mag_filter;
    BitField<u64, Filter, 2, 1> mip_filter;
    BitField<u64, AddressMode, 3, 2> address_u;
    BitField<u64, AddressMode, 5, 2> address_v;
    BitField<u64, AddressMode, 7, 2> address_w;
    BitField<u64, u8, 9, 5> anisotropy;
    BitField<u64, u8, 14, 4> min_lod;
    BitField<u64, u8, 18, 4> max_lod;
    BitField<u64, u32, 32, 32> border_color;
    u64 key;
  };

  GPUSampler();
  virtual ~GPUSampler();

  virtual void SetDebugName(const std::string_view& name) = 0;
};

enum class GPUShaderStage : u8
{
  Vertex,
  Fragment,
  Compute
};

class GPUShader
{
public:
  GPUShader(GPUShaderStage stage);
  virtual ~GPUShader();

  static const char* GetStageName(GPUShaderStage stage);

  ALWAYS_INLINE GPUShaderStage GetStage() const { return m_stage; }

  virtual void SetDebugName(const std::string_view& name) = 0;

protected:
  GPUShaderStage m_stage;
};

class GPUPipeline
{
public:
  enum class Layout : u8
  {
    // 128 byte UBO via push constants, 1 texture.
    SingleTexture,

    // 1 streamed UBO, 1 texture in PS.
    HWBatch,

    MaxCount
  };

  enum class Primitive : u8
  {
    Points,
    Lines,
    Triangles,
    TriangleStrips,

    MaxCount
  };

  union VertexAttribute
  {
    static constexpr u32 MaxAttributes = 16;

    enum class Type : u8
    {
      Float,
      UInt8,
      SInt8,
      UNorm8,
      UInt16,
      SInt16,
      UNorm16,
      UInt32,
      SInt32,

      MaxCount
    };

    BitField<u32, u8, 0, 4> index;
    BitField<u32, Type, 4, 4> type;
    BitField<u32, u8, 8, 3> components;
    BitField<u32, u16, 16, 16> offset;
    u32 key;

    // clang-format off
    ALWAYS_INLINE VertexAttribute& operator=(const VertexAttribute& rhs) { key = rhs.key; return *this; }
    ALWAYS_INLINE bool operator==(const VertexAttribute& rhs) const { return key == rhs.key; }
    ALWAYS_INLINE bool operator!=(const VertexAttribute& rhs) const { return key != rhs.key; }
    ALWAYS_INLINE bool operator<(const VertexAttribute& rhs) const { return key < rhs.key; }
    // clang-format on

    static constexpr VertexAttribute Make(u8 index, Type type, u8 components, u8 offset)
    {
      VertexAttribute ret = {};
#if 0
      ret.index = index;
      ret.type = type;
      ret.components = components;
      ret.offset = offset;
#else
      // Nasty :/ can't access an inactive element of a union here..
      ret.key = (static_cast<u32>(index) & 0xf) | ((static_cast<u32>(type) & 0xf) << 4) |
                ((static_cast<u32>(components) & 0x7) << 8) | ((static_cast<u32>(offset) & 0xffff) << 16);
#endif
      return ret;
    }
  };

  struct InputLayout
  {
    gsl::span<const VertexAttribute> vertex_attributes;
    u32 vertex_stride;

    bool operator==(const InputLayout& rhs) const;
    bool operator!=(const InputLayout& rhs) const;
  };

  struct InputLayoutHash
  {
    size_t operator()(const InputLayout& il) const;
  };

  enum class CullMode : u8
  {
    None,
    Front,
    Back,

    MaxCount
  };

  enum class DepthFunc : u8
  {
    Never,
    Always,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,

    MaxCount
  };

  enum class BlendFunc : u8
  {
    Zero,
    One,
    SrcColor,
    InvSrcColor,
    DstColor,
    InvDstColor,
    SrcAlpha,
    InvSrcAlpha,
    SrcAlpha1,
    InvSrcAlpha1,
    DstAlpha,
    InvDstAlpha,
    ConstantColor,
    InvConstantColor,

    MaxCount
  };

  enum class BlendOp : u8
  {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,

    MaxCount
  };

  union RasterizationState
  {
    BitField<u8, CullMode, 0, 2> cull_mode;
    u8 key;

    // clang-format off
    ALWAYS_INLINE RasterizationState& operator=(const RasterizationState& rhs) { key = rhs.key; return *this; }
    ALWAYS_INLINE bool operator==(const RasterizationState& rhs) const { return key == rhs.key; }
    ALWAYS_INLINE bool operator!=(const RasterizationState& rhs) const { return key != rhs.key; }
    ALWAYS_INLINE bool operator<(const RasterizationState& rhs) const { return key < rhs.key; }
    // clang-format on

    static RasterizationState GetNoCullState();
  };

  union DepthState
  {
    BitField<u8, DepthFunc, 0, 3> depth_test;
    BitField<u8, bool, 4, 1> depth_write;
    u8 key;

    // clang-format off
    ALWAYS_INLINE DepthState& operator=(const DepthState& rhs) { key = rhs.key; return *this; }
    ALWAYS_INLINE bool operator==(const DepthState& rhs) const { return key == rhs.key; }
    ALWAYS_INLINE bool operator!=(const DepthState& rhs) const { return key != rhs.key; }
    ALWAYS_INLINE bool operator<(const DepthState& rhs) const { return key < rhs.key; }
    // clang-format on

    static DepthState GetNoTestsState();
    static DepthState GetAlwaysWriteState();
  };

  union BlendState
  {
    BitField<u64, bool, 0, 1> enable;
    BitField<u64, BlendFunc, 1, 4> src_blend;
    BitField<u64, BlendFunc, 5, 4> src_alpha_blend;
    BitField<u64, BlendFunc, 9, 4> dst_blend;
    BitField<u64, BlendFunc, 13, 4> dst_alpha_blend;
    BitField<u64, BlendOp, 17, 3> blend_op;
    BitField<u64, BlendOp, 20, 3> alpha_blend_op;
    BitField<u64, bool, 24, 1> write_r;
    BitField<u64, bool, 25, 1> write_g;
    BitField<u64, bool, 26, 1> write_b;
    BitField<u64, bool, 27, 1> write_a;
    BitField<u64, u8, 24, 4> write_mask;
    BitField<u64, u32, 32, 32> constant;
    u64 key;

    // clang-format off
    ALWAYS_INLINE BlendState& operator=(const BlendState& rhs) { key = rhs.key; return *this; }
    ALWAYS_INLINE bool operator==(const BlendState& rhs) const { return key == rhs.key; }
    ALWAYS_INLINE bool operator!=(const BlendState& rhs) const { return key != rhs.key; }
    ALWAYS_INLINE bool operator<(const BlendState& rhs) const { return key < rhs.key; }
    // clang-format on

    static BlendState GetNoBlendingState();
    static BlendState GetAlphaBlendingState();
  };

  struct GraphicsConfig
  {
    Layout layout;

    Primitive primitive;
    InputLayout input_layout;

    RasterizationState rasterization;
    DepthState depth;
    BlendState blend;

    const GPUShader* vertex_shader;
    const GPUShader* pixel_shader;

    GPUTexture::Format color_format;
    GPUTexture::Format depth_format;
    u32 samples;
    bool per_sample_shading;
  };

  GPUPipeline();
  virtual ~GPUPipeline();

  virtual void SetDebugName(const std::string_view& name) = 0;
};

class GPUTextureBuffer
{
public:
  enum class Format
  {
    R16UI,

    MaxCount
  };

  GPUTextureBuffer(Format format, u32 size_in_elements);
  virtual ~GPUTextureBuffer();

  static u32 GetElementSize(Format format);

  ALWAYS_INLINE Format GetFormat() const { return m_format; }
  ALWAYS_INLINE u32 GetSizeInElements() const { return m_size_in_elements; }
  ALWAYS_INLINE u32 GetSizeInBytes() const { return m_size_in_elements * GetElementSize(m_format); }
  ALWAYS_INLINE u32 GetCurrentPosition() const { return m_current_position; }

  virtual void* Map(u32 required_elements) = 0;
  virtual void Unmap(u32 used_elements) = 0;

protected:
  Format m_format;
  u32 m_size_in_elements;
  u32 m_current_position;
};

class GPUDevice
{
public:
  // TODO: drop virtuals
  using DrawIndex = u16;

  struct Features
  {
    bool dual_source_blend : 1;
    bool per_sample_shading : 1;
    bool mipmapped_render_targets : 1;
    bool noperspective_interpolation : 1;
    bool supports_texture_buffers : 1;
    bool texture_buffers_emulated_with_ssbo : 1;
  };

  struct AdapterAndModeList
  {
    std::vector<std::string> adapter_names;
    std::vector<std::string> fullscreen_modes;
  };

  virtual ~GPUDevice();

  /// Returns the default/preferred API for the system.
  static RenderAPI GetPreferredAPI();

  /// Parses a fullscreen mode into its components (width * height @ refresh hz)
  static bool ParseFullscreenMode(const std::string_view& mode, u32* width, u32* height, float* refresh_rate);

  /// Converts a fullscreen mode to a string.
  static std::string GetFullscreenModeString(u32 width, u32 height, float refresh_rate);

  ALWAYS_INLINE const Features& GetFeatures() const { return m_features; }
  ALWAYS_INLINE u32 GetMaxTextureSize() const { return m_max_texture_size; }
  ALWAYS_INLINE u32 GetMaxMultisamples() const { return m_max_multisamples; }

  ALWAYS_INLINE const WindowInfo& GetWindowInfo() const { return m_window_info; }
  ALWAYS_INLINE s32 GetWindowWidth() const { return static_cast<s32>(m_window_info.surface_width); }
  ALWAYS_INLINE s32 GetWindowHeight() const { return static_cast<s32>(m_window_info.surface_height); }
  ALWAYS_INLINE float GetWindowScale() const { return m_window_info.surface_scale; }

  ALWAYS_INLINE GPUSampler* GetLinearSampler() const { return m_linear_sampler.get(); }
  ALWAYS_INLINE GPUSampler* GetPointSampler() const { return m_point_sampler.get(); }

  // Position is relative to the top-left corner of the window.
  ALWAYS_INLINE s32 GetMousePositionX() const { return m_mouse_position_x; }
  ALWAYS_INLINE s32 GetMousePositionY() const { return m_mouse_position_y; }
  ALWAYS_INLINE void SetMousePosition(s32 x, s32 y)
  {
    m_mouse_position_x = x;
    m_mouse_position_y = y;
  }

  ALWAYS_INLINE const void* GetDisplayTextureHandle() const { return m_display_texture; }
  ALWAYS_INLINE s32 GetDisplayWidth() const { return m_display_width; }
  ALWAYS_INLINE s32 GetDisplayHeight() const { return m_display_height; }
  ALWAYS_INLINE float GetDisplayAspectRatio() const { return m_display_aspect_ratio; }
  ALWAYS_INLINE bool IsGPUTimingEnabled() const { return m_gpu_timing_enabled; }

  virtual RenderAPI GetRenderAPI() const = 0;
  virtual void* GetDevice() const = 0;
  virtual void* GetContext() const = 0;

  virtual bool HasDevice() const = 0;
  virtual bool HasSurface() const = 0;

  virtual bool CreateDevice(const WindowInfo& wi, bool vsync) = 0;
  virtual bool SetupDevice();
  virtual bool MakeCurrent() = 0;
  virtual bool DoneCurrent() = 0;
  virtual void DestroySurface() = 0;
  virtual bool ChangeWindow(const WindowInfo& wi) = 0;
  virtual bool SupportsFullscreen() const = 0;
  virtual bool IsFullscreen() = 0;
  virtual bool SetFullscreen(bool fullscreen, u32 width, u32 height, float refresh_rate) = 0;
  virtual AdapterAndModeList GetAdapterAndModeList() = 0;
  virtual bool CreateResources();
  virtual void DestroyResources();

  virtual bool SetPostProcessingChain(const std::string_view& config);

  virtual std::string GetShaderCacheBaseName(const std::string_view& type, bool debug) const;

  /// Call when the window size changes externally to recreate any resources.
  virtual void ResizeWindow(s32 new_window_width, s32 new_window_height) = 0;

  /// Creates an abstracted RGBA8 texture. If dynamic, the texture can be updated with UpdateTexture() below.
  virtual std::unique_ptr<GPUTexture> CreateTexture(u32 width, u32 height, u32 layers, u32 levels, u32 samples,
                                                    GPUTexture::Type type, GPUTexture::Format format,
                                                    const void* data = nullptr, u32 data_stride = 0,
                                                    bool dynamic = false) = 0;
  virtual std::unique_ptr<GPUSampler> CreateSampler(const GPUSampler::Config& config);
  virtual std::unique_ptr<GPUTextureBuffer> CreateTextureBuffer(GPUTextureBuffer::Format format, u32 size_in_elements);

  virtual bool DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                               u32 out_data_stride) = 0;
  virtual void CopyTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                                 u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height);
  virtual void ResolveTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level,
                                    GPUTexture* src, u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width,
                                    u32 height);
  void ClearRenderTarget(GPUTexture* t, u32 c);
  void ClearDepth(GPUTexture* t, float d);
  void InvalidateRenderTarget(GPUTexture* t);

  /// Framebuffer abstraction.
  virtual std::unique_ptr<GPUFramebuffer> CreateFramebuffer(GPUTexture* rt = nullptr, u32 rt_layer = 0,
                                                            u32 rt_level = 0, GPUTexture* ds = nullptr,
                                                            u32 ds_layer = 0, u32 ds_level = 0);

  /// Shader abstraction.
  // TODO:  entry point? source format?
  std::unique_ptr<GPUShader> CreateShader(GPUShaderStage stage, const std::string_view& source);
  virtual std::unique_ptr<GPUPipeline> CreatePipeline(const GPUPipeline::GraphicsConfig& config);

  /// Debug messaging.
  virtual void PushDebugGroup(const char* fmt, ...);
  virtual void PopDebugGroup();
  virtual void InsertDebugMessage(const char* fmt, ...);

  /// Vertex/index buffer abstraction.
  virtual void MapVertexBuffer(u32 vertex_size, u32 vertex_count, void** map_ptr, u32* map_space, u32* map_base_vertex);
  virtual void UnmapVertexBuffer(u32 vertex_size, u32 vertex_count);
  virtual void MapIndexBuffer(u32 index_count, DrawIndex** map_ptr, u32* map_space, u32* map_base_index);
  virtual void UnmapIndexBuffer(u32 used_size);

  void UploadVertexBuffer(const void* vertices, u32 vertex_size, u32 vertex_count, u32* base_vertex);
  void UploadIndexBuffer(const DrawIndex* indices, u32 index_count, u32* base_index);

  /// Uniform buffer abstraction.
  virtual void PushUniformBuffer(const void* data, u32 data_size);
  virtual void* MapUniformBuffer(u32 size);
  virtual void UnmapUniformBuffer(u32 size);
  void UploadUniformBuffer(const void* data, u32 data_size);

  /// Drawing setup abstraction.
  virtual void SetFramebuffer(GPUFramebuffer* fb);
  virtual void SetPipeline(GPUPipeline* pipeline);
  virtual void SetTextureSampler(u32 slot, GPUTexture* texture, GPUSampler* sampler);
  virtual void SetTextureBuffer(u32 slot, GPUTextureBuffer* buffer);
  virtual void SetViewport(s32 x, s32 y, s32 width, s32 height);
  virtual void SetScissor(s32 x, s32 y, s32 width, s32 height);
  void SetViewportAndScissor(s32 x, s32 y, s32 width, s32 height);

  // Drawing abstraction.
  virtual void Draw(u32 vertex_count, u32 base_vertex);
  virtual void DrawIndexed(u32 index_count, u32 base_index, u32 base_vertex);

  /// Returns false if the window was completely occluded.
  virtual bool Render(bool skip_present) = 0;

  /// Renders the display with postprocessing to the specified image.
  bool RenderScreenshot(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect, std::vector<u32>* out_pixels,
                        u32* out_stride, GPUTexture::Format* out_format);

  ALWAYS_INLINE bool IsVsyncEnabled() const { return m_vsync_enabled; }
  virtual void SetVSync(bool enabled) = 0;

  bool UpdateImGuiFontTexture();
  bool UsesLowerLeftOrigin() const;
  void SetDisplayMaxFPS(float max_fps);
  bool ShouldSkipDisplayingFrame();
  void ThrottlePresentation();

  void ClearDisplayTexture();
  void SetDisplayTexture(GPUTexture* texture, s32 view_x, s32 view_y, s32 view_width, s32 view_height);
  void SetDisplayTextureRect(s32 view_x, s32 view_y, s32 view_width, s32 view_height);
  void SetDisplayParameters(s32 display_width, s32 display_height, s32 active_left, s32 active_top, s32 active_width,
                            s32 active_height, float display_aspect_ratio);

  virtual bool SupportsTextureFormat(GPUTexture::Format format) const = 0;

  virtual bool GetHostRefreshRate(float* refresh_rate);

  /// Enables/disables GPU frame timing.
  virtual bool SetGPUTimingEnabled(bool enabled);

  /// Returns the amount of GPU time utilized since the last time this method was called.
  virtual float GetAndResetAccumulatedGPUTime();

  /// Sets the software cursor to the specified texture. Ownership of the texture is transferred.
  void SetSoftwareCursor(std::unique_ptr<GPUTexture> texture, float scale = 1.0f);

  /// Sets the software cursor to the specified image.
  bool SetSoftwareCursor(const void* pixels, u32 width, u32 height, u32 stride, float scale = 1.0f);

  /// Sets the software cursor to the specified path (png image).
  bool SetSoftwareCursor(const char* path, float scale = 1.0f);

  /// Disables the software cursor.
  void ClearSoftwareCursor();

  /// Helper function for computing the draw rectangle in a larger window.
  std::tuple<s32, s32, s32, s32> CalculateDrawRect(s32 window_width, s32 window_height,
                                                   bool apply_aspect_ratio = true) const;

  /// Helper function for converting window coordinates to display coordinates.
  std::tuple<float, float> ConvertWindowCoordinatesToDisplayCoordinates(s32 window_x, s32 window_y, s32 window_width,
                                                                        s32 window_height) const;

  /// Helper function to save texture data to a PNG. If flip_y is set, the image will be flipped aka OpenGL.
  bool WriteTextureToFile(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, std::string filename,
                          bool clear_alpha = true, bool flip_y = false, u32 resize_width = 0, u32 resize_height = 0,
                          bool compress_on_thread = false);

  /// Helper function to save current display texture to PNG.
  bool WriteDisplayTextureToFile(std::string filename, bool full_resolution = true, bool apply_aspect_ratio = true,
                                 bool compress_on_thread = false);

  /// Helper function to save current display texture to a buffer.
  bool WriteDisplayTextureToBuffer(std::vector<u32>* buffer, u32 resize_width = 0, u32 resize_height = 0,
                                   bool clear_alpha = true);

  /// Helper function to save screenshot to PNG.
  bool WriteScreenshotToFile(std::string filename, bool internal_resolution = false, bool compress_on_thread = false);

protected:
  virtual std::unique_ptr<GPUShader> CreateShaderFromBinary(GPUShaderStage stage, gsl::span<const u8> data);
  virtual std::unique_ptr<GPUShader> CreateShaderFromSource(GPUShaderStage stage, const std::string_view& source,
                                                            std::vector<u8>* out_binary = nullptr);

  ALWAYS_INLINE bool HasSoftwareCursor() const { return static_cast<bool>(m_cursor_texture); }
  ALWAYS_INLINE bool HasDisplayTexture() const { return (m_display_texture != nullptr); }

  bool IsUsingLinearFiltering() const;

  void CalculateDrawRect(s32 window_width, s32 window_height, float* out_left, float* out_top, float* out_width,
                         float* out_height, float* out_left_padding, float* out_top_padding, float* out_scale,
                         float* out_x_scale, bool apply_aspect_ratio = true) const;

  std::tuple<s32, s32, s32, s32> CalculateSoftwareCursorDrawRect() const;
  std::tuple<s32, s32, s32, s32> CalculateSoftwareCursorDrawRect(s32 cursor_x, s32 cursor_y) const;

  void RenderImGui();

  void RenderDisplay();
  void RenderSoftwareCursor();

  void RenderDisplay(s32 left, s32 top, s32 width, s32 height, GPUTexture* texture, s32 texture_view_x,
                     s32 texture_view_y, s32 texture_view_width, s32 texture_view_height, bool linear_filter);
  void RenderSoftwareCursor(s32 left, s32 top, s32 width, s32 height, GPUTexture* texture);

  Features m_features = {};
  u32 m_max_texture_size = 0;
  u32 m_max_multisamples = 0;

  WindowInfo m_window_info;

  GPUShaderCache m_shader_cache;

  std::unique_ptr<GPUSampler> m_point_sampler;
  std::unique_ptr<GPUSampler> m_linear_sampler;
  std::unique_ptr<GPUSampler> m_border_sampler;

  u64 m_last_frame_displayed_time = 0;

  s32 m_mouse_position_x = 0;
  s32 m_mouse_position_y = 0;

  s32 m_display_width = 0;
  s32 m_display_height = 0;
  s32 m_display_active_left = 0;
  s32 m_display_active_top = 0;
  s32 m_display_active_width = 0;
  s32 m_display_active_height = 0;
  float m_display_aspect_ratio = 1.0f;
  float m_display_frame_interval = 0.0f;

  std::unique_ptr<GPUPipeline> m_display_pipeline;
  GPUTexture* m_display_texture = nullptr;
  s32 m_display_texture_view_x = 0;
  s32 m_display_texture_view_y = 0;
  s32 m_display_texture_view_width = 0;
  s32 m_display_texture_view_height = 0;

  std::unique_ptr<GPUPipeline> m_imgui_pipeline;
  std::unique_ptr<GPUTexture> m_imgui_font_texture;

  std::unique_ptr<GPUPipeline> m_cursor_pipeline;
  std::unique_ptr<GPUTexture> m_cursor_texture;
  float m_cursor_texture_scale = 1.0f;

  bool m_display_changed = false;
  bool m_gpu_timing_enabled = false;
  bool m_vsync_enabled = false;
};

/// Returns a pointer to the current host display abstraction. Assumes AcquireHostDisplay() has been called.
extern std::unique_ptr<GPUDevice> g_host_display;

namespace Host {
std::unique_ptr<GPUDevice> CreateDisplayForAPI(RenderAPI api);

/// Creates the host display. This may create a new window. The API used depends on the current configuration.
bool AcquireHostDisplay(RenderAPI api);

/// Destroys the host display. This may close the display window.
void ReleaseHostDisplay();

/// Returns false if the window was completely occluded. If frame_skip is set, the frame won't be
/// displayed, but the GPU command queue will still be flushed.
// bool BeginPresentFrame(bool frame_skip);

/// Presents the frame to the display, and renders OSD elements.
// void EndPresentFrame();

/// Provided by the host; renders the display.
void RenderDisplay(bool skip_present);
void InvalidateDisplay();
} // namespace Host

// Macros for debug messages.
#ifdef _DEBUG
struct GLAutoPop
{
  GLAutoPop(int dummy) {}
  ~GLAutoPop() { g_host_display->PopDebugGroup(); }
};

#define GL_SCOPE(...) GLAutoPop gl_auto_pop((g_host_display->PushDebugGroup(__VA_ARGS__), 0))
#define GL_PUSH(...) g_host_display->PushDebugGroup(__VA_ARGS__)
#define GL_POP() g_host_display->PopDebugGroup()
#define GL_INS(...) g_host_display->InsertDebugMessage(__VA_ARGS__)
#define GL_OBJECT_NAME(obj, ...) (obj)->SetDebugName(StringUtil::StdStringFromFormat(__VA_ARGS__))
#else
#define GL_SCOPE(...) (void)0
#define GL_PUSH(...) (void)0
#define GL_POP() (void)0
#define GL_INS(...) (void)0
#define GL_OBJECT_NAME(...) (void)0
#endif
