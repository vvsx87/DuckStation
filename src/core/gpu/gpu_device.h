// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

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
  GPUFramebuffer() = default;
  virtual ~GPUFramebuffer() = default;

  virtual void SetDebugName(const std::string_view& name) = 0;
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

  struct Config
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

  GPUSampler() = default;
  virtual ~GPUSampler() = default;

  virtual void SetDebugName(const std::string_view& name) = 0;
};

class GPUShader
{
public:
  enum class Stage
  {
    Vertex,
    Pixel,
    Compute
  };

  GPUShader(Stage stage) : m_stage(stage) {}
  virtual ~GPUShader() = default;

  ALWAYS_INLINE Stage GetStage() const { return m_stage; }

  virtual void SetDebugName(const std::string_view& name) = 0;

protected:
  Stage m_stage;
};

class GPUPipeline
{
public:
  enum class Layout : u8
  {
    // 128 byte UBO via push constants, 1 texture.
    SingleTexture,

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
    enum class Semantic : u8
    {
      Position,
      Texcoord,
      Color,

      MaxCount
    };

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

    BitField<u32, Semantic, 0, 3> semantic;
    BitField<u32, u8, 4, 8> semantic_index;
    BitField<u32, Type, 12, 4> type;
    BitField<u32, u8, 16, 3> components;
    BitField<u32, u8, 19, 8> offset;
    u32 key;

    // clang-format off
    ALWAYS_INLINE VertexAttribute& operator=(const VertexAttribute& rhs) { key = rhs.key; return *this; }
    ALWAYS_INLINE bool operator==(const VertexAttribute& rhs) const { return key == rhs.key; }
    ALWAYS_INLINE bool operator!=(const VertexAttribute& rhs) const { return key != rhs.key; }
    ALWAYS_INLINE bool operator<(const VertexAttribute& rhs) const { return key < rhs.key; }
    // clang-format on

    static constexpr VertexAttribute Make(Semantic semantic, u8 semantic_index, Type type, u8 components, u8 offset)
    {
      VertexAttribute ret = {};
#if 0
      ret.semantic = semantic;
      ret.semantic_index = semantic_index;
      ret.type = type;
      ret.components = components;
      ret.offset = offset;
#else
      // Nasty :/ can't access an inactive element of a union here..
      ret.key = (static_cast<u32>(semantic) & 0x7) | ((static_cast<u32>(semantic_index) & 0xff) << 4) |
                ((static_cast<u32>(type) & 0xf) << 12) | ((static_cast<u32>(components) & 0x7) << 16) |
                ((static_cast<u32>(offset) & 0xff) << 19);
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

  struct DepthState
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

  struct BlendState
  {
    BitField<u32, bool, 0, 1> enable;
    BitField<u32, BlendFunc, 1, 4> src_blend;
    BitField<u32, BlendFunc, 5, 4> src_alpha_blend;
    BitField<u32, BlendFunc, 9, 4> dst_blend;
    BitField<u32, BlendFunc, 13, 4> dst_alpha_blend;
    BitField<u32, BlendOp, 17, 3> blend_op;
    BitField<u32, BlendOp, 20, 3> alpha_blend_op;
    BitField<u32, bool, 24, 1> write_r;
    BitField<u32, bool, 25, 1> write_g;
    BitField<u32, bool, 26, 1> write_b;
    BitField<u32, bool, 27, 1> write_a;
    BitField<u32, u8, 24, 4> write_mask;
    u32 key;

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

  GPUPipeline() = default;
  virtual ~GPUPipeline() = default;

  virtual void SetDebugName(const std::string_view& name) = 0;
};

class GPUDevice
{
public:
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

  ALWAYS_INLINE const WindowInfo& GetWindowInfo() const { return m_window_info; }
  ALWAYS_INLINE s32 GetWindowWidth() const { return static_cast<s32>(m_window_info.surface_width); }
  ALWAYS_INLINE s32 GetWindowHeight() const { return static_cast<s32>(m_window_info.surface_height); }
  ALWAYS_INLINE float GetWindowScale() const { return m_window_info.surface_scale; }

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
  virtual bool SetupDevice() = 0;
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

  virtual bool SetPostProcessingChain(const std::string_view& config) = 0;

  /// Call when the window size changes externally to recreate any resources.
  virtual void ResizeWindow(s32 new_window_width, s32 new_window_height) = 0;

  /// Vertex/index buffer abstraction.
  virtual void MapVertexBuffer(u32 vertex_size, u32 vertex_count, void** map_ptr, u32* map_space, u32* map_base_vertex);
  virtual void UnmapVertexBuffer(u32 used_vertex_count);
  virtual void MapIndexBuffer(u32 index_count, u16** map_ptr, u32* map_space, u32* map_base_index);
  virtual void UnmapIndexBuffer(u32 used_index_count);

  void UploadVertexBuffer(const void* vertices, u32 vertex_size, u32 vertex_count, u32* base_vertex);
  void UploadIndexBuffer(const u16* indices, u32 index_count, u32* base_index);

  /// Uniform buffer abstraction.
  virtual void PushUniformBuffer(const void* data, u32 data_size);

  /// Drawing setup abstraction.
  virtual void SetFramebuffer(GPUFramebuffer* fb);
  virtual void SetPipeline(GPUPipeline* pipeline);
  virtual void SetTextureSampler(u32 slot, GPUTexture* texture, GPUSampler* sampler);
  virtual void SetViewport(u32 x, u32 y, u32 width, u32 height);
  virtual void SetScissor(u32 x, u32 y, u32 width, u32 height);
  void SetViewportAndScissor(u32 x, u32 y, u32 width, u32 height);

  // Drawing abstraction.
  virtual void Draw(u32 base_vertex, u32 vertex_count);
  virtual void DrawIndexed(u32 base_index, u32 index_count, u32 base_vertex);

  /// Creates an abstracted RGBA8 texture. If dynamic, the texture can be updated with UpdateTexture() below.
  virtual std::unique_ptr<GPUTexture> CreateTexture(u32 width, u32 height, u32 layers, u32 levels, u32 samples,
                                                    GPUTexture::Type type, GPUTexture::Format format,
                                                    const void* data = nullptr, u32 data_stride = 0,
                                                    bool dynamic = false) = 0;
  virtual std::unique_ptr<GPUSampler> CreateSampler(const GPUSampler::Config& config);

  virtual bool DownloadTexture(GPUTexture* texture, u32 x, u32 y, u32 width, u32 height, void* out_data,
                               u32 out_data_stride) = 0;
  virtual void CopyTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level, GPUTexture* src,
                                 u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width, u32 height);
  virtual void ResolveTextureRegion(GPUTexture* dst, u32 dst_x, u32 dst_y, u32 dst_layer, u32 dst_level,
                                    GPUTexture* src, u32 src_x, u32 src_y, u32 src_layer, u32 src_level, u32 width,
                                    u32 height);

  /// Framebuffer abstraction.
  virtual std::unique_ptr<GPUFramebuffer> CreateFramebuffer(GPUTexture* rt, u32 rt_layer, u32 rt_level, GPUTexture* ds,
                                                            u32 ds_layer, u32 ds_level);

  /// Shader abstraction.
  virtual std::unique_ptr<GPUShader> CreateShaderFromBinary(GPUShader::Stage stage, gsl::span<const u8> data);
  virtual std::unique_ptr<GPUShader> CreateShaderFromSource(GPUShader::Stage stage, const std::string_view& source,
                                                            std::vector<u8>* out_binary = nullptr);
  virtual std::unique_ptr<GPUPipeline> CreatePipeline(const GPUPipeline::GraphicsConfig& config);

  /// Returns false if the window was completely occluded.
  virtual bool Render(bool skip_present) = 0;

  /// Renders the display with postprocessing to the specified image.
  virtual bool RenderScreenshot(u32 width, u32 height, const Common::Rectangle<s32>& draw_rect,
                                std::vector<u32>* out_pixels, u32* out_stride, GPUTexture::Format* out_format) = 0;

  ALWAYS_INLINE bool IsVsyncEnabled() const { return m_vsync_enabled; }
  virtual void SetVSync(bool enabled) = 0;

  bool UpdateImGuiFontTexture();
  bool UsesLowerLeftOrigin() const;
  void SetDisplayMaxFPS(float max_fps);
  bool ShouldSkipDisplayingFrame();
  void ThrottlePresentation();

  void ClearDisplayTexture()
  {
    m_display_texture = nullptr;
    m_display_texture_view_x = 0;
    m_display_texture_view_y = 0;
    m_display_texture_view_width = 0;
    m_display_texture_view_height = 0;
    m_display_changed = true;
  }

  void SetDisplayTexture(GPUTexture* texture, s32 view_x, s32 view_y, s32 view_width, s32 view_height)
  {
    m_display_texture = texture;
    m_display_texture_view_x = view_x;
    m_display_texture_view_y = view_y;
    m_display_texture_view_width = view_width;
    m_display_texture_view_height = view_height;
    m_display_changed = true;
  }

  void SetDisplayTextureRect(s32 view_x, s32 view_y, s32 view_width, s32 view_height)
  {
    m_display_texture_view_x = view_x;
    m_display_texture_view_y = view_y;
    m_display_texture_view_width = view_width;
    m_display_texture_view_height = view_height;
    m_display_changed = true;
  }

  void SetDisplayParameters(s32 display_width, s32 display_height, s32 active_left, s32 active_top, s32 active_width,
                            s32 active_height, float display_aspect_ratio)
  {
    m_display_width = display_width;
    m_display_height = display_height;
    m_display_active_left = active_left;
    m_display_active_top = active_top;
    m_display_active_width = active_width;
    m_display_active_height = active_height;
    m_display_aspect_ratio = display_aspect_ratio;
    m_display_changed = true;
  }

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
  ALWAYS_INLINE bool HasSoftwareCursor() const { return static_cast<bool>(m_cursor_texture); }
  ALWAYS_INLINE bool HasDisplayTexture() const { return (m_display_texture != nullptr); }

  bool IsUsingLinearFiltering() const;

  void CalculateDrawRect(s32 window_width, s32 window_height, float* out_left, float* out_top, float* out_width,
                         float* out_height, float* out_left_padding, float* out_top_padding, float* out_scale,
                         float* out_x_scale, bool apply_aspect_ratio = true) const;

  std::tuple<s32, s32, s32, s32> CalculateSoftwareCursorDrawRect() const;
  std::tuple<s32, s32, s32, s32> CalculateSoftwareCursorDrawRect(s32 cursor_x, s32 cursor_y) const;

  bool CreateImGuiResources();
  void DestroyImGuiResources();

  WindowInfo m_window_info;

  std::unique_ptr<GPUSampler> m_point_sampler;
  std::unique_ptr<GPUSampler> m_linear_sampler;

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

  GPUTexture* m_display_texture = nullptr;
  s32 m_display_texture_view_x = 0;
  s32 m_display_texture_view_y = 0;
  s32 m_display_texture_view_width = 0;
  s32 m_display_texture_view_height = 0;

  std::unique_ptr<GPUPipeline> m_imgui_pipeline;
  std::unique_ptr<GPUTexture> m_imgui_font_texture;

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
