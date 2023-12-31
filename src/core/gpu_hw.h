// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu.h"
#include "texture_replacements.h"

#include "util/gpu_device.h"

#include "common/dimensional_array.h"
#include "common/heap_array.h"

#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

class GPU_SW_Backend;
struct GPUBackendCommand;
struct GPUBackendDrawCommand;

class GPU_HW final : public GPU
{
public:
  using Rect = Common::Rectangle<u32>; // TODO: Could be u16.

  // TODO: Shouldn't be public
  struct Source;
  template<typename T>
  struct TList;
  template<typename T>
  struct TListNode;
  struct HashCacheEntry;

  struct SourceKey
  {
    u8 page;
    GPUTextureMode mode;
    GPUTexturePaletteReg palette;

    SourceKey() = default;
    ALWAYS_INLINE constexpr SourceKey(u8 page_, GPUTexturePaletteReg palette_, GPUTextureMode mode_)
      : page(page_), mode(mode_), palette(palette_)
    {
    }
    ALWAYS_INLINE constexpr SourceKey(const SourceKey& k) : page(k.page), mode(k.mode), palette(k.palette) {}

    ALWAYS_INLINE bool HasPalette() const { return (mode < GPUTextureMode::Direct16Bit); }

    ALWAYS_INLINE SourceKey& operator=(const SourceKey& k)
    {
      page = k.page;
      mode = k.mode;
      palette.bits = k.palette.bits;
      return *this;
    }

    ALWAYS_INLINE bool operator==(const SourceKey& k) const { return (std::memcmp(&k, this, sizeof(SourceKey)) == 0); }
    ALWAYS_INLINE bool operator!=(const SourceKey& k) const { return (std::memcmp(&k, this, sizeof(SourceKey)) != 0); }
  };
  static_assert(sizeof(SourceKey) == 4);

  enum class BatchRenderMode : u8
  {
    TransparencyDisabled,
    TransparentAndOpaque,
    OnlyOpaque,
    OnlyTransparent,
    ShaderBlend
  };

  GPU_HW();
  ~GPU_HW() override;

  const Threading::Thread* GetSWThread() const override;
  bool IsHardwareRenderer() const override;

  bool Initialize() override;
  void Reset(bool clear_vram) override;
  bool DoState(StateWrapper& sw, GPUTexture** host_texture, bool update_display) override;

  void RestoreDeviceContext() override;

  void UpdateSettings(const Settings& old_settings) override;
  void UpdateResolutionScale() override final;
  std::tuple<u32, u32> GetEffectiveDisplayResolution(bool scaled = true) override final;
  std::tuple<u32, u32> GetFullDisplayResolution(bool scaled = true) override final;

  void UpdateDisplay() override;

private:
  enum : u32
  {
    MAX_BATCH_VERTEX_COUNTER_IDS = 65536 - 2,
    MAX_VERTICES_FOR_RECTANGLE = 6 * (((MAX_PRIMITIVE_WIDTH + (TEXTURE_PAGE_WIDTH - 1)) / TEXTURE_PAGE_WIDTH) + 1u) *
                                 (((MAX_PRIMITIVE_HEIGHT + (TEXTURE_PAGE_HEIGHT - 1)) / TEXTURE_PAGE_HEIGHT) + 1u)
  };

  static_assert(GPUDevice::MIN_TEXEL_BUFFER_ELEMENTS >= (VRAM_WIDTH * VRAM_HEIGHT));

  struct BatchVertex
  {
    float x;
    float y;
    float z;
    float w;
    u32 color;
    u32 texpage;
    u16 u; // 16-bit texcoords are needed for 256 extent rectangles
    u16 v;
    u32 uv_limits;

    void Set(float x_, float y_, float z_, float w_, u32 color_, u32 texpage_, u16 packed_texcoord, u32 uv_limits_);
    void Set(float x_, float y_, float z_, float w_, u32 color_, u32 texpage_, u16 u_, u16 v_, u32 uv_limits_);
    static u32 PackUVLimits(u32 min_u, u32 max_u, u32 min_v, u32 max_v);
    void SetUVLimits(u32 min_u, u32 max_u, u32 min_v, u32 max_v);
  };

  struct BatchConfig
  {
    GPUTextureMode texture_mode = GPUTextureMode::Disabled;
    GPUTransparencyMode transparency_mode = GPUTransparencyMode::Disabled;
    bool dithering = false;
    bool interlacing = false;
    bool set_mask_while_drawing = false;
    bool check_mask_before_draw = false;
    bool use_depth_buffer = false;

    bool use_texture_cache = false;
    SourceKey texture_cache_key = {};

    // Returns the render mode for this batch.
    BatchRenderMode GetRenderMode() const;
  };

  struct BatchUBOData
  {
    u32 u_texture_window_and[2];
    u32 u_texture_window_or[2];
    float u_src_alpha_factor;
    float u_dst_alpha_factor;
    u32 u_interlaced_displayed_field;
    u32 u_set_mask_while_drawing;
  };

  struct RendererStats
  {
    u32 num_batches;
    u32 num_vram_read_texture_updates;
    u32 num_uniform_buffer_updates;
  };

  /// Returns true if a depth buffer should be created.
  bool NeedsDepthBuffer() const;

  bool CreateBuffers();
  void ClearFramebuffer();
  void DestroyBuffers();

  bool CompilePipelines();
  void DestroyPipelines();

  void LoadVertices();

  void PrintSettingsToLog();
  void CheckSettings();

  void UpdateVRAMReadTexture();
  void UpdateDepthBufferFromMaskBit();
  void ClearDepthBuffer();
  void SetScissor();
  void SetVRAMRenderTarget();
  void MapGPUBuffer(u32 required_vertices, u32 required_indices);
  void UnmapGPUBuffer(u32 used_vertices, u32 used_indices);
  void DrawBatchVertices(BatchRenderMode render_mode, u32 num_indices, u32 base_index, u32 base_vertex,
                         const Source* texture);

  u32 CalculateResolutionScale() const;
  GPUDownsampleMode GetDownsampleMode(u32 resolution_scale) const;

  bool IsUsingMultisampling() const;
  bool IsUsingDownsampling() const;

  void SetFullVRAMDirtyRectangle();
  void AddVRAMDirtyRectangle(u32 left, u32 top, u32 right, u32 bottom);

  void AddWrittenRectangle(const Rect& rect);
  void AddDrawnRectangle(u32 left, u32 top, u32 right, u32 bottom);
  void CheckForTexPageOverlap(u32 texpage, u32 min_u, u32 min_v, u32 max_u, u32 max_v);

  bool IsFlushed() const;
  void EnsureVertexBufferSpace(u32 required_vertices, u32 required_indices);
  void EnsureVertexBufferSpaceForCurrentCommand();
  void ResetBatchVertexDepth();

  /// Returns the value to be written to the depth buffer for the current operation for mask bit emulation.
  float GetCurrentNormalizedVertexDepth() const;

  /// Returns if the draw needs to be broken into opaque/transparent passes.
  bool NeedsTwoPassRendering() const;

  /// Returns true if the draw is going to use shader blending/framebuffer fetch.
  bool NeedsShaderBlending(GPUTransparencyMode transparency, bool check_mask) const;

  void FillBackendCommandParameters(GPUBackendCommand* cmd) const;
  void FillDrawCommand(GPUBackendDrawCommand* cmd, GPURenderCommand rc) const;
  void UpdateSoftwareRenderer(bool copy_vram_from_hw);

  void FillVRAM(u32 x, u32 y, u32 width, u32 height, u32 color) override;
  void ReadVRAM(u32 x, u32 y, u32 width, u32 height) override;
  void UpdateVRAM(u32 x, u32 y, u32 width, u32 height, const void* data, bool set_mask, bool check_mask) override;
  void CopyVRAM(u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height) override;
  void DispatchRenderCommand() override;
  void FlushRender() override;
  void DrawRendererStats() override;

  bool BlitVRAMReplacementTexture(const TextureReplacementTexture* tex, u32 dst_x, u32 dst_y, u32 width, u32 height);

  /// Expands a line into two triangles.
  void DrawLine(float x0, float y0, u32 col0, float x1, float y1, u32 col1, float depth);

  /// Handles quads with flipped texture coordinate directions.
  void HandleFlippedQuadTextureCoordinates(BatchVertex* vertices);
  void ExpandLineTriangles(BatchVertex* vertices, u32 base_vertex);

  /// Computes polygon U/V boundaries.
  void ComputePolygonUVLimits(u32 texpage, BatchVertex* vertices, u32 num_vertices);

  /// Sets the depth test flag for PGXP depth buffering.
  void SetBatchDepthBuffer(bool enabled);
  void CheckForDepthClear(const BatchVertex* vertices, u32 num_vertices);

  /// Returns the number of mipmap levels used for adaptive smoothing.
  u32 GetAdaptiveDownsamplingMipLevels() const;

  void DownsampleFramebuffer(GPUTexture* source, u32 left, u32 top, u32 width, u32 height);
  void DownsampleFramebufferAdaptive(GPUTexture* source, u32 left, u32 top, u32 width, u32 height);
  void DownsampleFramebufferBoxFilter(GPUTexture* source, u32 left, u32 top, u32 width, u32 height);

  std::unique_ptr<GPUTexture> m_vram_texture;
  std::unique_ptr<GPUTexture> m_vram_depth_texture;
  std::unique_ptr<GPUTexture> m_vram_read_texture;
  std::unique_ptr<GPUTexture> m_vram_readback_texture;
  std::unique_ptr<GPUDownloadTexture> m_vram_readback_download_texture;
  std::unique_ptr<GPUTexture> m_vram_replacement_texture;

  std::unique_ptr<GPUTextureBuffer> m_vram_upload_buffer;
  std::unique_ptr<GPUTexture> m_vram_write_texture;

  std::unique_ptr<GPU_SW_Backend> m_sw_renderer;

  BatchVertex* m_batch_vertex_ptr = nullptr;
  u16* m_batch_index_ptr = nullptr;
  u32 m_batch_base_vertex = 0;
  u32 m_batch_base_index = 0;
  u16 m_batch_vertex_count = 0;
  u16 m_batch_index_count = 0;
  u16 m_batch_vertex_space = 0;
  u16 m_batch_index_space = 0;
  s32 m_current_depth = 0;
  float m_last_depth_z = 1.0f;

  u8 m_resolution_scale = 1;
  u8 m_multisamples = 1;

  bool m_supports_dual_source_blend : 1 = false;
  bool m_supports_framebuffer_fetch : 1 = false;
  bool m_per_sample_shading : 1 = false;
  bool m_scaled_dithering : 1 = false;
  bool m_disable_color_perspective : 1 = false;

  GPUTextureFilter m_texture_filtering = GPUTextureFilter::Nearest;
  GPULineDetectMode m_line_detect_mode = GPULineDetectMode::Disabled;
  GPUDownsampleMode m_downsample_mode = GPUDownsampleMode::Disabled;
  GPUWireframeMode m_wireframe_mode = GPUWireframeMode::Disabled;
  bool m_true_color : 1 = true;
  bool m_debanding : 1 = false;
  bool m_clamp_uvs : 1 = false;
  bool m_compute_uv_range : 1 = false;
  bool m_pgxp_depth_buffer : 1 = false;
  bool m_allow_shader_blend : 1 = false;
  bool m_prefer_shader_blend : 1 = false;

  BatchConfig m_batch;

  // Changed state
  s8 m_texpage_drawn_page = -1;
  bool m_batch_ubo_dirty = true;
  BatchUBOData m_batch_ubo_data = {};

  // Bounding box of VRAM area that the GPU has drawn into.
  Rect m_vram_dirty_rect;
  Rect m_current_draw_rect;
  Rect m_current_uv_rect;

  std::unique_ptr<GPUPipeline> m_wireframe_pipeline;

  // [wrapped][interlaced]
  DimensionalArray<std::unique_ptr<GPUPipeline>, 2, 2> m_vram_fill_pipelines{};

  // [depth_test]
  std::array<std::unique_ptr<GPUPipeline>, 2> m_vram_write_pipelines{};
  std::array<std::unique_ptr<GPUPipeline>, 2> m_vram_copy_pipelines{};

  std::unique_ptr<GPUPipeline> m_vram_readback_pipeline;
  std::unique_ptr<GPUPipeline> m_vram_update_depth_pipeline;
  std::unique_ptr<GPUPipeline> m_vram_write_replacement_pipeline;

  std::array<std::unique_ptr<GPUPipeline>, 2> m_vram_extract_pipeline; // [24bit]
  std::unique_ptr<GPUTexture> m_vram_extract_texture;

  std::unique_ptr<GPUTexture> m_downsample_texture;
  std::unique_ptr<GPUPipeline> m_downsample_first_pass_pipeline;
  std::unique_ptr<GPUPipeline> m_downsample_mid_pass_pipeline;
  std::unique_ptr<GPUPipeline> m_downsample_blur_pass_pipeline;
  std::unique_ptr<GPUPipeline> m_downsample_composite_pass_pipeline;
  std::unique_ptr<GPUSampler> m_downsample_lod_sampler;
  std::unique_ptr<GPUSampler> m_downsample_composite_sampler;
  u32 m_downsample_scale_or_levels = 0;

  //////////////////////////////////////////////////////////////////////////
  // Texture Cache
  //////////////////////////////////////////////////////////////////////////
  // TODO: UNDO-PUBLIC
public:
  // TODO: Should be u32 on ARM64/ARM32.
  using HashType = u64;

  static constexpr u32 VRAM_PAGE_WIDTH = 64;
  static constexpr u32 VRAM_PAGE_HEIGHT = 256;
  static constexpr u32 VRAM_PAGES_WIDE = VRAM_WIDTH / VRAM_PAGE_WIDTH;
  static constexpr u32 VRAM_PAGES_HIGH = VRAM_HEIGHT / VRAM_PAGE_HEIGHT;
  static constexpr u32 VRAM_PAGE_X_MASK = 0xf;  // 16 pages wide
  static constexpr u32 VRAM_PAGE_Y_MASK = 0x10; // 2 pages high
  static constexpr u32 NUM_PAGES = VRAM_PAGES_WIDE * VRAM_PAGES_HIGH;
  static_assert(NUM_PAGES == 32);

  /// 4 pages in C16 mode, 2+4 pages in P8 mode, 1+1 pages in P4 mode.
  static constexpr u32 MAX_PAGE_REFS_PER_SOURCE = 6;

  static constexpr u32 MAX_PAGE_REFS_PER_WRITE = 32;

  ALWAYS_INLINE static constexpr u32 PageIndex(u32 px, u32 py) { return ((py * VRAM_PAGES_WIDE) + px); }
  ALWAYS_INLINE static constexpr Rect PageRect(u32 px, u32 py)
  {
    return Rect(px * VRAM_PAGE_WIDTH, py * VRAM_PAGE_HEIGHT, (px + 1) * VRAM_PAGE_WIDTH, (py + 1) * VRAM_PAGE_HEIGHT);
  }
  ALWAYS_INLINE static constexpr Rect PageRect(u32 pn)
  {
    // TODO: Put page rects in a LUT instead?
    return PageRect(pn % VRAM_PAGES_WIDE, pn / VRAM_PAGES_WIDE);
  }

  ALWAYS_INLINE static constexpr u32 VRAMCoordinateToPage(u32 x, u32 y)
  {
    return PageIndex(x / VRAM_PAGE_WIDTH, y / VRAM_PAGE_HEIGHT);
  }

  ALWAYS_INLINE static constexpr u32 PageStartX(u32 pn)
  {
    return (pn % GPU_HW::VRAM_PAGES_WIDE) * GPU_HW::VRAM_PAGE_WIDTH;
  }

  ALWAYS_INLINE static constexpr u32 PageStartY(u32 pn)
  {
    return (pn / GPU_HW::VRAM_PAGES_WIDE) * GPU_HW::VRAM_PAGE_HEIGHT;
  }

  ALWAYS_INLINE static constexpr u32 PageWidthForMode(GPUTextureMode mode)
  {
    return TEXTURE_PAGE_WIDTH >> ((mode < GPUTextureMode::Direct16Bit) ? (2 - static_cast<u8>(mode)) : 0);
  }

  ALWAYS_INLINE static constexpr u32 VRAMWidthForMode(GPUTextureMode mode, u32 width)
  {
    switch (mode)
    {
      case GPUTextureMode::Palette4Bit:
        return (width + 3) / 4;
      case GPUTextureMode::Palette8Bit:
        return (width + 1) / 2;
      default:
        return width;
    }
  }

  ALWAYS_INLINE static constexpr u32 TextureWidthForMode(GPUTextureMode mode, u32 vram_width)
  {
    return vram_width << ((mode < GPUTextureMode::Direct16Bit) ? (2 - static_cast<u8>(mode)) : 0);
  }

  ALWAYS_INLINE static constexpr u32 TexturePageCountForMode(GPUTextureMode mode)
  {
    return ((mode < GPUTextureMode::Direct16Bit) ? (1 + static_cast<u8>(mode)) : 4);
  }

  ALWAYS_INLINE static constexpr u32 PalettePageCountForMode(GPUTextureMode mode)
  {
    return (mode == GPUTextureMode::Palette4Bit) ? 1 : 4;
  }

  ALWAYS_INLINE static bool DrawModeHasPalette(GPUTextureMode mode) { return (mode < GPUTextureMode::Direct16Bit); }

  ALWAYS_INLINE static u32 PalettePageNumber(GPUTexturePaletteReg reg)
  {
    return VRAMCoordinateToPage(reg.GetXBase(), reg.GetYBase());
  }

  ALWAYS_INLINE static Rect GetTextureRect(u32 pn, GPUTextureMode mode)
  {
    // TODO: Wrong doesn't handle wrapping
    return Rect::FromExtents(PageStartX(pn), PageStartY(pn), PageWidthForMode(mode), VRAM_PAGE_HEIGHT);
  }

  ALWAYS_INLINE static Rect GetPaletteRect(GPUTexturePaletteReg palette, GPUTextureMode mode)
  {
    // TODO: Wrong doesn't handle wrapping
    return Rect::FromExtents(palette.GetXBase(), palette.GetYBase(), palette.GetWidth(mode), 1);
  }

  template<typename T>
  struct TList
  {
    TListNode<T>* head;
    TListNode<T>* tail;
  };

  template<typename T>
  struct TListNode
  {
    // why inside itself? because we have 3 lists
    T* ref;
    TList<T>* list;
    TListNode<T>* prev;
    TListNode<T>* next;
  };

  // TODO: Pool objects
  struct Source
  {
    SourceKey key;
    u32 num_page_refs;
    GPUTexture* texture;
    HashCacheEntry* from_hash_cache;
    Rect texture_rect;
    Rect palette_rect;

    std::array<TListNode<Source>, MAX_PAGE_REFS_PER_SOURCE> page_refs;
  };

  struct VRAMWrite
  {
    Rect rect;
    HashType hash;

    u32 num_page_refs;
    std::array<TListNode<VRAMWrite>, MAX_PAGE_REFS_PER_WRITE> page_refs;
  };

  struct PageEntry
  {
    TList<Source> sources;
    TList<VRAMWrite> writes;
    Rect draw_rect; // NOTE: In global VRAM space.
    bool is_drawn = false;
  };

  struct HashCacheKey
  {
    HashType texture_hash;
    HashType palette_hash;
    HashType mode;

    ALWAYS_INLINE bool operator==(const HashCacheKey& k) const
    {
      return (std::memcmp(&k, this, sizeof(HashCacheKey)) == 0);
    }
    ALWAYS_INLINE bool operator!=(const HashCacheKey& k) const
    {
      return (std::memcmp(&k, this, sizeof(HashCacheKey)) != 0);
    }
  };
  struct HashCacheKeyHash
  {
    size_t operator()(const HashCacheKey& k) const;
  };

  struct HashCacheEntry
  {
    std::unique_ptr<GPUTexture> texture;
    u32 ref_count;
    u32 age;
  };

  struct DumpedTextureKey
  {
    HashType tex_hash;
    HashType pal_hash;
    u32 width, height;
    GPUTextureMode mode;
    u8 pad[7];

    ALWAYS_INLINE bool operator==(const DumpedTextureKey& k) const
    {
      return (std::memcmp(&k, this, sizeof(DumpedTextureKey)) == 0);
    }
    ALWAYS_INLINE bool operator!=(const DumpedTextureKey& k) const
    {
      return (std::memcmp(&k, this, sizeof(DumpedTextureKey)) != 0);
    }
  };
  struct DumpedTextureKeyHash
  {
    size_t operator()(const DumpedTextureKey& k) const;
  };

private:
  const Source* LookupSource(SourceKey key);

  bool IsPageDrawn(u32 page_index) const;
  bool IsPageDrawn(u32 page_index, const Rect& rect) const;
  bool IsRectDrawn(const Rect& rect) const;

  void InvalidateTextureCache();
  void InvalidatePageSources(u32 pn);
  void InvalidatePageSources(u32 pn, const Rect& rc);

  void AgeSources();
  void AgeHashCache();

  using HashCache = std::unordered_map<HashCacheKey, HashCacheEntry, HashCacheKeyHash>;

  template<typename F>
  void LoopRectPages(u32 left, u32 top, u32 right, u32 bottom, const F& f) const;
  template<typename F>
  void LoopRectPages(const Rect& rc, const F& f) const;
  template<typename F>
  void LoopXWrappedPages(u32 page, u32 num_pages, const F& f) const;
  template<typename F>
  void LoopPages(u32 x, u32 y, u32 width, u32 height, const F& f);

  const Source* CreateSource(SourceKey key);

  HashCacheEntry* LookupHashCache(SourceKey key, HashType tex_hash, HashType pal_hash);
  void RemoveFromHashCache(HashCache::iterator it);

  static HashType HashPage(u8 page, GPUTextureMode mode);
  static HashType HashPalette(GPUTexturePaletteReg palette, GPUTextureMode mode);
  static HashType HashRect(const Rect& rc);

  void TrackVRAMWrite(const Rect& rect);
  void RemoveVRAMWrite(VRAMWrite* entry);
  void DumpVRAMWrite(const VRAMWrite& it, HashType pal_hash, GPUTextureMode mode, GPUTexturePaletteReg palette);
  void DumpSourceVRAMWrites(const Source* source, const Rect& uv_rect);

  HashCache m_hash_cache;

  std::array<PageEntry, NUM_PAGES> m_pages = {};

  /// List of candidates for purging when the hash cache gets too large.
  std::vector<std::pair<HashCache::iterator, s32>> s_hash_cache_purge_list;
  std::unordered_set<DumpedTextureKey, DumpedTextureKeyHash> s_dumped_textures;

  //////////////////////////////////////////////////////////////////////////
  // Pipeline Storage
  //////////////////////////////////////////////////////////////////////////

  // [depth_test][transparency_mode][render_mode][texture_mode][dithering][interlacing][check_mask]
  DimensionalArray<std::unique_ptr<GPUPipeline>, 2, 2, 2, 9, 5, 5, 2> m_batch_pipelines{};
};
