// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_texture.h"

#include "common/bitfield.h"
#include "common/types.h"

#include "gsl/span"

#include <string_view>

class GPUShader;

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
    BitField<u32, u8, 16, 2> components;
    BitField<u32, u8, 18, 8> offset;
    u32 key;
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

    static RasterizationState GetNoCullState();
  };

  struct DepthState
  {
    BitField<u8, DepthFunc, 0, 3> depth_test;
    BitField<u8, bool, 4, 1> depth_write;
    u8 key;

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
