// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "d3d12_pipeline.h"
#include "d3d12_builders.h"
#include "d3d12_device.h"
#include "d3d_common.h"

#include "common/assert.h"
#include "common/log.h"

#include <d3dcompiler.h>

Log_SetChannel(D3D12Device);

static u32 s_next_bad_shader_id = 1;

D3D12Shader::D3D12Shader(GPUShaderStage stage, Bytecode bytecode) : GPUShader(stage), m_bytecode(std::move(bytecode))
{
}

D3D12Shader::~D3D12Shader() = default;

void D3D12Shader::SetDebugName(const std::string_view& name)
{
}

std::unique_ptr<GPUShader> D3D12Device::CreateShaderFromBinary(GPUShaderStage stage, gsl::span<const u8> data)
{
  // Can't do much at this point.
  std::vector bytecode(data.begin(), data.end());
  return std::unique_ptr<GPUShader>(new D3D12Shader(stage, std::move(bytecode)));
}

std::unique_ptr<GPUShader> D3D12Device::CreateShaderFromSource(GPUShaderStage stage, const std::string_view& source,
                                                               std::vector<u8>* out_binary /*= nullptr*/)
{
  std::optional<std::vector<u8>> bytecode = D3DCommon::CompileShader(m_feature_level, m_debug_device, stage, source);
  if (!bytecode.has_value())
    return {};

  std::unique_ptr<GPUShader> ret = CreateShaderFromBinary(stage, bytecode.value());
  if (ret && out_binary)
    *out_binary = std::move(bytecode.value());

  return ret;
}

//////////////////////////////////////////////////////////////////////////

D3D12Pipeline::D3D12Pipeline(Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline, Layout layout,
                             D3D12_PRIMITIVE_TOPOLOGY topology, u32 vertex_stride, u32 blend_constants)
  : GPUPipeline(), m_pipeline(std::move(pipeline)), m_layout(layout), m_topology(topology),
    m_vertex_stride(vertex_stride), m_blend_constants(blend_constants),
    m_blend_constants_f(GPUDevice::RGBA8ToFloat(blend_constants))
{
}

D3D12Pipeline::~D3D12Pipeline()
{
  D3D12Device::GetInstance().DeferObjectDestruction(std::move(m_pipeline));
}

void D3D12Pipeline::SetDebugName(const std::string_view& name)
{
  D3D12::SetObjectName(m_pipeline.Get(), name);
}

std::unique_ptr<GPUPipeline> D3D12Device::CreatePipeline(const GPUPipeline::GraphicsConfig& config)
{
  static constexpr std::array<D3D12_PRIMITIVE_TOPOLOGY, static_cast<u32>(GPUPipeline::Primitive::MaxCount)> primitives =
    {{
      D3D_PRIMITIVE_TOPOLOGY_POINTLIST,     // Points
      D3D_PRIMITIVE_TOPOLOGY_LINELIST,      // Lines
      D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,  // Triangles
      D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, // TriangleStrips
    }};
  static constexpr std::array<D3D12_PRIMITIVE_TOPOLOGY_TYPE, static_cast<u32>(GPUPipeline::Primitive::MaxCount)>
    primitive_types = {{
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,    // Points
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,     // Lines
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, // Triangles
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, // TriangleStrips
    }};

  static constexpr u32 MAX_COMPONENTS = 4;
  static constexpr const DXGI_FORMAT
    format_mapping[static_cast<u8>(GPUPipeline::VertexAttribute::Type::MaxCount)][MAX_COMPONENTS] = {
      {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
       DXGI_FORMAT_R32G32B32A32_FLOAT},                                                                       // Float
      {DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UINT},           // UInt8
      {DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_SINT},           // SInt8
      {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_UNORM},        // UNorm8
      {DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_UINT},    // UInt16
      {DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_SINT},    // SInt16
      {DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_UNORM}, // UNorm16
      {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32G32B32A32_UINT},    // UInt32
      {DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32G32B32A32_SINT},    // SInt32
    };

  static constexpr std::array<D3D12_CULL_MODE, static_cast<u32>(GPUPipeline::CullMode::MaxCount)> cull_mapping = {{
    D3D12_CULL_MODE_NONE,  // None
    D3D12_CULL_MODE_FRONT, // Front
    D3D12_CULL_MODE_BACK,  // Back
  }};

  static constexpr std::array<D3D12_COMPARISON_FUNC, static_cast<u32>(GPUPipeline::DepthFunc::MaxCount)>
    compare_mapping = {{
      D3D12_COMPARISON_FUNC_NEVER,         // Never
      D3D12_COMPARISON_FUNC_ALWAYS,        // Always
      D3D12_COMPARISON_FUNC_LESS,          // Less
      D3D12_COMPARISON_FUNC_LESS_EQUAL,    // LessEqual
      D3D12_COMPARISON_FUNC_GREATER,       // Greater
      D3D12_COMPARISON_FUNC_GREATER_EQUAL, // GreaterEqual
      D3D12_COMPARISON_FUNC_EQUAL,         // Equal
    }};

  static constexpr std::array<D3D12_BLEND, static_cast<u32>(GPUPipeline::BlendFunc::MaxCount)> blend_mapping = {{
    D3D12_BLEND_ZERO,             // Zero
    D3D12_BLEND_ONE,              // One
    D3D12_BLEND_SRC_COLOR,        // SrcColor
    D3D12_BLEND_INV_SRC_COLOR,    // InvSrcColor
    D3D12_BLEND_DEST_COLOR,       // DstColor
    D3D12_BLEND_INV_DEST_COLOR,   // InvDstColor
    D3D12_BLEND_SRC_ALPHA,        // SrcAlpha
    D3D12_BLEND_INV_SRC_ALPHA,    // InvSrcAlpha
    D3D12_BLEND_SRC1_ALPHA,       // SrcAlpha1
    D3D12_BLEND_INV_SRC1_ALPHA,   // InvSrcAlpha1
    D3D12_BLEND_DEST_ALPHA,       // DstAlpha
    D3D12_BLEND_INV_DEST_ALPHA,   // InvDstAlpha
    D3D12_BLEND_BLEND_FACTOR,     // ConstantColor
    D3D12_BLEND_INV_BLEND_FACTOR, // InvConstantColor
  }};

  static constexpr std::array<D3D12_BLEND_OP, static_cast<u32>(GPUPipeline::BlendOp::MaxCount)> op_mapping = {{
    D3D12_BLEND_OP_ADD,          // Add
    D3D12_BLEND_OP_SUBTRACT,     // Subtract
    D3D12_BLEND_OP_REV_SUBTRACT, // ReverseSubtract
    D3D12_BLEND_OP_MIN,          // Min
    D3D12_BLEND_OP_MAX,          // Max
  }};

  D3D12::GraphicsPipelineBuilder gpb;
  gpb.SetRootSignature(m_root_signatures[static_cast<u8>(config.layout)].Get());
  gpb.SetVertexShader(static_cast<const D3D12Shader*>(config.vertex_shader)->GetBytecodeData(),
                      static_cast<const D3D12Shader*>(config.vertex_shader)->GetBytecodeSize());
  gpb.SetPixelShader(static_cast<const D3D12Shader*>(config.fragment_shader)->GetBytecodeData(),
                     static_cast<const D3D12Shader*>(config.fragment_shader)->GetBytecodeSize());
  gpb.SetPrimitiveTopologyType(primitive_types[static_cast<u8>(config.primitive)]);

  if (!config.input_layout.vertex_attributes.empty())
  {
    for (u32 i = 0; i < static_cast<u32>(config.input_layout.vertex_attributes.size()); i++)
    {
      const GPUPipeline::VertexAttribute& va = config.input_layout.vertex_attributes[i];
      DebugAssert(va.components > 0 && va.components <= MAX_COMPONENTS);
      gpb.AddVertexAttribute(
        "ATTR", i, format_mapping[static_cast<u8>(va.type.GetValue())][static_cast<u8>(va.components.GetValue() - 1)],
        0, va.offset);
    }
  }

  gpb.SetRasterizationState(D3D12_FILL_MODE_SOLID,
                            cull_mapping[static_cast<u8>(config.rasterization.cull_mode.GetValue())], false);
  if (config.samples > 1)
    gpb.SetMultisamples(config.samples);
  gpb.SetDepthState(config.depth.depth_test != GPUPipeline::DepthFunc::Always || config.depth.depth_write,
                    config.depth.depth_write, compare_mapping[static_cast<u8>(config.depth.depth_test.GetValue())]);
  gpb.SetNoStencilState();

  gpb.SetBlendState(0, config.blend.enable, blend_mapping[static_cast<u8>(config.blend.src_blend.GetValue())],
                    blend_mapping[static_cast<u8>(config.blend.dst_blend.GetValue())],
                    op_mapping[static_cast<u8>(config.blend.blend_op.GetValue())],
                    blend_mapping[static_cast<u8>(config.blend.src_alpha_blend.GetValue())],
                    blend_mapping[static_cast<u8>(config.blend.dst_alpha_blend.GetValue())],
                    op_mapping[static_cast<u8>(config.blend.alpha_blend_op.GetValue())], config.blend.write_mask);

  if (config.color_format != GPUTexture::Format::Unknown)
  {
    DXGI_FORMAT color_format;
    LookupNativeFormat(config.color_format, nullptr, nullptr, &color_format, nullptr);
    gpb.SetRenderTarget(0, color_format);
  }
  if (config.depth_format != GPUTexture::Format::Unknown)
  {
    DXGI_FORMAT depth_format;
    LookupNativeFormat(config.depth_format, nullptr, nullptr, nullptr, &depth_format);
    gpb.SetDepthStencilFormat(depth_format);
  }

  /* TODO: PIPELINE CACHE */
  ComPtr<ID3D12PipelineState> pipeline = gpb.Create(m_device.Get(), false);
  if (!pipeline)
    return {};

  return std::unique_ptr<GPUPipeline>(new D3D12Pipeline(
    pipeline, config.layout, primitives[static_cast<u8>(config.primitive)],
    config.input_layout.vertex_attributes.empty() ? 0 : config.input_layout.vertex_stride, config.blend.constant));
}
