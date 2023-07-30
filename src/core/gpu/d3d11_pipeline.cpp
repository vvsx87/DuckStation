// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "d3d11_pipeline.h"
#include "d3d11_device.h"
#include "d3d11_shader.h"

#include "common/assert.h"
#include "common/log.h"

#include <array>
#include <malloc.h>

Log_SetChannel(D3D11Device);

D3D11Pipeline::D3D11Pipeline(ComPtr<ID3D11RasterizerState> rs, ComPtr<ID3D11DepthStencilState> ds,
                             ComPtr<ID3D11BlendState> bs, ComPtr<ID3D11InputLayout> il, ComPtr<ID3D11VertexShader> vs,
                             ComPtr<ID3D11PixelShader> ps, D3D11_PRIMITIVE_TOPOLOGY topology)
  : m_rs(std::move(rs)), m_ds(std::move(ds)), m_bs(std::move(bs)), m_il(std::move(il)), m_vs(std::move(vs)),
    m_ps(std::move(ps)), m_topology(topology)
{
}

D3D11Pipeline::~D3D11Pipeline() = default;

void D3D11Pipeline::SetDebugName(const std::string_view& name)
{
  UnreachableCode();
}

void D3D11Pipeline::Bind(ID3D11DeviceContext* context)
{
  context->IASetInputLayout(GetInputLayout());
  context->IASetPrimitiveTopology(GetPrimitiveTopology());
  context->RSSetState(GetRasterizerState());
  context->OMSetDepthStencilState(GetDepthStencilState(), 0);
  context->OMSetBlendState(GetBlendState(), nullptr, 0xFFFFFFFFu);
  context->VSSetShader(GetVertexShader(), nullptr, 0);
  context->PSSetShader(GetPixelShader(), nullptr, 0);
}

D3D11Device::ComPtr<ID3D11RasterizerState> D3D11Device::GetRasterizationState(const GPUPipeline::RasterizationState& rs)
{
  ComPtr<ID3D11RasterizerState> drs;

  const auto it = m_rasterization_states.find(rs.key);
  if (it != m_rasterization_states.end())
  {
    drs = it->second;
    return drs;
  }

  static constexpr std::array<D3D11_CULL_MODE, static_cast<u32>(GPUPipeline::CullMode::MaxCount)> cull_mapping = {{
    D3D11_CULL_NONE,  // None
    D3D11_CULL_FRONT, // Front
    D3D11_CULL_BACK,  // Back
  }};

  D3D11_RASTERIZER_DESC desc = {};
  desc.FillMode = D3D11_FILL_SOLID;
  desc.CullMode = cull_mapping[static_cast<u8>(rs.cull_mode.GetValue())];
  desc.ScissorEnable = TRUE;
  // desc.MultisampleEnable ???

  HRESULT hr = m_device->CreateRasterizerState(&desc, drs.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create depth state with %08X", hr);

  m_rasterization_states.emplace(rs.key, drs);
  return drs;
}

D3D11Device::ComPtr<ID3D11DepthStencilState> D3D11Device::GetDepthState(const GPUPipeline::DepthState& ds)
{
  ComPtr<ID3D11DepthStencilState> dds;

  const auto it = m_depth_states.find(ds.key);
  if (it != m_depth_states.end())
  {
    dds = it->second;
    return dds;
  }

  static constexpr std::array<D3D11_COMPARISON_FUNC, static_cast<u32>(GPUPipeline::DepthFunc::MaxCount)> func_mapping =
    {{
      D3D11_COMPARISON_NEVER,         // Never
      D3D11_COMPARISON_ALWAYS,        // Always
      D3D11_COMPARISON_LESS,          // Less
      D3D11_COMPARISON_LESS_EQUAL,    // LessEqual
      D3D11_COMPARISON_GREATER,       // Greater
      D3D11_COMPARISON_GREATER_EQUAL, // GreaterEqual
      D3D11_COMPARISON_EQUAL,         // Equal
    }};

  D3D11_DEPTH_STENCIL_DESC desc = {};
  desc.DepthEnable = ds.depth_test != GPUPipeline::DepthFunc::Never;
  desc.DepthFunc = func_mapping[static_cast<u8>(ds.depth_test.GetValue())];
  desc.DepthWriteMask = ds.depth_write ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;

  HRESULT hr = m_device->CreateDepthStencilState(&desc, dds.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create depth state with %08X", hr);

  m_depth_states.emplace(ds.key, dds);
  return dds;
}

D3D11Device::ComPtr<ID3D11BlendState> D3D11Device::GetBlendState(const GPUPipeline::BlendState& bs)
{
  ComPtr<ID3D11BlendState> dbs;

  const auto it = m_blend_states.find(bs.key);
  if (it != m_blend_states.end())
  {
    dbs = it->second;
    return dbs;
  }

  static constexpr std::array<D3D11_BLEND, static_cast<u32>(GPUPipeline::BlendFunc::MaxCount)> blend_mapping = {{
    D3D11_BLEND_ZERO,           // Zero
    D3D11_BLEND_ONE,            // One
    D3D11_BLEND_SRC_COLOR,      // SrcColor
    D3D11_BLEND_INV_SRC_COLOR,  // InvSrcColor
    D3D11_BLEND_DEST_COLOR,     // DstColor
    D3D11_BLEND_INV_DEST_COLOR, // InvDstColor
    D3D11_BLEND_SRC_ALPHA,      // SrcAlpha
    D3D11_BLEND_INV_SRC_ALPHA,  // InvSrcAlpha
    D3D11_BLEND_SRC1_ALPHA,     // SrcAlpha1
    D3D11_BLEND_INV_SRC1_ALPHA, // InvSrcAlpha1
    D3D11_BLEND_DEST_ALPHA,     // DstAlpha
    D3D11_BLEND_INV_DEST_ALPHA, // InvDstAlpha
  }};

  static constexpr std::array<D3D11_BLEND_OP, static_cast<u32>(GPUPipeline::BlendOp::MaxCount)> op_mapping = {{
    D3D11_BLEND_OP_ADD,          // Add
    D3D11_BLEND_OP_SUBTRACT,     // Subtract
    D3D11_BLEND_OP_REV_SUBTRACT, // ReverseSubtract
    D3D11_BLEND_OP_MIN,          // Min
    D3D11_BLEND_OP_MAX,          // Max
  }};

  D3D11_BLEND_DESC blend_desc = {};
  D3D11_RENDER_TARGET_BLEND_DESC& tgt_desc = blend_desc.RenderTarget[0];
  tgt_desc.BlendEnable = bs.enable;
  tgt_desc.RenderTargetWriteMask = bs.write_mask;
  if (bs.enable)
  {
    tgt_desc.SrcBlend = blend_mapping[static_cast<u8>(bs.src_blend.GetValue())];
    tgt_desc.DestBlend = blend_mapping[static_cast<u8>(bs.dst_blend.GetValue())];
    tgt_desc.BlendOp = op_mapping[static_cast<u8>(bs.blend_op.GetValue())];
    tgt_desc.SrcBlendAlpha = blend_mapping[static_cast<u8>(bs.src_alpha_blend.GetValue())];
    tgt_desc.DestBlendAlpha = blend_mapping[static_cast<u8>(bs.dst_alpha_blend.GetValue())];
    tgt_desc.BlendOpAlpha = op_mapping[static_cast<u8>(bs.alpha_blend_op.GetValue())];
  }

  HRESULT hr = m_device->CreateBlendState(&blend_desc, dbs.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create blend state with %08X", hr);

  m_blend_states.emplace(bs.key, dbs);
  return dbs;
}

D3D11Device::ComPtr<ID3D11InputLayout> D3D11Device::GetInputLayout(const GPUPipeline::InputLayout& il,
                                                                   const D3D11Shader* vs)
{
  ComPtr<ID3D11InputLayout> dil;
  const auto it = m_input_layouts.find(il);
  if (it != m_input_layouts.end())
  {
    dil = it->second;
    return dil;
  }

  static constexpr std::array<const char*, static_cast<u32>(GPUPipeline::VertexAttribute::Semantic::MaxCount)>
    semantics = {{
      "POSITION", // Position
      "TEXCOORD", // Texcoord
      "COLOR",    // Color
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

  D3D11_INPUT_ELEMENT_DESC* elems =
    static_cast<D3D11_INPUT_ELEMENT_DESC*>(alloca(sizeof(D3D11_INPUT_ELEMENT_DESC) * il.vertex_attributes.size()));
  for (size_t i = 0; i < il.vertex_attributes.size(); i++)
  {
    const GPUPipeline::VertexAttribute& va = il.vertex_attributes[i];
    Assert(va.components > 0 && va.components < MAX_COMPONENTS);

    D3D11_INPUT_ELEMENT_DESC& elem = elems[i];
    elem.SemanticName = semantics[static_cast<u8>(va.semantic.GetValue())];
    elem.SemanticIndex = va.semantic_index;
    elem.Format = format_mapping[static_cast<u8>(va.type.GetValue())][va.components - 1];
    elem.InputSlot = 0;
    elem.AlignedByteOffset = va.offset;
    elem.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    elem.InstanceDataStepRate = 0;
  }

  HRESULT hr = m_device->CreateInputLayout(elems, static_cast<UINT>(il.vertex_attributes.size()),
                                           vs->GetBytecode().data(), vs->GetBytecode().size(), dil.GetAddressOf());
  if (FAILED(hr))
    Log_ErrorPrintf("Failed to create input layout with %08X", hr);

  m_input_layouts.emplace(il, dil);
  return dil;
}

std::unique_ptr<GPUPipeline> D3D11Device::CreatePipeline(const GPUPipeline::GraphicsConfig& config)
{
  ComPtr<ID3D11RasterizerState> rs = GetRasterizationState(config.rasterization);
  ComPtr<ID3D11DepthStencilState> ds = GetDepthState(config.depth);
  ComPtr<ID3D11BlendState> bs = GetBlendState(config.blend);

  static constexpr std::array<D3D11_PRIMITIVE_TOPOLOGY, static_cast<u32>(GPUPipeline::Primitive::MaxCount)> primitives =
    {{
      D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,     // Points
      D3D11_PRIMITIVE_TOPOLOGY_LINELIST,      // Lines
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,  // Triangles
      D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, // TriangleStrips
    }};

  return std::unique_ptr<GPUPipeline>(
    new D3D11Pipeline(std::move(rs), std::move(ds), std::move(bs), nullptr,
                      static_cast<const D3D11Shader*>(config.vertex_shader)->GetD3DVertexShader(),
                      static_cast<const D3D11Shader*>(config.pixel_shader)->GetD3DPixelShader(),
                      primitives[static_cast<u8>(config.primitive)]));
}
