// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "d3d11_shader.h"
#include "d3d11/shader_compiler.h"
#include "d3d11_device.h"

#include "common/assert.h"

D3D11Shader::D3D11Shader(Stage stage, Microsoft::WRL::ComPtr<ID3D11DeviceChild> shader)
  : GPUShader(stage), m_shader(std::move(shader))
{
}

D3D11Shader::~D3D11Shader() = default;

ID3D11VertexShader* D3D11Shader::GetD3DVertexShader() const
{
  DebugAssert(m_stage == Stage::Vertex);
  return static_cast<ID3D11VertexShader*>(m_shader.Get());
}

ID3D11PixelShader* D3D11Shader::GetD3DPixelShader() const
{
  DebugAssert(m_stage == Stage::Pixel);
  return static_cast<ID3D11PixelShader*>(m_shader.Get());
}

ID3D11ComputeShader* D3D11Shader::GetD3DComputeShader() const
{
  DebugAssert(m_stage == Stage::Compute);
  return static_cast<ID3D11ComputeShader*>(m_shader.Get());
}

void D3D11Shader::SetDebugName(const std::string_view& name)
{
  Panic("Implement me");
}

std::unique_ptr<GPUShader> D3D11Device::CreateShaderFromBinary(GPUShader::Stage stage, gsl::span<const u8> data)
{
  ComPtr<ID3D11DeviceChild> shader;
  std::vector<u8> bytecode;
  switch (stage)
  {
    case GPUShader::Stage::Vertex:
      shader = D3D11::ShaderCompiler::CreateVertexShader(D3D11Device::GetD3DDevice(), data.data(), data.size());
      bytecode.resize(data.size());
      std::memcpy(bytecode.data(), data.data(), data.size());
      break;

    case GPUShader::Stage::Pixel:
      shader = D3D11::ShaderCompiler::CreatePixelShader(D3D11Device::GetD3DDevice(), data.data(), data.size());
      break;

    case GPUShader::Stage::Compute:
      shader = D3D11::ShaderCompiler::CreateComputeShader(D3D11Device::GetD3DDevice(), data.data(), data.size());
      break;

    default:
      UnreachableCode();
      break;
  }

  if (!shader)
    return {};

  return std::unique_ptr<GPUShader>(new D3D11Shader(stage, std::move(shader), std::move(bytecode)));
}

std::unique_ptr<GPUShader> D3D11Device::CreateShaderFromSource(GPUShader::Stage stage, const std::string_view& source,
                                                               std::vector<u8>* out_binary /* = nullptr */)
{
  // TODO: This shouldn't be dependent on build type.
#ifdef _DEBUG
  constexpr bool debug = true;
#else
  constexpr bool debug = false;
#endif

  ComPtr<ID3DBlob> blob;
  switch (stage)
  {
    case GPUShader::Stage::Vertex:
      blob = D3D11::ShaderCompiler::CompileShader(D3D11::ShaderCompiler::Type::Vertex, m_device->GetFeatureLevel(),
                                                  source, debug);
      break;

    case GPUShader::Stage::Pixel:
      blob = D3D11::ShaderCompiler::CompileShader(D3D11::ShaderCompiler::Type::Pixel, m_device->GetFeatureLevel(),
                                                  source, debug);
      break;

    case GPUShader::Stage::Compute:
      blob = D3D11::ShaderCompiler::CompileShader(D3D11::ShaderCompiler::Type::Compute, m_device->GetFeatureLevel(),
                                                  source, debug);
      break;

    default:
      UnreachableCode();
      break;
  }

  if (out_binary)
  {
    const size_t size = blob->GetBufferSize();
    out_binary->resize(size);
    std::memcpy(out_binary->data(), blob->GetBufferPointer(), size);
  }

  return CreateShaderFromBinary(
    stage, gsl::span<const u8>(static_cast<const u8*>(blob->GetBufferPointer()), blob->GetBufferSize()));
}
