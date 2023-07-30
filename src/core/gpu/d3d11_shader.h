// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_shader.h"

#include "common/windows_headers.h"

#include <memory>
#include <vector>

#include <d3d11.h>
#include <wrl/client.h>

#include "gsl/span"

class D3D11Device;

class D3D11Shader final : public GPUShader
{
  friend D3D11Device;

public:
  ~D3D11Shader() override;

  ID3D11VertexShader* GetD3DVertexShader() const;
  ID3D11PixelShader* GetD3DPixelShader() const;
  ID3D11ComputeShader* GetD3DComputeShader() const;

  ALWAYS_INLINE const std::vector<u8>& GetBytecode() const { return m_bytecode; }

  void SetDebugName(const std::string_view& name) override;

private:
  D3D11Shader(Stage stage, Microsoft::WRL::ComPtr<ID3D11DeviceChild> shader, std::vector<u8> bytecode);

  Microsoft::WRL::ComPtr<ID3D11DeviceChild> m_shader;
  std::vector<u8> m_bytecode; // only for VS
};
