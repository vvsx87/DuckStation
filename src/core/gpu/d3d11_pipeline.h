// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_pipeline.h"

#include "common/windows_headers.h"

#include <memory>

#include <d3d11.h>
#include <wrl/client.h>

#include "gsl/span"

class D3D11Device;

class D3D11Pipeline final : public GPUPipeline
{
  friend D3D11Device;

  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

public:
  ~D3D11Pipeline() override;

  void SetDebugName(const std::string_view& name) override;

  ALWAYS_INLINE ID3D11RasterizerState* GetRasterizerState() const { return m_rs.Get(); }
  ALWAYS_INLINE ID3D11DepthStencilState* GetDepthStencilState() const { return m_ds.Get(); }
  ALWAYS_INLINE ID3D11BlendState* GetBlendState() const { return m_bs.Get(); }
  ALWAYS_INLINE ID3D11InputLayout* GetInputLayout() const { return m_il.Get(); }
  ALWAYS_INLINE ID3D11VertexShader* GetVertexShader() const { return m_vs.Get(); }
  ALWAYS_INLINE ID3D11PixelShader* GetPixelShader() const { return m_ps.Get(); }
  ALWAYS_INLINE D3D11_PRIMITIVE_TOPOLOGY GetPrimitiveTopology() const { return m_topology; }

  void Bind(ID3D11DeviceContext* context);

private:
  D3D11Pipeline(ComPtr<ID3D11RasterizerState> rs, ComPtr<ID3D11DepthStencilState> ds, ComPtr<ID3D11BlendState> bs,
                ComPtr<ID3D11InputLayout> il, ComPtr<ID3D11VertexShader> vs, ComPtr<ID3D11PixelShader> ps,
                D3D11_PRIMITIVE_TOPOLOGY topology);

  ComPtr<ID3D11RasterizerState> m_rs;
  ComPtr<ID3D11DepthStencilState> m_ds;
  ComPtr<ID3D11BlendState> m_bs;
  ComPtr<ID3D11InputLayout> m_il;
  ComPtr<ID3D11VertexShader> m_vs;
  ComPtr<ID3D11PixelShader> m_ps;
  D3D11_PRIMITIVE_TOPOLOGY m_topology;
};
