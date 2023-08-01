// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once
#include "common/windows_headers.h"
#include "gpu_texture.h"
#include <d3d11.h>
#include <wrl/client.h>

class D3D11Device;

class D3D11Texture final : public GPUTexture
{
  friend D3D11Device;

public:
  template<typename T>
  using ComPtr = Microsoft::WRL::ComPtr<T>;

  D3D11Texture();
  D3D11Texture(ComPtr<ID3D11Texture2D> texture, ComPtr<ID3D11ShaderResourceView> srv, ComPtr<ID3D11View> rtv);
  ~D3D11Texture();

  static DXGI_FORMAT GetDXGIFormat(Format format);
  static Format LookupBaseFormat(DXGI_FORMAT dformat);

  ALWAYS_INLINE ID3D11Texture2D* GetD3DTexture() const { return m_texture.Get(); }
  ALWAYS_INLINE ID3D11ShaderResourceView* GetD3DSRV() const { return m_srv.Get(); }
  ALWAYS_INLINE ID3D11RenderTargetView* GetD3DRTV() const
  {
    return static_cast<ID3D11RenderTargetView*>(m_rtv_dsv.Get());
  }
  ALWAYS_INLINE ID3D11DepthStencilView* GetD3DDSV() const
  {
    return static_cast<ID3D11DepthStencilView*>(m_rtv_dsv.Get());
  }
  ALWAYS_INLINE ID3D11ShaderResourceView* const* GetD3DSRVArray() const { return m_srv.GetAddressOf(); }
  ALWAYS_INLINE ID3D11RenderTargetView* const* GetD3DRTVArray() const
  {
    return reinterpret_cast<ID3D11RenderTargetView* const*>(m_rtv_dsv.GetAddressOf());
  }
  ALWAYS_INLINE DXGI_FORMAT GetDXGIFormat() const { return GetDXGIFormat(m_format); }
  ALWAYS_INLINE bool IsDynamic() const { return m_dynamic; }

  ALWAYS_INLINE operator ID3D11Texture2D*() const { return m_texture.Get(); }
  ALWAYS_INLINE operator ID3D11ShaderResourceView*() const { return m_srv.Get(); }
  ALWAYS_INLINE operator ID3D11RenderTargetView*() const
  {
    return static_cast<ID3D11RenderTargetView*>(m_rtv_dsv.Get());
  }
  ALWAYS_INLINE operator ID3D11DepthStencilView*() const
  {
    return static_cast<ID3D11DepthStencilView*>(m_rtv_dsv.Get());
  }
  ALWAYS_INLINE operator bool() const { return static_cast<bool>(m_texture); }

  bool Create(ID3D11Device* device, u32 width, u32 height, u32 layers, u32 levels, u32 samples, Type type,
              Format format, const void* initial_data = nullptr, u32 initial_data_stride = 0, bool dynamic = false);
  bool Adopt(ID3D11Device* device, ComPtr<ID3D11Texture2D> texture);

  void Destroy();

  D3D11_TEXTURE2D_DESC GetDesc() const;
  void CommitClear(ID3D11DeviceContext* context);

  bool IsValid() const override;

  bool Update(u32 x, u32 y, u32 width, u32 height, const void* data, u32 pitch, u32 layer = 0, u32 level = 0) override;
  bool Map(void** map, u32* map_stride, u32 x, u32 y, u32 width, u32 height, u32 layer = 0, u32 level = 0) override;
  void Unmap() override;

private:
  ComPtr<ID3D11Texture2D> m_texture;
  ComPtr<ID3D11ShaderResourceView> m_srv;
  ComPtr<ID3D11View> m_rtv_dsv;
  u32 m_mapped_subresource = 0;
  bool m_dynamic = false;
};
