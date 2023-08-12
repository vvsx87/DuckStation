// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "gpu_device.h"

#include "common/types.h"
#include "common/windows_headers.h"

#include <d3dcommon.h>
#include <dxgiformat.h>
#include <optional>
#include <string>
#include <vector>
#include <wrl/client.h>

struct IDXGIFactory5;
struct IDXGIAdapter1;
struct IDXGIOutput;
struct DXGI_MODE_DESC;

namespace D3DCommon {
// create a dxgi factory
Microsoft::WRL::ComPtr<IDXGIFactory5> CreateFactory(bool debug);

// returns a list of all adapter names
std::vector<std::string> GetAdapterNames(IDXGIFactory5* factory);

// returns a list of fullscreen modes for the specified adapter
std::vector<std::string> GetFullscreenModes(IDXGIFactory5* factory, const std::string_view& adapter_name);

// returns the fullscreen mode to use for the specified dimensions
bool GetRequestedExclusiveFullscreenModeDesc(IDXGIFactory5* factory, const RECT& window_rect, u32 width, u32 height,
                                             float refresh_rate, DXGI_FORMAT format, DXGI_MODE_DESC* fullscreen_mode,
                                             IDXGIOutput** output);

// get an adapter based on name
Microsoft::WRL::ComPtr<IDXGIAdapter1> GetAdapterByName(IDXGIFactory5* factory, const std::string_view& name);

// returns the first adapter in the system
Microsoft::WRL::ComPtr<IDXGIAdapter1> GetFirstAdapter(IDXGIFactory5* factory);

// returns the adapter specified in the configuration, or the default
Microsoft::WRL::ComPtr<IDXGIAdapter1> GetChosenOrFirstAdapter(IDXGIFactory5* factory, const std::string_view& name);

// returns a utf-8 string of the specified adapter's name
std::string GetAdapterName(IDXGIAdapter1* adapter);

// returns the driver version from the registry as a string
std::string GetDriverVersionFromLUID(const LUID& luid);

std::optional<std::vector<u8>> CompileShader(D3D_FEATURE_LEVEL feature_level, bool debug_device, GPUShaderStage stage,
                                             const std::string_view& source);
} // namespace D3DCommon
