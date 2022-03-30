// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

// Enable usage of SSE4.1 in MSVC.
#ifndef __SSE4_1__
#define __SSE4_1__
#endif

#include "gpu_sw_rasterizer.h"

#include "common/assert.h"
#include "common/gsvector.h"

namespace GPU_SW_Rasterizer::SSE4 {
#define USE_VECTOR 1
#include "gpu_sw_rasterizer.inl"
}
