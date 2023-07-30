// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include <string_view>

#include "common/types.h"

class GPUShader
{
public:
  enum class Stage
  {
    Vertex,
    Pixel,
    Compute
  };

  GPUShader(Stage stage) : m_stage(stage) {}
  virtual ~GPUShader() = default;

  ALWAYS_INLINE Stage GetStage() const { return m_stage; }

  virtual void SetDebugName(const std::string_view& name) = 0;

protected:
  Stage m_stage;
};
