// SPDX-FileCopyrightText: 2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#pragma once

#include "common/hash_combine.h"
#include "common/types.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class GPUShaderStage : u8;

class GPUShaderCache
{
public:
  using ShaderBinary = std::vector<u8>;

  struct CacheIndexKey
  {
    u32 shader_type;
    u32 source_length;
    u64 source_hash_low;
    u64 source_hash_high;
    u64 entry_point_low;
    u64 entry_point_high;

    bool operator==(const CacheIndexKey& key) const;
    bool operator!=(const CacheIndexKey& key) const;
  };
  static_assert(sizeof(CacheIndexKey) == 40);

  struct CacheIndexEntryHash
  {
    std::size_t operator()(const CacheIndexKey& e) const noexcept;
  };

  GPUShaderCache();
  ~GPUShaderCache();

  bool IsOpen() const { return (m_index_file != nullptr); }

  bool Open(const std::string_view& base_filename);
  void Close();

  static CacheIndexKey GetCacheKey(GPUShaderStage stage, const std::string_view& shader_code,
                                   const std::string_view& entry_point);

  bool Lookup(const CacheIndexKey& key, ShaderBinary* binary);
  bool Insert(const CacheIndexKey& key, const void* data, u32 data_size);
  void Clear();

private:
  struct CacheIndexData
  {
    u32 file_offset;
    u32 blob_size;
  };

  using CacheIndex = std::unordered_map<CacheIndexKey, CacheIndexData, CacheIndexEntryHash>;

  bool CreateNew(const std::string& index_filename, const std::string& blob_filename);
  bool ReadExisting(const std::string& index_filename, const std::string& blob_filename);

  std::string m_base_filename;
  std::FILE* m_index_file = nullptr;
  std::FILE* m_blob_file = nullptr;

  CacheIndex m_index;
};
