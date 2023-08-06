// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "opengl_pipeline.h"
#include "../settings.h"
#include "../shadergen.h"
#include "opengl_device.h"
#include "opengl_stream_buffer.h"

#include "common/assert.h"
#include "common/file_system.h"
#include "common/hash_combine.h"
#include "common/log.h"
#include "common/path.h"

Log_SetChannel(OpenGLPipeline);

static unsigned s_next_bad_shader_id = 1;

static GLenum GetGLShaderType(GPUShaderStage stage)
{
  static constexpr std::array<GLenum, static_cast<u32>(GPUShaderStage::MaxCount)> mapping = {{
    GL_VERTEX_SHADER,   // Vertex
    GL_FRAGMENT_SHADER, // Fragment
    GL_COMPUTE_SHADER,  // Compute
  }};

  return mapping[static_cast<u32>(stage)];
}

OpenGLShader::OpenGLShader(GPUShaderStage stage, GLuint id, const GPUShaderCache::CacheIndexKey& key)
  : GPUShader(stage), m_id(id), m_key(key)
{
}

OpenGLShader::~OpenGLShader() = default;

void OpenGLShader::SetDebugName(const std::string_view& name)
{
#ifdef _DEBUG
  if (glObjectLabel)
    glObjectLabel(GL_SHADER, m_id, static_cast<GLsizei>(name.length()), static_cast<const GLchar*>(name.data()));
#endif
}

std::unique_ptr<GPUShader> OpenGLDevice::CreateShaderFromBinary(GPUShaderStage stage, gsl::span<const u8> data)
{
  // Not supported.. except spir-v maybe? but no point really...
  return {};
}

std::unique_ptr<GPUShader> OpenGLDevice::CreateShaderFromSource(GPUShaderStage stage, const std::string_view& source,
                                                                std::vector<u8>* out_binary)
{
  glGetError();

  GLuint shader = glCreateShader(GetGLShaderType(stage));
  if (GLenum err = glGetError(); err != GL_NO_ERROR)
  {
    Log_ErrorPrintf("glCreateShader() failed: %u", err);
    return {};
  }

  const GLchar* string = source.data();
  const GLint length = static_cast<GLint>(source.length());
  glShaderSource(shader, 1, &string, &length);
  glCompileShader(shader);

  GLint status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

  GLint info_log_length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);

  if (status == GL_FALSE || info_log_length > 0)
  {
    std::string info_log;
    info_log.resize(info_log_length + 1);
    glGetShaderInfoLog(shader, info_log_length, &info_log_length, &info_log[0]);

    if (status == GL_TRUE)
    {
      Log_ErrorPrintf("Shader compiled with warnings:\n%s", info_log.c_str());
    }
    else
    {
      Log_ErrorPrintf("Shader failed to compile:\n%s", info_log.c_str());

      auto fp = FileSystem::OpenManagedCFile(
        Path::Combine(EmuFolders::DataRoot, fmt::format("bad_shader_{}.txt", s_next_bad_shader_id++)).c_str(), "wb");
      if (fp)
      {
        std::fwrite(source.data(), source.size(), 1, fp.get());
        std::fprintf(fp.get(), "\n\nCompile %s shader failed\n", GPUShader::GetStageName(stage));
        std::fwrite(info_log.c_str(), info_log_length, 1, fp.get());
      }

      glDeleteShader(shader);
      return {};
    }
  }

  return std::unique_ptr<GPUShader>(
    new OpenGLShader(stage, shader, GPUShaderCache::GetCacheKey(stage, source, "main")));
}

//////////////////////////////////////////////////////////////////////////

bool OpenGLPipeline::VertexArrayCacheKey::operator==(const VertexArrayCacheKey& rhs) const
{
  return (std::memcmp(this, &rhs, sizeof(*this)) == 0);
}

bool OpenGLPipeline::VertexArrayCacheKey::operator!=(const VertexArrayCacheKey& rhs) const
{
  return (std::memcmp(this, &rhs, sizeof(*this)) != 0);
}

size_t OpenGLPipeline::VertexArrayCacheKeyHash::operator()(const VertexArrayCacheKey& k) const
{
  std::size_t h = 0;
  hash_combine(h, k.num_vertex_attributes, k.vertex_attribute_stride);
  for (const VertexAttribute& va : k.vertex_attributes)
    hash_combine(h, va.key);
  return h;
}

bool OpenGLPipeline::ProgramCacheKey::operator==(const ProgramCacheKey& rhs) const
{
  return (std::memcmp(this, &rhs, sizeof(*this)) == 0);
}

bool OpenGLPipeline::ProgramCacheKey::operator!=(const ProgramCacheKey& rhs) const
{
  return (std::memcmp(this, &rhs, sizeof(*this)) != 0);
}

size_t OpenGLPipeline::ProgramCacheKeyHash::operator()(const ProgramCacheKey& k) const
{
  // TODO: maybe use xxhash here...
  std::size_t h = 0;
  hash_combine(h, k.vs_key.entry_point_low, k.vs_key.entry_point_high, k.vs_key.source_hash_low,
               k.vs_key.source_hash_high, k.vs_key.source_length, k.vs_key.shader_type);
  hash_combine(h, k.fs_key.entry_point_low, k.fs_key.entry_point_high, k.fs_key.source_hash_low,
               k.fs_key.source_hash_high, k.fs_key.source_length, k.fs_key.shader_type);
  hash_combine(h, k.va_key.num_vertex_attributes, k.va_key.vertex_attribute_stride);
  for (const VertexAttribute& va : k.va_key.vertex_attributes)
    hash_combine(h, va.key);
  return h;
}

OpenGLPipeline::ProgramCacheKey OpenGLPipeline::GetProgramCacheKey(const GraphicsConfig& plconfig)
{
  Assert(plconfig.input_layout.vertex_attributes.size() <= MAX_VERTEX_ATTRIBUTES);

  ProgramCacheKey ret;
  ret.vs_key = static_cast<const OpenGLShader*>(plconfig.vertex_shader)->GetKey();
  ret.fs_key = static_cast<const OpenGLShader*>(plconfig.fragment_shader)->GetKey();

  ret.va_key.num_vertex_attributes = static_cast<u32>(plconfig.input_layout.vertex_attributes.size());
  ret.va_key.vertex_attribute_stride = plconfig.input_layout.vertex_stride;
  std::memset(ret.va_key.vertex_attributes, 0, sizeof(ret.va_key.vertex_attributes));
  if (ret.va_key.num_vertex_attributes > 0)
  {
    std::memcpy(ret.va_key.vertex_attributes, plconfig.input_layout.vertex_attributes.data(),
                sizeof(VertexAttribute) * ret.va_key.num_vertex_attributes);
  }

  return ret;
}

GLuint OpenGLDevice::LookupProgramCache(const OpenGLPipeline::ProgramCacheKey& key,
                                        const GPUPipeline::GraphicsConfig& plconfig)
{
  auto it = m_program_cache.find(key);
  if (it != m_program_cache.end())
  {
    if (it->second.program_id == 0)
    {
      // TODO: read it from file instead
    }
    else
    {
      it->second.reference_count++;
    }

    return it->second.program_id;
  }

  OpenGLPipeline::ProgramCacheItem item;
  item.program_id = CompileProgram(plconfig);
  item.reference_count = 0;
  item.file_offset = std::numeric_limits<decltype(item.file_offset)>::max();
  if (item.program_id != 0)
  {
    item.reference_count++;
  }

  // Insert into cache even if we failed, so we don't compile it again, but don't increment reference count.
  m_program_cache.emplace(key, item);
  return item.program_id;
}

GLuint OpenGLDevice::CompileProgram(const GPUPipeline::GraphicsConfig& plconfig)
{
  glGetError();
  const GLuint program_id = glCreateProgram();
  if (glGetError() != GL_NO_ERROR)
  {
    Log_ErrorPrintf("Failed to create program object.");
    return 0;
  }

  Assert(plconfig.vertex_shader && plconfig.fragment_shader);
  glAttachShader(program_id, static_cast<const OpenGLShader*>(plconfig.vertex_shader)->GetGLId());
  glAttachShader(program_id, static_cast<const OpenGLShader*>(plconfig.fragment_shader)->GetGLId());

  if (!ShaderGen::UseGLSLBindingLayout())
  {
    // TODO: fixme, need semantic type...
#if 0
    for (u32 i = 0; i < static_cast<u32>(plconfig.input_layout.vertex_attributes.size()); i++)
      glBindAttribLocation(program_id, i, fmt::format(""));

    glBindFragDataLocation(program_id, 0, "o_col0");

    if (m_features.dual_source_blend)
    {
      if (GLAD_GL_VERSION_3_3 || GLAD_GL_ARB_blend_func_extended)
      {
        glBindFragDataLocationIndexed(program_id, 1, 0, name);
        return;
      }
      else if (GLAD_GL_EXT_blend_func_extended)
      {
        glBindFragDataLocationIndexedEXT(program_id, 1, 0, name);
        return;
      }

      Log_ErrorPrintf("BindFragDataIndexed() called without ARB or EXT extension, we'll probably crash.");
      glBindFragDataLocationIndexed(program_id, 1, 0, name);
    }
#else
    Panic("Fixme");
#endif
  }

  glLinkProgram(program_id);

  GLint status = GL_FALSE;
  glGetProgramiv(program_id, GL_LINK_STATUS, &status);

  GLint info_log_length = 0;
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);

  if (status == GL_FALSE || info_log_length > 0)
  {
    std::string info_log;
    info_log.resize(info_log_length + 1);
    glGetProgramInfoLog(program_id, info_log_length, &info_log_length, &info_log[0]);

    if (status == GL_TRUE)
    {
      Log_ErrorPrintf("Program linked with warnings:\n%s", info_log.c_str());
    }
    else
    {
      Log_ErrorPrintf("Program failed to link:\n%s", info_log.c_str());
      glDeleteProgram(program_id);
      return 0;
    }
  }

  return program_id;
}

void OpenGLDevice::UnrefProgram(const OpenGLPipeline::ProgramCacheKey& key)
{
  auto it = m_program_cache.find(key);
  Assert(it != m_program_cache.end() && it->second.program_id != 0 && it->second.reference_count > 0);

  if ((--it->second.reference_count) > 0)
    return;

  if (m_last_program == it->second.program_id)
  {
    m_last_program = 0;
    glUseProgram(0);
  }

  glDeleteProgram(it->second.program_id);
  it->second.program_id = 0;
}

GLuint OpenGLDevice::LookupVAOCache(const OpenGLPipeline::VertexArrayCacheKey& key)
{
  auto it = m_vao_cache.find(key);
  if (it != m_vao_cache.end())
  {
    it->second.reference_count++;
    return it->second.vao_id;
  }

  OpenGLPipeline::VertexArrayCacheItem item;
  item.vao_id =
    CreateVAO(gsl::span<const GPUPipeline::VertexAttribute>(key.vertex_attributes, key.num_vertex_attributes),
              key.vertex_attribute_stride);
  if (item.vao_id == 0)
    return 0;

  item.reference_count = 1;
  m_vao_cache.emplace(key, item);
  return item.vao_id;
}

GLuint OpenGLDevice::CreateVAO(gsl::span<const GPUPipeline::VertexAttribute> attributes, u32 stride)
{
  glGetError();
  GLuint vao;
  glGenVertexArrays(1, &vao);
  if (const GLenum err = glGetError(); err != GL_NO_ERROR)
  {
    Log_ErrorPrintf("Failed to create vertex array object: %u", vao);
    return 0;
  }

  glBindVertexArray(vao);
  m_vertex_buffer->Bind();
  m_index_buffer->Bind();

  struct VAMapping
  {
    GLenum type;
    GLboolean normalized;
    GLboolean integer;
  };
  static constexpr u32 MAX_COMPONENTS = 4;
  static constexpr const std::array<VAMapping, static_cast<u8>(GPUPipeline::VertexAttribute::Type::MaxCount)>
    format_mapping = {{
      {GL_FLOAT, GL_FALSE, GL_FALSE},         // Float
      {GL_UNSIGNED_BYTE, GL_FALSE, GL_TRUE},  // UInt8
      {GL_BYTE, GL_FALSE, GL_TRUE},           // SInt8
      {GL_UNSIGNED_BYTE, GL_TRUE, GL_FALSE},  // UNorm8
      {GL_UNSIGNED_SHORT, GL_FALSE, GL_TRUE}, // UInt16
      {GL_SHORT, GL_FALSE, GL_TRUE},          // SInt16
      {GL_UNSIGNED_SHORT, GL_TRUE, GL_FALSE}, // UNorm16
      {GL_UNSIGNED_INT, GL_FALSE, GL_TRUE},   // UInt32
      {GL_INT, GL_FALSE, GL_TRUE},            // SInt32
    }};

  for (u32 i = 0; i < static_cast<u32>(attributes.size()); i++)
  {
    const GPUPipeline::VertexAttribute& va = attributes[i];
    const VAMapping& m = format_mapping[static_cast<u8>(va.type.GetValue())];
    const void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(va.offset.GetValue()));
    glEnableVertexAttribArray(i);
    if (m.integer)
      glVertexAttribIPointer(i, va.components, m.type, stride, ptr);
    else
      glVertexAttribPointer(i, va.components, m.type, m.normalized, stride, ptr);
  }

  glBindVertexArray(m_last_vao);

  return vao;
}

void OpenGLDevice::UnrefVAO(const OpenGLPipeline::VertexArrayCacheKey& key)
{
  auto it = m_vao_cache.find(key);
  Assert(it != m_vao_cache.end() && it->second.reference_count > 0);

  if ((--it->second.reference_count) > 0)
    return;

  if (m_last_vao == it->second.vao_id)
  {
    m_last_vao = 0;
    glBindVertexArray(0);
  }

  glDeleteVertexArrays(1, &it->second.vao_id);
  m_vao_cache.erase(it);
}

OpenGLPipeline::OpenGLPipeline(const ProgramCacheKey& key, GLuint program, GLuint vao, const RasterizationState& rs,
                               const DepthState& ds, const BlendState& bs, GLenum topology)
  : m_key(key), m_program(program), m_vao(vao), m_rasterization_state(rs), m_depth_state(ds), m_blend_state(bs),
    m_topology(topology)
{
}

OpenGLPipeline::~OpenGLPipeline()
{
  OpenGLDevice::GetInstance().UnbindPipeline(this);
}

void OpenGLPipeline::SetDebugName(const std::string_view& name)
{
#ifdef _DEBUG
  if (glObjectLabel)
    glObjectLabel(GL_PROGRAM, m_program, static_cast<u32>(name.length()), name.data());
#endif
}

std::unique_ptr<GPUPipeline> OpenGLDevice::CreatePipeline(const GPUPipeline::GraphicsConfig& config)
{
  const OpenGLPipeline::ProgramCacheKey pkey = OpenGLPipeline::GetProgramCacheKey(config);

  const GLuint program_id = LookupProgramCache(pkey, config);
  if (program_id == 0)
    return {};

  const GLuint vao_id = LookupVAOCache(pkey.va_key);
  if (vao_id == 0)
  {
    UnrefProgram(pkey);
    return {};
  }

  static constexpr std::array<GLenum, static_cast<u32>(GPUPipeline::Primitive::MaxCount)> primitives = {{
    GL_POINTS,         // Points
    GL_LINES,          // Lines
    GL_TRIANGLES,      // Triangles
    GL_TRIANGLE_STRIP, // TriangleStrips
  }};

  return std::unique_ptr<GPUPipeline>(new OpenGLPipeline(pkey, program_id, vao_id, config.rasterization, config.depth,
                                                         config.blend, primitives[static_cast<u8>(config.primitive)]));
}

ALWAYS_INLINE static void ApplyRasterizationState(const GPUPipeline::RasterizationState& rs)
{
  if (rs.cull_mode == GPUPipeline::CullMode::None)
  {
    glDisable(GL_CULL_FACE);
  }
  else
  {
    glEnable(GL_CULL_FACE);
    glCullFace((rs.cull_mode == GPUPipeline::CullMode::Front) ? GL_FRONT : GL_BACK);
  }

  // TODO: always enabled, should be done at init time
  glEnable(GL_SCISSOR_TEST);
}

ALWAYS_INLINE static void ApplyDepthState(const GPUPipeline::DepthState& ds)
{
  static constexpr std::array<GLenum, static_cast<u32>(GPUPipeline::DepthFunc::MaxCount)> func_mapping = {{
    GL_NEVER,   // Never
    GL_ALWAYS,  // Always
    GL_LESS,    // Less
    GL_LEQUAL,  // LessEqual
    GL_GREATER, // Greater
    GL_GEQUAL,  // GreaterEqual
    GL_EQUAL,   // Equal
  }};

  (ds.depth_test != GPUPipeline::DepthFunc::Never) ? glEnable(GL_DEPTH_TEST) : glDisable(GL_DEPTH_TEST);
  glDepthFunc(func_mapping[static_cast<u8>(ds.depth_test.GetValue())]);
  glDepthMask(ds.depth_write);
}

ALWAYS_INLINE static void ApplyBlendState(const GPUPipeline::BlendState& bs)
{
  static constexpr std::array<GLenum, static_cast<u32>(GPUPipeline::BlendFunc::MaxCount)> blend_mapping = {{
    GL_ZERO,                     // Zero
    GL_ONE,                      // One
    GL_SRC_COLOR,                // SrcColor
    GL_ONE_MINUS_SRC_COLOR,      // InvSrcColor
    GL_DST_COLOR,                // DstColor
    GL_ONE_MINUS_DST_COLOR,      // InvDstColor
    GL_SRC_ALPHA,                // SrcAlpha
    GL_ONE_MINUS_SRC_ALPHA,      // InvSrcAlpha
    GL_SRC1_ALPHA,               // SrcAlpha1
    GL_ONE_MINUS_SRC1_ALPHA,     // InvSrcAlpha1
    GL_DST_ALPHA,                // DstAlpha
    GL_ONE_MINUS_DST_ALPHA,      // InvDstAlpha
    GL_CONSTANT_COLOR,           // ConstantColor
    GL_ONE_MINUS_CONSTANT_COLOR, // InvConstantColor
  }};

  static constexpr std::array<GLenum, static_cast<u32>(GPUPipeline::BlendOp::MaxCount)> op_mapping = {{
    GL_FUNC_ADD,              // Add
    GL_FUNC_SUBTRACT,         // Subtract
    GL_FUNC_REVERSE_SUBTRACT, // ReverseSubtract
    GL_MIN,                   // Min
    GL_MAX,                   // Max
  }};

  // TODO: driver bugs
  // TODO: rdoc and look for redundant calls

  bs.enable ? glEnable(GL_BLEND) : glDisable(GL_BLEND);

  if (bs.enable)
  {
    glBlendFuncSeparate(blend_mapping[static_cast<u8>(bs.src_blend.GetValue())],
                        blend_mapping[static_cast<u8>(bs.dst_blend.GetValue())],
                        blend_mapping[static_cast<u8>(bs.src_alpha_blend.GetValue())],
                        blend_mapping[static_cast<u8>(bs.dst_alpha_blend.GetValue())]);
    glBlendEquationSeparate(op_mapping[static_cast<u8>(bs.blend_op.GetValue())],
                            op_mapping[static_cast<u8>(bs.alpha_blend_op.GetValue())]);

    // TODO: cache this to avoid calls?
    glBlendColor(bs.GetConstantRed(), bs.GetConstantGreen(), bs.GetConstantBlue(), bs.GetConstantAlpha());
  }

  glColorMask(bs.write_r, bs.write_g, bs.write_b, bs.write_a);
}

void OpenGLDevice::SetPipeline(GPUPipeline* pipeline)
{
  if (m_current_pipeline == pipeline)
    return;

  OpenGLPipeline* const P = static_cast<OpenGLPipeline*>(pipeline);
  m_current_pipeline = P;

  if (m_last_rasterization_state != P->GetRasterizationState())
  {
    m_last_rasterization_state = P->GetRasterizationState();
    ApplyRasterizationState(m_last_rasterization_state);
  }
  if (m_last_depth_state != P->GetDepthState())
  {
    m_last_depth_state = P->GetDepthState();
    ApplyDepthState(m_last_depth_state);
  }
  if (m_last_blend_state != P->GetBlendState())
  {
    m_last_blend_state = P->GetBlendState();
    ApplyBlendState(m_last_blend_state);
  }
  if (m_last_vao != P->GetVAO())
  {
    m_last_vao = P->GetVAO();
    glBindVertexArray(m_last_vao);
  }
  if (m_last_program != P->GetProgram())
  {
    m_last_program = P->GetProgram();
    glUseProgram(m_last_program);
  }
}
