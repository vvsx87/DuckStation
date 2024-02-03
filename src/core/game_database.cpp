// SPDX-FileCopyrightText: 2019-2023 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "game_database.h"
#include "host.h"
#include "system.h"

#include "util/cd_image.h"
#include "util/imgui_manager.h"

#include "common/assert.h"
#include "common/byte_stream.h"
#include "common/heterogeneous_containers.h"
#include "common/log.h"
#include "common/path.h"
#include "common/string_util.h"
#include "common/timer.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "ryml.hpp"
#include "ryml_std.hpp" // TODO: Remove this?

#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>

#include "IconsFontAwesome5.h"

Log_SetChannel(GameDatabase);

ALWAYS_INLINE std::string_view to_stringview(const c4::csubstr& s)
{
  return std::string_view(s.data(), s.size());
}

ALWAYS_INLINE std::string_view to_stringview(const c4::substr& s)
{
  return std::string_view(s.data(), s.size());
}

ALWAYS_INLINE c4::csubstr to_csubstr(const std::string_view& sv)
{
  return c4::csubstr(sv.data(), sv.length());
}

namespace GameDatabase {

enum : u32
{
  GAME_DATABASE_CACHE_SIGNATURE = 0x45434C48,
  GAME_DATABASE_CACHE_VERSION = 6,
};

static Entry* GetMutableEntry(const std::string_view& serial);
static const Entry* GetEntryForId(const std::string_view& code);

static bool LoadFromCache();
static bool SaveToCache();

static void SetRymlCallbacks();
static bool LoadGameDBJson();
static bool LoadGameDBYaml();
static bool ParseJsonEntry(Entry* entry, const rapidjson::Value& value);
static bool ParseJsonCodes(u32 index, const rapidjson::Value& value);
static bool ParseYamlEntry(Entry* entry, const ryml::ConstNodeRef& value);
static bool ParseYamlCodes(u32 index, const ryml::ConstNodeRef& value, std::string_view serial);
static bool LoadTrackHashes();

static std::array<const char*, static_cast<int>(CompatibilityRating::Count)> s_compatibility_rating_names = {
  {"Unknown", "DoesntBoot", "CrashesInIntro", "CrashesInGame", "GraphicalAudioIssues", "NoIssues"}};

static constexpr std::array<const char*, static_cast<size_t>(CompatibilityRating::Count)>
  s_compatibility_rating_display_names = {{TRANSLATE_NOOP("GameListCompatibilityRating", "Unknown"),
                                           TRANSLATE_NOOP("GameListCompatibilityRating", "Doesn't Boot"),
                                           TRANSLATE_NOOP("GameListCompatibilityRating", "Crashes In Intro"),
                                           TRANSLATE_NOOP("GameListCompatibilityRating", "Crashes In-Game"),
                                           TRANSLATE_NOOP("GameListCompatibilityRating", "Graphical/Audio Issues"),
                                           TRANSLATE_NOOP("GameListCompatibilityRating", "No Issues")}};

static std::array<const char*, static_cast<u32>(GameDatabase::Trait::Count)> s_trait_names = {{
  "forceInterpreter",
  "forceSoftwareRenderer",
  "forceSoftwareRendererForReadbacks",
  "forceInterlacing",
  "disableTrueColor",
  "disableUpscaling",
  "disableTextureFiltering",
  "disableScaledDithering",
  "disableForceNTSCTimings",
  "disableWidescreen",
  "disablePGXP",
  "disablePGXPCulling",
  "disablePGXPTextureCorrection",
  "disablePGXPColorCorrection",
  "disablePGXPDepthBuffer",
  "forcePGXPVertexCache",
  "forcePGXPCPUMode",
  "forceRecompilerMemoryExceptions",
  "forceRecompilerICache",
  "forceRecompilerLUTFastmem",
  "isLibCryptProtected",
}};

static std::array<const char*, static_cast<u32>(GameDatabase::Trait::Count)> s_jsontrait_names = {{
  "ForceInterpreter",
  "ForceSoftwareRenderer",
  "ForceSoftwareRendererForReadbacks",
  "ForceInterlacing",
  "DisableTrueColor",
  "DisableUpscaling",
  "DisableTextureFiltering",
  "DisableScaledDithering",
  "DisableForceNTSCTimings",
  "DisableWidescreen",
  "DisablePGXP",
  "DisablePGXPCulling",
  "DisablePGXPTextureCorrection",
  "DisablePGXPColorCorrection",
  "DisablePGXPDepthBuffer",
  "ForcePGXPVertexCache",
  "ForcePGXPCPUMode",
  "ForceRecompilerMemoryExceptions",
  "ForceRecompilerICache",
  "ForceRecompilerLUTFastmem",
  "IsLibCryptProtected",
}};

static bool s_loaded = false;
static bool s_track_hashes_loaded = false;

static std::vector<GameDatabase::Entry> s_entries;
static PreferUnorderedStringMap<u32> s_code_lookup;

static TrackHashesMap s_track_hashes_map;
} // namespace GameDatabase

void GameDatabase::EnsureLoaded()
{
  if (s_loaded)
    return;

  Common::Timer timer;

  s_loaded = true;

  LoadGameDBYaml();

  auto entries = std::move(s_entries);
  auto code_lookup = std::move(s_code_lookup);
  s_entries = {};
  s_code_lookup = {};

  LoadGameDBJson();

  Assert(code_lookup.size() == s_code_lookup.size());
  for (const auto& it : code_lookup)
  {
    const auto oit = s_code_lookup.find(it.first);
    Assert(oit != s_code_lookup.end() && oit->second == it.second);
  }

  Assert(entries.size() == s_entries.size());
  for (size_t i = 0; i < entries.size(); i++)
  {
    const GameDatabase::Entry& ee = s_entries[i];
    const GameDatabase::Entry& oe = entries[i];

#define CHE(f) Assert(ee.f == oe.f)
    CHE(serial);
    CHE(title);
    CHE(genre);
    CHE(developer);
    CHE(publisher);
    CHE(release_date);
    CHE(min_players);
    CHE(max_players);
    CHE(min_blocks);
    CHE(max_blocks);

    CHE(supported_controllers);
    CHE(compatibility);
    CHE(traits);
    CHE(display_active_start_offset);
    CHE(display_active_end_offset);
    CHE(display_line_start_offset);
    CHE(display_line_end_offset);
    CHE(dma_max_slice_ticks);
    CHE(dma_halt_ticks);
    CHE(gpu_fifo_size);
    CHE(gpu_max_run_ahead);
    CHE(gpu_pgxp_tolerance);
    CHE(gpu_pgxp_depth_threshold);
    CHE(disc_set_name);
    CHE(disc_set_serials);
#undef CHE
  }

  LoadTrackHashes();

#if 0
  if (!LoadFromCache())
  {
    s_entries = {};
    s_code_lookup = {};

    LoadGameDBJson();
    SaveToCache();
  }
#endif

  Log_InfoPrintf("Database load took %.2f ms", timer.GetTimeMilliseconds());
}

void GameDatabase::Unload()
{
  s_entries = {};
  s_code_lookup = {};
  s_loaded = false;
}

const GameDatabase::Entry* GameDatabase::GetEntryForId(const std::string_view& code)
{
  if (code.empty())
    return nullptr;

  EnsureLoaded();

  auto iter = s_code_lookup.find(code);
  return (iter != s_code_lookup.end()) ? &s_entries[iter->second] : nullptr;
}

std::string GameDatabase::GetSerialForDisc(CDImage* image)
{
  std::string ret;

  const GameDatabase::Entry* entry = GetEntryForDisc(image);
  if (entry)
    ret = entry->serial;

  return ret;
}

std::string GameDatabase::GetSerialForPath(const char* path)
{
  std::string ret;

  if (System::IsLoadableFilename(path) && !System::IsExeFileName(path) && !System::IsPsfFileName(path))
  {
    std::unique_ptr<CDImage> image(CDImage::Open(path, false, nullptr));
    if (image)
      ret = GetSerialForDisc(image.get());
  }

  return ret;
}

const GameDatabase::Entry* GameDatabase::GetEntryForDisc(CDImage* image)
{
  std::string id;
  System::GameHash hash;
  System::GetGameDetailsFromImage(image, &id, &hash);
  const Entry* entry = GetEntryForGameDetails(id, hash);
  if (entry)
    return entry;

  Log_WarningPrintf("No entry found for disc '%s'", id.c_str());
  return nullptr;
}

const GameDatabase::Entry* GameDatabase::GetEntryForGameDetails(const std::string& id, u64 hash)
{
  const Entry* entry;

  if (!id.empty())
  {
    entry = GetEntryForId(id);
    if (entry)
      return entry;
  }

  // some games with invalid serials use the hash
  entry = GetEntryForId(System::GetGameHashId(hash));
  if (entry)
    return entry;

  return nullptr;
}

const GameDatabase::Entry* GameDatabase::GetEntryForSerial(const std::string_view& serial)
{
  EnsureLoaded();

  return GetMutableEntry(serial);
}

GameDatabase::Entry* GameDatabase::GetMutableEntry(const std::string_view& serial)
{
  for (Entry& entry : s_entries)
  {
    if (entry.serial == serial)
      return &entry;
  }

  return nullptr;
}

const char* GameDatabase::GetCompatibilityRatingName(CompatibilityRating rating)
{
  return s_compatibility_rating_names[static_cast<int>(rating)];
}

const char* GameDatabase::GetCompatibilityRatingDisplayName(CompatibilityRating rating)
{
  return (rating >= CompatibilityRating::Unknown && rating < CompatibilityRating::Count) ?
           Host::TranslateToCString("GameListCompatibilityRating",
                                    s_compatibility_rating_display_names[static_cast<int>(rating)]) :
           "";
}

void GameDatabase::Entry::ApplySettings(Settings& settings, bool display_osd_messages) const
{
  constexpr float osd_duration = Host::OSD_INFO_DURATION;

  if (display_active_start_offset.has_value())
    settings.display_active_start_offset = display_active_start_offset.value();
  if (display_active_end_offset.has_value())
    settings.display_active_end_offset = display_active_end_offset.value();
  if (display_line_start_offset.has_value())
    settings.display_line_start_offset = display_line_start_offset.value();
  if (display_line_end_offset.has_value())
    settings.display_line_end_offset = display_line_end_offset.value();
  if (dma_max_slice_ticks.has_value())
    settings.dma_max_slice_ticks = dma_max_slice_ticks.value();
  if (dma_halt_ticks.has_value())
    settings.dma_halt_ticks = dma_halt_ticks.value();
  if (gpu_fifo_size.has_value())
    settings.gpu_fifo_size = gpu_fifo_size.value();
  if (gpu_max_run_ahead.has_value())
    settings.gpu_max_run_ahead = gpu_max_run_ahead.value();
  if (gpu_pgxp_tolerance.has_value())
    settings.gpu_pgxp_tolerance = gpu_pgxp_tolerance.value();
  if (gpu_pgxp_depth_threshold.has_value())
    settings.SetPGXPDepthClearThreshold(gpu_pgxp_depth_threshold.value());

  if (HasTrait(Trait::ForceInterpreter))
  {
    if (display_osd_messages && settings.cpu_execution_mode != CPUExecutionMode::Interpreter)
    {
      Host::AddIconOSDMessage("gamedb_force_interpreter", ICON_FA_MICROCHIP,
                              TRANSLATE_STR("OSDMessage", "CPU interpreter forced by compatibility settings."),
                              osd_duration);
    }

    settings.cpu_execution_mode = CPUExecutionMode::Interpreter;
  }

  if (HasTrait(Trait::ForceSoftwareRenderer))
  {
    if (display_osd_messages && settings.gpu_renderer != GPURenderer::Software)
    {
      Host::AddIconOSDMessage("gamedb_force_software", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "Software renderer forced by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_renderer = GPURenderer::Software;
  }

  if (HasTrait(Trait::ForceSoftwareRendererForReadbacks))
  {
    if (display_osd_messages && settings.gpu_renderer != GPURenderer::Software)
    {
      Host::AddIconOSDMessage(
        "gamedb_force_software_rb", ICON_FA_MAGIC,
        TRANSLATE_STR("OSDMessage", "Using software renderer for readbacks based on compatibility settings."),
        osd_duration);
    }

    settings.gpu_use_software_renderer_for_readbacks = true;
  }

  if (HasTrait(Trait::ForceInterlacing))
  {
    if (display_osd_messages && settings.gpu_disable_interlacing)
    {
      Host::AddIconOSDMessage("gamedb_force_interlacing", ICON_FA_TV,
                              TRANSLATE_STR("OSDMessage", "Interlacing forced by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_disable_interlacing = false;
  }

  if (HasTrait(Trait::DisableTrueColor))
  {
    if (display_osd_messages && settings.gpu_true_color)
    {
      Host::AddIconOSDMessage("gamedb_disable_true_color", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "True color disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_true_color = false;
  }

  if (HasTrait(Trait::DisableUpscaling))
  {
    if (display_osd_messages && settings.gpu_resolution_scale > 1)
    {
      Host::AddIconOSDMessage("gamedb_disable_upscaling", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "Upscaling disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_resolution_scale = 1;
  }

  if (HasTrait(Trait::DisableTextureFiltering))
  {
    if (display_osd_messages && settings.gpu_texture_filter != GPUTextureFilter::Nearest)
    {
      Host::AddIconOSDMessage("gamedb_disable_upscaling", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "Texture filtering disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_texture_filter = GPUTextureFilter::Nearest;
  }

  if (HasTrait(Trait::DisableScaledDithering))
  {
    if (display_osd_messages && settings.gpu_scaled_dithering)
    {
      Host::AddIconOSDMessage("gamedb_disable_scaled_dithering", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "Scaled dithering disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_scaled_dithering = false;
  }

  if (HasTrait(Trait::DisableWidescreen))
  {
    if (display_osd_messages && settings.gpu_widescreen_hack)
    {
      Host::AddIconOSDMessage("gamedb_disable_widescreen", ICON_FA_TV,
                              TRANSLATE_STR("OSDMessage", "Widescreen rendering disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_widescreen_hack = false;
  }

  if (HasTrait(Trait::DisableForceNTSCTimings))
  {
    if (display_osd_messages && settings.gpu_force_ntsc_timings)
    {
      Host::AddIconOSDMessage("gamedb_disable_force_ntsc_timings", ICON_FA_TV,
                              TRANSLATE_STR("OSDMessage", "Forcing NTSC Timings disallowed by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_force_ntsc_timings = false;
  }

  if (HasTrait(Trait::DisablePGXP))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable)
    {
      Host::AddIconOSDMessage(
        "gamedb_disable_pgxp", ICON_FA_MAGIC,
        TRANSLATE_STR("OSDMessage", "PGXP geometry correction disabled by compatibility settings."), osd_duration);
    }

    settings.gpu_pgxp_enable = false;
  }

  if (HasTrait(Trait::DisablePGXPCulling))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && settings.gpu_pgxp_culling)
    {
      Host::AddIconOSDMessage("gamedb_disable_pgxp_culling", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "PGXP culling disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_pgxp_culling = false;
  }

  if (HasTrait(Trait::DisablePGXPTextureCorrection))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && settings.gpu_pgxp_texture_correction)
    {
      Host::AddIconOSDMessage(
        "gamedb_disable_pgxp_texture", ICON_FA_MAGIC,
        TRANSLATE_STR("OSDMessage", "PGXP perspective corrected textures disabled by compatibility settings."),
        osd_duration);
    }

    settings.gpu_pgxp_texture_correction = false;
  }

  if (HasTrait(Trait::DisablePGXPColorCorrection))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && settings.gpu_pgxp_texture_correction &&
        settings.gpu_pgxp_color_correction)
    {
      Host::AddIconOSDMessage(
        "gamedb_disable_pgxp_texture", ICON_FA_MAGIC,
        TRANSLATE_STR("OSDMessage", "PGXP perspective corrected colors disabled by compatibility settings."),
        osd_duration);
    }

    settings.gpu_pgxp_color_correction = false;
  }

  if (HasTrait(Trait::ForcePGXPVertexCache))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && !settings.gpu_pgxp_vertex_cache)
    {
      Host::AddIconOSDMessage("gamedb_force_pgxp_vertex_cache", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "PGXP vertex cache forced by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_pgxp_vertex_cache = true;
  }

  if (HasTrait(Trait::ForcePGXPCPUMode))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && !settings.gpu_pgxp_cpu)
    {
#ifndef __ANDROID__
      Host::AddIconOSDMessage("gamedb_force_pgxp_cpu", ICON_FA_MICROCHIP,
                              TRANSLATE_STR("OSDMessage", "PGXP CPU mode forced by compatibility settings."),
                              osd_duration);
#else
      Host::AddIconOSDMessage(
        "gamedb_force_pgxp_cpu", ICON_FA_MICROCHIP,
        "This game requires PGXP CPU mode, which increases system requirements.\n" ICON_FA_EXCLAMATION_TRIANGLE
        "  If the game runs too slow, disable PGXP for this game.",
        Host::OSD_WARNING_DURATION);
#endif
    }

    settings.gpu_pgxp_cpu = true;
  }

  if (HasTrait(Trait::DisablePGXPDepthBuffer))
  {
    if (display_osd_messages && settings.gpu_pgxp_enable && settings.gpu_pgxp_depth_buffer)
    {
      Host::AddIconOSDMessage("gamedb_disable_pgxp_depth", ICON_FA_MAGIC,
                              TRANSLATE_STR("OSDMessage", "PGXP Depth Buffer disabled by compatibility settings."),
                              osd_duration);
    }

    settings.gpu_pgxp_depth_buffer = false;
  }

  if (HasTrait(Trait::ForceRecompilerMemoryExceptions))
  {
    Log_WarningPrint("Memory exceptions for recompiler forced by compatibility settings.");
    settings.cpu_recompiler_memory_exceptions = true;
  }

  if (HasTrait(Trait::ForceRecompilerICache))
  {
    Log_WarningPrint("ICache for recompiler forced by compatibility settings.");
    settings.cpu_recompiler_icache = true;
  }

  if (settings.cpu_fastmem_mode == CPUFastmemMode::MMap && HasTrait(Trait::ForceRecompilerLUTFastmem))
  {
    Log_WarningPrint("LUT fastmem for recompiler forced by compatibility settings.");
    settings.cpu_fastmem_mode = CPUFastmemMode::LUT;
  }

#define BIT_FOR(ctype) (static_cast<u16>(1) << static_cast<u32>(ctype))

  if (supported_controllers != 0 && supported_controllers != static_cast<u16>(-1))
  {
    for (u32 i = 0; i < NUM_CONTROLLER_AND_CARD_PORTS; i++)
    {
      const ControllerType ctype = settings.controller_types[i];
      if (ctype == ControllerType::None)
        continue;

      if (supported_controllers & BIT_FOR(ctype))
        continue;

      // Special case: Dualshock is permitted when not supported as long as it's in digital mode.
      if (ctype == ControllerType::AnalogController &&
          (supported_controllers & BIT_FOR(ControllerType::DigitalController)) != 0)
      {
        settings.controller_disable_analog_mode_forcing = true;
        continue;
      }

      if (display_osd_messages)
      {
        SmallString supported_controller_string;
        for (u32 j = 0; j < static_cast<u32>(ControllerType::Count); j++)
        {
          const ControllerType supported_ctype = static_cast<ControllerType>(j);
          if ((supported_controllers & BIT_FOR(supported_ctype)) == 0)
            continue;

          if (!supported_controller_string.empty())
            supported_controller_string.append(", ");

          supported_controller_string.append(Settings::GetControllerTypeDisplayName(supported_ctype));
        }

        Host::AddKeyedOSDMessage(
          "gamedb_controller_unsupported",
          fmt::format(
            TRANSLATE_FS("OSDMessage", "Controller in port {0} ({1}) is not supported for {2}.\nSupported controllers: "
                                       "{3}\nPlease configure a supported controller from the list above."),
            i + 1u, Settings::GetControllerTypeDisplayName(ctype), System::GetGameTitle(), supported_controller_string),
          Host::OSD_CRITICAL_ERROR_DURATION);
      }
    }
  }

#undef BIT_FOR
}

template<typename T>
bool ReadOptionalFromStream(ByteStream* stream, std::optional<T>* dest)
{
  bool has_value;
  if (!stream->Read2(&has_value, sizeof(has_value)))
    return false;

  if (!has_value)
    return true;

  T value;
  if (!stream->Read2(&value, sizeof(T)))
    return false;

  *dest = value;
  return true;
}

template<typename T>
bool WriteOptionalToStream(ByteStream* stream, const std::optional<T>& src)
{
  const bool has_value = src.has_value();
  if (!stream->Write2(&has_value, sizeof(has_value)))
    return false;

  if (!has_value)
    return true;

  return stream->Write2(&src.value(), sizeof(T));
}

static std::string GetCacheFile()
{
  return Path::Combine(EmuFolders::Cache, "gamedb.cache");
}

bool GameDatabase::LoadFromCache()
{
  std::unique_ptr<ByteStream> stream(
    ByteStream::OpenFile(GetCacheFile().c_str(), BYTESTREAM_OPEN_READ | BYTESTREAM_OPEN_STREAMED));
  if (!stream)
  {
    Log_DevPrintf("Cache does not exist, loading full database.");
    return false;
  }

  const u64 gamedb_ts = Host::GetResourceFileTimestamp("gamedb.yaml", false).value_or(0);

  u32 signature, version, num_entries, num_codes;
  u64 file_gamedb_ts;
  if (!stream->ReadU32(&signature) || !stream->ReadU32(&version) || !stream->ReadU64(&file_gamedb_ts) ||
      !stream->ReadU32(&num_entries) || !stream->ReadU32(&num_codes) || signature != GAME_DATABASE_CACHE_SIGNATURE ||
      version != GAME_DATABASE_CACHE_VERSION)
  {
    Log_DevPrintf("Cache header is corrupted or version mismatch.");
    return false;
  }

  if (gamedb_ts != file_gamedb_ts)
  {
    Log_DevPrintf("Cache is out of date, recreating.");
    return false;
  }

  s_entries.reserve(num_entries);

  for (u32 i = 0; i < num_entries; i++)
  {
    Entry& entry = s_entries.emplace_back();

    constexpr u32 num_bytes = (static_cast<u32>(Trait::Count) + 7) / 8;
    std::array<u8, num_bytes> bits;
    u8 compatibility;
    u32 num_disc_set_serials;

    if (!stream->ReadSizePrefixedString(&entry.serial) || !stream->ReadSizePrefixedString(&entry.title) ||
        !stream->ReadSizePrefixedString(&entry.genre) || !stream->ReadSizePrefixedString(&entry.developer) ||
        !stream->ReadSizePrefixedString(&entry.publisher) || !stream->ReadU64(&entry.release_date) ||
        !stream->ReadU8(&entry.min_players) || !stream->ReadU8(&entry.max_players) ||
        !stream->ReadU8(&entry.min_blocks) || !stream->ReadU8(&entry.max_blocks) ||
        !stream->ReadU16(&entry.supported_controllers) || !stream->ReadU8(&compatibility) ||
        compatibility >= static_cast<u8>(GameDatabase::CompatibilityRating::Count) ||
        !stream->Read2(bits.data(), num_bytes) ||
        !ReadOptionalFromStream(stream.get(), &entry.display_active_start_offset) ||
        !ReadOptionalFromStream(stream.get(), &entry.display_active_end_offset) ||
        !ReadOptionalFromStream(stream.get(), &entry.display_line_start_offset) ||
        !ReadOptionalFromStream(stream.get(), &entry.display_line_end_offset) ||
        !ReadOptionalFromStream(stream.get(), &entry.dma_max_slice_ticks) ||
        !ReadOptionalFromStream(stream.get(), &entry.dma_halt_ticks) ||
        !ReadOptionalFromStream(stream.get(), &entry.gpu_fifo_size) ||
        !ReadOptionalFromStream(stream.get(), &entry.gpu_max_run_ahead) ||
        !ReadOptionalFromStream(stream.get(), &entry.gpu_pgxp_tolerance) ||
        !ReadOptionalFromStream(stream.get(), &entry.gpu_pgxp_depth_threshold) ||
        !stream->ReadSizePrefixedString(&entry.disc_set_name) || !stream->ReadU32(&num_disc_set_serials))
    {
      Log_DevPrintf("Cache entry is corrupted.");
      return false;
    }

    if (num_disc_set_serials > 0)
    {
      entry.disc_set_serials.reserve(num_disc_set_serials);
      for (u32 j = 0; j < num_disc_set_serials; j++)
      {
        if (!stream->ReadSizePrefixedString(&entry.disc_set_serials.emplace_back()))
        {
          Log_DevPrintf("Cache entry is corrupted.");
          return false;
        }
      }
    }

    entry.compatibility = static_cast<GameDatabase::CompatibilityRating>(compatibility);
    entry.traits.reset();
    for (u32 j = 0; j < static_cast<int>(Trait::Count); j++)
    {
      if ((bits[j / 8] & (1u << (j % 8))) != 0)
        entry.traits[j] = true;
    }
  }

  for (u32 i = 0; i < num_codes; i++)
  {
    std::string code;
    u32 index;
    if (!stream->ReadSizePrefixedString(&code) || !stream->ReadU32(&index) ||
        index >= static_cast<u32>(s_entries.size()))
    {
      Log_DevPrintf("Cache code entry is corrupted.");
      return false;
    }

    s_code_lookup.emplace(std::move(code), index);
  }

  return true;
}

bool GameDatabase::SaveToCache()
{
  const u64 gamedb_ts = Host::GetResourceFileTimestamp("gamedb.yaml", false).value_or(0);

  std::unique_ptr<ByteStream> stream(
    ByteStream::OpenFile(GetCacheFile().c_str(), BYTESTREAM_OPEN_CREATE | BYTESTREAM_OPEN_WRITE |
                                                   BYTESTREAM_OPEN_TRUNCATE | BYTESTREAM_OPEN_STREAMED));
  if (!stream)
    return false;

  bool result = stream->WriteU32(GAME_DATABASE_CACHE_SIGNATURE);
  result = result && stream->WriteU32(GAME_DATABASE_CACHE_VERSION);
  result = result && stream->WriteU64(static_cast<u64>(gamedb_ts));

  result = result && stream->WriteU32(static_cast<u32>(s_entries.size()));
  result = result && stream->WriteU32(static_cast<u32>(s_code_lookup.size()));

  for (const Entry& entry : s_entries)
  {
    result = result && stream->WriteSizePrefixedString(entry.serial);
    result = result && stream->WriteSizePrefixedString(entry.title);
    result = result && stream->WriteSizePrefixedString(entry.genre);
    result = result && stream->WriteSizePrefixedString(entry.developer);
    result = result && stream->WriteSizePrefixedString(entry.publisher);
    result = result && stream->WriteU64(entry.release_date);
    result = result && stream->WriteU8(entry.min_players);
    result = result && stream->WriteU8(entry.max_players);
    result = result && stream->WriteU8(entry.min_blocks);
    result = result && stream->WriteU8(entry.max_blocks);
    result = result && stream->WriteU16(entry.supported_controllers);
    result = result && stream->WriteU8(static_cast<u8>(entry.compatibility));

    constexpr u32 num_bytes = (static_cast<u32>(Trait::Count) + 7) / 8;
    std::array<u8, num_bytes> bits;
    bits.fill(0);
    for (u32 j = 0; j < static_cast<int>(Trait::Count); j++)
    {
      if (entry.traits[j])
        bits[j / 8] |= (1u << (j % 8));
    }

    result = result && stream->Write2(bits.data(), num_bytes);

    result = result && WriteOptionalToStream(stream.get(), entry.display_active_start_offset);
    result = result && WriteOptionalToStream(stream.get(), entry.display_active_end_offset);
    result = result && WriteOptionalToStream(stream.get(), entry.display_line_start_offset);
    result = result && WriteOptionalToStream(stream.get(), entry.display_line_end_offset);
    result = result && WriteOptionalToStream(stream.get(), entry.dma_max_slice_ticks);
    result = result && WriteOptionalToStream(stream.get(), entry.dma_halt_ticks);
    result = result && WriteOptionalToStream(stream.get(), entry.gpu_fifo_size);
    result = result && WriteOptionalToStream(stream.get(), entry.gpu_max_run_ahead);
    result = result && WriteOptionalToStream(stream.get(), entry.gpu_pgxp_tolerance);
    result = result && WriteOptionalToStream(stream.get(), entry.gpu_pgxp_depth_threshold);

    result = result && stream->WriteSizePrefixedString(entry.disc_set_name);
    result = result && stream->WriteU32(static_cast<u32>(entry.disc_set_serials.size()));
    for (const std::string& serial : entry.disc_set_serials)
      result = result && stream->WriteSizePrefixedString(serial);
  }

  for (const auto& it : s_code_lookup)
  {
    result = result && stream->WriteSizePrefixedString(it.first);
    result = result && stream->WriteU32(it.second);
  }

  result = result && stream->Flush();
  return true;
}

//////////////////////////////////////////////////////////////////////////
// JSON Parsing
//////////////////////////////////////////////////////////////////////////

static bool GetStringFromObject(const rapidjson::Value& object, const char* key, std::string* dest)
{
  dest->clear();
  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsString())
    return false;

  dest->assign(member->value.GetString(), member->value.GetStringLength());
  return true;
}

static bool GetStringFromObject(const ryml::ConstNodeRef& object, std::string_view key, std::string* dest)
{
  dest->clear();

  const ryml::ConstNodeRef member = object.find_child(to_csubstr(key));
  if (!member.valid())
    return false;

  const c4::csubstr val = member.val();
  if (!val.empty())
    dest->assign(val.data(), val.size());

  return true;
}

static bool GetBoolFromObject(const rapidjson::Value& object, const char* key, bool* dest)
{
  *dest = false;

  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsBool())
    return false;

  *dest = member->value.GetBool();
  return true;
}

static bool GetBoolFromObject(const ryml::ConstNodeRef& object, std::string_view key, bool* dest)
{
  *dest = false;

  const ryml::ConstNodeRef member = object.find_child(to_csubstr(key));
  if (!member.valid())
    return false;

  const c4::csubstr val = member.val();
  if (val.empty())
  {
    Log_ErrorFmt("Unexpected empty value in {}", key);
    return false;
  }

  const std::optional<bool> opt_value = StringUtil::FromChars<bool>(to_stringview(val));
  if (!opt_value.has_value())
  {
    Log_ErrorFmt("Unexpected non-bool value in {}", key);
    return false;
  }

  *dest = opt_value.value();
  return true;
}

template<typename T>
static bool GetUIntFromObject(const rapidjson::Value& object, const char* key, T* dest)
{
  *dest = 0;

  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsUint())
    return false;

  *dest = static_cast<T>(member->value.GetUint());
  return true;
}

template<typename T>
static bool GetUIntFromObject(const ryml::ConstNodeRef& object, std::string_view key, T* dest)
{
  *dest = 0;

  const ryml::ConstNodeRef member = object.find_child(to_csubstr(key));
  if (!member.valid())
    return false;

  const c4::csubstr val = member.val();
  if (val.empty())
  {
    Log_ErrorFmt("Unexpected empty value in {}", key);
    return false;
  }

  // TODO: Check for int??
  const std::optional<T> opt_value = StringUtil::FromChars<T>(to_stringview(val));
  if (!opt_value.has_value())
  {
    Log_ErrorFmt("Unexpected non-uint value in {}", key);
    return false;
  }

  *dest = opt_value.value();
  return true;
}

template<typename T>
static std::optional<T> GetOptionalIntFromObject(const rapidjson::Value& object, const char* key)
{
  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsInt())
    return std::nullopt;

  return static_cast<T>(member->value.GetInt());
}

template<typename T>
static std::optional<T> GetOptionalUIntFromObject(const rapidjson::Value& object, const char* key)
{
  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsUint())
    return std::nullopt;

  return static_cast<T>(member->value.GetUint());
}

static std::optional<float> GetOptionalFloatFromObject(const rapidjson::Value& object, const char* key)
{
  auto member = object.FindMember(key);
  if (member == object.MemberEnd() || !member->value.IsFloat())
    return std::nullopt;

  return member->value.GetFloat();
}

template<typename T>
static std::optional<T> GetOptionalTFromObject(const ryml::ConstNodeRef& object, std::string_view key)
{
  std::optional<T> ret;

  const ryml::ConstNodeRef member = object.find_child(to_csubstr(key));
  if (member.valid())
  {
    const c4::csubstr val = member.val();
    if (!val.empty())
    {
      ret = StringUtil::FromChars<T>(to_stringview(val));
      if (!ret.has_value())
      {
        if constexpr (std::is_floating_point_v<T>)
          Log_ErrorFmt("Unexpected non-float value in {}", key);
        else if constexpr (std::is_integral_v<T>)
          Log_ErrorFmt("Unexpected non-int value in {}", key);
      }
    }
    else
    {
      Log_ErrorFmt("Unexpected empty value in {}", key);
    }
  }

  return ret;
}

bool GameDatabase::LoadGameDBJson()
{
  std::optional<std::string> gamedb_data(Host::ReadResourceFileToString("gamedb.json", false));
  if (!gamedb_data.has_value())
  {
    Log_ErrorPrintf("Failed to read game database");
    return false;
  }

  // TODO: Parse in-place, avoid string allocations.
  std::unique_ptr<rapidjson::Document> json = std::make_unique<rapidjson::Document>();
  json->Parse(gamedb_data->c_str(), gamedb_data->size());
  if (json->HasParseError())
  {
    Log_ErrorPrintf("Failed to parse game database: %s at offset %zu",
                    rapidjson::GetParseError_En(json->GetParseError()), json->GetErrorOffset());
    return false;
  }

  if (!json->IsArray())
  {
    Log_ErrorPrintf("Document is not an array");
    return false;
  }

  const auto& jarray = json->GetArray();
  s_entries.reserve(jarray.Size());

  for (const rapidjson::Value& current : json->GetArray())
  {
    // TODO: binary sort
    const u32 index = static_cast<u32>(s_entries.size());
    Entry& entry = s_entries.emplace_back();
    if (!ParseJsonEntry(&entry, current))
    {
      s_entries.pop_back();
      continue;
    }

    ParseJsonCodes(index, current);
  }

  Log_InfoPrintf("Loaded %zu entries and %zu codes from database", s_entries.size(), s_code_lookup.size());
  return true;
}

void GameDatabase::SetRymlCallbacks()
{
  ryml::Callbacks callbacks = ryml::get_callbacks();
  callbacks.m_error = [](const char* msg, size_t msg_len, ryml::Location loc, void* userdata) {
    Log_ErrorFmt("Parse error at {}:{} (bufpos={}): {}", loc.line, loc.col, loc.offset, std::string_view(msg, msg_len));
  };
  ryml::set_callbacks(callbacks);
  c4::set_error_callback(
    [](const char* msg, size_t msg_size) { Log_ErrorFmt("C4 error: {}", std::string_view(msg, msg_size)); });
}

bool GameDatabase::LoadGameDBYaml()
{
  Common::Timer timer;

  const std::optional<std::string> gamedb_data = Host::ReadResourceFileToString("gamedb.yaml", false);
  if (!gamedb_data.has_value())
  {
    Log_ErrorPrint("Failed to read game database");
    return false;
  }

  SetRymlCallbacks();

  const ryml::Tree tree = ryml::parse_in_arena(c4::to_csubstr(gamedb_data.value()));
  const ryml::ConstNodeRef root = tree.rootref();
  s_entries.reserve(root.num_children());

  for (const ryml::ConstNodeRef& current : root.children())
  {
    // TODO: binary sort
    const u32 index = static_cast<u32>(s_entries.size());
    Entry& entry = s_entries.emplace_back();
    if (!ParseYamlEntry(&entry, current))
    {
      s_entries.pop_back();
      continue;
    }

    ParseYamlCodes(index, current, entry.serial);
  }

  ryml::reset_callbacks();

  Log_InfoFmt("Loaded {} entries and {} codes from database in {:.0f}ms.", s_entries.size(), s_code_lookup.size(),
              timer.GetTimeMilliseconds());
  return !s_entries.empty();
}

bool GameDatabase::ParseYamlEntry(Entry* entry, const ryml::ConstNodeRef& value)
{
  entry->serial = to_stringview(value.key());
  if (entry->serial.empty())
  {
    Log_ErrorPrint("Missing serial for entry.");
    return false;
  }

  GetStringFromObject(value, "name", &entry->title);

  if (const ryml::ConstNodeRef metadata = value.find_child(to_csubstr("metadata")); metadata.valid())
  {
    GetStringFromObject(metadata, "genre", &entry->genre);
    GetStringFromObject(metadata, "developer", &entry->developer);
    GetStringFromObject(metadata, "publisher", &entry->publisher);

    GetUIntFromObject(metadata, "minPlayers", &entry->min_players);
    GetUIntFromObject(metadata, "maxPlayers", &entry->max_players);
    GetUIntFromObject(metadata, "minBlocks", &entry->min_blocks);
    GetUIntFromObject(metadata, "maxBlocks", &entry->max_blocks);

    entry->release_date = 0;
    {
      std::string release_date;
      if (GetStringFromObject(metadata, "releaseDate", &release_date))
      {
        std::istringstream iss(release_date);
        struct tm parsed_time = {};
        iss >> std::get_time(&parsed_time, "%Y-%m-%d");
        if (!iss.fail())
        {
          parsed_time.tm_isdst = 0;
#ifdef _WIN32
          entry->release_date = _mkgmtime(&parsed_time);
#else
          entry->release_date = timegm(&parsed_time);
#endif
        }
      }
    }
  }

  entry->supported_controllers = static_cast<u16>(~0u);

  if (const ryml::ConstNodeRef controllers = value.find_child(to_csubstr("controllers"));
      controllers.valid() && controllers.has_children())
  {
    bool first = true;
    for (const ryml::ConstNodeRef& controller : controllers.children())
    {
      const std::string_view controller_str = to_stringview(controller.val());
      if (controller_str.empty())
      {
        Log_WarningFmt("controller is not a string in {}", entry->serial);
        return false;
      }

      std::optional<ControllerType> ctype = Settings::ParseControllerTypeName(controller_str);
      if (!ctype.has_value())
      {
        Log_WarningFmt("Invalid controller type {} in {}", controller_str, entry->serial);
        continue;
      }

      if (first)
      {
        entry->supported_controllers = 0;
        first = false;
      }

      entry->supported_controllers |= (1u << static_cast<u16>(ctype.value()));
    }
  }

  if (const ryml::ConstNodeRef compatibility = value.find_child(to_csubstr("compatibility"));
      compatibility.valid() && compatibility.has_children())
  {
    const ryml::ConstNodeRef rating = compatibility.find_child(to_csubstr("rating"));
    if (rating.valid())
    {
      const std::string_view rating_str = to_stringview(rating.val());

      const auto iter = std::find(s_compatibility_rating_names.begin(), s_compatibility_rating_names.end(), rating_str);
      if (iter != s_compatibility_rating_names.end())
      {
        const size_t rating_idx = static_cast<size_t>(std::distance(s_compatibility_rating_names.begin(), iter));
        DebugAssert(rating_idx < static_cast<size_t>(CompatibilityRating::Count));
        entry->compatibility = static_cast<CompatibilityRating>(rating_idx);
      }
      else
      {
        Log_WarningFmt("Unknown compatibility rating {} in {}", rating_str, entry->serial);
      }
    }
  }

  if (const ryml::ConstNodeRef traits = value.find_child(to_csubstr("traits")); traits.valid() && traits.has_children())
  {
    for (const ryml::ConstNodeRef& trait : traits.children())
    {
      const std::string_view trait_str = to_stringview(trait.val());
      if (trait_str.empty())
      {
        Log_WarningFmt("Empty trait in {}", entry->serial);
        continue;
      }

      const auto iter = std::find(s_trait_names.begin(), s_trait_names.end(), trait_str);
      if (iter == s_trait_names.end())
      {
        Log_WarningFmt("Unknown trait {} in {}", trait_str, entry->serial);
        continue;
      }

      const size_t trait_idx = static_cast<size_t>(std::distance(s_trait_names.begin(), iter));
      DebugAssert(trait_idx < static_cast<size_t>(Trait::Count));
      entry->traits[trait_idx] = true;
    }
  }

  if (const ryml::ConstNodeRef settings = value.find_child(to_csubstr("settings"));
      settings.valid() && settings.has_children())
  {
    entry->display_active_start_offset = GetOptionalTFromObject<s16>(settings, "displayActiveStartOffset");
    entry->display_active_end_offset = GetOptionalTFromObject<s16>(settings, "displayActiveEndOffset");
    entry->display_line_start_offset = GetOptionalTFromObject<s8>(settings, "displayLineStartOffset");
    entry->display_line_end_offset = GetOptionalTFromObject<s8>(settings, "displayLineEndOffset");
    entry->dma_max_slice_ticks = GetOptionalTFromObject<u32>(settings, "dmaMaxSliceTicks");
    entry->dma_halt_ticks = GetOptionalTFromObject<u32>(settings, "dmaHaltTicks");
    entry->gpu_fifo_size = GetOptionalTFromObject<u32>(settings, "gpuFIFOSize");
    entry->gpu_max_run_ahead = GetOptionalTFromObject<u32>(settings, "gpuMaxRunAhead");
    entry->gpu_pgxp_tolerance = GetOptionalTFromObject<float>(settings, "gpuPGXPTolerance");
    entry->gpu_pgxp_depth_threshold = GetOptionalTFromObject<float>(settings, "gpuPGXPDepthThreshold");
  }

  if (const ryml::ConstNodeRef disc_set = value.find_child("discSet"); disc_set.valid() && disc_set.has_children())
  {
    GetStringFromObject(disc_set, "name", &entry->disc_set_name);

    if (const ryml::ConstNodeRef set_serials = disc_set.find_child("serials");
        set_serials.valid() && set_serials.has_children())
    {
      entry->disc_set_serials.reserve(set_serials.num_children());
      for (const ryml::ConstNodeRef& serial : set_serials)
      {
        const std::string_view serial_str = to_stringview(serial.val());
        if (serial_str.empty())
        {
          Log_WarningFmt("Empty disc set serial in {}", entry->serial);
          continue;
        }

        if (std::find(entry->disc_set_serials.begin(), entry->disc_set_serials.end(), serial_str) !=
            entry->disc_set_serials.end())
        {
          Log_WarningFmt("Duplicate serial {} in disc set serials for {}", serial_str, entry->serial);
          continue;
        }

        entry->disc_set_serials.emplace_back(serial_str);
      }
    }
  }

  return true;
}

bool GameDatabase::ParseYamlCodes(u32 index, const ryml::ConstNodeRef& value, std::string_view serial)
{
  const ryml::ConstNodeRef& codes = value.find_child(to_csubstr("codes"));
  if (!codes.valid() || !codes.has_children())
  {
    // use serial instead
    auto iter = s_code_lookup.find(serial);
    if (iter != s_code_lookup.end())
    {
      Log_WarningFmt("Duplicate code '{}'", serial);
      return false;
    }

    s_code_lookup.emplace(serial, index);
    return true;
  }

  u32 added = 0;
  for (const ryml::ConstNodeRef& current_code : codes)
  {
    const std::string_view current_code_str = to_stringview(current_code.val());
    if (current_code_str.empty())
    {
      Log_WarningFmt("code is not a string in {}", serial);
      continue;
    }

    auto iter = s_code_lookup.find(current_code_str);
    if (iter != s_code_lookup.end())
    {
      Log_WarningFmt("Duplicate code '{}' in {}", current_code_str, serial);
      continue;
    }

    s_code_lookup.emplace(current_code_str, index);
    added++;
  }

  return (added > 0);
}

bool GameDatabase::ParseJsonEntry(Entry* entry, const rapidjson::Value& value)
{
  if (!value.IsObject())
  {
    Log_WarningPrintf("entry is not an object");
    return false;
  }

  if (!GetStringFromObject(value, "serial", &entry->serial) || !GetStringFromObject(value, "name", &entry->title) ||
      entry->serial.empty())
  {
    Log_ErrorPrintf("Missing serial or title for entry");
    return false;
  }

  GetStringFromObject(value, "genre", &entry->genre);
  GetStringFromObject(value, "developer", &entry->developer);
  GetStringFromObject(value, "publisher", &entry->publisher);

  GetUIntFromObject(value, "minPlayers", &entry->min_players);
  GetUIntFromObject(value, "maxPlayers", &entry->max_players);
  GetUIntFromObject(value, "minBlocks", &entry->min_blocks);
  GetUIntFromObject(value, "maxBlocks", &entry->max_blocks);

  entry->release_date = 0;
  {
    std::string release_date;
    if (GetStringFromObject(value, "releaseDate", &release_date))
    {
      std::istringstream iss(release_date);
      struct tm parsed_time = {};
      iss >> std::get_time(&parsed_time, "%Y-%m-%d");
      if (!iss.fail())
      {
        parsed_time.tm_isdst = 0;
#ifdef _WIN32
        entry->release_date = _mkgmtime(&parsed_time);
#else
        entry->release_date = timegm(&parsed_time);
#endif
      }
    }
  }

  entry->supported_controllers = static_cast<u16>(~0u);
  const auto controllers = value.FindMember("controllers");
  if (controllers != value.MemberEnd())
  {
    if (controllers->value.IsArray())
    {
      bool first = true;
      for (const rapidjson::Value& controller : controllers->value.GetArray())
      {
        if (!controller.IsString())
        {
          Log_WarningPrintf("controller is not a string");
          return false;
        }

        std::optional<ControllerType> ctype = Settings::ParseControllerTypeName(controller.GetString());
        if (!ctype.has_value())
        {
          Log_WarningPrintf("Invalid controller type '%s'", controller.GetString());
          continue;
        }

        if (first)
        {
          entry->supported_controllers = 0;
          first = false;
        }

        entry->supported_controllers |= (1u << static_cast<u16>(ctype.value()));
      }
    }
    else
    {
      Log_WarningPrintf("controllers is not an array");
    }
  }

  const auto compatibility = value.FindMember("compatibility");
  if (compatibility != value.MemberEnd())
  {
    if (compatibility->value.IsObject())
    {
      u32 rating;
      if (GetUIntFromObject(compatibility->value, "rating", &rating) &&
          rating < static_cast<u32>(CompatibilityRating::Count))
      {
        entry->compatibility = static_cast<CompatibilityRating>(rating);
      }
    }
    else
    {
      Log_WarningPrintf("compatibility is not an object");
    }
  }

  const auto traits = value.FindMember("traits");
  if (traits != value.MemberEnd())
  {
    if (traits->value.IsObject())
    {
      const auto& traitsobj = traits->value;
      for (u32 trait = 0; trait < static_cast<u32>(Trait::Count); trait++)
      {
        bool bvalue;
        if (GetBoolFromObject(traitsobj, s_jsontrait_names[trait], &bvalue) && bvalue)
          entry->traits[trait] = bvalue;
      }

      entry->display_active_start_offset = GetOptionalIntFromObject<s16>(traitsobj, "DisplayActiveStartOffset");
      entry->display_active_end_offset = GetOptionalIntFromObject<s16>(traitsobj, "DisplayActiveEndOffset");
      entry->display_line_start_offset = GetOptionalIntFromObject<s8>(traitsobj, "DisplayLineStartOffset");
      entry->display_line_end_offset = GetOptionalIntFromObject<s8>(traitsobj, "DisplayLineEndOffset");
      entry->dma_max_slice_ticks = GetOptionalUIntFromObject<u32>(traitsobj, "DMAMaxSliceTicks");
      entry->dma_halt_ticks = GetOptionalUIntFromObject<u32>(traitsobj, "DMAHaltTicks");
      entry->gpu_fifo_size = GetOptionalUIntFromObject<u32>(traitsobj, "GPUFIFOSize");
      entry->gpu_max_run_ahead = GetOptionalUIntFromObject<u32>(traitsobj, "GPUMaxRunAhead");
      entry->gpu_pgxp_tolerance = GetOptionalFloatFromObject(traitsobj, "GPUPGXPTolerance");
      entry->gpu_pgxp_depth_threshold = GetOptionalFloatFromObject(traitsobj, "GPUPGXPDepthThreshold");
    }
    else
    {
      Log_WarningPrintf("traits is not an object");
    }
  }

  GetStringFromObject(value, "discSetName", &entry->disc_set_name);
  const auto disc_set_serials = value.FindMember("discSetSerials");
  if (disc_set_serials != value.MemberEnd())
  {
    if (disc_set_serials->value.IsArray())
    {
      const auto disc_set_serials_array = disc_set_serials->value.GetArray();
      entry->disc_set_serials.reserve(disc_set_serials_array.Size());
      for (const rapidjson::Value& serial : disc_set_serials_array)
      {
        if (serial.IsString())
        {
          entry->disc_set_serials.emplace_back(serial.GetString(), serial.GetStringLength());
        }
        else
        {
          Log_WarningPrintf("discSetSerial is not a string");
        }
      }
    }
    else
    {
      Log_WarningPrintf("discSetSerials is not an array");
    }
  }

  return true;
}

bool GameDatabase::ParseJsonCodes(u32 index, const rapidjson::Value& value)
{
  auto member = value.FindMember("codes");
  if (member == value.MemberEnd())
  {
    Log_WarningPrintf("codes member is missing");
    return false;
  }

  if (!member->value.IsArray())
  {
    Log_WarningPrintf("codes is not an array");
    return false;
  }

  u32 added = 0;
  for (const rapidjson::Value& current_code : member->value.GetArray())
  {
    if (!current_code.IsString())
    {
      Log_WarningPrintf("code is not a string");
      continue;
    }

    const std::string_view code(current_code.GetString(), current_code.GetStringLength());
    auto iter = s_code_lookup.find(code);
    if (iter != s_code_lookup.end())
    {
      Log_WarningPrintf("Duplicate code '%.*s'", static_cast<int>(code.size()), code.data());
      continue;
    }

    s_code_lookup.emplace(code, index);
    added++;
  }

  return (added > 0);
}

void GameDatabase::EnsureTrackHashesMapLoaded()
{
  if (s_track_hashes_loaded)
    return;

  s_track_hashes_loaded = true;
  LoadTrackHashes();
}

bool GameDatabase::LoadTrackHashes()
{
  Common::Timer load_timer;

  std::optional<std::string> gamedb_data(Host::ReadResourceFileToString("discdb.yaml", false));
  if (!gamedb_data.has_value())
  {
    Log_ErrorPrint("Failed to read game database");
    return false;
  }

  SetRymlCallbacks();

  // TODO: Parse in-place, avoid string allocations.
  const ryml::Tree tree = ryml::parse_in_arena(c4::to_csubstr(gamedb_data.value()));
  const ryml::ConstNodeRef root = tree.rootref();

  s_track_hashes_map = {};

  size_t serials = 0;
  for (const ryml::ConstNodeRef& current : root.children())
  {
    const std::string_view serial = to_stringview(current.key());
    if (serial.empty() || !current.has_children())
    {
      Log_WarningPrint("entry is not an object");
      continue;
    }

    const ryml::ConstNodeRef track_data = current.find_child(to_csubstr("trackData"));
    if (!track_data.valid() || !track_data.has_children())
    {
      Log_WarningFmt("trackData is missing in {}", serial);
      continue;
    }

    u32 revision = 0;
    for (const ryml::ConstNodeRef& track_revisions : track_data.children())
    {
      const ryml::ConstNodeRef tracks = track_revisions.find_child(to_csubstr("tracks"));
      if (!tracks.valid() || !tracks.has_children())
      {
        Log_WarningFmt("tracks member is missing in {}", serial);
        continue;
      }

      std::string revision_string;
      GetStringFromObject(track_revisions, "version", &revision_string);

      for (const ryml::ConstNodeRef& track : tracks)
      {
        const ryml::ConstNodeRef md5 = track.find_child("md5");
        std::string_view md5_str;
        if (!md5.valid() || (md5_str = to_stringview(md5.val())).empty())
        {
          Log_WarningFmt("md5 is missing in track in {}", serial);
          continue;
        }

        const std::optional<CDImageHasher::Hash> md5o = CDImageHasher::HashFromString(md5_str);
        if (md5o.has_value())
        {
          s_track_hashes_map.emplace(std::piecewise_construct, std::forward_as_tuple(md5o.value()),
                                     std::forward_as_tuple(std::string(serial), revision_string, revision));
        }
        else
        {
          Log_WarningFmt("invalid md5 in {}", serial);
        }
      }
      revision++;
    }

    serials++;
  }

  ryml::reset_callbacks();
  Log_InfoFmt("Loaded {} track hashes from {} serials in {:.0f}ms.", s_track_hashes_map.size(), serials,
              load_timer.GetTimeMilliseconds());
  return !s_track_hashes_map.empty();
}

const GameDatabase::TrackHashesMap& GameDatabase::GetTrackHashesMap()
{
  EnsureTrackHashesMapLoaded();
  return s_track_hashes_map;
}
