add_library(core
    achievements.h
    analog_controller.cpp
    analog_controller.h
    analog_joystick.cpp
    analog_joystick.h
    bios.cpp
    bios.h
    bus.cpp
    bus.h
    cdrom.cpp
    cdrom.h
    cdrom_async_reader.cpp
    cdrom_async_reader.h
    cheats.cpp
    cheats.h
    controller.cpp
    controller.h
    cpu_code_cache.cpp
    cpu_code_cache.h
    cpu_core.cpp
    cpu_core.h
    cpu_core_private.h
    cpu_disasm.cpp
    cpu_disasm.h
    cpu_types.cpp
    cpu_types.h
    digital_controller.cpp
    digital_controller.h
    dma.cpp
    dma.h
    game_database.cpp
    game_database.h
    gdb_protocol.cpp
    gdb_protocol.h
    gpu.cpp
    gpu.h
    gpu_backend.cpp
    gpu_backend.h
    gpu_commands.cpp
    gpu_hw.cpp
    gpu_hw.h
    gpu_hw_shadergen.cpp
    gpu_hw_shadergen.h
    gpu_sw.cpp
    gpu_sw.h
    gpu_sw_backend.cpp
    gpu_sw_backend.h
    gpu_types.h
    guncon.cpp
    guncon.h
    gte.cpp
    gte.h
    gte_types.h
    host.cpp
    host.h
    host_interface_progress_callback.cpp
    host_interface_progress_callback.h
    host_settings.h
    input_types.h
    interrupt_controller.cpp
    interrupt_controller.h
    libcrypt_serials.cpp
    libcrypt_serials.h
    mdec.cpp
    mdec.h
    memory_card.cpp
    memory_card.h
    memory_card_image.cpp
    memory_card_image.h
    multitap.cpp
    multitap.h
    negcon.cpp
    negcon.h
    pad.cpp
    pad.h
    pcdrv.cpp
    pcdrv.h
    pgxp.cpp
    pgxp.h
    playstation_mouse.cpp
    playstation_mouse.h
    psf_loader.cpp
    psf_loader.h
    resources.cpp
    resources.h
    save_state_version.h
    settings.cpp
    settings.h
    shader_cache_version.h
    shadergen.cpp
    shadergen.h
    sio.cpp
    sio.h
    spu.cpp
    spu.h
    system.cpp
    system.h
    texture_replacements.cpp
    texture_replacements.h
    timers.cpp
    timers.h
    timing_event.cpp
    timing_event.h
    types.h
)

set(RECOMPILER_SRCS
    cpu_recompiler_code_generator.cpp
    cpu_recompiler_code_generator.h
    cpu_recompiler_code_generator_generic.cpp
    cpu_recompiler_register_cache.cpp
    cpu_recompiler_register_cache.h
    cpu_recompiler_thunks.h
    cpu_recompiler_types.h
)

target_include_directories(core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(core PUBLIC Threads::Threads common util zlib)
target_link_libraries(core PRIVATE stb xxhash imgui rapidjson tinyxml2)

target_sources(core PRIVATE
  gpu/gpu_device.cpp
  gpu/gpu_device.h
  gpu/gpu_shader_cache.cpp
  gpu/gpu_shader_cache.h
  gpu/gpu_texture.cpp
  gpu/gpu_texture.h
  gpu/postprocessing_chain.cpp
  gpu/postprocessing_chain.h
  gpu/postprocessing_shader.cpp
  gpu/postprocessing_shader.h
  gpu/postprocessing_shadergen.cpp
  gpu/postprocessing_shadergen.h
)

if(WIN32)
  target_sources(core PRIVATE
    d3d12/context.cpp
    d3d12/context.h
    d3d12/descriptor_heap_manager.cpp
    d3d12/descriptor_heap_manager.h
    d3d12/shader_cache.cpp
    d3d12/shader_cache.h
    d3d12/staging_texture.cpp
    d3d12/staging_texture.h
    d3d12/stream_buffer.cpp
    d3d12/stream_buffer.h
    d3d12/texture.cpp
    d3d12/texture.h
    d3d12/util.cpp
    d3d12/util.h
    d3d11/shader_cache.cpp
    d3d11/shader_cache.h
    d3d11/shader_compiler.cpp
    d3d11/shader_compiler.h
    d3d11/stream_buffer.cpp
    d3d11/stream_buffer.h
    d3d11/texture.cpp
    d3d11/texture.h
  )
endif()
if(WIN32)
  target_sources(core PRIVATE
    gpu_hw_d3d12.cpp
    gpu_hw_d3d12.h
    gpu_hw_d3d11.cpp
    gpu_hw_d3d11.h
  )
  target_link_libraries(core PRIVATE winmm.lib)
endif()

if(USE_X11)
  target_sources(common PRIVATE
      gl/x11_window.cpp
      gl/x11_window.h
  )
  target_compile_definitions(common PRIVATE "-DUSE_X11=1")
  target_include_directories(common PRIVATE "${X11_INCLUDE_DIR}" "${X11_Xrandr_INCLUDE_PATH}")
  target_link_libraries(common PRIVATE "${X11_LIBRARIES}" "${X11_Xrandr_LIB}")
endif()

if(USE_WAYLAND)
  target_compile_definitions(common PRIVATE "-DUSE_WAYLAND=1")
elseif(SUPPORTS_WAYLAND)
  message(WARNING "Wayland support for renderers is disabled.\nDuckStation will FAIL to start on Wayland.")
endif()

if(USE_DRMKMS)
  target_sources(common PRIVATE
    drm_display.cpp
    drm_display.h
  )
  target_link_libraries(common PUBLIC Libdrm::Libdrm)
endif()

if(ENABLE_OPENGL)
  target_sources(core PRIVATE
    gpu/gl/context.cpp
    gpu/gl/context.h
    gpu/opengl_device.cpp
    gpu/opengl_device.h
    gpu/opengl_loader.h
    gpu/opengl_pipeline.cpp
    gpu/opengl_pipeline.h
    gpu/opengl_stream_buffer.cpp
    gpu/opengl_stream_buffer.h
    gpu/opengl_texture.cpp
    gpu/opengl_texture.h
  )
  target_compile_definitions(core PUBLIC "WITH_OPENGL=1")
  target_link_libraries(core PRIVATE glad)

  if(WIN32)
    target_sources(core PRIVATE
      gl/context_wgl.cpp
      gl/context_wgl.h
    )
  endif()

  if(USE_EGL)
    target_sources(core PRIVATE
      gl/context_egl.cpp
      gl/context_egl.h
    )
    target_compile_definitions(core PRIVATE "-DUSE_EGL=1")

    if(USE_X11)
      target_sources(core PRIVATE
        gl/context_egl_x11.cpp
        gl/context_egl_x11.h
      )

      # We set EGL_NO_X11 because otherwise X comes in with its macros and breaks
      # a bunch of files from compiling, if we include the EGL headers. This just
      # makes the data types opaque, we can still use it with X11 if needed.
      target_compile_definitions(core PRIVATE "-DEGL_NO_X11=1")
    endif()
    if(ANDROID AND USE_EGL)
      target_sources(core PRIVATE
        gl/context_egl_android.cpp
        gl/context_egl_android.h
      )
    endif()
    if(USE_DRMKMS)
      target_compile_definitions(core PRIVATE "-DUSE_GBM=1")
      target_sources(core PRIVATE
        gl/context_egl_gbm.cpp
        gl/context_egl_gbm.h
      )
      target_link_libraries(core PUBLIC GBM::GBM)
    endif()
    if(USE_FBDEV)
      target_compile_definitions(core PRIVATE "-DUSE_FBDEV=1")
      target_sources(core PRIVATE
        gl/context_egl_fbdev.cpp
        gl/context_egl_fbdev.h
      )
    endif()
  endif()

  if(USE_X11)
    target_sources(core PRIVATE
      gl/context_glx.cpp
      gl/context_glx.h
    )
    target_compile_definitions(core PRIVATE "-DUSE_GLX=1")
  endif()

  if(USE_WAYLAND)
    target_sources(core PRIVATE
      gl/context_egl_wayland.cpp
      gl/context_egl_wayland.h
    )
  endif()

  if(APPLE)
    target_sources(core PRIVATE
      gpu/gl/context_agl.mm
      gpu/gl/context_agl.h
    )
  endif()
endif()

if(ENABLE_VULKAN)
  target_sources(core PRIVATE
    gpu/vulkan/builders.cpp
    gpu/vulkan/builders.h
    gpu/vulkan/context.cpp
    gpu/vulkan/context.h
    gpu/vulkan/loader.h
    gpu/vulkan/loader.cpp
    gpu/vulkan/shader_cache.cpp
    gpu/vulkan/shader_cache.h
    gpu/vulkan/shader_compiler.cpp
    gpu/vulkan/shader_compiler.h
    gpu/vulkan/stream_buffer.cpp
    gpu/vulkan/stream_buffer.h
    gpu/vulkan/swap_chain.cpp
    gpu/vulkan/swap_chain.h
    gpu/vulkan/texture.cpp
    gpu/vulkan/texture.h
    gpu/vulkan/util.cpp
    gpu/vulkan/util.h
    gpu/vulkan_gpu_device.cpp
    gpu/vulkan_gpu_device.h
    gpu/imgui_impl_vulkan.cpp
    gpu/imgui_impl_vulkan.h
  )
  target_compile_definitions(core PUBLIC "WITH_VULKAN=1")
  target_link_libraries(core PRIVATE glslang)

  if(APPLE)
    # Needed for Vulkan Swap Chain.
    target_link_libraries(core PRIVATE "objc")
  endif()
endif()
if(ENABLE_VULKAN)
  target_sources(core PRIVATE
    gpu_hw_vulkan.cpp
    gpu_hw_vulkan.h
  )
endif()

if(ENABLE_CUBEB)
  target_compile_definitions(core PUBLIC "WITH_CUBEB=1")
endif()

if(${CPU_ARCH} STREQUAL "x64")
  target_include_directories(core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../dep/xbyak/xbyak")
  target_compile_definitions(core PUBLIC "WITH_RECOMPILER=1" "WITH_MMAP_FASTMEM=1")
  target_sources(core PRIVATE ${RECOMPILER_SRCS}
    cpu_recompiler_code_generator_x64.cpp
  )
  message("Building x64 recompiler")
elseif(${CPU_ARCH} STREQUAL "aarch32")
  target_compile_definitions(core PUBLIC "WITH_RECOMPILER=1")
  target_sources(core PRIVATE ${RECOMPILER_SRCS}
    cpu_recompiler_code_generator_aarch32.cpp
  )
  target_link_libraries(core PUBLIC vixl)
  message("Building AArch32 recompiler")
elseif(${CPU_ARCH} STREQUAL "aarch64")
  target_compile_definitions(core PUBLIC "WITH_RECOMPILER=1")
  if (NOT APPLE)
    # Disabled until we support 16K pages.
    target_compile_definitions(core PUBLIC "WITH_MMAP_FASTMEM=1")
  endif()
  target_sources(core PRIVATE ${RECOMPILER_SRCS}
    cpu_recompiler_code_generator_aarch64.cpp
  )
  target_link_libraries(core PUBLIC vixl)
  message("Building AArch64 recompiler")
else()
  message("Not building recompiler")
endif()

if(ENABLE_CHEEVOS)
  target_compile_definitions(core PRIVATE -DWITH_CHEEVOS=1)
endif()
