function(disable_compiler_warnings_for_target target)
	if(MSVC)
		target_compile_options(${target} PRIVATE "/W0")
	else()
		target_compile_options(${target} PRIVATE "-w")
	endif()
endfunction()

function(detect_page_size)
  # This is only needed for ARM64, or if the user hasn't overridden it explicitly.
  if(NOT CPU_ARCH_ARM64 OR HOST_PAGE_SIZE)
    return()
  endif()

  if(NOT LINUX OR ANDROID)
    # For universal Apple builds, we use preprocessor macros to determine page size.
    # Similar for Windows, except it's always 4KB.
    return()
  endif()

  if(CMAKE_CROSSCOMPILING)
    message(WARNING "Cross-compiling and can't determine page size, assuming default.")
    return()
  endif()

  set(detect_page_size_c_code "
#include <stdio.h>
#include <unistd.h>
int main() {
  int res = sysconf(_SC_PAGESIZE);
  printf(\"%d\", res);
  return (res > 0) ? 1 : 0;
}")
  try_run(
    detect_page_size_run_result detect_page_size_compile_result
    SOURCE_FROM_VAR detect_page_size.c detect_page_size_c_code
    RUN_OUTPUT_STDOUT_VARIABLE detect_page_size_output)
  if(NOT detect_page_size_compile_result OR NOT detect_page_size_run_result)
    message(FATAL_ERROR "Could not determine host page size.")
  else()
    message(STATUS "Page size: ${detect_page_size_output}")
    set(HOST_PAGE_SIZE ${detect_page_size_output} PARENT_SCOPE)
  endif()
endfunction()
