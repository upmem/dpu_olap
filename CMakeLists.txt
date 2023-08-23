cmake_minimum_required(VERSION 3.16)
project(upmem-query VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0135 NEW)

include(Dependencies.cmake)
myproject_setup_dependencies()

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

include_directories(PUBLIC "${CMAKE_SOURCE_DIR}/shared")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

enable_testing()
add_subdirectory(host)
add_subdirectory(dpu)

add_dependencies(upmem-query-host arrow_shared)
add_dependencies(upmem-query-host benchmark)
