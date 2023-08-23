cmake_minimum_required(VERSION 3.5.0)
project(upmem-query-dpu-filter VERSION 0.1.0)

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -Og")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -g0 -O3")

if(NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

set(DPU_FILTER_SOURCES main.c ${PROJECT_DPU_SOURCE_DIR}/shared/kernels/filter.c)

add_executable(kernel-filter ${DPU_FILTER_SOURCES})
target_compile_definitions(kernel-filter PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(kernel-filter PUBLIC -DNR_TASKLETS=${NR_TASKLETS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
