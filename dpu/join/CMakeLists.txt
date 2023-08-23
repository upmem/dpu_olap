cmake_minimum_required(VERSION 3.5.0)
project(upmem-query-dpu-join VERSION 0.1.0)

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -Og")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -g0 -O3")

if(NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

# DPU Join Kernels
set(DPU_JOIN_SOURCES
    main.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/kernels/partition.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/kernels/hash_build.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/kernels/hash_probe.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/kernels/take.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/mram_alloc.c
    ${PROJECT_DPU_SOURCE_DIR}/shared/hashtable/hashtable.c)
add_executable(kernel-join ${DPU_JOIN_SOURCES})
target_compile_definitions(kernel-join PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(kernel-join PUBLIC -DNR_TASKLETS=${NR_TASKLETS})

# Hashmap tests
add_executable(
  ${PROJECT_NAME}-hashmap-test
  ${PROJECT_DPU_SOURCE_DIR}/shared/hashtable/hashtable_test.c
  ${PROJECT_DPU_SOURCE_DIR}/shared/mram_alloc.c
  ${PROJECT_DPU_SOURCE_DIR}/shared/hashtable/hashtable.c)
target_compile_definitions(${PROJECT_NAME}-hashmap-test
                           PUBLIC NR_TASKLETS=${NR_TASKLETS})
target_link_options(${PROJECT_NAME}-hashmap-test PUBLIC
                    -DNR_TASKLETS=${NR_TASKLETS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
