cmake_minimum_required(VERSION 3.16)
project(upmem-query-dpu VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PROJECT_DPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

include_directories(PUBLIC "${PROJECT_DPU_SOURCE_DIR}/shared")

add_subdirectory(filter)
add_subdirectory(join)
add_subdirectory(partition)
add_subdirectory(take)
add_subdirectory(aggr)
