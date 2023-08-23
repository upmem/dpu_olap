cmake_minimum_required(VERSION 3.5.0)
project(upmem-query-host VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include("${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake")

set(CMAKE_CXX_FLAGS "--std=c++17")
# Available extra options for profiling/debugging: -DENABLE_LOG
# -DACTIVATE_JOIN_TIMERS
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -Wall -Wextra -Werror -g3 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
link_directories("${DPU_HOST_LINK_DIRECTORIES}")

# set NR_DPUS if not set
if(NOT NR_DPUS)
  message(STATUS "NR_DPUS not set, defaulting to 2048")
  set(NR_DPUS 2048)
endif()

file(GLOB_RECURSE benchmark_source_list "*.cc" "*.h" "*.hpp")
list(FILTER benchmark_source_list EXCLUDE REGEX ".*_test.cc")
add_executable(${PROJECT_NAME} ${benchmark_source_list})
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC "${DPU_HOST_INCLUDE_DIRECTORIES}"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${ARROW_INCLUDE_DIR}")
target_compile_definitions(${PROJECT_NAME} PUBLIC NR_DPUS=${NR_DPUS})
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC ${DPU_HOST_LIBRARIES}
  PRIVATE arrow_shared benchmark benchmark_main)

enable_testing()
file(GLOB_RECURSE test_source_list "*.cc" "*.h" "*.hpp")
list(FILTER test_source_list EXCLUDE REGEX ".*_benchmark.cc")
add_executable(${PROJECT_NAME}_test ${test_source_list})
target_include_directories(
  ${PROJECT_NAME}_test
  PUBLIC "${DPU_HOST_INCLUDE_DIRECTORIES}"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(${PROJECT_NAME}_test SYSTEM
                           PUBLIC ${arrow_SOURCE_DIRS})
target_compile_definitions(${PROJECT_NAME}_test PUBLIC NR_DPUS=${NR_DPUS})
target_link_libraries(
  ${PROJECT_NAME}_test
  PUBLIC ${DPU_HOST_LIBRARIES}
  PRIVATE arrow_shared gtest_main)

include(GoogleTest)
gtest_discover_tests(
  ${PROJECT_NAME}_test
  WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}_test>
  PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
