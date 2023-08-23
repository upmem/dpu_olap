include(cmake/CPM.cmake)
if(NOT CPM_USE_LOCAL_PACKAGES)
  set(CPM_USE_LOCAL_PACKAGES ON)
endif()

# Done as a function so that updates to variables like CMAKE_CXX_FLAGS don't
# propagate out to other targets
function(myproject_setup_dependencies)
  # For each dependency, see if it's already been provided to us by a parent
  # project
  if(NOT TARGET gtest)
    cpmaddpackage(
      NAME
      gooogletest
      GITHUB_REPOSITORY
      google/googletest
      VERSION
      1.14.0
      OPTIONS
      "INSTALL_GTEST Off"
      "gtest_force_shared_crt On")
  endif()

  if(NOT TARGET benchmark)
    cpmaddpackage(
      NAME
      benchmark
      GITHUB_REPOSITORY
      google/benchmark
      VERSION
      1.7.1
      OPTIONS
      "BENCHMARK_ENABLE_TESTING Off"
      "CMAKE_BUILD_TYPE Release"
      "BENCHMARK_DOWNLOAD_DEPENDENCIES ON")

    if(benchmark_ADDED)
      # patch benchmark target
      set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
    endif()
  endif()

  # Arrow doesn't play nicely with CPM and FetchContent so we build it as an
  # external project, like in arrow/matlab/CMakeLists.txt We still use CPM to
  # download the source code and check it it is already available
  if(NOT TARGET arrow_shared)
    cpmaddpackage(
      NAME
      arrow
      GITHUB_REPOSITORY
      apache/arrow
      GIT_TAG
      apache-arrow-8.0.0
      DOWNLOAD_ONLY
      TRUE)

    if(arrow_ADDED)
      set(arrow_prefix "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix")
      set(arrow_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-build")
      set(arrow_cmake_args
          "-DCMAKE_INSTALL_PREFIX=${arrow_prefix}"
          "-DCMAKE_INSTALL_LIBDIR=lib"
          "-DARROW_BUILD_STATIC=OFF"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DARROW_WITH_RE2=OFF"
          "-DARROW_WITH_UTF8PROC=OFF"
          "-DARROW_COMPUTE=ON"
          "-DARROW_TESTING=ON"
          "-DARROW_JSON=ON"
          "-DCMAKE_POLICY_DEFAULT_CMP0135:STRING=NEW")
      add_library(arrow_shared SHARED IMPORTED)
      set(arrow_library_target arrow_shared)

      # Set the runtime shared library (.dll, .so, or .dylib)
      set(arrow_share_library_dir "${arrow_prefix}/lib")

      set(arrow_shared_lib_filename
          "${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}")
      set(arrow_shared_lib
          "${arrow_share_library_dir}/${arrow_shared_lib_filename}")

      set_target_properties(arrow_shared PROPERTIES IMPORTED_LOCATION
                                                    ${arrow_shared_lib})

      # Set the include directories
      set(arrow_include_dir "${arrow_prefix}/include")
      file(MAKE_DIRECTORY "${arrow_include_dir}")

      set_target_properties(
        arrow_shared PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                ${arrow_include_dir})

      # Set the build byproducts for the ExternalProject build The appropriate
      # libraries need to be guaranteed to be available when linking the test
      # executables.
      set(arrow_build_byproducts "${arrow_shared_lib}")

      # Building the Arrow C++ libraries and bundled GoogleTest binaries
      # requires ExternalProject.
      include(ExternalProject)

      ExternalProject_Add(
        arrow_ep
        SOURCE_DIR ${arrow_SOURCE_DIR}/cpp
        BINARY_DIR "${arrow_binary_dir}"
        CMAKE_ARGS "${arrow_cmake_args}"
        BUILD_BYPRODUCTS "${arrow_build_byproducts}")

      add_dependencies(${arrow_library_target} arrow_ep)
    endif()
  endif()

endfunction()
