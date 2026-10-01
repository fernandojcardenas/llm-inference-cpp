cmake_minimum_required(VERSION 3.24)
project(llm_inference_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LLMI_BUILD_TESTS "Build unit tests" ON)
option(LLMI_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LLMI_BUILD_FUZZERS "Build libFuzzer targets (requires Clang)" OFF)
option(LLMI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(llmi_warnings INTERFACE)
target_compile_options(llmi_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wnon-virtual-dtor
  $<$<BOOL:${LLMI_WARNINGS_AS_ERRORS}>:-Werror>)

if(LLMI_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

if(LLMI_BUILD_FUZZERS)
  # Coverage instrumentation everywhere; the fuzzer main is linked only into the fuzz targets.
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(llmi
  src/util/utf8.cpp
  src/util/json.cpp
  src/model/safetensors.cpp
  src/model/model.cpp
  src/tokenizer/tokenizer.cpp)
target_include_directories(llmi PUBLIC include PRIVATE src/tokenizer)
target_link_libraries(llmi PRIVATE llmi_warnings)

add_executable(llmi-inspect apps/llmi-inspect/main.cpp)
target_link_libraries(llmi-inspect PRIVATE llmi llmi_warnings)
add_executable(llmi-tokenize apps/llmi-tokenize/main.cpp)
target_link_libraries(llmi-tokenize PRIVATE llmi llmi_warnings)

set(LLMI_TESTDATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/testdata")

if(LLMI_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(llmi_tests
    tests/utf8_test.cpp
    tests/json_test.cpp
    tests/safetensors_test.cpp
    tests/model_test.cpp
    tests/tokenizer_test.cpp)
  target_compile_definitions(llmi_tests PRIVATE LLMI_TESTDATA_DIR="${LLMI_TESTDATA_DIR}")
  target_link_libraries(llmi_tests PRIVATE llmi llmi_warnings GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(llmi_tests)
endif()

if(LLMI_BUILD_FUZZERS)
  foreach(target fuzz_json fuzz_safetensors fuzz_config fuzz_tokenizer)
    add_executable(${target} fuzz/${target}.cpp)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
    target_compile_definitions(${target} PRIVATE LLMI_TESTDATA_DIR="${LLMI_TESTDATA_DIR}")
    target_link_libraries(${target} PRIVATE llmi)
  endforeach()
endif()
