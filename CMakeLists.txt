cmake_minimum_required (VERSION 3.1)

project(clahe)

find_package(OpenCL REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(clahe
  clahe.c)

target_include_directories(clahe
  PRIVATE OpenCL::OpenCL)

target_link_libraries(clahe
  OpenCL::OpenCL
  m)
