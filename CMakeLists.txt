cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(actor)


find_package (Eigen3 3.3.7 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(actor actor.cpp)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(actor Eigen3::Eigen)
set_property(TARGET actor PROPERTY CXX_STANDARD 17)


