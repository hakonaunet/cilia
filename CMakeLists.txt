cmake_minimum_required(VERSION 3.10)

project(Kuramoto)

set(CMAKE_CXX_STANDARD 17)

find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(Kuramoto src/main.cpp src/Oscillator.cpp src/Grid.cpp)

target_link_libraries(Kuramoto sfml-graphics sfml-audio)