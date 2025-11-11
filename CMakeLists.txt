cmake_minimum_required(VERSION 3.20)
project(simple-path-tracer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "d:/git/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Find Embree package
find_package(embree CONFIG REQUIRED)

# Create executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link Embree
target_link_libraries(${PROJECT_NAME} embree)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)