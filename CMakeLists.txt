cmake_minimum_required(VERSION 3.20)

# Set vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "d:/git/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(simple-path-tracer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find packages
find_package(embree CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)

# Create executable
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/Camera.cpp 
    src/EmbreeScene.cpp
    src/PathTracer.cpp
)

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Stb_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} embree glm::glm-header-only glfw opengl32 GLEW::GLEW)


# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
