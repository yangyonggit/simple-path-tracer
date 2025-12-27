cmake_minimum_required(VERSION 3.20)

# Set vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "d:/git/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
    "Generate compile_commands.json for clangd" FORCE)
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
find_package(TBB CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

# Create executable
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/Camera.cpp 
    src/EmbreeScene.cpp
    src/PathTracer.cpp
    src/Light.cpp
    src/Material.cpp
    src/MaterialManager.cpp
    src/EnvironmentManager.cpp
    src/GLRenderer.cpp
    src/Cubemap.cpp
    src/GLTFLoader.cpp
    src/MeshIntegrator.cpp
    src/wavefront/wf_integrator_cpu.cpp
)

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/wavefront
    ${Stb_INCLUDE_DIR}
    ${TINYGLTF_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} embree glm::glm-header-only glfw opengl32 GLEW::GLEW TBB::tbb)


# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy assets to build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
    COMMENT "Copying assets to build directory"
)
