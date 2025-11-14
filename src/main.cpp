#include <iostream>
#include <thread>
#include <tbb/global_control.h>
#include "Camera.h"
#include "EmbreeScene.h"
#include "PathTracer.h"
#include "GLRenderer.h"

// Setup lighting for the scene
void setupLights(PathTracer& path_tracer) {
    LightManager& light_manager = path_tracer.getLightManager();
    
    // Add directional light (like sun)
    light_manager.addDirectionalLight(
        glm::vec3(-0.5f, -1.0f, 0.3f), // Direction: from above and slightly to the side
        glm::vec3(1.0f, 0.95f, 0.8f),  // Warm white color
        2.0f                           // Intensity
    );
}

// Setup camera with optimal positioning
Camera setupCamera() {
    const glm::vec3 camera_pos{0.0f, 3.0f, 8.0f};     // Higher and further back to see all spheres
    const glm::vec3 camera_target{0.0f, 1.0f, 0.0f};  // Look at sphere center height
    const auto aspect_ratio = static_cast<float>(GLRenderer::IMAGE_WIDTH) / static_cast<float>(GLRenderer::IMAGE_HEIGHT);
    
    return Camera(camera_pos, camera_target, glm::vec3(0.0f, 1.0f, 0.0f), 60.0f, aspect_ratio);
}

// Setup PathTracer with optimized settings
PathTracer setupPathTracer() {
    PathTracer::Settings settings;
    settings.samples_per_pixel = 4;  // Optimized for performance
    settings.max_depth = 6;          // Optimized for performance
    settings.enable_path_tracing = true;
    
    return PathTracer(settings);
}

int main() {
    std::cout << "Starting path tracer with Embree4...\n";

    // Limit TBB threads to leave one core free for system (unless only 1 core available)
    const auto total_cores = std::thread::hardware_concurrency();
    const auto num_threads = (total_cores <= 1) ? 1 : total_cores - 1;
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, num_threads);
    std::cout << "TBB using " << num_threads << " of " << total_cores << " available threads";
    if (total_cores > 1) {
        std::cout << " (reserved 1 core for system)";
    }
    std::cout << "\n";

    // Create scene using helper class
    EmbreeScene embree_scene;
    if (!embree_scene.isValid()) {
        std::cerr << "Failed to create Embree scene!\n";
        return -1;
    }

    std::cout << "Scene created. Initializing renderer...\n";

    // Create and initialize GL renderer
    GLRenderer renderer;
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize GL renderer!\n";
        return -1;
    }

    // Setup camera and path tracer
    Camera camera = setupCamera();
    PathTracer path_tracer = setupPathTracer();
    setupLights(path_tracer);

    std::cout << "Renderer initialized successfully. Starting real-time rendering with camera controls...\n";
    std::cout << "Rendering mode: Monte Carlo Path Tracing (TBB Parallel + Accumulation)\n";
    std::cout << "Path tracing settings: 4 samples/pixel, max depth 6 (optimized for performance)\n";
    std::cout << "Parallel processing: Intel TBB with " << GLRenderer::TILE_SIZE << "x" << GLRenderer::TILE_SIZE << " tiles\n";
    std::cout << "Accumulation: Progressive sampling when camera is stationary\n";

    // Start the main rendering loop
    renderer.renderLoop(embree_scene, camera, path_tracer);

    std::cout << "Path tracer completed successfully!\n";
    return 0;
}
