#include <iostream>
#include <thread>
#include <fstream>
#include <tbb/global_control.h>
#include "Camera.h"
#include "EmbreeScene.h"
#include "PathTracer.h"
#include "GLRenderer.h"

// Command line options structure
struct CommandLineOptions {
    std::string gltf_filename;
    std::string skybox_filename;
    bool use_custom_scene = false;
    bool show_help = false;
};

// Parse command line arguments
CommandLineOptions parseCommandLine(int argc, char* argv[]) {
    CommandLineOptions options;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
        }
        else if (arg == "--i" || arg == "-i") {
            if (i + 1 < argc) {
                options.gltf_filename = argv[++i];
                options.use_custom_scene = true;
            } else {
                std::cerr << "Error: --i requires a filename argument" << std::endl;
                options.show_help = true;
            }
        }
        else if (arg == "--s" || arg == "-s") {
            if (i + 1 < argc) {
                options.skybox_filename = argv[++i];
            } else {
                std::cerr << "Error: --s requires a filename argument" << std::endl;
                options.show_help = true;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            options.show_help = true;
        }
    }
    
    return options;
}

// Display help information
void showHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Simple Path Tracer - Real-time Monte Carlo path tracing with Embree4" << std::endl;
    std::cout << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --i, -i <filename>    Load GLTF model from specified file" << std::endl;
    std::cout << "                        (replaces default scene)" << std::endl;
    std::cout << "  --s, -s <filename>    Load skybox from specified HDR file" << std::endl;
    std::cout << "                        (replaces default environment)" << std::endl;
    std::cout << "  --help, -h            Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  " << program_name << "                                    # Use default scene" << std::endl;
    std::cout << "  " << program_name << " --i model.gltf                    # Load custom model" << std::endl;
    std::cout << "  " << program_name << " --s environment.hdr               # Use custom skybox" << std::endl;
    std::cout << "  " << program_name << " --i model.gltf --s env.hdr        # Custom model and skybox" << std::endl;
    std::cout << std::endl;
    std::cout << "CONTROLS:" << std::endl;
    std::cout << "  WASD      - Move camera" << std::endl;
    std::cout << "  Mouse     - Look around" << std::endl;
    std::cout << "  ESC       - Exit" << std::endl;
    std::cout << std::endl;
}

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

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CommandLineOptions options = parseCommandLine(argc, argv);
    
    // Show help if requested
    if (options.show_help) {
        showHelp(argv[0]);
        return 0;
    }
    
    std::cout << "Starting path tracer with Embree4..." << std::endl;

    // Limit TBB threads to leave one core free for system (unless only 1 core available)
    const auto total_cores = std::thread::hardware_concurrency();
    const auto num_threads = (total_cores <= 1) ? 1 : total_cores - 1;
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, num_threads);
    std::cout << "TBB using " << num_threads << " of " << total_cores << " available threads";
    if (total_cores > 1) {
        std::cout << " (reserved 1 core for system)";
    }
    std::cout << "\n";

    // Create scene based on command line options
    EmbreeScene embree_scene(options.use_custom_scene ? false : true);
    
    // If using custom scene, load the GLTF file instead of default scene
    if (options.use_custom_scene) {
        std::cout << "Loading custom GLTF model: " << options.gltf_filename << std::endl;
        
        // Check if file exists
        std::ifstream file_check(options.gltf_filename);
        if (!file_check.good()) {
            std::cerr << "Error: GLTF file does not exist: " << options.gltf_filename << std::endl;
            std::cerr << "Falling back to default scene" << std::endl;
        } else {
            file_check.close();
            // Load the custom GLTF file with material ID 0 (default material)
            std::cout << "File exists, attempting to load..." << std::endl;
            if (!embree_scene.loadGLTF(options.gltf_filename, glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, 0)) {
                std::cerr << "Failed to load GLTF model: " << options.gltf_filename << std::endl;
                std::cerr << "Falling back to default scene" << std::endl;
                // Continue with default scene if custom loading fails
            } else {
                std::cout << "Successfully loaded custom GLTF model" << std::endl;
                std::cout << "Note: Using custom scene - default spheres will not be loaded" << std::endl;
            }
        }
    } else {
        std::cout << "Using default scene with multiple spheres and test objects" << std::endl;
    }
    
    if (!embree_scene.isValid()) {
        std::cerr << "Failed to create Embree scene!" << std::endl;
        return -1;
    }

    std::cout << "Scene created. Initializing renderer..." << std::endl;
    
    // Create and initialize GL renderer
    GLRenderer renderer;
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize GL renderer!" << std::endl;
        return -1;
    }
    
    // Setup environment with custom skybox if specified
    if (!options.skybox_filename.empty()) {
        std::cout << "Loading custom skybox: " << options.skybox_filename << std::endl;
        // Note: This would require extending EnvironmentManager to support custom HDR files
        // For now, we'll use the default environment
    }

    // Setup camera and path tracer
    Camera camera = setupCamera();
    PathTracer path_tracer = setupPathTracer();
    setupLights(path_tracer);
    
    // Load custom environment if specified
    if (!options.skybox_filename.empty()) {
        std::cout << "Loading custom skybox: " << options.skybox_filename << std::endl;
        auto& env_manager = path_tracer.getEnvironmentManager();
        if (!env_manager.loadCubemap(options.skybox_filename)) {
            std::cerr << "Warning: Failed to load custom skybox: " << options.skybox_filename << std::endl;
            std::cout << "Falling back to default environment" << std::endl;
        } else {
            std::cout << "Successfully loaded custom skybox" << std::endl;
        }
    }
    
    std::cout << "Renderer initialized successfully. Starting real-time rendering with camera controls..." << std::endl;
    std::cout << "Rendering mode: Monte Carlo Path Tracing (TBB Parallel + Accumulation)" << std::endl;
    std::cout << "Path tracing settings: 4 samples/pixel, max depth 6 (optimized for performance)" << std::endl;
    std::cout << "Parallel processing: Intel TBB with " << GLRenderer::TILE_SIZE << "x" << GLRenderer::TILE_SIZE << " tiles" << std::endl;
    std::cout << "Accumulation: Progressive sampling when camera is stationary" << std::endl;
    
    // Display loaded content info
    if (options.use_custom_scene) {
        std::cout << "Scene: Custom GLTF model (" << options.gltf_filename << ")" << std::endl;
    } else {
        std::cout << "Scene: Default test scene with spheres and objects" << std::endl;
    }
    
    if (!options.skybox_filename.empty()) {
        std::cout << "Environment: Custom HDR skybox (" << options.skybox_filename << ")" << std::endl;
    } else {
        std::cout << "Environment: Default HDR environment" << std::endl;
    }
    std::cout << std::endl;
    
    // Display loaded content info
    if (options.use_custom_scene) {
        std::cout << "Scene: Custom GLTF model (" << options.gltf_filename << ")" << std::endl;
    } else {
        std::cout << "Scene: Default test scene with spheres and objects" << std::endl;
    }
    
    if (!options.skybox_filename.empty()) {
        std::cout << "Environment: Custom HDR skybox (" << options.skybox_filename << ")" << std::endl;
    } else {
        std::cout << "Environment: Default HDR environment" << std::endl;
    }
    std::cout << std::endl;

    // Start the main rendering loop
    renderer.renderLoop(embree_scene, camera, path_tracer);

    std::cout << "Path tracer completed successfully!" << std::endl;
    return 0;
}
