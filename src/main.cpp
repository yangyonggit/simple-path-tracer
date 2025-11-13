#include <embree4/rtcore.h>
#include <iostream>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include "Camera.h"
#include "EmbreeScene.h"
#include "PathTracer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Image dimensions
const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 600;

// Tile rendering parameters (smaller tiles for better responsiveness)
const int TILE_SIZE = 32;

// Progress tracking
std::atomic<int> g_tiles_completed{0};

// Accumulation variables
std::vector<glm::vec3> g_accumulation_buffer; // High precision accumulation
int g_accumulated_samples = 0;
bool g_camera_moved = true;

// Input state
bool keys[1024] = {false};

// Mouse control
bool firstMouse = true;
float lastX = IMAGE_WIDTH / 2.0f;
float lastY = IMAGE_HEIGHT / 2.0f;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Global camera pointer for callbacks
Camera* g_camera = nullptr;

// Keyboard callback
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }
}

// Mouse callback
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if (g_camera) {
        g_camera->processMouseMovement(xoffset, yoffset);
    }
}

// Process input
void processInput() {
    if (!g_camera) return;
    
    if (keys[GLFW_KEY_W])
        g_camera->processKeyboard(Camera::FORWARD, deltaTime);
    if (keys[GLFW_KEY_S])
        g_camera->processKeyboard(Camera::BACKWARD, deltaTime);
    if (keys[GLFW_KEY_A])
        g_camera->processKeyboard(Camera::LEFT, deltaTime);
    if (keys[GLFW_KEY_D])
        g_camera->processKeyboard(Camera::RIGHT, deltaTime);
}

// Setup PathTracer with lights and scene
void setupPathTracerWithLights(PathTracer& path_tracer) {
    // Set up lighting
    LightManager& light_manager = path_tracer.getLightManager();
    
    // Add a directional light (like sun)
    light_manager.addDirectionalLight(
        glm::vec3(0.5f, -1.0f, 0.3f), // Direction: from above and slightly to the side
        glm::vec3(1.0f, 0.95f, 0.8f),  // Warm white color
        2.0f                           // Intensity
    );
    
    // // Add a point light above the scene
    // light_manager.addPointLight(
    //     glm::vec3(1.5f, 2.0f, 1.5f),  // Position above and to the side
    //     glm::vec3(0.8f, 0.9f, 1.0f),  // Cool white color
    //     8.0f                           // Intensity
    // );
}

// Initialize OpenGL texture and rendering setup
GLuint initializeGLTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Ensure correct row alignment for 3-byte RGB data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    return texture;
}

// Render frame to OpenGL texture and display
void renderGLFrame(GLuint texture, const std::vector<unsigned char>& image) {
    // Upload texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());

    // Render
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    // Flip top/bottom vertex positions as requested (invert Y for each vertex)
    // Texture coordinates remain the same; only vertex positions swap vertically.
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glEnd();
}

// Function to initialize OpenGL and create a window
GLFWwindow* initializeOpenGL(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!\n";
        return nullptr;
    }

    // Request an older compatibility profile so we can use immediate-mode
    // and fixed-function pipeline (glBegin/glEnd, glEnable(GL_TEXTURE_2D)).
    // Modern core profiles remove these functions which can cause nothing
    // to be rendered when using the simple immediate-mode code below.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    // Do NOT request CORE profile so compatibility/fixed pipeline is available.

    GLFWwindow* window = glfwCreateWindow(width, height, "Path Tracer Output", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // Set input callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW!\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    glViewport(0, 0, width, height);
    return window;
}

// Function to render the image using OpenGL with real-time camera control
void renderImageWithOpenGL(GLFWwindow* window, RTCScene scene, Camera& camera, PathTracer& path_tracer) {
    // Initialize OpenGL texture
    GLuint texture = initializeGLTexture();

    // Allocate image buffer (RGB format)
    std::vector<unsigned char> image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    
    // Initialize accumulation buffer
    g_accumulation_buffer.resize(IMAGE_WIDTH * IMAGE_HEIGHT, glm::vec3(0.0f));
    g_accumulated_samples = 0;

    while (!glfwWindowShouldClose(window)) {
        // Calculate deltaTime
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        glfwPollEvents();
        processInput();

        // Check if camera has moved using Camera class method
        g_camera_moved = camera.hasMovedSinceLastCheck();
        
        if (g_camera_moved) {
            // Reset accumulation when camera moves
            g_accumulated_samples = 0;
            std::fill(g_accumulation_buffer.begin(), g_accumulation_buffer.end(), glm::vec3(0.0f));
        }
        
        // Increment sample count
        g_accumulated_samples++;

        // Render image using PathTracer's parallel processing
        path_tracer.renderImage(image, IMAGE_WIDTH, IMAGE_HEIGHT, camera, scene,
                               g_accumulation_buffer, g_accumulated_samples,
                               g_camera_moved, g_tiles_completed);
        
        // Show accumulation progress
        if (g_accumulated_samples == 1) {
            std::cout << "\nFrame " << g_accumulated_samples << " completed (new view)                    \n" << std::flush;
        } else {
            std::cout << "\nFrame " << g_accumulated_samples << " completed (accumulating)                    \n" << std::flush;
        }

        // Render to OpenGL
        renderGLFrame(texture, image);
        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    std::cout << "Starting path tracer with Embree4...\n";

    // Limit TBB threads to leave one core free for system (unless only 1 core available)
    unsigned int total_cores = std::thread::hardware_concurrency();
    unsigned int num_threads = (total_cores <= 1) ? 1 : total_cores - 1;
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

    std::cout << "Scene created. Initializing OpenGL...\n";

    // Initialize OpenGL and create a window
    GLFWwindow* window = initializeOpenGL(IMAGE_WIDTH, IMAGE_HEIGHT);
    if (!window) {
        return -1;
    }

    // Create camera positioned to see all 9 spheres spread across the scene
    // Spheres are at: front row (-3,-1,1,3), middle row (-2,0,2), back row (-1,1)
    glm::vec3 camera_pos(0.0f, 3.0f, 8.0f);  // Higher and further back to see all spheres
    glm::vec3 camera_target(0.0f, 1.0f, 0.0f);  // Look at sphere center height
    float aspect_ratio = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);
    Camera camera(camera_pos, camera_target, glm::vec3(0.0f, 1.0f, 0.0f), 60.0f, aspect_ratio); // Wider FOV
    
    // Set global camera pointer for callbacks
    g_camera = &camera;
    
    // Create and setup PathTracer with lights (reduced settings for better performance)
    PathTracer::Settings settings;
    settings.samples_per_pixel = 4;  // Reduced from 16 for better performance
    settings.max_depth = 6;          // Reduced from 8 for better performance
    settings.enable_path_tracing = true;
    
    PathTracer path_tracer(settings);
    setupPathTracerWithLights(path_tracer);  // Add this crucial line!

    std::cout << "OpenGL initialized. Starting real-time rendering with camera controls...\n";
    std::cout << "Rendering mode: Monte Carlo Path Tracing (TBB Parallel + Accumulation)\n";
    std::cout << "Path tracing settings: 4 samples/pixel, max depth 6 (optimized for performance)\n";
    std::cout << "Parallel processing: Intel TBB with " << TILE_SIZE << "x" << TILE_SIZE << " tiles\n";
    std::cout << "Accumulation: Progressive sampling when camera is stationary\n";
    std::cout << "Controls: WASD to move, mouse to look around, ESC to exit\n";

    // Render the scene with real-time camera control
    renderImageWithOpenGL(window, embree_scene.getScene(), camera, path_tracer);

    std::cout << "Path tracer completed successfully!\n";
    return 0;
}
