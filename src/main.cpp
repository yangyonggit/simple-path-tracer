#include <embree4/rtcore.h>
#include <iostream>
#include <vector>
#include <memory>
#include <atomic>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
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

// Tile rendering parameters
const int TILE_SIZE = 64;

// Progress tracking
std::atomic<int> g_tiles_completed{0};

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

// Render a single tile using TBB
void renderTileTask(int tileIndex, int threadIndex, std::vector<unsigned char>& pixels, 
                   int width, int height, const Camera& camera, RTCScene scene, 
                   const PathTracer& pathTracer, int numTilesX, int numTilesY) {
    // Calculate tile coordinates
    int tileX = tileIndex % numTilesX;
    int tileY = tileIndex / numTilesX;
    
    int x0 = tileX * TILE_SIZE;
    int y0 = tileY * TILE_SIZE;
    int x1 = std::min(x0 + TILE_SIZE, width);
    int y1 = std::min(y0 + TILE_SIZE, height);
    
    // Render pixels in this tile
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            // Calculate normalized pixel coordinates
            float u = float(x) / float(width);
            float v = float(y) / float(height);
            
            // Get ray direction from camera
            glm::vec3 ray_dir = camera.getRayDirection(u, v);
            
            // Trace ray using PathTracer
            glm::vec3 color = pathTracer.traceRay(scene, camera.getPosition(), ray_dir);
            
            // Clamp color to valid range
            color = glm::clamp(color, 0.0f, 1.0f);
            
            // Convert to 8-bit color and store in image
            int pixel_index = (y * width + x) * 3;
            pixels[pixel_index + 0] = static_cast<unsigned char>(color.r * 255); // R
            pixels[pixel_index + 1] = static_cast<unsigned char>(color.g * 255); // G
            pixels[pixel_index + 2] = static_cast<unsigned char>(color.b * 255); // B
        }
    }
    
    // Update progress atomically
    int completed = ++g_tiles_completed;
    if (completed % 8 == 0) { // Show progress every 8 tiles
        int total = numTilesX * numTilesY;
        std::cout << "Rendered " << completed << "/" << total << " tiles (" 
                 << int(100.0f * completed / total) << "%)\r" << std::flush;
    }
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
void renderImageWithOpenGL(GLFWwindow* window, RTCScene scene, Camera& camera) {
    // Create PathTracer with default settings
    PathTracer::Settings settings;
    settings.samples_per_pixel = 16;
    settings.max_depth = 8;
    settings.enable_path_tracing = true;
    PathTracer path_tracer(settings);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Ensure correct row alignment for 3-byte RGB data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Allocate image buffer (RGB format)
    std::vector<unsigned char> image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    while (!glfwWindowShouldClose(window)) {
        // Calculate deltaTime
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        glfwPollEvents();
        processInput();

        // Calculate tile dimensions
        int numTilesX = (IMAGE_WIDTH + TILE_SIZE - 1) / TILE_SIZE;
        int numTilesY = (IMAGE_HEIGHT + TILE_SIZE - 1) / TILE_SIZE;
        int totalTiles = numTilesX * numTilesY;
        
        // Reset progress counter
        g_tiles_completed = 0;
        
        // Parallel tile rendering using TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalTiles), 
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); i++) {
                    renderTileTask((int)i, 0, image, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                  camera, scene, path_tracer, numTilesX, numTilesY);
                }
            });
        
        std::cout << "\nFrame completed                    \n" << std::flush;

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

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    std::cout << "Starting path tracer with Embree4...\n";

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

    // Create camera positioned to see ground plane, box, cube, and sphere
    glm::vec3 camera_pos(2.0f, 1.0f, 2.5f);  // Position above and away from scene
    glm::vec3 camera_target(0.0f, -0.3f, 0.0f);  // Look toward the scene center
    float aspect_ratio = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);
    Camera camera(camera_pos, camera_target, glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, aspect_ratio);
    
    // Set global camera pointer for callbacks
    g_camera = &camera;

    std::cout << "OpenGL initialized. Starting real-time rendering with camera controls...\n";
    std::cout << "Rendering mode: Monte Carlo Path Tracing (TBB Parallel)\n";
    std::cout << "Path tracing settings: 16 samples/pixel, max depth 8\n";
    std::cout << "Parallel processing: Intel TBB with " << TILE_SIZE << "x" << TILE_SIZE << " tiles\n";
    std::cout << "Controls: WASD to move, mouse to look around, ESC to exit\n";

    // Render the scene with real-time camera control
    renderImageWithOpenGL(window, embree_scene.getScene(), camera);

    std::cout << "Path tracer completed successfully!\n";
    return 0;
}
