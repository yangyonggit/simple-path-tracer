#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <atomic>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

struct RTCSceneTy;
typedef RTCSceneTy* RTCScene;

namespace wf {
    class WavefrontPathTracerCPU;
}

// Forward declarations
class Camera;
class PathTracer;

namespace backends {
    class EmbreeBackend;
}

namespace scene {
    struct SceneDesc;
}

class GLRenderer {
public:
    static constexpr int IMAGE_WIDTH = 800;
    static constexpr int IMAGE_HEIGHT = 600;
    static constexpr int TILE_SIZE = 32;

private:
    GLFWwindow* m_window = nullptr;
    GLuint m_texture = 0;
    
    // Rendering state
    std::vector<glm::vec3> m_accumulation_buffer;
    int m_accumulated_samples = 0;
    bool m_camera_moved = true;
    std::atomic<int> m_tiles_completed{0};
    bool m_test_wavefront = false;  // Flag to test wavefront integrator
    std::unique_ptr<wf::WavefrontPathTracerCPU> m_wf_path_tracer;
    
    // Input state
    bool m_keys[1024] = {false};
    bool m_mouse_buttons[8] = {false};  // Track mouse button states
    bool m_first_mouse = true;
    float m_last_x = IMAGE_WIDTH / 2.0f;
    float m_last_y = IMAGE_HEIGHT / 2.0f;
    
    // Timing
    float m_delta_time = 0.0f;
    float m_last_frame = 0.0f;
    
    // Camera reference (not owned)
    Camera* m_camera = nullptr;
    
    // Static instance for GLFW callbacks
    static GLRenderer* s_instance;

public:
    GLRenderer();
    ~GLRenderer();
    
    // Non-copyable
    GLRenderer(const GLRenderer&) = delete;
    GLRenderer& operator=(const GLRenderer&) = delete;
    
    // Initialization and cleanup
    bool initialize();
    void cleanup();
    
    // Main rendering loop
    void renderLoop(backends::EmbreeBackend& backend, const scene::SceneDesc& sceneDesc,
                   Camera& camera, PathTracer& path_tracer);
    
    // Camera management
    void setCamera(Camera* camera) { m_camera = camera; }
    
    // Window and context management
    GLFWwindow* getWindow() const { return m_window; }
    bool shouldClose() const;
    
private:
    // OpenGL setup
    bool initializeOpenGL();
    GLuint createTexture();
    void setupCallbacks();
    
    // Rendering
    void renderFrame(const std::vector<unsigned char>& image);
    void renderWavefront(std::vector<unsigned char>& image, const Camera& camera, 
                        RTCScene scene, PathTracer& path_tracer,
                        std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples);
    void updateTiming();
    void processInput();
    bool hasCameraMoved();
    
    // Wavefront tile rendering
    static void renderWavefrontTileTask(int tileIndex, std::vector<unsigned char>& image,
                                       int width, int height, const Camera& camera, RTCScene scene,
                                       const GLRenderer& renderer, PathTracer& path_tracer, int numTilesX, int numTilesY,
                                       std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples);
    
    // GLFW callbacks (static)
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void errorCallback(int error, const char* description);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};