#include "GLRenderer.h"
#include "Camera.h"
#include "PathTracer.h"
#include "wavefront/wf_integrator_cpu.h"
#include "wavefront/wf_pt_cpu.h"
#include "wavefront/wf_math.h"
#include "EmbreeScene.h"
#include <iostream>
#include <algorithm>

// Static member initialization
GLRenderer* GLRenderer::s_instance = nullptr;

GLRenderer::GLRenderer() {
    // Initialize accumulation buffer
    m_accumulation_buffer.resize(IMAGE_WIDTH * IMAGE_HEIGHT, glm::vec3(0.0f));
    // Initialize wavefront path tracer
    m_wf_path_tracer = std::make_unique<wf::WavefrontPathTracerCPU>();
    s_instance = this;
}

GLRenderer::~GLRenderer() {
    cleanup();
    s_instance = nullptr;
}

bool GLRenderer::initialize() {
    return initializeOpenGL();
}

bool GLRenderer::initializeOpenGL() {
    // Set error callback before glfwInit
    glfwSetErrorCallback(errorCallback);
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // Create window
    m_window = glfwCreateWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Simple Path Tracer", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // Make context current
    glfwMakeContextCurrent(m_window);
    
    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }
    
    // Set viewport
    glViewport(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Set framebuffer size callback
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    
    // Create texture for rendering
    m_texture = createTexture();
    
    // Setup input callbacks
    setupCallbacks();
    
    // Disable VSync for maximum performance
    glfwSwapInterval(0);
    
    std::cout << "OpenGL initialized successfully" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    return true;
}

GLuint GLRenderer::createTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    // Ensure correct row alignment for 3-byte RGB data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    return texture;
}

void GLRenderer::setupCallbacks() {
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetCursorPosCallback(m_window, mouseCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
}

void GLRenderer::renderLoop(EmbreeScene& embree_scene, Camera& camera, PathTracer& path_tracer) {
    setCamera(&camera);
    
    // Allocate image buffer (RGB format)
    std::vector<unsigned char> image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    
    std::cout << "Starting rendering loop..." << std::endl;
    std::cout << "Controls: WASD to move, mouse to look around, ESC to exit" << std::endl;
    
    while (!shouldClose()) {
        // Update timing
        updateTiming();
        
        // Process input
        glfwPollEvents();
        processInput();
        
        // Check for camera movement
        if (hasCameraMoved()) {
            // Reset accumulation when camera moves
            m_accumulated_samples = 0;
            std::fill(m_accumulation_buffer.begin(), m_accumulation_buffer.end(), glm::vec3(0.0f));
        }
        
        // Increment sample count
        ++m_accumulated_samples;
        
        // Test wavefront path tracer if enabled
        if (m_test_wavefront) {
            renderWavefront(image, camera, embree_scene.getScene(), path_tracer,
                           m_accumulation_buffer, m_accumulated_samples);
        } else {
            // Render image using PathTracer's parallel processing
            path_tracer.renderImage(image, IMAGE_WIDTH, IMAGE_HEIGHT, camera, embree_scene.getScene(),
                                   m_accumulation_buffer, m_accumulated_samples,
                                   m_camera_moved, m_tiles_completed);
        }
        
        // Render to OpenGL
        renderFrame(image);
        glfwSwapBuffers(m_window);
        
        // Optional: Display progress info every second
        static float last_info_time = 0.0f;
        if (glfwGetTime() - last_info_time > 5.0f) {
            std::cout << "Samples: " << m_accumulated_samples << ", FPS: " << (1.0f / m_delta_time) << std::endl;
            last_info_time = static_cast<float>(glfwGetTime());
        }
    }
}

void GLRenderer::renderFrame(const std::vector<unsigned char>& image) {
    // Upload texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    
    // Render
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    glBegin(GL_QUADS);
    // Flip top/bottom vertex positions as requested (invert Y for each vertex)
    // Texture coordinates remain the same; only vertex positions swap vertically.
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glEnd();
}

void GLRenderer::updateTiming() {
    const auto currentFrame = static_cast<float>(glfwGetTime());
    m_delta_time = currentFrame - m_last_frame;
    m_last_frame = currentFrame;
}

void GLRenderer::processInput() {
    if (!m_camera) return;
    
    if (m_keys[GLFW_KEY_W])
        m_camera->processKeyboard(Camera::FORWARD, m_delta_time);
    if (m_keys[GLFW_KEY_S])
        m_camera->processKeyboard(Camera::BACKWARD, m_delta_time);
    if (m_keys[GLFW_KEY_A])
        m_camera->processKeyboard(Camera::LEFT, m_delta_time);
    if (m_keys[GLFW_KEY_D])
        m_camera->processKeyboard(Camera::RIGHT, m_delta_time);
}

bool GLRenderer::hasCameraMoved() {
    if (!m_camera) return false;
    
    // Check if camera has moved using Camera class method with enhanced sensitivity for glass effects
    m_camera_moved = m_camera->hasMovedSinceLastCheck(0.0005f, 0.1f);
    return m_camera_moved;
}

bool GLRenderer::shouldClose() const {
    return m_window && glfwWindowShouldClose(m_window);
}

void GLRenderer::cleanup() {
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }
    
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    
    glfwTerminate();
}

// Static callback functions
void GLRenderer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (!s_instance) return;
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    
    // Toggle wavefront integrator test with 'T' key
    if (key == GLFW_KEY_T && action == GLFW_PRESS) {
        s_instance->m_test_wavefront = !s_instance->m_test_wavefront;
        s_instance->m_accumulated_samples = 0;
        std::fill(s_instance->m_accumulation_buffer.begin(), s_instance->m_accumulation_buffer.end(), glm::vec3(0.0f));
        std::cout << "Wavefront integrator test: " << (s_instance->m_test_wavefront ? "ON" : "OFF") << std::endl;
    }
    
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            s_instance->m_keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            s_instance->m_keys[key] = false;
        }
    }
}

void GLRenderer::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!s_instance || !s_instance->m_camera) return;
    
    // Only process mouse movement if left mouse button is pressed
    if (!s_instance->m_mouse_buttons[GLFW_MOUSE_BUTTON_LEFT]) return;
    
    if (s_instance->m_first_mouse) {
        s_instance->m_last_x = static_cast<float>(xpos);
        s_instance->m_last_y = static_cast<float>(ypos);
        s_instance->m_first_mouse = false;
    }
    
    const auto xoffset = static_cast<float>(xpos) - s_instance->m_last_x;
    const auto yoffset = s_instance->m_last_y - static_cast<float>(ypos); // Reversed since y-coordinates range from bottom to top
    s_instance->m_last_x = static_cast<float>(xpos);
    s_instance->m_last_y = static_cast<float>(ypos);
    
    s_instance->m_camera->processMouseMovement(xoffset, yoffset);
}

void GLRenderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (!s_instance) return;
    
    if (button >= 0 && button < 8) {
        if (action == GLFW_PRESS) {
            s_instance->m_mouse_buttons[button] = true;
            // Reset first mouse when starting to drag
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                s_instance->m_first_mouse = true;
            }
        } else if (action == GLFW_RELEASE) {
            s_instance->m_mouse_buttons[button] = false;
        }
    }
}

void GLRenderer::errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

void GLRenderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void GLRenderer::renderWavefront(std::vector<unsigned char>& image, const Camera& camera,
                                 RTCScene scene, PathTracer& path_tracer,
                                 std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples) {
    constexpr int TILE_SIZE = 32;  // Match the tile size from PathTracer
    
    // Calculate tile dimensions
    const auto numTilesX = (IMAGE_WIDTH + TILE_SIZE - 1) / TILE_SIZE;
    const auto numTilesY = (IMAGE_HEIGHT + TILE_SIZE - 1) / TILE_SIZE;
    const auto totalTiles = numTilesX * numTilesY;
    
    // Parallel tile rendering using TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, totalTiles), 
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                renderWavefrontTileTask((int)i, image, IMAGE_WIDTH, IMAGE_HEIGHT,
                                       camera, scene, *this, path_tracer, numTilesX, numTilesY,
                                       accumulation_buffer, accumulated_samples);
            }
        });
}

void GLRenderer::renderWavefrontTileTask(int tileIndex, std::vector<unsigned char>& image,
                                         int width, int height, const Camera& camera, RTCScene scene,
                                         const GLRenderer& renderer, PathTracer& path_tracer, int numTilesX, int numTilesY,
                                         std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples) {
    constexpr int TILE_SIZE = 32;
    
    // Calculate tile coordinates
    const auto tileX = tileIndex % numTilesX;
    const auto tileY = tileIndex / numTilesX;
    
    const auto x0 = tileX * TILE_SIZE;
    const auto y0 = tileY * TILE_SIZE;
    const auto x1 = std::min(x0 + TILE_SIZE, width);
    const auto y1 = std::min(y0 + TILE_SIZE, height);
    
    // Setup wavefront context once per tile
    wf::WFContext ctx;
    ctx.scene = scene;
    ctx.materials = &path_tracer.getMaterialManager();
    ctx.lights = &path_tracer.getLightManager();
    ctx.env = &path_tracer.getEnvironmentManager();
    
    // Note: We can't use lambda with PathTracer reference in parallel context safely,
    // so we'll use the default implementation from wf_math.h
    
    // Setup parameters
    wf::WFParams wp;
    wp.spp = 1;
    wp.max_depth = path_tracer.getSettings().max_depth;
    
    // Render pixels in this tile
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            uint32_t pixel_seed = uint32_t(y * width + x);
            
            // Generate one sample with subpixel jitter per frame
            uint32_t rng = wf::wang_hash(pixel_seed ^ accumulated_samples * 9781u);
            
            // Subpixel jitter
            float jx = wf::default_rand01(rng);
            float jy = wf::default_rand01(rng);
            
            // Generate jittered primary ray
            float u = (float(x) + jx) / float(width);
            float v = (float(y) + jy) / float(height);
            glm::vec3 ray_dir = camera.getRayDirection(u, v);
            glm::vec3 origin = camera.getPosition();
            
            // Trace ray
            glm::vec3 sample_color = renderer.m_wf_path_tracer->traceRay(
                ctx,
                wp,
                origin,
                ray_dir,
                pixel_seed ^ accumulated_samples
            );
            
            // Accumulate the sample
            int pixel_index = y * width + x;
            if (pixel_index >= 0 && pixel_index < accumulation_buffer.size()) {
                accumulation_buffer[pixel_index] += sample_color;
                
                // Get averaged color for display
                glm::vec3 averaged_color = accumulation_buffer[pixel_index] / float(accumulated_samples);
                
                // Tone mapping & gamma correction
                averaged_color = path_tracer.getEnvironmentManager().acesToneMapping(averaged_color);
                averaged_color = glm::pow(averaged_color, glm::vec3(1.0f / 2.2f));
                
                // Clamp and convert to 8-bit
                averaged_color = glm::clamp(averaged_color, 0.0f, 1.0f);
                
                // Write to image buffer
                const size_t idx = pixel_index * 3;
                if (idx + 2 < image.size()) {
                    image[idx + 0] = static_cast<unsigned char>(averaged_color.r * 255.0f);
                    image[idx + 1] = static_cast<unsigned char>(averaged_color.g * 255.0f);
                    image[idx + 2] = static_cast<unsigned char>(averaged_color.b * 255.0f);
                }
            }
        }
    }
}