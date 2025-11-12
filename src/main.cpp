#include <embree4/rtcore.h>
#include <iostream>
#include <vector>
#include <memory>
#include "Camera.h"
#include "EmbreeScene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Image dimensions
const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 600;

// Function to get color based on geometry ID
glm::vec3 getColorFromGeometryID(int geomID) {
    switch(geomID) {
        case 0: return glm::vec3(0.8f, 0.8f, 0.8f); // Ground plane - Light gray
        case 1: return glm::vec3(1.0f, 0.2f, 0.2f); // Test box - Red
        case 2: return glm::vec3(0.2f, 1.0f, 0.2f); // Cube - Green
        case 3: return glm::vec3(0.2f, 0.2f, 1.0f); // Sphere - Blue
        default: return glm::vec3(0.0f, 0.0f, 0.0f); // Background - Black
    }
}

// Function to trace a ray and return shaded color with Lambertian shading
glm::vec3 traceRayWithShading(RTCScene scene, const glm::vec3& origin, const glm::vec3& direction) {
    RTCRayHit rayhit;
    
    // Set ray origin and direction
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    
    // Set ray parameters
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray intersection
    rtcIntersect1(scene, &rayhit);

    // If no hit, return background color
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return glm::vec3(0.0f, 0.0f, 0.0f); // Black background
    }

    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);

    // Define directional light direction (light comes FROM this direction)
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));

    // Compute Lambertian diffuse intensity: max(dot(normal, -lightDir), 0)
    // We use -lightDir because we want the direction TO the light source
    float diffuseIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

    // Get base color from geometry ID
    glm::vec3 baseColor = getColorFromGeometryID(rayhit.hit.geomID);

    // Apply shading: multiply base color by diffuse intensity (white light)
    return baseColor * diffuseIntensity;
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

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW!\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    glViewport(0, 0, width, height);
    return window;
}

// Function to render the image using OpenGL
void renderImageWithOpenGL(GLFWwindow* window, const std::vector<unsigned char>& image, int width, int height) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Ensure correct row alignment for 3-byte RGB data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    while (!glfwWindowShouldClose(window)) {
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
        glfwPollEvents();
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

    // Create camera positioned to see ground plane, box, cube, and sphere
    glm::vec3 camera_pos(2.0f, 1.0f, 2.5f);  // Position above and away from scene
    glm::vec3 camera_target(0.0f, -0.3f, 0.0f);  // Look toward the scene center
    float aspect_ratio = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT);
    Camera camera(camera_pos, camera_target, glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, aspect_ratio);

    std::cout << "Camera created. Starting raytracing...\n";

    // Allocate image buffer (RGB format)
    std::vector<unsigned char> image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    // Raytrace each pixel
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            // Calculate normalized pixel coordinates
            float u = float(x) / float(IMAGE_WIDTH);
            // Use straightforward v (0..1) across image rows
            float v = float(y) / float(IMAGE_HEIGHT);
            
            // Get ray direction from camera
            glm::vec3 ray_dir = camera.getRayDirection(u, v);
            
            // Trace ray and get shaded color with Lambertian shading
            glm::vec3 color = traceRayWithShading(embree_scene.getScene(), camera.getPosition(), ray_dir);
            
            // Convert to 8-bit color and store in image
            int pixel_index = (y * IMAGE_WIDTH + x) * 3;
            image[pixel_index + 0] = static_cast<unsigned char>(color.r * 255); // R
            image[pixel_index + 1] = static_cast<unsigned char>(color.g * 255); // G
            image[pixel_index + 2] = static_cast<unsigned char>(color.b * 255); // B
        }
        
        // Print progress
        if (y % 100 == 0) {
            std::cout << "Rendered " << y << "/" << IMAGE_HEIGHT << " rows\n";
        }
    }

    std::cout << "Raytracing complete. Rendering image in OpenGL window...\n";
    // Create a flipped copy (rows reversed) for PNG and OpenGL upload so
    // the on-disk PNG and the GL texture both display upright.
    std::vector<unsigned char> flipped(image.size());
    const int rowBytes = IMAGE_WIDTH * 3;
    for (int yy = 0; yy < IMAGE_HEIGHT; ++yy) {
        const unsigned char* srcRow = image.data() + (IMAGE_HEIGHT - 1 - yy) * rowBytes;
        unsigned char* dstRow = flipped.data() + yy * rowBytes;
        memcpy(dstRow, srcRow, rowBytes);
    }

    // Saving to PNG temporarily disabled while debugging display
    /*
    std::cout << "Saving image to 'output.png'...\n";
    if (stbi_write_png("output.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, flipped.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Image saved successfully as 'output.png'\n";
    } else {
        std::cerr << "Failed to save image to 'output.png'\n";
    }
    */

    // Initialize OpenGL and create a window
    GLFWwindow* window = initializeOpenGL(IMAGE_WIDTH, IMAGE_HEIGHT);
    if (!window) {
        return -1;
    }

    // Render the image using OpenGL
    renderImageWithOpenGL(window, image, IMAGE_WIDTH, IMAGE_HEIGHT);

    std::cout << "Path tracer completed successfully!\n";
    return 0;
}
