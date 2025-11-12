#include <embree4/rtcore.h>
#include <iostream>
#include <vector>
#include <memory>
#include "Camera.h"
#include "EmbreeScene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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

    std::cout << "Raytracing complete. Saving image...\n";

    // Save image as PNG
    if (stbi_write_png("output.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, image.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Image saved successfully as 'output.png'\n";
    } else {
        std::cerr << "Failed to save image!\n";
    }

    std::cout << "Path tracer completed successfully!\n";
    return 0;
}
