#include "PathTracer.h"
#include "Camera.h"
#include "Light.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <atomic>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Thread local static member definitions
thread_local std::mt19937 PathTracer::s_rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> PathTracer::s_uniform_dist(0.0f, 1.0f);

PathTracer::PathTracer(const Settings& settings) : m_settings(settings) {
}

void PathTracer::initializeRandomSeed() {
    s_rng.seed(std::random_device{}());
}

float PathTracer::randomFloat() {
    return s_uniform_dist(s_rng);
}

float PathTracer::randomFloat(float min, float max) {
    return min + (max - min) * randomFloat();
}

glm::vec3 PathTracer::randomUnitSphere() {
    glm::vec3 p;
    do {
        p = 2.0f * glm::vec3(randomFloat(), randomFloat(), randomFloat()) - glm::vec3(1.0f);
    } while (glm::dot(p, p) >= 1.0f);
    return p;
}

glm::vec3 PathTracer::randomUnitHemisphere(const glm::vec3& normal) {
    glm::vec3 in_unit_sphere = randomUnitSphere();
    if (glm::dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

glm::vec3 PathTracer::cosineHemisphereSample(const glm::vec3& normal) {
    // Generate random numbers
    float u1 = randomFloat();
    float u2 = randomFloat();
    
    // Convert to spherical coordinates with cosine weighting
    float phi = 2.0f * M_PI * u1;
    float cos_theta = sqrt(u2);
    float sin_theta = sqrt(1.0f - u2);
    
    // Local coordinates (z-up)
    glm::vec3 local_dir(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    
    // Create orthonormal basis around normal
    glm::vec3 nt = (abs(normal.x) > 0.1f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 u = glm::normalize(glm::cross(normal, nt));
    glm::vec3 v = glm::cross(normal, u);
    
    // Transform to world space
    return local_dir.x * u + local_dir.y * v + local_dir.z * normal;
}

glm::vec3 PathTracer::getColorFromGeometryID(int geomID) {
    switch(geomID) {
        case 0: return glm::vec3(0.8f, 0.8f, 0.8f); // Ground plane - Light gray
        case 1: return glm::vec3(1.0f, 0.2f, 0.2f); // Test box - Red
        case 2: return glm::vec3(0.2f, 1.0f, 0.2f); // Cube - Green
        case 3: return glm::vec3(0.2f, 0.2f, 1.0f); // Sphere - Blue
        default: return glm::vec3(0.0f, 0.0f, 0.0f); // Background - Black
    }
}

glm::vec3 PathTracer::getMaterialAlbedo(int geomID) {
    switch(geomID) {
        case 0: return glm::vec3(0.7f, 0.7f, 0.7f); // Ground plane - Light gray diffuse
        case 1: return glm::vec3(0.8f, 0.3f, 0.3f); // Test box - Red diffuse
        case 2: return glm::vec3(0.3f, 0.8f, 0.3f); // Cube - Green diffuse
        case 3: return glm::vec3(0.3f, 0.3f, 0.8f); // Sphere - Blue diffuse
        default: return glm::vec3(0.0f, 0.0f, 0.0f); // Background - Black
    }
}

glm::vec3 PathTracer::tracePathMonteCarlo(RTCScene scene, const glm::vec3& origin, 
                                         const glm::vec3& direction, int depth) const {
    // Stop if we've reached max depth
    if (depth <= 0) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    RTCRayHit rayhit;
    
    // Set ray origin and direction
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    
    // Set ray parameters
    rayhit.ray.tnear = 0.001f; // Small offset to avoid self-intersection
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray intersection
    rtcIntersect1(scene, &rayhit);

    // If no hit, return background color (sky)
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        // Simple sky gradient
        float t = 0.5f * (glm::normalize(direction).y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }

    // Calculate hit point
    glm::vec3 hit_point = origin + rayhit.ray.tfar * direction;
    
    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);
    
    // Make sure normal faces the incoming ray
    if (glm::dot(normal, direction) > 0.0f) {
        normal = -normal;
    }

    // Get material properties
    glm::vec3 albedo = getMaterialAlbedo(rayhit.hit.geomID);
    
    // Calculate direct lighting contribution using light manager
    glm::vec3 view_dir = -direction; // Direction towards camera
    glm::vec3 direct_lighting = m_light_manager.calculateDirectLighting(
        hit_point, normal, view_dir, albedo, scene);
    
    // Simple emission for light sources (none in this scene)
    glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // Generate random direction using cosine-weighted hemisphere sampling
    glm::vec3 scatter_direction = cosineHemisphereSample(normal);
    
    // Recursive ray tracing for indirect lighting
    glm::vec3 indirect_light = tracePathMonteCarlo(scene, hit_point, scatter_direction, depth - 1);
    
    // BRDF for Lambertian diffuse: albedo / π
    // The cosine term is already accounted for in the cosine-weighted sampling
    // The π factor cancels out with the π in the Monte Carlo estimator
    glm::vec3 indirect_contribution = albedo * indirect_light;
    
    // Combine direct and indirect lighting
    return emission + direct_lighting + indirect_contribution;
}

glm::vec3 PathTracer::traceRaySimple(RTCScene scene, const glm::vec3& origin, 
                                    const glm::vec3& direction) const {
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
        float t = 0.5f * (glm::normalize(direction).y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }

    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);
    
    // Make sure normal faces the incoming ray
    if (glm::dot(normal, direction) > 0.0f) {
        normal = -normal;
    }

    // Get material properties
    glm::vec3 albedo = getMaterialAlbedo(rayhit.hit.geomID);
    
    // Calculate hit point
    glm::vec3 hit_point = origin + rayhit.ray.tfar * direction;
    
    // Use the new lighting system for direct lighting
    glm::vec3 view_dir = -direction; // Direction towards camera
    glm::vec3 direct_lighting = m_light_manager.calculateDirectLighting(
        hit_point, normal, view_dir, albedo, scene);
    
    return direct_lighting;
}

glm::vec3 PathTracer::traceRay(RTCScene scene, const glm::vec3& origin, 
                              const glm::vec3& direction) const {
    if (!m_settings.enable_path_tracing) {
        return traceRaySimple(scene, origin, direction);
    }
    
    glm::vec3 color(0.0f, 0.0f, 0.0f);
    
    // Shoot multiple samples and average
    for (int s = 0; s < m_settings.samples_per_pixel; ++s) {
        color += tracePathMonteCarlo(scene, origin, direction, m_settings.max_depth);
    }
    
    // Average the samples
    color /= float(m_settings.samples_per_pixel);
    
    // Simple tone mapping (gamma correction)
    color = glm::sqrt(color);
    
    return color;
}

void PathTracer::renderTileTask(int tileIndex, int threadIndex, std::vector<unsigned char>& pixels,
                               int width, int height, const Camera& camera, RTCScene scene,
                               const PathTracer& pathTracer, int numTilesX, int numTilesY,
                               std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                               bool camera_moved, std::atomic<int>& tiles_completed) {
    const int TILE_SIZE = 64;  // Match the tile size from main.cpp
    
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
            
            int pixel_index = y * width + x;
            
            if (camera_moved) {
                // Camera moved - start new accumulation
                accumulation_buffer[pixel_index] = color;
            } else {
                // Camera stationary - accumulate samples
                accumulation_buffer[pixel_index] += color;
            }
            
            // Get averaged color for display
            glm::vec3 averaged_color = accumulation_buffer[pixel_index] / float(accumulated_samples);
            
            // Clamp color to valid range
            averaged_color = glm::clamp(averaged_color, 0.0f, 1.0f);
            
            // Convert to 8-bit color and store in image
            int rgb_index = pixel_index * 3;
            pixels[rgb_index + 0] = static_cast<unsigned char>(averaged_color.r * 255); // R
            pixels[rgb_index + 1] = static_cast<unsigned char>(averaged_color.g * 255); // G
            pixels[rgb_index + 2] = static_cast<unsigned char>(averaged_color.b * 255); // B
        }
    }
    
    // Update progress atomically
    int completed = ++tiles_completed;
    if (completed % 8 == 0) { // Show progress every 8 tiles
        int total = numTilesX * numTilesY;
        std::cout << "Rendered " << completed << "/" << total << " tiles (" 
                 << int(100.0f * completed / total) << "%)\r" << std::flush;
    }
}