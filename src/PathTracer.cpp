#include "PathTracer.h"
#include "Camera.h"
#include "Light.h"
#include "Material.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <atomic>
#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Thread local static member definitions
thread_local std::mt19937 PathTracer::s_rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> PathTracer::s_uniform_dist(0.0f, 1.0f);

PathTracer::PathTracer(const Settings& settings) : m_settings(settings) {
    // Try to load cubemap (should auto-detect cross layout)
    if (!m_environment_manager.loadCubemap("assets/Cubemap/brown_photostudio_02_4k.hdr")) {
        std::cout << "Failed to load cubemap, using procedural sky" << std::endl;
    }
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
    glm::vec3 in_unit_sphere = glm::normalize(randomUnitSphere());
    if (glm::dot(in_unit_sphere, normal) > 0.0f) {
        return in_unit_sphere;
    } else {
        return -in_unit_sphere;
    }
}

glm::vec3 PathTracer::cosineHemisphereSample(const glm::vec3& normal) {
    // Cosine-weighted hemisphere sampling
    float r1 = randomFloat();
    float r2 = randomFloat();
    
    float cos_theta = sqrt(r1);
    float sin_theta = sqrt(1.0f - r1);
    float phi = 2.0f * M_PI * r2;
    
    glm::vec3 sample_dir(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
    
    // Build tangent space around normal
    glm::vec3 up = (abs(normal.x) < 0.9f) ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    
    return tangent * sample_dir.x + normal * sample_dir.y + bitangent * sample_dir.z;
}

bool PathTracer::intersectRay(RTCScene scene, const glm::vec3& origin, const glm::vec3& direction,
                             RTCRayHit& rayhit) const {
    // Initialize ray
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    rayhit.ray.tnear = 1e-4f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    
    // Initialize hit
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    
    // Perform intersection
    rtcIntersect1(scene, &rayhit);
    
    return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

glm::vec3 PathTracer::calculateSafeRayOrigin(const glm::vec3& hit_point, const glm::vec3& normal, 
                                           bool offset_forward) const {
    // Calculate epsilon based on hit point magnitude for numerical stability
    float eps = 1e-4f * glm::max(1.0f, glm::max(glm::max(abs(hit_point.x), abs(hit_point.y)), abs(hit_point.z)));
    
    if (offset_forward) {
        return hit_point + normal * eps;
    } else {
        return hit_point - normal * eps;
    }
}

glm::vec3 PathTracer::tracePathMonteCarlo(RTCScene scene, const glm::vec3& origin, 
                                         const glm::vec3& direction, int depth) const {
    // Stop if we've reached max depth
    if (depth <= 0) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    RTCRayHit rayhit;
    
    // Perform ray intersection
    if (!intersectRay(scene, origin, direction, rayhit)) {
        return m_environment_manager.getEnvironmentColor(glm::normalize(direction));
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
    
    // Get material for this geometry
    const Material& material = m_material_manager.getMaterialFromHit(scene, rayhit);
    
    // Add emission if material is emissive
    glm::vec3 color = material.emission;
    
    // Calculate direct lighting contribution using material BRDF
    glm::vec3 view_dir = -direction;
    const auto& lights = m_light_manager;
    
    // Sample all lights for direct lighting
    for (size_t i = 0; i < lights.getLightCount(); ++i) {
        const Light& light = lights.getLight(i);
        
        glm::vec3 light_direction;
        float light_distance;
        glm::vec3 light_radiance = light.getRadiance(hit_point, normal, light_direction, light_distance);
        
        float cos_theta = glm::max(glm::dot(normal, light_direction), 0.0f);
        if (cos_theta > 0.0f) {
            // Check for shadows
            bool occluded = light.isOccluded(hit_point, normal, light_direction, light_distance, scene);
            if (!occluded) {
                // Use material BRDF for realistic shading
                glm::vec3 brdf = material.evaluateBRDF(normal, view_dir, light_direction);
                color += brdf * light_radiance * cos_theta;
            }
        }
    }
    
    // Handle different material types for indirect lighting
    if (material.metallic > 0.5f) {
        // Metallic reflection
        glm::vec3 reflect_dir = glm::reflect(direction, normal);
        glm::vec3 offset_origin = calculateSafeRayOrigin(hit_point, normal, true);
        glm::vec3 indirect_light = tracePathMonteCarlo(scene, offset_origin, reflect_dir, depth - 1);
        color += material.albedo * indirect_light * material.metallic;
    } 
    else if (material.isTransparent()) {
        // Glass/transparent material
        float ior = material.ior;
        float cosine = -glm::dot(direction, normal);
        float eta = cosine > 0.0f ? (1.0f / ior) : ior;
        float transparency = material.getTransparency();
        
        glm::vec3 refracted = refract(direction, normal, eta);
        if (glm::length(refracted) > 0.0f && !shouldTransmit(material, cosine)) {
            // Refraction
            glm::vec3 refract_origin = calculateSafeRayOrigin(hit_point, normal, false);
            glm::vec3 transmission_light = tracePathMonteCarlo(scene, refract_origin, refracted, depth - 1);
            color += transmission_light * transparency;
        } else {
            // Total internal reflection
            glm::vec3 reflect_dir = glm::reflect(direction, normal);
            glm::vec3 reflect_origin = calculateSafeRayOrigin(hit_point, normal, true);
            glm::vec3 reflection_light = tracePathMonteCarlo(scene, reflect_origin, reflect_dir, depth - 1);
            color += reflection_light;
        }
    }
    else {
        // Diffuse material
        glm::vec3 scatter_direction = cosineHemisphereSample(normal);
        glm::vec3 offset_origin = calculateSafeRayOrigin(hit_point, normal, true);
        
        // Russian roulette for path termination
        float survival_probability = glm::max(glm::max(material.albedo.r, material.albedo.g), material.albedo.b);
        if (randomFloat() < survival_probability) {
            glm::vec3 sample_light = tracePathMonteCarlo(scene, offset_origin, scatter_direction, depth - 1);
            color += (material.albedo * sample_light) / survival_probability;
        }
    }
    
    return color;
}

glm::vec3 PathTracer::traceRaySimple(RTCScene scene, const glm::vec3& origin, 
                                    const glm::vec3& direction) const {
    RTCRayHit rayhit;
    
    // Perform ray intersection
    if (!intersectRay(scene, origin, direction, rayhit)) {
        return m_environment_manager.getEnvironmentColor(glm::normalize(direction));
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
    
    // Get material for this geometry
    const Material& material = m_material_manager.getMaterialFromHit(scene, rayhit);
    
    // Use the lighting system with PBR for direct lighting
    glm::vec3 view_dir = -direction; // Direction towards camera
    
    glm::vec3 direct_lighting(0.0f);
    const auto& lights = m_light_manager;
    
    // Sample all lights for direct lighting
    for (size_t i = 0; i < lights.getLightCount(); ++i) {
        const Light& light = lights.getLight(i);
        
        glm::vec3 light_direction;
        float light_distance;
        glm::vec3 light_radiance = light.getRadiance(hit_point, normal, light_direction, light_distance);
        
        float cos_theta = glm::max(glm::dot(normal, light_direction), 0.0f);
        if (cos_theta > 0.0f) {
            // Check for shadows
            bool occluded = light.isOccluded(hit_point, normal, light_direction, light_distance, scene);
            if (!occluded) {
                // Use material BRDF
                glm::vec3 brdf = material.evaluateBRDF(normal, view_dir, light_direction);
                direct_lighting += brdf * light_radiance;
            }
        }
    }
    
    // Add material emission
    return material.emission + direct_lighting;
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
    
    // ACES Filmic tone mapping for natural exposure
    color = m_environment_manager.acesToneMapping(color);
    
    // Gamma correction
    color = glm::pow(color, glm::vec3(1.0f / 2.2f));
    
    return color;
}

void PathTracer::renderImage(std::vector<unsigned char>& pixels, int width, int height,
                            const Camera& camera, RTCScene scene,
                            std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                            bool camera_moved, std::atomic<int>& tiles_completed) const {
    constexpr int TILE_SIZE = 32;  // Match the tile size from main.cpp
    
    // Calculate tile dimensions
    const auto numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    const auto numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    const auto totalTiles = numTilesX * numTilesY;
    
    // Reset progress counter
    tiles_completed = 0;
    
    // Parallel tile rendering using TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, totalTiles), 
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                renderTileTask((int)i, 0, pixels, width, height, 
                              camera, scene, *this, numTilesX, numTilesY,
                              accumulation_buffer, accumulated_samples,
                              camera_moved, tiles_completed);
            }
        });
}

void PathTracer::renderTileTask(int tileIndex, int threadIndex, std::vector<unsigned char>& pixels,
                               int width, int height, const Camera& camera, RTCScene scene,
                               const PathTracer& pathTracer, int numTilesX, int numTilesY,
                               std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                               bool camera_moved, std::atomic<int>& tiles_completed) {
    constexpr int TILE_SIZE = 32;  // Match the tile size from main.cpp
    
    // Calculate tile coordinates
    const auto tileX = tileIndex % numTilesX;
    const auto tileY = tileIndex / numTilesX;
    
    const auto x0 = tileX * TILE_SIZE;
    const auto y0 = tileY * TILE_SIZE;
    const auto x1 = std::min(x0 + TILE_SIZE, width);
    const auto y1 = std::min(y0 + TILE_SIZE, height);
    
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
            
            // Add bounds checking for safety
            if (pixel_index >= 0 && pixel_index < accumulation_buffer.size()) {
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
                if (rgb_index >= 0 && rgb_index + 2 < pixels.size()) {
                    pixels[rgb_index + 0] = static_cast<unsigned char>(averaged_color.r * 255); // R
                    pixels[rgb_index + 1] = static_cast<unsigned char>(averaged_color.g * 255); // G
                    pixels[rgb_index + 2] = static_cast<unsigned char>(averaged_color.b * 255); // B
                }
            }
        }
    }
    
    // Increment tile completion counter
    ++tiles_completed;
}

float PathTracer::schlickFresnel(float cosine, float ior) const {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

glm::vec3 PathTracer::refract(const glm::vec3& incident, const glm::vec3& normal, float eta) const {
    float cos_i = -glm::dot(incident, normal);
    float sin_t2 = eta * eta * (1.0f - cos_i * cos_i);
    
    if (sin_t2 >= 1.0f) {
        return glm::vec3(0.0f); // Total internal reflection
    }
    
    float cos_t = sqrt(1.0f - sin_t2);
    return eta * incident + (eta * cos_i - cos_t) * normal;
}

bool PathTracer::shouldTransmit(const Material& material, float cosine) const {
    float fresnel = schlickFresnel(abs(cosine), material.ior);
    return randomFloat() > fresnel;
}
