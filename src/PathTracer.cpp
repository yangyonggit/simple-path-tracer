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
    setupDefaultMaterials();
    
    // Try to load cubemap (should auto-detect cross layout)
    if (!loadCubemap("assets/Cubemap/derelict_airfield_02_4k.hdr")) {
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

void PathTracer::setupDefaultMaterials() {
    m_materials.clear();
    
    // Material 0: Ground plane - Concrete
    m_materials.push_back(Materials::Concrete());
    
    // Material 1: Test box - Red plastic  
    m_materials.push_back(Material(glm::vec3(0.8f, 0.2f, 0.2f), 0.0f, 0.4f));
    
    // Material 2: Cube - Copper metal
    m_materials.push_back(Materials::Copper());
    
    // Material 3: Sphere - Gold metal
    m_materials.push_back(Materials::Gold());
}

void PathTracer::setMaterial(int index, const Material& material) {
    if (index >= 0 && index < m_materials.size()) {
        m_materials[index] = material;
    }
}

const Material& PathTracer::getMaterial(int index) const {
    if (index >= 0 && index < m_materials.size()) {
        return m_materials[index];
    }
    // Return default material if index is out of bounds
    static Material defaultMaterial;
    return defaultMaterial;
}

const Material& PathTracer::getMaterialByID(int geomID) const {
    return getMaterial(geomID);
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
    rayhit.ray.tnear = 1e-4f; // Use robust tnear for secondary rays
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray intersection
    rtcIntersect1(scene, &rayhit);

    // If no hit, return background color (sky)
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return getCubemapColor(glm::normalize(direction));
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
    const Material& material = getMaterialByID(rayhit.hit.geomID);
    
    // Add emission if material is emissive
    glm::vec3 emission = material.emission;
    
    // Calculate direct lighting contribution using PBR
    glm::vec3 view_dir = -direction; // Direction towards camera
    
    glm::vec3 direct_lighting(0.0f);
    const auto& lights = m_light_manager;
    
    // Sample all lights for direct lighting
    for (size_t i = 0; i < lights.getLightCount(); ++i) {
        const Light* light = lights.getLight(i);
        if (!light) continue;
        
        glm::vec3 light_direction;
        float light_distance;
        glm::vec3 light_radiance = light->getRadiance(hit_point, normal, light_direction, light_distance);
        
        float cos_theta = glm::max(glm::dot(normal, light_direction), 0.0f);
        if (cos_theta > 0.0f) {
            // Check for shadows
            bool occluded = light->isOccluded(hit_point, normal, light_direction, light_distance, scene);
            if (!occluded) {
                // Use Cook-Torrance BRDF for realistic shading
                glm::vec3 brdf = material.evaluateBRDF(normal, view_dir, light_direction);
                direct_lighting += brdf * light_radiance;
            }
        }
    }
    
    // Generate random direction using cosine-weighted hemisphere sampling
    glm::vec3 scatter_direction = cosineHemisphereSample(normal);
    
    // Recursive ray tracing for indirect lighting
    glm::vec3 indirect_light(0.0f);
    glm::vec3 indirect_contribution(0.0f);
    
    if (material.metallic > 0.5f) {
        // Metals: Sample environment using reflection direction for better specular highlights
        glm::vec3 reflect_dir = glm::reflect(-view_dir, normal);
        
        // Add some roughness-based perturbation for realistic metal reflection
        if (material.roughness > 0.01f) {
            glm::vec3 roughened_reflect = reflect_dir + randomUnitSphere() * material.roughness * 0.3f;
            reflect_dir = glm::normalize(roughened_reflect);
        }
        
        // Use robust epsilon for secondary ray origin offset
        float eps = 1e-4f * glm::max(1.0f, glm::max(glm::max(abs(hit_point.x), abs(hit_point.y)), abs(hit_point.z)));
        glm::vec3 offset_origin = hit_point + normal * eps;
        indirect_light = tracePathMonteCarlo(scene, offset_origin, reflect_dir, depth - 1);
        
        // For metals, use simpler reflection model instead of full BRDF to avoid artifacts
        // Metals mainly reflect the environment directly
        glm::vec3 fresnel = material.getF0(); // Use material's F0 value
        float ndotv = glm::max(glm::dot(normal, view_dir), 0.0f);
        
        // Simple fresnel approximation
        glm::vec3 F = fresnel + (glm::vec3(1.0f) - fresnel) * pow(1.0f - ndotv, 5.0f);
        
        // Metal contribution - increase to make reflections more visible
        indirect_contribution = F * indirect_light * 0.8f; // Increased from 0.4f to make reflections clearer
        
    } else {
        // Dielectrics: Use importance sampling for global illumination
        // Sample multiple directions for better color bleeding
        glm::vec3 total_indirect(0.0f);
        
        // Exponential sampling reduction: use more samples at shallow depths for better quality
        // while maintaining performance by reducing samples at deeper levels
        // Formula: samples = max(1, initial_samples >> (depth_threshold_passed))
        const int max_samples = 2;
        const int depth_threshold = m_settings.max_depth / 2; // Use half of max_depth as threshold
        int depth_exceeded = std::max(0, depth - depth_threshold);
        int indirect_samples = std::max(1, max_samples >> depth_exceeded); // Bit shift for power-of-2 reduction
        
        for (int i = 0; i < indirect_samples; i++) {
            glm::vec3 scatter_direction = cosineHemisphereSample(normal);
            
            // Use robust epsilon for secondary ray origin offset
            float eps = 1e-4f * glm::max(1.0f, glm::max(glm::max(abs(hit_point.x), abs(hit_point.y)), abs(hit_point.z)));
            glm::vec3 offset_origin = hit_point + normal * eps;
            
            glm::vec3 sample_light = tracePathMonteCarlo(scene, offset_origin, scatter_direction, depth - 1);
            total_indirect += sample_light;
        }
        
        indirect_light = total_indirect / float(indirect_samples);
        
        // Enhanced global illumination strength for realistic color bleeding
        float indirect_strength = 0.3f; // Increased for more visible global illumination
        indirect_contribution = material.getDiffuseColor() * indirect_light * indirect_strength;
    }
    
    // Combine emission, direct lighting, and indirect lighting
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
        return getCubemapColor(glm::normalize(direction));
    }

    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);
    
    // Make sure normal faces the incoming ray
    if (glm::dot(normal, direction) > 0.0f) {
        normal = -normal;
    }

    // Get material properties
    const Material& material = getMaterialByID(rayhit.hit.geomID);
    
    // Calculate hit point
    glm::vec3 hit_point = origin + rayhit.ray.tfar * direction;
    
    // Use the new lighting system with PBR for direct lighting
    glm::vec3 view_dir = -direction; // Direction towards camera
    
    glm::vec3 direct_lighting(0.0f);
    const auto& lights = m_light_manager;
    
    // Sample all lights for direct lighting
    for (size_t i = 0; i < lights.getLightCount(); ++i) {
        const Light* light = lights.getLight(i);
        if (!light) continue;
        
        glm::vec3 light_direction;
        float light_distance;
        glm::vec3 light_radiance = light->getRadiance(hit_point, normal, light_direction, light_distance);
        
        float cos_theta = glm::max(glm::dot(normal, light_direction), 0.0f);
        if (cos_theta > 0.0f) {
            // Check for shadows
            bool occluded = light->isOccluded(hit_point, normal, light_direction, light_distance, scene);
            if (!occluded) {
                // Use Cook-Torrance BRDF
                glm::vec3 brdf = material.evaluateBRDF(normal, view_dir, light_direction);
                direct_lighting += brdf * light_radiance;
            }
        }
    }
    
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
    color = acesToneMapping(color);
    
    // Gamma correction
    color = glm::pow(color, glm::vec3(1.0f / 2.2f));
    
    return color;
}

void PathTracer::renderImage(std::vector<unsigned char>& pixels, int width, int height,
                            const Camera& camera, RTCScene scene,
                            std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                            bool camera_moved, std::atomic<int>& tiles_completed) const {
    const int TILE_SIZE = 64;  // Match the tile size from main.cpp
    
    // Calculate tile dimensions
    int numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    int numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    int totalTiles = numTilesX * numTilesY;
    
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

// Environment/Sky color implementation with enhanced environment mapping
glm::vec3 PathTracer::getSkyColor(const glm::vec3& direction) {
    // Normalized direction (should already be normalized)
    glm::vec3 dir = glm::normalize(direction);
    
    // Sun position (slightly elevated, pointing from southeast)
    glm::vec3 sun_direction = glm::normalize(glm::vec3(0.3f, 0.5f, -0.8f));
    
    // Calculate angle to sun
    float sun_dot = glm::max(glm::dot(dir, sun_direction), 0.0f);
    
    // Base sky colors - very conservative to prevent artificial brightness
    glm::vec3 horizon_color(0.6f, 0.55f, 0.5f);  // Subdued warm horizon
    glm::vec3 zenith_color(0.2f, 0.35f, 0.6f);   // Gentle blue sky  
    glm::vec3 sun_color(1.5f, 1.3f, 1.1f);       // Mild sun
    glm::vec3 ground_color(0.08f, 0.06f, 0.04f); // Very subtle ground bounce
    
    // Sky gradient based on height
    float t = 0.5f * (dir.y + 1.0f); // Map from [-1, 1] to [0, 1]
    
    if (dir.y > 0.0f) {
        // Sky - blend from horizon to zenith
        glm::vec3 sky_color = glm::mix(horizon_color, zenith_color, t * t);
        
        // Add environmental variation for better reflections (but keep it subtle)
        // Create some directional variation to make reflections more interesting
        float azimuth = atan2(dir.z, dir.x) / (2.0f * 3.14159f) + 0.5f; // [0, 1]
        float variation = sin(azimuth * 8.0f) * 0.05f + sin(azimuth * 16.0f) * 0.02f; // Much smaller variation
        sky_color *= (1.0f + variation);
        
        // Add sun disk
        if (sun_dot > 0.998f) { // Very close to sun direction
            float sun_intensity = (sun_dot - 0.998f) / 0.002f; // Normalize to [0, 1]
            sky_color = glm::mix(sky_color, sun_color, sun_intensity);
        }
        // Add sun glow
        else if (sun_dot > 0.95f) {
            float glow_intensity = (sun_dot - 0.95f) / 0.048f; // Normalize to [0, 1]
            glm::vec3 glow_color = glm::mix(horizon_color, sun_color * 0.3f, glow_intensity);
            sky_color = glm::mix(sky_color, glow_color, glow_intensity * 0.5f);
        }
        
        return sky_color;
    } else {
        // Below horizon - darker ground reflection with some variation
        float ground_t = -dir.y; // How far below horizon
        float azimuth_ground = atan2(dir.z, dir.x) / (2.0f * 3.14159f) + 0.5f;
        float ground_variation = sin(azimuth_ground * 4.0f) * 0.05f;
        glm::vec3 varied_ground = ground_color * (1.0f + ground_variation);
        return glm::mix(horizon_color * 0.6f, varied_ground, ground_t * 0.7f);
    }
}

bool PathTracer::loadCubemap(const std::string& filename) {
    std::cout << "Loading cubemap: " << filename << std::endl;
    return m_cubemap.loadFromFile(filename);
}

glm::vec3 PathTracer::getCubemapColor(const glm::vec3& direction) const {
    if (m_cubemap.isLoaded()) {
        // Use cubemap for realistic environment lighting
        glm::vec3 color = m_cubemap.sample(direction);
        
        // Much milder tone mapping to preserve cubemap appearance
        // First clamp extreme values only
        color = glm::min(color, glm::vec3(5.0f)); // More reasonable clamp
        
        // Light exposure adjustment - keep most of original brightness
        color *= 0.7f; // Mild reduction instead of 0.15
        
        // Optional light Reinhard if still too bright
        // color = color / (color + glm::vec3(1.0f));
        
        return color;
    } else {
        // Fallback to procedural sky
        return getSkyColor(direction);
    }
}

// ACES Filmic tone mapping implementation
glm::vec3 PathTracer::acesToneMapping(const glm::vec3& color) {
    // ACES constants
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    // Apply ACES formula: (color * (a * color + b)) / (color * (c * color + d) + e)
    glm::vec3 numerator = color * (a * color + glm::vec3(b));
    glm::vec3 denominator = color * (c * color + glm::vec3(d)) + glm::vec3(e);
    
    // Avoid division by zero
    denominator = glm::max(denominator, glm::vec3(0.0001f));
    
    return glm::clamp(numerator / denominator, glm::vec3(0.0f), glm::vec3(1.0f));
}