#include "Light.h"
#include <embree4/rtcore.h>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Base Light class implementation
Light::Light(Type type, const glm::vec3& color, float intensity)
    : m_type(type), m_color(color), m_intensity(intensity) {
}

bool Light::isOccluded(const glm::vec3& hit_point, const glm::vec3& normal,
                      const glm::vec3& light_dir, float light_distance, RTCScene scene) const {
    RTCRay shadow_ray;
    
    // Offset the ray origin along the surface normal to avoid self-intersection
    const float epsilon = 0.001f;
    glm::vec3 offset_origin = hit_point + normal * epsilon;
    
    // Set up shadow ray
    shadow_ray.org_x = offset_origin.x;
    shadow_ray.org_y = offset_origin.y;
    shadow_ray.org_z = offset_origin.z;
    shadow_ray.dir_x = light_dir.x;
    shadow_ray.dir_y = light_dir.y;
    shadow_ray.dir_z = light_dir.z;
    shadow_ray.tnear = 0.0f; // Start from offset origin
    shadow_ray.tfar = light_distance - epsilon; // Stop just before light
    shadow_ray.mask = 0xFFFFFFFF;
    shadow_ray.flags = 0;
    
    // Check for occlusion
    rtcOccluded1(scene, &shadow_ray);
    
    return shadow_ray.tfar < 0.0f; // Ray was occluded if tfar is negative
}

// DirectionalLight implementation
DirectionalLight::DirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity)
    : Light(DIRECTIONAL, color, intensity), m_direction(glm::normalize(-direction)) {
    // Note: we store the direction TO the light (opposite of light rays)
}

glm::vec3 DirectionalLight::getRadiance(const glm::vec3& hit_point, const glm::vec3& normal,
                                       glm::vec3& light_direction, float& distance) const {
    light_direction = m_direction; // Direction TO the light
    distance = std::numeric_limits<float>::infinity(); // Directional lights are infinitely far
    
    // No attenuation for directional lights
    return m_color * m_intensity;
}

// PointLight implementation
PointLight::PointLight(const glm::vec3& position, const glm::vec3& color, float intensity,
                      float constant, float linear, float quadratic)
    : Light(POINT, color, intensity), m_position(position),
      m_constant_attenuation(constant), m_linear_attenuation(linear), 
      m_quadratic_attenuation(quadratic) {
}

glm::vec3 PointLight::getRadiance(const glm::vec3& hit_point, const glm::vec3& normal,
                                 glm::vec3& light_direction, float& distance) const {
    // Calculate direction and distance to light
    glm::vec3 light_vector = m_position - hit_point;
    distance = glm::length(light_vector);
    light_direction = light_vector / distance; // Normalize
    
    // Calculate attenuation
    float attenuation = m_constant_attenuation + 
                       m_linear_attenuation * distance + 
                       m_quadratic_attenuation * distance * distance;
    
    // Return attenuated radiance
    return (m_color * m_intensity) / attenuation;
}

// LightManager implementation
LightManager::LightManager() {
}

void LightManager::addDirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity) {
    m_lights.push_back(std::make_unique<DirectionalLight>(direction, color, intensity));
}

void LightManager::addPointLight(const glm::vec3& position, const glm::vec3& color, float intensity) {
    m_lights.push_back(std::make_unique<PointLight>(position, color, intensity));
}

glm::vec3 LightManager::calculateDirectLighting(const glm::vec3& hit_point, const glm::vec3& normal,
                                               const glm::vec3& view_dir, const glm::vec3& albedo,
                                               RTCScene scene) const {
    glm::vec3 total_lighting(0.0f);
    
    for (const auto& light : m_lights) {
        glm::vec3 light_direction;
        float light_distance;
        
        // Get light radiance and direction
        glm::vec3 light_radiance = light->getRadiance(hit_point, normal, light_direction, light_distance);
        
        // Calculate the cosine of the angle between surface normal and light direction
        float cos_theta = glm::max(glm::dot(normal, light_direction), 0.0f);
        
        if (cos_theta > 0.0f) {
            // Check for shadows (occlusion)
            bool occluded = light->isOccluded(hit_point, normal, light_direction, light_distance, scene);
            
            if (!occluded) {
                // Lambertian BRDF: albedo / π
                // The π factor is often omitted in real-time rendering for simplicity
                glm::vec3 brdf = albedo / float(M_PI);
                
                // Add this light's contribution
                // L_out = BRDF * L_in * cos(θ)
                total_lighting += brdf * light_radiance * cos_theta;
            }
        }
    }
    
    return total_lighting;
}

const Light* LightManager::getLight(size_t index) const {
    if (index < m_lights.size()) {
        return m_lights[index].get();
    }
    return nullptr;
}

void LightManager::clearLights() {
    m_lights.clear();
}