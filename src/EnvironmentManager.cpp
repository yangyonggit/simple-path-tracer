#include "EnvironmentManager.h"
#include <iostream>
#include <algorithm>

bool EnvironmentManager::loadCubemap(const std::string& filename) {
    return m_cubemap.loadFromFile(filename);
}

glm::vec3 EnvironmentManager::getCubemapColor(const glm::vec3& direction) const {
    if (m_cubemap.isLoaded()) {
        // Use cubemap for realistic environment lighting
        glm::vec3 color = m_cubemap.sample(direction);
        
        // Much milder tone mapping to preserve cubemap appearance
        // First clamp extreme values only
        color = glm::min(color, glm::vec3(m_env_max_clamp)); // More reasonable clamp
        
        // Light exposure adjustment - increase for stronger global lighting
        color *= m_env_intensity; // Increased from 0.4f to boost environment lighting
        
        // Optional light Reinhard if still too bright
        // color = color / (color + glm::vec3(1.0f));
        
        return color;
    } else {
        // Fallback to procedural sky
        return getSkyColor(direction);
    }
}

glm::vec3 EnvironmentManager::getEnvironmentColor(const glm::vec3& direction) const {
    return getCubemapColor(direction);
}

glm::vec3 EnvironmentManager::getSkyColor(const glm::vec3& direction) {
    // Create a simple procedural sky
    float t = 0.5f * (direction.y + 1.0f); // Map y from [-1,1] to [0,1]
    
    // Smooth horizon gradient
    t = glm::smoothstep(0.0f, 1.0f, t);
    
    // Sky colors
    glm::vec3 horizon_color(0.7f, 0.8f, 0.9f);  // Light blue-gray
    glm::vec3 zenith_color(0.2f, 0.4f, 0.8f);   // Deep blue
    
    // Interpolate between horizon and zenith
    glm::vec3 sky_color = glm::mix(horizon_color, zenith_color, t);
    
    // Add some sun glow effect
    glm::vec3 sun_direction = glm::normalize(glm::vec3(0.3f, 0.6f, -0.8f));
    float sun_dot = glm::max(glm::dot(direction, sun_direction), 0.0f);
    
    // Sun disk and glow
    float sun_intensity = pow(sun_dot, 64.0f);  // Sharp sun disk
    float sun_glow = pow(sun_dot, 8.0f) * 0.3f; // Softer glow
    
    glm::vec3 sun_color(1.0f, 0.9f, 0.7f);  // Warm sun color
    sky_color += sun_color * (sun_intensity + sun_glow);
    
    return sky_color * 0.8f; // Scale down the overall intensity
}

glm::vec3 EnvironmentManager::acesToneMapping(const glm::vec3& color) {
    // ACES Filmic tone mapping
    // Reference: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    return glm::clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
}