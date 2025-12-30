#pragma once

#include "Cubemap.h"
#include <glm/glm.hpp>
#include <string>

class EnvironmentManager {
private:
    Cubemap m_cubemap;

    // Keep CPU behavior identical to the current hard-coded values in getCubemapColor().
    float m_env_intensity = 0.8f;
    float m_env_max_clamp = 5.0f;

public:
    EnvironmentManager() = default;
    ~EnvironmentManager() = default;
    
    // Non-copyable but movable
    EnvironmentManager(const EnvironmentManager&) = delete;
    EnvironmentManager& operator=(const EnvironmentManager&) = delete;
    EnvironmentManager(EnvironmentManager&&) = default;
    EnvironmentManager& operator=(EnvironmentManager&&) = default;
    
    // Cubemap management
    bool loadCubemap(const std::string& filename);
    glm::vec3 getCubemapColor(const glm::vec3& direction) const;
    bool hasCubemap() const { return m_cubemap.isLoaded(); }

    // GPU-facing environment map access (equirectangular HDR source, if available)
    bool hasEquirectangularEnvironment() const { return m_cubemap.hasEquirectangular(); }
    int getEquirectangularWidth() const { return m_cubemap.getEquirectWidth(); }
    int getEquirectangularHeight() const { return m_cubemap.getEquirectHeight(); }
    const float* getEquirectangularRGBA() const { return m_cubemap.getEquirectRGBA(); }
    uint64_t getEquirectangularRevision() const { return m_cubemap.getEquirectRevision(); }
    float getEnvironmentIntensity() const { return m_env_intensity; }
    float getEnvironmentMaxClamp() const { return m_env_max_clamp; }
    
    // Environment lighting
    glm::vec3 getEnvironmentColor(const glm::vec3& direction) const;
    
    // Fallback sky functions
    static glm::vec3 getSkyColor(const glm::vec3& direction);
    
    // Advanced tone mapping
    static glm::vec3 acesToneMapping(const glm::vec3& color);
};