#pragma once

#include "Cubemap.h"
#include <glm/glm.hpp>
#include <string>

class EnvironmentManager {
private:
    Cubemap m_cubemap;

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
    
    // Environment lighting
    glm::vec3 getEnvironmentColor(const glm::vec3& direction) const;
    
    // Fallback sky functions
    static glm::vec3 getSkyColor(const glm::vec3& direction);
    
    // Advanced tone mapping
    static glm::vec3 acesToneMapping(const glm::vec3& color);
};