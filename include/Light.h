#pragma once

#include <glm/glm.hpp>
#include <embree4/rtcore.h>
#include <vector>
#include <memory>

// Base light class
class Light {
public:
    enum Type {
        DIRECTIONAL,
        POINT,
        AREA
    };

protected:
    Type m_type;
    glm::vec3 m_color;
    float m_intensity;

public:
    Light(Type type, const glm::vec3& color, float intensity);
    virtual ~Light() = default;

    // Get light contribution at a point
    virtual glm::vec3 getRadiance(const glm::vec3& hit_point, const glm::vec3& normal,
                                  glm::vec3& light_direction, float& distance) const = 0;
    
    // Check if light is occluded by scene geometry
    virtual bool isOccluded(const glm::vec3& hit_point, const glm::vec3& normal,
                           const glm::vec3& light_dir, float light_distance, RTCScene scene) const;

    // Accessors
    Type getType() const { return m_type; }
    const glm::vec3& getColor() const { return m_color; }
    float getIntensity() const { return m_intensity; }
    
    void setColor(const glm::vec3& color) { m_color = color; }
    void setIntensity(float intensity) { m_intensity = intensity; }
};

// Directional light (like sun)
class DirectionalLight : public Light {
private:
    glm::vec3 m_direction; // Direction TO the light (opposite of light rays)

public:
    DirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity);
    
    glm::vec3 getRadiance(const glm::vec3& hit_point, const glm::vec3& normal,
                         glm::vec3& light_direction, float& distance) const override;
    
    void setDirection(const glm::vec3& direction) { m_direction = glm::normalize(direction); }
    const glm::vec3& getDirection() const { return m_direction; }
};

// Point light
class PointLight : public Light {
private:
    glm::vec3 m_position;
    float m_constant_attenuation;
    float m_linear_attenuation;
    float m_quadratic_attenuation;

public:
    PointLight(const glm::vec3& position, const glm::vec3& color, float intensity,
               float constant = 1.0f, float linear = 0.09f, float quadratic = 0.032f);
    
    glm::vec3 getRadiance(const glm::vec3& hit_point, const glm::vec3& normal,
                         glm::vec3& light_direction, float& distance) const override;
    
    void setPosition(const glm::vec3& position) { m_position = position; }
    const glm::vec3& getPosition() const { return m_position; }
    
    void setAttenuation(float constant, float linear, float quadratic) {
        m_constant_attenuation = constant;
        m_linear_attenuation = linear;
        m_quadratic_attenuation = quadratic;
    }
};

// Light manager class
class LightManager {
private:
    std::vector<std::unique_ptr<Light>> m_lights;

public:
    LightManager();
    ~LightManager() = default;

    // Add lights
    void addDirectionalLight(const glm::vec3& direction, const glm::vec3& color, float intensity);
    void addPointLight(const glm::vec3& position, const glm::vec3& color, float intensity);
    
    // Calculate direct lighting contribution
    glm::vec3 calculateDirectLighting(const glm::vec3& hit_point, const glm::vec3& normal,
                                     const glm::vec3& view_dir, const glm::vec3& albedo,
                                     RTCScene scene) const;
    
    // Accessors
    size_t getLightCount() const { return m_lights.size(); }
    const Light& getLight(size_t index) const;
    void clearLights();
};