#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
private:
    glm::vec3 m_position;
    glm::vec3 m_target;
    glm::vec3 m_up;
    float m_fov;
    float m_aspect_ratio;
    
    // Camera coordinate system
    glm::vec3 m_forward;
    glm::vec3 m_right;
    glm::vec3 m_camera_up;
    
    // Screen parameters
    float m_half_width;
    float m_half_height;
    
    void updateCameraVectors();
    
public:
    Camera(const glm::vec3& position, 
           const glm::vec3& target, 
           const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f),
           float fov = 45.0f,
           float aspect_ratio = 1.0f);
    
    // Generate a ray direction for given pixel coordinates
    glm::vec3 getRayDirection(float x, float y) const;
    
    const glm::vec3& getPosition() const { return m_position; }
    
    void setAspectRatio(float aspect_ratio);
};