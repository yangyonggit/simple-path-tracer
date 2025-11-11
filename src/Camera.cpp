#include "Camera.h"
#include <cmath>

Camera::Camera(const glm::vec3& position, 
               const glm::vec3& target, 
               const glm::vec3& up,
               float fov,
               float aspect_ratio)
    : m_position(position)
    , m_target(target)
    , m_up(up)
    , m_fov(fov)
    , m_aspect_ratio(aspect_ratio)
{
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    m_forward = glm::normalize(m_target - m_position);
    m_right = glm::normalize(glm::cross(m_forward, m_up));
    m_camera_up = glm::cross(m_right, m_forward);
    
    // Calculate screen dimensions based on FOV
    m_half_height = std::tan(glm::radians(m_fov) * 0.5f);
    m_half_width = m_half_height * m_aspect_ratio;
}

glm::vec3 Camera::getRayDirection(float x, float y) const {
    // Convert from [0,1] to [-1,1] and flip y to match screen coordinates
    float normalized_x = (x - 0.5f) * 2.0f;
    float normalized_y = -(y - 0.5f) * 2.0f; // Flip Y axis
    
    // Calculate ray direction in camera space
    glm::vec3 ray_dir = m_forward 
                      + normalized_x * m_half_width * m_right
                      + normalized_y * m_half_height * m_camera_up;
    
    return glm::normalize(ray_dir);
}

void Camera::setAspectRatio(float aspect_ratio) {
    m_aspect_ratio = aspect_ratio;
    updateCameraVectors();
}