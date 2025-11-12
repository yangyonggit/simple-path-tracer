#include "Camera.h"
#include <cmath>
#include <algorithm>

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
    , m_movement_speed(2.5f)
    , m_mouse_sensitivity(0.1f)
{
    // Calculate initial yaw and pitch from position and target
    glm::vec3 direction = glm::normalize(target - position);
    m_yaw = glm::degrees(atan2(direction.z, direction.x));
    m_pitch = glm::degrees(asin(direction.y));
    
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    // Update m_forward based on yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_forward = glm::normalize(front);
    
    // Update right and up vectors
    m_right = glm::normalize(glm::cross(m_forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    m_camera_up = glm::normalize(glm::cross(m_right, m_forward));
    
    // Update target based on new forward direction
    m_target = m_position + m_forward;
    
    // Calculate screen dimensions based on FOV
    m_half_height = std::tan(glm::radians(m_fov) * 0.5f);
    m_half_width = m_half_height * m_aspect_ratio;
}

void Camera::processKeyboard(int direction, float deltaTime) {
    float velocity = m_movement_speed * deltaTime;
    
    switch (direction) {
        case FORWARD:
            m_position += m_forward * velocity;
            break;
        case BACKWARD:
            m_position -= m_forward * velocity;
            break;
        case LEFT:
            m_position -= m_right * velocity;
            break;
        case RIGHT:
            m_position += m_right * velocity;
            break;
    }
    
    // Update target to maintain forward direction
    m_target = m_position + m_forward;
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= m_mouse_sensitivity;
    yoffset *= m_mouse_sensitivity;

    m_yaw += xoffset;
    m_pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch) {
        m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
    }

    // Update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void Camera::setPosition(const glm::vec3& position) {
    m_position = position;
    m_target = m_position + m_forward;
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