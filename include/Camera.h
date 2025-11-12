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
    
    // Camera controls
    float m_yaw;
    float m_pitch;
    float m_movement_speed;
    float m_mouse_sensitivity;
    
    // Movement detection
    glm::vec3 m_last_position;
    float m_last_yaw;
    float m_last_pitch;
    bool m_first_movement_check;
    
    void updateCameraVectors();
    
public:
    Camera(const glm::vec3& position, 
           const glm::vec3& target, 
           const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f),
           float fov = 45.0f,
           float aspect_ratio = 1.0f);
    
    // Generate a ray direction for given pixel coordinates
    glm::vec3 getRayDirection(float x, float y) const;
    
    // Camera control methods
    void processKeyboard(int direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    void setPosition(const glm::vec3& position);
    void setMovementSpeed(float speed) { m_movement_speed = speed; }
    void setMouseSensitivity(float sensitivity) { m_mouse_sensitivity = sensitivity; }
    
    // Getters
    const glm::vec3& getPosition() const { return m_position; }
    const glm::vec3& getFront() const { return m_forward; }
    const glm::vec3& getRight() const { return m_right; }
    const glm::vec3& getUp() const { return m_camera_up; }
    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }
    
    // Movement detection
    bool hasMovedSinceLastCheck(float position_threshold = 0.001f, float rotation_threshold = 0.1f);
    void resetMovementTracking();
    
    void setAspectRatio(float aspect_ratio);
    
    // Camera movement directions
    enum Movement {
        FORWARD = 0,
        BACKWARD = 1,
        LEFT = 2,
        RIGHT = 3
    };
};