#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <vector>

class EmbreeScene {
private:
    RTCDevice m_device;
    RTCScene m_scene;
    
    // Sphere parameters for user-defined geometry
    struct SphereData {
        float center_x, center_y, center_z;
        float radius;
        unsigned int materialID;
    };
    std::vector<SphereData> m_spheres;
    
public:
    EmbreeScene();
    ~EmbreeScene();
    
    // Non-copyable
    EmbreeScene(const EmbreeScene&) = delete;
    EmbreeScene& operator=(const EmbreeScene&) = delete;
    
    // Get the scene handle for ray intersections
    RTCScene getScene() const { return m_scene; }
    RTCDevice getDevice() const { return m_device; }
    
    bool isValid() const {
        return m_device != nullptr && m_scene != nullptr;
    }
    
private:
    void initialize();
    void addGroundPlane();
    void addTestBox();
    void addCube();
    void addSphere();
    void addSphereWithMaterial(unsigned int materialID, const glm::vec3& position, float radius);
    void cleanup();
    
    // Helper function to create a cube with specified parameters
    void createCube(float pos_x, float pos_y, float pos_z, float size_x, float size_y, float size_z);
    
    // User-defined geometry callbacks
    static void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);
    static void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
    static void sphereBoundsFunc(const RTCBoundsFunctionArguments* args);
};