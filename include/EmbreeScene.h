#pragma once

#include <embree4/rtcore.h>

class EmbreeScene {
private:
    RTCDevice m_device;
    RTCScene m_scene;
    
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
    void cleanup();
};