#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include "GLTFLoader.h"
#include "MeshIntegrator.h"

class EmbreeScene {
public:
    // Sphere parameters for user-defined geometry (public for PathTracer access)
    struct SphereData {
        float center_x, center_y, center_z;
        float radius;
        unsigned int materialID;
        
        // Default constructor
        SphereData() = default;
        
        // Constructor with parameters
        SphereData(float cx, float cy, float cz, float r, unsigned int matID)
            : center_x(cx), center_y(cy), center_z(cz), radius(r), materialID(matID) {}
        
        // Move constructor and assignment (for improved performance)
        SphereData(SphereData&& other) noexcept = default;
        SphereData& operator=(SphereData&& other) noexcept = default;
        
        // Copy constructor and assignment
        SphereData(const SphereData& other) = default;
        SphereData& operator=(const SphereData& other) = default;
    };

private:
    RTCDevice m_device;
    RTCScene m_scene;
    std::vector<SphereData> m_spheres;
    
    // GLTF mesh integration
    std::unique_ptr<MeshIntegrator> m_meshIntegrator;
    std::unique_ptr<GLTFLoader> m_gltfLoader;
    
public:
    EmbreeScene();
    EmbreeScene(bool loadDefaultScene);
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
    
    // GLTF mesh loading (public interface)
    bool loadGLTF(const std::string& filepath, const glm::vec3& position = glm::vec3(0.0f), float scale = 1.0f, unsigned int materialID = 0);
    
private:
    void initialize();
    void initialize(bool loadDefaultScene);
    void addGroundPlane();
    void addTestBox();
    void addCube();
    void addSphere();
    void addSphereWithMaterial(unsigned int materialID, const glm::vec3& position, float radius);
    void addCubeWithMaterial(unsigned int materialID, const glm::vec3& position, float size);
    void cleanup();
    
    // GLTF mesh loading
    bool loadGLTF(const std::string& filepath, unsigned int materialID = 0);
    bool loadGLTFWithTransform(const std::string& filepath, const glm::mat4& transform, unsigned int materialID = 0);
    
    // Get material ID for geometry (includes both spheres and meshes)
    unsigned int getGeometryMaterialID(unsigned int geomID) const;
    
    // Helper function to create a cube with specified parameters
    void createCube(float pos_x, float pos_y, float pos_z, float size_x, float size_y, float size_z);
    
    // User-defined geometry callbacks
    static void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);
    static void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
    static void sphereBoundsFunc(const RTCBoundsFunctionArguments* args);
};