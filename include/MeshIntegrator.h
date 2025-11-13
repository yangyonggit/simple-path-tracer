#pragma once

#include "GLTFLoader.h"
#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <vector>

class MeshIntegrator {
public:
    MeshIntegrator(RTCDevice device, RTCScene scene);
    ~MeshIntegrator();
    
    // Add mesh from GLTF loader to Embree scene
    bool addMeshToScene(const Mesh& mesh, unsigned int materialID);
    
    // Add all meshes from GLTF loader
    bool addAllMeshes(const GLTFLoader& loader);
    
    // Get material ID for a geometry ID
    unsigned int getMaterialID(unsigned int geomID) const;
    
    // Clear all added meshes
    void clear();
    
private:
    RTCDevice m_device;
    RTCScene m_scene;
    
    // Map geometry ID to material ID
    std::vector<unsigned int> m_materialMapping;
    
    // Helper functions
    RTCGeometry createEmbreeGeometry(const Mesh& mesh);
    void updateMaterialMapping(unsigned int geomID, unsigned int materialID);
};