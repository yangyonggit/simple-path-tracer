#include "MeshIntegrator.h"
#include <iostream>

MeshIntegrator::MeshIntegrator(RTCDevice device, RTCScene scene) 
    : m_device(device), m_scene(scene) {
}

MeshIntegrator::~MeshIntegrator() {
    clear();
}

bool MeshIntegrator::addMeshToScene(const Mesh& mesh, unsigned int materialID) {
    if (!m_device || !m_scene) {
        std::cerr << "Invalid Embree device or scene" << std::endl;
        return false;
    }
    
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        std::cerr << "Mesh has no vertices or indices" << std::endl;
        return false;
    }
    
    // Create Embree geometry
    RTCGeometry geometry = createEmbreeGeometry(mesh);
    if (!geometry) {
        std::cerr << "Failed to create Embree geometry" << std::endl;
        return false;
    }
    
    // Commit geometry
    rtcCommitGeometry(geometry);
    
    // Attach to scene and get geometry ID
    unsigned int geomID = rtcAttachGeometry(m_scene, geometry);
    
    // Release geometry (scene holds reference)
    rtcReleaseGeometry(geometry);
    
    // Update material mapping
    updateMaterialMapping(geomID, materialID);
    
    std::cout << "Added mesh to scene with geometry ID: " << geomID 
              << ", material ID: " << materialID << std::endl;
    
    return true;
}

bool MeshIntegrator::addAllMeshes(const GLTFLoader& loader) {
    const auto& meshes = loader.getMeshes();
    
    if (meshes.empty()) {
        std::cout << "No meshes to add from GLTF loader" << std::endl;
        return true;
    }
    
    bool success = true;
    for (size_t i = 0; i < meshes.size(); ++i) {
        const Mesh& mesh = meshes[i];
        if (!addMeshToScene(mesh, mesh.materialID)) {
            std::cerr << "Failed to add mesh " << i << " to scene" << std::endl;
            success = false;
        }
    }
    
    std::cout << "Added " << meshes.size() << " meshes to scene" << std::endl;
    return success;
}

unsigned int MeshIntegrator::getMaterialID(unsigned int geomID) const {
    if (geomID < m_materialMapping.size()) {
        return m_materialMapping[geomID];
    }
    return 0; // Default material
}

void MeshIntegrator::clear() {
    m_materialMapping.clear();
}

RTCGeometry MeshIntegrator::createEmbreeGeometry(const Mesh& mesh) {
    // Create triangle geometry
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (!geometry) {
        return nullptr;
    }
    
    // Set vertex buffer
    size_t vertexCount = mesh.vertices.size();
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), vertexCount);
    
    if (!vertices) {
        rtcReleaseGeometry(geometry);
        return nullptr;
    }
    
    // Copy vertices and apply transform
    for (size_t i = 0; i < vertexCount; ++i) {
        glm::vec4 transformedVertex = mesh.transform * glm::vec4(mesh.vertices[i], 1.0f);
        glm::vec3 vertex = glm::vec3(transformedVertex);
        
        vertices[i * 3 + 0] = vertex.x;
        vertices[i * 3 + 1] = vertex.y;
        vertices[i * 3 + 2] = vertex.z;
    }
    
    // Set index buffer
    size_t triangleCount = mesh.indices.size() / 3;
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), triangleCount);
    
    if (!indices) {
        rtcReleaseGeometry(geometry);
        return nullptr;
    }
    
    // Copy indices
    for (size_t i = 0; i < mesh.indices.size(); ++i) {
        indices[i] = mesh.indices[i];
    }
    
    // Set material ID as user data so raycast can access it
    rtcSetGeometryUserData(geometry, (void*)(uintptr_t)mesh.materialID);
    
    return geometry;
}

void MeshIntegrator::updateMaterialMapping(unsigned int geomID, unsigned int materialID) {
    // Ensure the mapping vector is large enough
    if (geomID >= m_materialMapping.size()) {
        m_materialMapping.resize(geomID + 1, 0);
    }
    
    m_materialMapping[geomID] = materialID;
}