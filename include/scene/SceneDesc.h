#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace scene {

// ========================================
// Material - Backend-agnostic material description
// ========================================
struct Material {
    glm::vec3 baseColor = glm::vec3(0.8f, 0.8f, 0.8f);  // Albedo/diffuse color
    glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.0f);   // Emissive color
    
    float metallic = 0.0f;        // 0 = dielectric, 1 = metal
    float roughness = 0.5f;       // Surface roughness
    float ior = 1.5f;             // Index of refraction
    float transparency = 0.0f;    // 0 = opaque, 1 = fully transparent
    
    // Default constructor
    Material() = default;
    
    // Convenience constructor for basic materials
    Material(const glm::vec3& color, const glm::vec3& emissive = glm::vec3(0.0f))
        : baseColor(color), emission(emissive) {}
};

// ========================================
// SphereData - Analytical sphere (user-defined geometry)
// ========================================
struct SphereData {
    glm::vec3 center = glm::vec3(0.0f);
    float radius = 0.5f;
    uint32_t materialId = 0;
    
    SphereData() = default;
    SphereData(const glm::vec3& c, float r, uint32_t mat = 0)
        : center(c), radius(r), materialId(mat) {}
};

// ========================================
// MeshData - Triangle mesh geometry
// ========================================
struct MeshData {
    std::vector<glm::vec3> positions;   // Vertex positions (required)
    std::vector<glm::vec3> normals;     // Vertex normals (optional but recommended)
    std::vector<glm::vec2> texcoords;   // Texture coordinates (optional)
    std::vector<glm::uvec3> indices;    // Triangle indices (3 per triangle)
    
    uint32_t materialId = 0;            // Material index for this mesh
    
    // Helper: Check if mesh has valid data
    bool isValid() const {
        return !positions.empty() && !indices.empty();
    }
    
    // Helper: Get triangle count
    size_t triangleCount() const {
        return indices.size();
    }
    
    // Helper: Get vertex count
    size_t vertexCount() const {
        return positions.size();
    }
};

// ========================================
// InstanceData - Mesh instance with transformation
// ========================================
struct InstanceData {
    uint32_t meshId = 0;                          // Index into SceneDesc::meshes
    glm::mat4 worldFromObject = glm::mat4(1.0f);  // Object-to-world transformation
    uint32_t materialId = 0;                      // Material override (if different from mesh default)
    
    // Default constructor
    InstanceData() = default;
    
    // Convenience constructor
    InstanceData(uint32_t mesh, const glm::mat4& transform = glm::mat4(1.0f), uint32_t mat = 0)
        : meshId(mesh), worldFromObject(transform), materialId(mat) {}
};

// ========================================
// SceneDesc - Complete scene description
// ========================================
struct SceneDesc {
    std::vector<Material> materials;
    std::vector<MeshData> meshes;
    std::vector<InstanceData> instances;
    std::vector<SphereData> spheres;  // Analytical spheres (user-defined geometry in Embree)
    
    // Helper: Add material and return its index
    uint32_t addMaterial(const Material& mat) {
        materials.push_back(mat);
        return static_cast<uint32_t>(materials.size() - 1);
    }
    
    // Helper: Add mesh and return its index
    uint32_t addMesh(const MeshData& mesh) {
        meshes.push_back(mesh);
        return static_cast<uint32_t>(meshes.size() - 1);
    }
    
    // Helper: Add mesh with move semantics
    uint32_t addMesh(MeshData&& mesh) {
        meshes.push_back(std::move(mesh));
        return static_cast<uint32_t>(meshes.size() - 1);
    }
    
    // Helper: Add instance and return its index
    uint32_t addInstance(const InstanceData& inst) {
        instances.push_back(inst);
        return static_cast<uint32_t>(instances.size() - 1);
    }
    
    // Helper: Create instance directly
    uint32_t addInstance(uint32_t meshId, const glm::mat4& transform = glm::mat4(1.0f), uint32_t materialId = 0) {
        return addInstance(InstanceData(meshId, transform, materialId));
    }
    
    // Helper: Add sphere and return its index
    uint32_t addSphere(const SphereData& sphere) {
        spheres.push_back(sphere);
        return static_cast<uint32_t>(spheres.size() - 1);
    }
    
    // Helper: Add sphere directly
    uint32_t addSphere(const glm::vec3& center, float radius, uint32_t materialId = 0) {
        return addSphere(SphereData(center, radius, materialId));
    }
    
    // Helper: Clear all scene data
    void clear() {
        materials.clear();
        meshes.clear();
        instances.clear();
        spheres.clear();
    }
    
    // Helper: Get scene statistics
    size_t totalTriangles() const {
        size_t total = 0;
        for (const auto& mesh : meshes) {
            total += mesh.triangleCount();
        }
        return total * instances.size(); // Rough estimate with instancing
    }
    
    size_t totalVertices() const {
        size_t total = 0;
        for (const auto& mesh : meshes) {
            total += mesh.vertexCount();
        }
        return total;
    }
};

// ========================================
// Helper functions for creating common primitives
// ========================================

// Create a unit cube mesh centered at origin
inline MeshData createCubeMesh(uint32_t materialId = 0) {
    MeshData mesh;
    mesh.materialId = materialId;
    
    // 8 vertices of a unit cube (same as old EmbreeScene::createCube)
    // Vertex layout: 0-3 bottom face, 4-7 top face
    mesh.positions = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},  // Bottom face
        {0.5f, -0.5f,  0.5f}, {-0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f, -0.5f}, {0.5f,  0.5f, -0.5f},  // Top face
        {0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
    };
    
    // 12 triangles (6 faces * 2 triangles) - same order as old EmbreeScene::createCube
    mesh.indices = {
        {0, 2, 1}, {0, 3, 2},  // Bottom face
        {4, 5, 6}, {4, 6, 7},  // Top face
        {0, 1, 5}, {0, 5, 4},  // Front face
        {2, 3, 7}, {2, 7, 6},  // Back face
        {3, 0, 4}, {3, 4, 7},  // Left face
        {1, 2, 6}, {1, 6, 5}   // Right face
    };
    
    return mesh;
}

// Create a ground plane (large quad)
inline MeshData createGroundPlaneMesh(float size = 10.0f, uint32_t materialId = 0) {
    MeshData mesh;
    mesh.materialId = materialId;
    
    float half = size * 0.5f;
    
    // 4 vertices for a quad
    mesh.positions = {
        {-half, 0.0f, -half},
        { half, 0.0f, -half},
        { half, 0.0f,  half},
        {-half, 0.0f,  half}
    };
    
    // Normals pointing up
    mesh.normals = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };
    
    // 2 triangles
    mesh.indices = {
        {0, 2, 1},
        {0, 3, 2}
    };
    
    return mesh;
}

// Create a UV sphere mesh
inline MeshData createSphereMesh(uint32_t stacks = 32, uint32_t slices = 64, float radius = 0.5f, uint32_t materialId = 0) {
    MeshData mesh;
    mesh.materialId = materialId;
    
    constexpr float PI = 3.14159265358979323846f;
    
    // Generate vertices
    for (uint32_t stack = 0; stack <= stacks; ++stack) {
        float phi = PI * static_cast<float>(stack) / static_cast<float>(stacks);
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        
        for (uint32_t slice = 0; slice <= slices; ++slice) {
            float theta = 2.0f * PI * static_cast<float>(slice) / static_cast<float>(slices);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            
            // Position
            glm::vec3 position(
                radius * sinPhi * cosTheta,
                radius * cosPhi,
                radius * sinPhi * sinTheta
            );
            
            // Normal (normalized position for unit sphere)
            glm::vec3 normal = glm::normalize(position);
            
            // Texture coordinates
            glm::vec2 texcoord(
                static_cast<float>(slice) / static_cast<float>(slices),
                static_cast<float>(stack) / static_cast<float>(stacks)
            );
            
            mesh.positions.push_back(position);
            mesh.normals.push_back(normal);
            mesh.texcoords.push_back(texcoord);
        }
    }
    
    // Generate indices (triangles)
    for (uint32_t stack = 0; stack < stacks; ++stack) {
        for (uint32_t slice = 0; slice < slices; ++slice) {
            uint32_t first = stack * (slices + 1) + slice;
            uint32_t second = first + slices + 1;
            
            // First triangle
            mesh.indices.push_back(glm::uvec3(first, second, first + 1));
            
            // Second triangle
            mesh.indices.push_back(glm::uvec3(second, second + 1, first + 1));
        }
    }
    
    return mesh;
}

} // namespace scene
