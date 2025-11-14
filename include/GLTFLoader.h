#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Material.h"

// Forward declarations
namespace tinygltf {
    class Model;
    class Node;
    class Mesh;
}

// Simple mesh data structure
struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<unsigned int> indices;
    
    // Transform matrix for positioning
    glm::mat4 transform;
    
    // Assigned material ID in our system
    unsigned int materialID;
    
    // Default constructor
    Mesh() : transform(1.0f), materialID(0) {}
    
    // Move constructor
    Mesh(Mesh&& other) noexcept
        : vertices(std::move(other.vertices)),
          normals(std::move(other.normals)),
          texcoords(std::move(other.texcoords)),
          indices(std::move(other.indices)),
          transform(other.transform),
          materialID(other.materialID) {}
    
    // Move assignment operator
    Mesh& operator=(Mesh&& other) noexcept {
        if (this != &other) {
            vertices = std::move(other.vertices);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
            transform = other.transform;
            materialID = other.materialID;
        }
        return *this;
    }
    
    // Copy constructor and copy assignment - explicitly defaulted
    Mesh(const Mesh& other) = default;
    Mesh& operator=(const Mesh& other) = default;
};

class GLTFLoader {
public:
    GLTFLoader();
    ~GLTFLoader();
    
    // Load GLTF file and extract mesh data
    bool loadGLTF(const std::string& filepath);
    
    // Get loaded meshes
    const std::vector<Mesh>& getMeshes() const { return m_meshes; }
    
    // Clear loaded data
    void clear();
    
    // Get bounding box of all meshes
    void getBoundingBox(glm::vec3& min, glm::vec3& max) const;
    
    // Transform all meshes (scale, translate, rotate)
    void transformMeshes(const glm::mat4& transform);
    
    // Assign material to all meshes or specific mesh
    void assignMaterial(unsigned int materialID);
    void assignMaterial(size_t meshIndex, unsigned int materialID);
    
private:
    std::vector<Mesh> m_meshes;
    
    // Helper functions
    void createTestCube();
    void calculateNormals(Mesh& mesh);
    
    // Helper functions for parsing
    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4& transform);
    void processNode(const tinygltf::Model& model, const tinygltf::Node& node, const glm::mat4& parentTransform);
    
    // Utility functions
    glm::mat4 getNodeTransform(const tinygltf::Node& node);
};