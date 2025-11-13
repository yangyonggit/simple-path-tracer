#include "GLTFLoader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
// Do not define STB implementations here as they are already defined in other files
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// Disable JSON schema validation to avoid dependency issues
#define TINYGLTF_NO_JSON_SCHEMA
#include "tiny_gltf.h"

// For now, we'll implement a simple GLTF loader
// You can replace this with tinygltf or cgltf for full GLTF support

GLTFLoader::GLTFLoader() {
}

GLTFLoader::~GLTFLoader() {
    clear();
}

bool GLTFLoader::loadGLTF(const std::string& filepath) {
    clear();
    
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    
    bool ret;
    if (filepath.find(".glb") != std::string::npos) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filepath);
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    }
    
    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }
    
    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
        return false;
    }
    
    if (!ret) {
        std::cerr << "Failed to parse GLTF file: " << filepath << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded GLTF file: " << filepath << std::endl;
    std::cout << "Meshes: " << model.meshes.size() << ", Nodes: " << model.nodes.size() << std::endl;
    
    // Process all scenes (usually just one)
    for (const auto& scene : model.scenes) {
        for (int nodeIndex : scene.nodes) {
            processNode(model, model.nodes[nodeIndex], glm::mat4(1.0f));
        }
    }
    
    std::cout << "Extracted " << m_meshes.size() << " meshes from GLTF file" << std::endl;
    return true;
}

void GLTFLoader::clear() {
    m_meshes.clear();
}

void GLTFLoader::getBoundingBox(glm::vec3& min, glm::vec3& max) const {
    if (m_meshes.empty()) {
        min = max = glm::vec3(0.0f);
        return;
    }
    
    min = glm::vec3(FLT_MAX);
    max = glm::vec3(-FLT_MAX);
    
    for (const auto& mesh : m_meshes) {
        for (const auto& vertex : mesh.vertices) {
            // Apply transform
            glm::vec4 transformedVertex = mesh.transform * glm::vec4(vertex, 1.0f);
            glm::vec3 pos = glm::vec3(transformedVertex);
            
            min = glm::min(min, pos);
            max = glm::max(max, pos);
        }
    }
}

void GLTFLoader::transformMeshes(const glm::mat4& transform) {
    for (auto& mesh : m_meshes) {
        mesh.transform = transform * mesh.transform;
    }
}

void GLTFLoader::assignMaterial(unsigned int materialID) {
    for (auto& mesh : m_meshes) {
        mesh.materialID = materialID;
    }
}

void GLTFLoader::assignMaterial(size_t meshIndex, unsigned int materialID) {
    if (meshIndex < m_meshes.size()) {
        m_meshes[meshIndex].materialID = materialID;
    }
}

void GLTFLoader::createTestCube() {
    Mesh cube;
    
    // Cube vertices
    cube.vertices = {
        // Front face
        {-1.0f, -1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f},
        { 1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f},
        
        // Back face
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f},
        { 1.0f, -1.0f, -1.0f},
        
        // Top face
        {-1.0f,  1.0f, -1.0f},
        {-1.0f,  1.0f,  1.0f},
        { 1.0f,  1.0f,  1.0f},
        { 1.0f,  1.0f, -1.0f},
        
        // Bottom face
        {-1.0f, -1.0f, -1.0f},
        { 1.0f, -1.0f, -1.0f},
        { 1.0f, -1.0f,  1.0f},
        {-1.0f, -1.0f,  1.0f},
        
        // Right face
        { 1.0f, -1.0f, -1.0f},
        { 1.0f,  1.0f, -1.0f},
        { 1.0f,  1.0f,  1.0f},
        { 1.0f, -1.0f,  1.0f},
        
        // Left face
        {-1.0f, -1.0f, -1.0f},
        {-1.0f, -1.0f,  1.0f},
        {-1.0f,  1.0f,  1.0f},
        {-1.0f,  1.0f, -1.0f}
    };
    
    // Cube indices
    cube.indices = {
        0,  1,  2,   0,  2,  3,    // front
        4,  5,  6,   4,  6,  7,    // back
        8,  9,  10,  8,  10, 11,   // top
        12, 13, 14,  12, 14, 15,   // bottom
        16, 17, 18,  16, 18, 19,   // right
        20, 21, 22,  20, 22, 23    // left
    };
    
    // Calculate normals
    calculateNormals(cube);
    
    // Generate basic texture coordinates
    cube.texcoords.resize(cube.vertices.size());
    for (size_t i = 0; i < cube.texcoords.size(); ++i) {
        cube.texcoords[i] = glm::vec2((i % 4) / 3.0f, (i / 4) / 5.0f);
    }
    
    cube.materialID = 0; // Default material
    
    m_meshes.push_back(cube);
}

void GLTFLoader::calculateNormals(Mesh& mesh) {
    mesh.normals.resize(mesh.vertices.size(), glm::vec3(0.0f));
    
    // Calculate face normals and accumulate
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        unsigned int i0 = mesh.indices[i];
        unsigned int i1 = mesh.indices[i + 1];
        unsigned int i2 = mesh.indices[i + 2];
        
        glm::vec3 v0 = mesh.vertices[i0];
        glm::vec3 v1 = mesh.vertices[i1];
        glm::vec3 v2 = mesh.vertices[i2];
        
        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        
        mesh.normals[i0] += normal;
        mesh.normals[i1] += normal;
        mesh.normals[i2] += normal;
    }
    
    // Normalize accumulated normals
    for (auto& normal : mesh.normals) {
        normal = glm::normalize(normal);
    }
}

void GLTFLoader::processNode(const tinygltf::Model& model, const tinygltf::Node& node, const glm::mat4& parentTransform) {
    glm::mat4 nodeTransform = getNodeTransform(node);
    glm::mat4 worldTransform = parentTransform * nodeTransform;
    
    // Process mesh if this node has one
    if (node.mesh >= 0 && node.mesh < model.meshes.size()) {
        processMesh(model, model.meshes[node.mesh], worldTransform);
    }
    
    // Recursively process children
    for (int childIndex : node.children) {
        if (childIndex >= 0 && childIndex < model.nodes.size()) {
            processNode(model, model.nodes[childIndex], worldTransform);
        }
    }
}

void GLTFLoader::processMesh(const tinygltf::Model& model, const tinygltf::Mesh& gltfMesh, const glm::mat4& transform) {
    for (const auto& primitive : gltfMesh.primitives) {
        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
            continue; // Skip non-triangle primitives
        }
        
        Mesh mesh;
        mesh.transform = transform;
        
        // Get position accessor
        auto posIt = primitive.attributes.find("POSITION");
        if (posIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[posIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            
            const float* positions = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            
            mesh.vertices.resize(accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                mesh.vertices[i] = glm::vec3(
                    positions[i * 3 + 0],
                    positions[i * 3 + 1], 
                    positions[i * 3 + 2]
                );
            }
        }
        
        // Get normal accessor if available
        auto normalIt = primitive.attributes.find("NORMAL");
        if (normalIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[normalIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            
            const float* normals = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            
            mesh.normals.resize(accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                mesh.normals[i] = glm::vec3(
                    normals[i * 3 + 0],
                    normals[i * 3 + 1],
                    normals[i * 3 + 2]
                );
            }
        }
        
        // Get texture coordinates if available
        auto texcoordIt = primitive.attributes.find("TEXCOORD_0");
        if (texcoordIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[texcoordIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            
            const float* texcoords = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            
            mesh.texcoords.resize(accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                mesh.texcoords[i] = glm::vec2(
                    texcoords[i * 2 + 0],
                    texcoords[i * 2 + 1]
                );
            }
        }
        
        // Get indices
        if (primitive.indices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            
            mesh.indices.resize(accessor.count);
            
            if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const uint16_t* indices = reinterpret_cast<const uint16_t*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    mesh.indices[i] = static_cast<unsigned int>(indices[i]);
                }
            } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const uint32_t* indices = reinterpret_cast<const uint32_t*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    mesh.indices[i] = indices[i];
                }
            } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                const uint8_t* indices = reinterpret_cast<const uint8_t*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    mesh.indices[i] = static_cast<unsigned int>(indices[i]);
                }
            }
        }
        
        // Calculate normals if not provided
        if (mesh.normals.empty() && !mesh.vertices.empty() && !mesh.indices.empty()) {
            calculateNormals(mesh);
        }
        
        // Generate basic texture coordinates if not provided
        if (mesh.texcoords.empty() && !mesh.vertices.empty()) {
            mesh.texcoords.resize(mesh.vertices.size());
            for (size_t i = 0; i < mesh.texcoords.size(); ++i) {
                mesh.texcoords[i] = glm::vec2(0.0f, 0.0f);
            }
        }
        
        mesh.materialID = 0; // Default material (will be assigned later)
        m_meshes.push_back(mesh);
    }
}

glm::mat4 GLTFLoader::getNodeTransform(const tinygltf::Node& node) {
    glm::mat4 transform(1.0f);
    
    if (!node.matrix.empty()) {
        // Use matrix directly if provided
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                transform[i][j] = static_cast<float>(node.matrix[i * 4 + j]);
            }
        }
    } else {
        // Build matrix from TRS components
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);
        
        if (!node.translation.empty()) {
            translation = glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2])
            );
        }
        
        if (!node.rotation.empty()) {
            rotation = glm::quat(
                static_cast<float>(node.rotation[3]), // w
                static_cast<float>(node.rotation[0]), // x
                static_cast<float>(node.rotation[1]), // y
                static_cast<float>(node.rotation[2])  // z
            );
        }
        
        if (!node.scale.empty()) {
            scale = glm::vec3(
                static_cast<float>(node.scale[0]),
                static_cast<float>(node.scale[1]),
                static_cast<float>(node.scale[2])
            );
        }
        
        // Build transform matrix: T * R * S
        transform = glm::translate(glm::mat4(1.0f), translation);
        transform = transform * glm::mat4_cast(rotation);
        transform = glm::scale(transform, scale);
    }
    
    return transform;
}