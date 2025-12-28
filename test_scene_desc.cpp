// Test file to verify SceneDesc compiles correctly
#include "scene/SceneDesc.h"
#include <iostream>

int main() {
    using namespace scene;
    
    // Create a simple scene
    SceneDesc scene;
    
    // Add materials
    Material red(glm::vec3(1.0f, 0.0f, 0.0f));
    Material emissive(glm::vec3(0.0f), glm::vec3(1.0f, 1.0f, 0.5f));
    
    uint32_t matRed = scene.addMaterial(red);
    uint32_t matLight = scene.addMaterial(emissive);
    
    // Add meshes
    uint32_t cubeMeshId = scene.addMesh(createCubeMesh(matRed));
    uint32_t groundMeshId = scene.addMesh(createGroundPlaneMesh(20.0f, matRed));
    
    // Add instances
    glm::mat4 cubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    scene.addInstance(cubeMeshId, cubeTransform, matRed);
    
    scene.addInstance(groundMeshId);
    
    // Print statistics
    std::cout << "Scene created successfully!" << std::endl;
    std::cout << "Materials: " << scene.materials.size() << std::endl;
    std::cout << "Meshes: " << scene.meshes.size() << std::endl;
    std::cout << "Instances: " << scene.instances.size() << std::endl;
    std::cout << "Total vertices: " << scene.totalVertices() << std::endl;
    
    return 0;
}
