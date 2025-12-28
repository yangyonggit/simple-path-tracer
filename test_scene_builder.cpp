// Test file demonstrating SceneBuilder usage
// This file is not compiled - it's for documentation purposes

#include "scene/SceneBuilder.h"
#include <iostream>

int main() {
    // Build the default scene
    scene::SceneDesc scene = scene::BuildDefaultScene();
    
    // Print scene statistics
    std::cout << "Scene created successfully!\n";
    std::cout << "Materials: " << scene.materials.size() << "\n";
    std::cout << "Meshes: " << scene.meshes.size() << "\n";
    std::cout << "Instances: " << scene.instances.size() << "\n";
    std::cout << "Total triangles: " << scene.totalTriangles() << "\n";
    std::cout << "Total vertices: " << scene.totalVertices() << "\n";
    
    // Example: Iterate through materials
    std::cout << "\nMaterials:\n";
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto& mat = scene.materials[i];
        std::cout << "  Material " << i << ": "
                  << "color(" << mat.baseColor.r << ", " << mat.baseColor.g << ", " << mat.baseColor.b << ") "
                  << "metallic=" << mat.metallic << " "
                  << "roughness=" << mat.roughness;
        
        if (glm::length(mat.emission) > 0.0f) {
            std::cout << " [EMISSIVE]";
        }
        std::cout << "\n";
    }
    
    // Example: Iterate through instances
    std::cout << "\nInstances:\n";
    for (size_t i = 0; i < scene.instances.size(); ++i) {
        const auto& inst = scene.instances[i];
        std::cout << "  Instance " << i << ": "
                  << "mesh=" << inst.meshId << " "
                  << "material=" << inst.materialId << "\n";
    }
    
    return 0;
}
