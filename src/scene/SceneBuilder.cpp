#include "scene/SceneBuilder.h"
#include <glm/gtc/matrix_transform.hpp>

namespace scene {

// ========================================
// Build default scene
// ========================================
SceneDesc BuildDefaultScene() {
    SceneDesc scene;
    
    // ========================================
    // 1. Define Materials (matching EmbreeScene material IDs)
    // ========================================
    
    Material mat;
    
    // Material 0: Gold (metallic, warm color)
    mat = Material(glm::vec3(1.0f, 0.71f, 0.29f), glm::vec3(0.0f));
    mat.metallic = 1.0f;
    mat.roughness = 0.05f;
    mat.ior = 1.5f;
    scene.addMaterial(mat);
    
    // Material 1: Silver (metallic, cool color)
    mat = Material(glm::vec3(0.95f, 0.93f, 0.88f), glm::vec3(0.0f));
    mat.metallic = 1.0f;
    mat.roughness = 0.02f;
    mat.ior = 1.5f;
    scene.addMaterial(mat);
    
    // Material 2: Copper (metallic, orange-red)
    mat = Material(glm::vec3(0.95f, 0.64f, 0.54f), glm::vec3(0.0f));
    mat.metallic = 1.0f;
    mat.roughness = 0.08f;
    mat.ior = 1.5f;
    scene.addMaterial(mat);
    
    // Material 3: Iron (metallic, dark gray)
    mat = Material(glm::vec3(0.56f, 0.57f, 0.58f), glm::vec3(0.0f));
    mat.metallic = 1.0f;
    mat.roughness = 0.3f;
    mat.ior = 1.5f;
    scene.addMaterial(mat);
    
    // Material 4: Glass (transparent dielectric)
    mat = Material(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f));
    mat.metallic = 0.0f;
    mat.roughness = 0.0f;
    mat.ior = 1.5f;
    mat.transparency = 0.95f;
    scene.addMaterial(mat);
    
    // Material 5: Plastic (smooth dielectric, red)
    mat = Material(glm::vec3(0.8f, 0.2f, 0.2f), glm::vec3(0.0f));
    mat.metallic = 0.0f;
    mat.roughness = 0.4f;
    mat.ior = 1.2f;
    mat.transparency = 0.0f;
    scene.addMaterial(mat);
    
    // Material 6: Rubber (rough dielectric, dark gray)
    mat = Material(glm::vec3(0.3f, 0.3f, 0.3f), glm::vec3(0.0f));
    mat.metallic = 0.0f;
    mat.roughness = 0.8f;
    mat.ior = 1.1f;
    mat.transparency = 0.0f;
    scene.addMaterial(mat);
    
    // Material 7: Wood (diffuse, brown)
    mat = Material(glm::vec3(0.4f, 0.25f, 0.1f), glm::vec3(0.0f));
    mat.metallic = 0.0f;
    mat.roughness = 0.7f;
    mat.ior = 1.0f;
    mat.transparency = 0.0f;
    scene.addMaterial(mat);
    
    // Material 8: Concrete (rough diffuse, gray)
    mat = Material(glm::vec3(0.6f, 0.6f, 0.6f), glm::vec3(0.0f));
    mat.metallic = 0.0f;
    mat.roughness = 0.9f;
    mat.ior = 1.0f;
    mat.transparency = 0.0f;
    scene.addMaterial(mat);
    
    // ========================================
    // 2. Create Meshes
    // ========================================
    
    // Mesh 0: Unit cube (1x1x1, centered at origin)
    uint32_t cubeMeshId = scene.addMesh(createCubeMesh(0));
    
    // ========================================
    // 3. Add Spheres (analytical - will be user geometry in Embree)
    // ========================================
    
    // Metal spheres - front row (radius 1.0)
    scene.addSphere(glm::vec3(-3.0f, 1.0f, 0.0f), 1.0f, 0);  // Gold
    scene.addSphere(glm::vec3(-1.0f, 1.0f, 0.0f), 1.0f, 1);  // Silver
    scene.addSphere(glm::vec3(1.0f, 1.0f, 0.0f), 1.0f, 2);   // Copper
    scene.addSphere(glm::vec3(3.0f, 1.0f, 0.0f), 1.0f, 3);   // Iron
    
    // Dielectric spheres - back row (radius 1.0)
    scene.addSphere(glm::vec3(-2.0f, 1.0f, -2.0f), 1.0f, 5); // Plastic
    scene.addSphere(glm::vec3(0.0f, 1.0f, -2.0f), 1.0f, 6);  // Rubber
    
    // Mixed material spheres - back row right (radius 1.0)
    scene.addSphere(glm::vec3(2.0f, 1.0f, -2.0f), 1.0f, 7);  // Wood
    scene.addSphere(glm::vec3(0.0f, 1.0f, -4.0f), 1.0f, 8);  // Concrete
    
    // ========================================
    // 4. Create Cube Instances
    // ========================================
    
    // Glass cube - middle position (size 1.5)
    glm::mat4 glassCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 2.0f));
    glassCubeTransform = glm::scale(glassCubeTransform, glm::vec3(1.5f));
    scene.addInstance(cubeMeshId, glassCubeTransform, 4);
    
    return scene;
}

// ========================================
// Build minimal triangle test scene
// ========================================
SceneDesc BuildTestTriangleScene() {
    SceneDesc scene;

    // 1) Single material (bright red)
    Material red(glm::vec3(1.0f, 0.0f, 0.0f));
    scene.addMaterial(red);

    // 2) Single triangle mesh
    MeshData tri;
    tri.materialId = 0;
    tri.positions = {
        glm::vec3(-1.0f, 0.0f, -3.0f),
        glm::vec3( 1.0f, 0.0f, -3.0f),
        glm::vec3( 0.0f, 1.0f, -3.0f)
    };
    tri.indices = {
        glm::uvec3(0, 1, 2)
    };
    const uint32_t triMeshId = scene.addMesh(tri);

    // 3) Two instances for transform validation
    //    a) identity
    scene.addInstance(triMeshId, glm::mat4(1.0f), 0);
    //    b) translate(1.2, 0, 0) + scale(0.5)
    glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(1.2f, 0.0f, 0.0f));
    t = glm::scale(t, glm::vec3(0.5f));
    scene.addInstance(triMeshId, t, 0);

    // 4) Single analytical sphere for OptiX sphere GAS validation
    // NOTE: OptiX backend colors spheres green via SBT geomType, independent of material.
    scene.addSphere(glm::vec3(0.0f, -0.5f, -3.0f), 0.5f, 0);

    return scene;
}

} // namespace scene