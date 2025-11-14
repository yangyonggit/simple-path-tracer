#include "EmbreeScene.h"
#include "GLTFLoader.h"
#include "MeshIntegrator.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

EmbreeScene::EmbreeScene() : m_device(nullptr), m_scene(nullptr) {
    initialize();
}

EmbreeScene::~EmbreeScene() {
    cleanup();
}

void EmbreeScene::initialize() {
    // Create Embree device
    m_device = rtcNewDevice(nullptr);
    if (!m_device) {
        std::cerr << "Failed to create Embree device!\n";
        return;
    }
    
    // Create scene
    m_scene = rtcNewScene(m_device);
    if (!m_scene) {
        std::cerr << "Failed to create Embree scene!\n";
        rtcReleaseDevice(m_device);
        m_device = nullptr;
        return;
    }
    
    // Initialize GLTF components after device and scene are created
    m_gltfLoader = std::make_unique<GLTFLoader>();
    m_meshIntegrator = std::make_unique<MeshIntegrator>(m_device, m_scene);
    std::cout << "GLTF components initialized successfully.\n";
    
    // Reserve space for all geometry to ensure no reallocation
    m_spheres.reserve(8);  // 8 spheres + 1 cube
    
    // Metal spheres - front row
    addSphereWithMaterial(0, glm::vec3(-3.0f, 1.0f, 0.0f), 1.0f);  // Gold
    addSphereWithMaterial(1, glm::vec3(-1.0f, 1.0f, 0.0f), 1.0f);  // Silver  
    addSphereWithMaterial(2, glm::vec3(1.0f, 1.0f, 0.0f), 1.0f);   // Copper
    addSphereWithMaterial(3, glm::vec3(3.0f, 1.0f, 0.0f), 1.0f);   // Iron
    
    // Glass cube - middle position
    addCubeWithMaterial(4, glm::vec3(0.0f, 1.0f, 2.0f), 1.5f);    // Glass Cube
    
    // Dielectric spheres - back row
    addSphereWithMaterial(5, glm::vec3(-2.0f, 1.0f, -2.0f), 1.0f); // Plastic
    addSphereWithMaterial(6, glm::vec3(0.0f, 1.0f, -2.0f), 1.0f);  // Rubber
    
    // Mixed material spheres - back row right
    addSphereWithMaterial(7, glm::vec3(2.0f, 1.0f, -2.0f), 1.0f);  // Wood
    addSphereWithMaterial(8, glm::vec3(0.0f, 1.0f, -4.0f), 1.0f);  // Concrete
    
    // Commit scene after basic geometry to update BVH first
    rtcCommitScene(m_scene);
    
    std::cout << "EmbreeScene initialized successfully with 8 spheres and 1 glass cube (no ground plane).\n";
    
    // Add GLTF model after all basic geometry is created and committed
    if (m_gltfLoader && m_meshIntegrator) {
        // Create a transform matrix to position the GLTF model
        glm::mat4 gltfTransform = glm::mat4(1.0f);
        gltfTransform = glm::translate(gltfTransform, glm::vec3(5.0f, 0.5f, 0.0f)); // Lower position
        gltfTransform = glm::scale(gltfTransform, glm::vec3(5.0f)); // Scale to 5x size
        
        // Load GLTF file with real parsing, use copper material (material ID 2) - very visible metal
        if (loadGLTFWithTransform("assets/models/rattan_dining_chair/scene.gltf", gltfTransform, 2)) { // Use copper material (material ID 2)
            std::cout << "Added GLTF model at position (5, 0.5, 0) with copper material and 5x scale.\n";
        } else {
            std::cout << "Note: GLTF model loading failed.\n";
        }
    }
}

void EmbreeScene::addGroundPlane() {
    // Create a quad (ground plane) using two triangles
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Define quad vertices (4 vertices for ground plane)
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 4);
    
    // Ground plane vertices (Y = -1.0, large quad)
    vertices[0] = -3.0f; vertices[1] = -1.0f; vertices[2] = -3.0f;  // vertex 0
    vertices[3] = 3.0f;  vertices[4] = -1.0f; vertices[5] = -3.0f;  // vertex 1
    vertices[6] = 3.0f;  vertices[7] = -1.0f; vertices[8] = 3.0f;   // vertex 2
    vertices[9] = -3.0f; vertices[10] = -1.0f; vertices[11] = 3.0f; // vertex 3

    // Define triangle indices (2 triangles forming a quad)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 2);
    
    // Triangle 1: vertices 0, 2, 1 (counter-clockwise from above for upward normal)
    indices[0] = 0; indices[1] = 2; indices[2] = 1;
    // Triangle 2: vertices 0, 3, 2 (counter-clockwise from above for upward normal)
    indices[3] = 0; indices[4] = 3; indices[5] = 2;

    // Commit geometry and attach to scene
    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::addTestBox() {
    // Create a smaller box positioned to the left
    createCube(-1.0f, -1.0f + 0.15f + 0.1f, 0.0f, 0.3f, 0.15f, 0.3f);
}

void EmbreeScene::addCube() {
    // Create a larger cube positioned to the right
    createCube(1.0f, -1.0f + 0.4f + 0.1f, 0.0f, 0.4f, 0.4f, 0.4f);
}

void EmbreeScene::createCube(float pos_x, float pos_y, float pos_z, float size_x, float size_y, float size_z) {
    // Create a cube using 12 triangles (6 faces * 2 triangles each)
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Define cube vertices (8 vertices for a cube)
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 8);
    
    // Calculate half sizes for easier vertex positioning
    float half_x = size_x;
    float half_y = size_y;
    float half_z = size_z;
    
    // Bottom face vertices
    vertices[0] = pos_x - half_x; vertices[1] = pos_y; vertices[2] = pos_z - half_z;  // 0
    vertices[3] = pos_x + half_x; vertices[4] = pos_y; vertices[5] = pos_z - half_z;  // 1
    vertices[6] = pos_x + half_x; vertices[7] = pos_y; vertices[8] = pos_z + half_z;   // 2
    vertices[9] = pos_x - half_x; vertices[10] = pos_y; vertices[11] = pos_z + half_z; // 3
    
    // Top face vertices
    vertices[12] = pos_x - half_x; vertices[13] = pos_y + half_y * 2; vertices[14] = pos_z - half_z; // 4
    vertices[15] = pos_x + half_x; vertices[16] = pos_y + half_y * 2; vertices[17] = pos_z - half_z; // 5
    vertices[18] = pos_x + half_x; vertices[19] = pos_y + half_y * 2; vertices[20] = pos_z + half_z;  // 6
    vertices[21] = pos_x - half_x; vertices[22] = pos_y + half_y * 2; vertices[23] = pos_z + half_z;  // 7

    // Define triangle indices (12 triangles for 6 faces)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 12);
    
    int idx = 0;
    // Bottom face
    indices[idx++] = 0; indices[idx++] = 2; indices[idx++] = 1;
    indices[idx++] = 0; indices[idx++] = 3; indices[idx++] = 2;
    // Top face
    indices[idx++] = 4; indices[idx++] = 5; indices[idx++] = 6;
    indices[idx++] = 4; indices[idx++] = 6; indices[idx++] = 7;
    // Front face
    indices[idx++] = 0; indices[idx++] = 1; indices[idx++] = 5;
    indices[idx++] = 0; indices[idx++] = 5; indices[idx++] = 4;
    // Back face
    indices[idx++] = 2; indices[idx++] = 3; indices[idx++] = 7;
    indices[idx++] = 2; indices[idx++] = 7; indices[idx++] = 6;
    // Left face
    indices[idx++] = 3; indices[idx++] = 0; indices[idx++] = 4;
    indices[idx++] = 3; indices[idx++] = 4; indices[idx++] = 7;
    // Right face
    indices[idx++] = 1; indices[idx++] = 2; indices[idx++] = 6;
    indices[idx++] = 1; indices[idx++] = 6; indices[idx++] = 5;

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::addSphere() {
    // Legacy function - use addSphereWithMaterial instead
    addSphereWithMaterial(0, glm::vec3(0.0f, 1.0f, 0.0f), 0.5f);
}

void EmbreeScene::addSphereWithMaterial(unsigned int materialID, const glm::vec3& position, float radius) {
    // Create a user-defined geometry for analytical sphere
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_USER);
    
    // Create sphere data using emplace_back for better performance
    m_spheres.emplace_back(position.x, position.y, position.z, radius, materialID);
    
    // Use pointer to element in vector (safe since we reserve space and don't resize)
    rtcSetGeometryUserData(geometry, &m_spheres.back());
    
    // Set callback functions
    rtcSetGeometryUserPrimitiveCount(geometry, 1);
    rtcSetGeometryIntersectFunction(geometry, sphereIntersectFunc);
    rtcSetGeometryOccludedFunction(geometry, sphereOccludedFunc);
    rtcSetGeometryBoundsFunction(geometry, sphereBoundsFunc, nullptr);
    
    rtcCommitGeometry(geometry);
    
    // Attach geometry with specific material ID
    unsigned int geomID = rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
    
    std::cout << "Added sphere at (" << position.x << ", " << position.y << ", " << position.z 
              << ") with radius " << radius << " and material ID " << materialID 
              << " (Embree geomID: " << geomID << ")" << std::endl;
}

void EmbreeScene::cleanup() {
    // Clear our bookkeeping vector
    m_spheres.clear();
    
    // Embree will clean up geometry user data when scene is released
    if (m_scene) {
        rtcReleaseScene(m_scene);
        m_scene = nullptr;
    }
    if (m_device) {
        rtcReleaseDevice(m_device);
        m_device = nullptr;
    }
}

// User-defined geometry callbacks for analytical sphere intersection
void EmbreeScene::sphereIntersectFunc(const RTCIntersectFunctionNArguments* args) {
    int* valid = args->valid;
    void* ptr = args->geometryUserPtr;
    RTCRayHit* rayhit = (RTCRayHit*)args->rayhit;
    unsigned int primID = args->primID;
    
    assert(args->N == 1);
    
    // Add safety checks for Release mode
    if (!valid || !valid[0] || !ptr || !rayhit) return;
    
    // Get sphere data by pointer
    const SphereData* sphere = (const SphereData*)ptr;
    
    // Ray-sphere intersection using quadratic formula
    // Ray: P = O + t*D
    // Sphere: |P - C|² = r²
    // Substituting: |O + t*D - C|² = r²
    
    RTCRay* ray = &rayhit->ray;
    RTCHit* hit = &rayhit->hit;
    
    // Vector from ray origin to sphere center
    float oc_x = ray->org_x - sphere->center_x;
    float oc_y = ray->org_y - sphere->center_y;
    float oc_z = ray->org_z - sphere->center_z;
    
    // Quadratic equation coefficients: a*t² + b*t + c = 0
    float a = ray->dir_x * ray->dir_x + ray->dir_y * ray->dir_y + ray->dir_z * ray->dir_z;
    float b = 2.0f * (oc_x * ray->dir_x + oc_y * ray->dir_y + oc_z * ray->dir_z);
    float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - sphere->radius * sphere->radius;
    
    // Discriminant
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0.0f) return; // No intersection
    
    // Calculate the two possible intersection distances
    float sqrt_discriminant = std::sqrt(discriminant);
    float t1 = (-b - sqrt_discriminant) / (2.0f * a);
    float t2 = (-b + sqrt_discriminant) / (2.0f * a);
    
    // Choose the nearest intersection within valid range
    float t = -1.0f;
    if (t1 > ray->tnear && t1 < ray->tfar) {
        t = t1;
    } else if (t2 > ray->tnear && t2 < ray->tfar) {
        t = t2;
    }
    
    if (t > 0.0f && t < ray->tfar) {
        // Update hit information
        ray->tfar = t;
        hit->primID = primID;
        hit->geomID = args->geomID;
        
        // Calculate hit point
        float hit_x = ray->org_x + t * ray->dir_x;
        float hit_y = ray->org_y + t * ray->dir_y;
        float hit_z = ray->org_z + t * ray->dir_z;
        
        // Calculate normal (hit point - sphere center) / radius
        hit->Ng_x = (hit_x - sphere->center_x) / sphere->radius;
        hit->Ng_y = (hit_y - sphere->center_y) / sphere->radius;
        hit->Ng_z = (hit_z - sphere->center_z) / sphere->radius;
        
        // Set UV coordinates (simple spherical mapping)
        float nx = hit->Ng_x;
        float ny = hit->Ng_y;
        float nz = hit->Ng_z;
        
        hit->u = 0.5f + std::atan2(nz, nx) / (2.0f * M_PI);
        hit->v = 0.5f - std::asin(ny) / M_PI;
    }
}

void EmbreeScene::sphereOccludedFunc(const RTCOccludedFunctionNArguments* args) {
    const int* valid = args->valid;
    void* ptr = args->geometryUserPtr;
    RTCRay* ray = (RTCRay*)args->ray;
    unsigned int primID = args->primID;
    
    assert(args->N == 1);
    
    // Add safety checks for Release mode
    if (!valid || !valid[0] || !ptr || !ray) return;
    
    const SphereData* sphere = (const SphereData*)ptr;
    
    // Ray-sphere intersection logic for occlusion test
    float oc_x = ray->org_x - sphere->center_x;
    float oc_y = ray->org_y - sphere->center_y;
    float oc_z = ray->org_z - sphere->center_z;
    
    float a = ray->dir_x * ray->dir_x + ray->dir_y * ray->dir_y + ray->dir_z * ray->dir_z;
    float b = 2.0f * (oc_x * ray->dir_x + oc_y * ray->dir_y + oc_z * ray->dir_z);
    float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - sphere->radius * sphere->radius;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant >= 0.0f) {
        float sqrt_discriminant = std::sqrt(discriminant);
        float t1 = (-b - sqrt_discriminant) / (2.0f * a);
        float t2 = (-b + sqrt_discriminant) / (2.0f * a);
        
        // Find the closest valid intersection
        float t = -1.0f;
        if (t1 > ray->tnear && t1 < ray->tfar) {
            t = t1;
        } else if (t2 > ray->tnear && t2 < ray->tfar) {
            t = t2;
        }
        
        if (t > 0.0f) {
            // Set tfar to indicate occlusion (Embree convention)
            ray->tfar = -1.0f;
        }
    }
}

void EmbreeScene::sphereBoundsFunc(const RTCBoundsFunctionArguments* args) {
    // Add safety checks for Release mode
    if (!args || !args->geometryUserPtr || !args->bounds_o) return;
    
    const SphereData* sphere = (const SphereData*)args->geometryUserPtr;
    RTCBounds* bounds_o = args->bounds_o;
    
    // Set bounding box for the sphere
    bounds_o->lower_x = sphere->center_x - sphere->radius;
    bounds_o->lower_y = sphere->center_y - sphere->radius;
    bounds_o->lower_z = sphere->center_z - sphere->radius;
    bounds_o->upper_x = sphere->center_x + sphere->radius;
    bounds_o->upper_y = sphere->center_y + sphere->radius;
    bounds_o->upper_z = sphere->center_z + sphere->radius;
}

void EmbreeScene::addCubeWithMaterial(unsigned int materialID, const glm::vec3& position, float size) {
    // Create a cube using 12 triangles (6 faces * 2 triangles each)
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Set the material ID for the geometry
    rtcSetGeometryUserData(geometry, (void*)(uintptr_t)materialID);
    
    // Define cube vertices (8 vertices for a cube)
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 8);
    
    // Calculate half size for easier vertex positioning
    float half = size * 0.5f;
    
    // Bottom face vertices (y = position.y - half)
    vertices[0] = position.x - half; vertices[1] = position.y - half; vertices[2] = position.z - half;  // 0
    vertices[3] = position.x + half; vertices[4] = position.y - half; vertices[5] = position.z - half;  // 1
    vertices[6] = position.x + half; vertices[7] = position.y - half; vertices[8] = position.z + half;   // 2
    vertices[9] = position.x - half; vertices[10] = position.y - half; vertices[11] = position.z + half; // 3
    
    // Top face vertices (y = position.y + half)
    vertices[12] = position.x - half; vertices[13] = position.y + half; vertices[14] = position.z - half; // 4
    vertices[15] = position.x + half; vertices[16] = position.y + half; vertices[17] = position.z - half; // 5
    vertices[18] = position.x + half; vertices[19] = position.y + half; vertices[20] = position.z + half;  // 6
    vertices[21] = position.x - half; vertices[22] = position.y + half; vertices[23] = position.z + half;  // 7

    // Define triangle indices (12 triangles for 6 faces)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 12);
    
    int idx = 0;
    // Bottom face (y = position.y - half)
    indices[idx++] = 0; indices[idx++] = 2; indices[idx++] = 1;
    indices[idx++] = 0; indices[idx++] = 3; indices[idx++] = 2;
    // Top face (y = position.y + half)
    indices[idx++] = 4; indices[idx++] = 5; indices[idx++] = 6;
    indices[idx++] = 4; indices[idx++] = 6; indices[idx++] = 7;
    // Front face (z = position.z - half)
    indices[idx++] = 0; indices[idx++] = 1; indices[idx++] = 5;
    indices[idx++] = 0; indices[idx++] = 5; indices[idx++] = 4;
    // Back face (z = position.z + half)
    indices[idx++] = 2; indices[idx++] = 3; indices[idx++] = 7;
    indices[idx++] = 2; indices[idx++] = 7; indices[idx++] = 6;
    // Left face (x = position.x - half)
    indices[idx++] = 3; indices[idx++] = 0; indices[idx++] = 4;
    indices[idx++] = 3; indices[idx++] = 4; indices[idx++] = 7;
    // Right face (x = position.x + half)
    indices[idx++] = 1; indices[idx++] = 2; indices[idx++] = 6;
    indices[idx++] = 1; indices[idx++] = 6; indices[idx++] = 5;

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

bool EmbreeScene::loadGLTF(const std::string& filepath, unsigned int materialID) {
    return loadGLTFWithTransform(filepath, glm::mat4(1.0f), materialID);
}

bool EmbreeScene::loadGLTFWithTransform(const std::string& filepath, const glm::mat4& transform, unsigned int materialID) {
    if (!m_gltfLoader || !m_meshIntegrator) {
        std::cerr << "GLTF components not initialized" << std::endl;
        return false;
    }
    
    // Load GLTF file
    if (!m_gltfLoader->loadGLTF(filepath)) {
        std::cerr << "Failed to load GLTF file: " << filepath << std::endl;
        return false;
    }
    
    // Apply transform and material
    m_gltfLoader->transformMeshes(transform);
    m_gltfLoader->assignMaterial(materialID);
    
    // Add meshes to scene
    if (!m_meshIntegrator->addAllMeshes(*m_gltfLoader)) {
        std::cerr << "Failed to add GLTF meshes to scene" << std::endl;
        return false;
    }
    
    // Commit scene to update BVH
    rtcCommitScene(m_scene);
    
    std::cout << "Successfully loaded GLTF file: " << filepath 
              << " with material ID: " << materialID << std::endl;
    
    return true;
}

unsigned int EmbreeScene::getGeometryMaterialID(unsigned int geomID) const {
    // For spheres, the geometry ID directly maps to material ID (first N geometries)
    if (geomID < m_spheres.size()) {
        return m_spheres[geomID].materialID;
    }
    
    // For meshes, use the mesh integrator mapping
    if (m_meshIntegrator) {
        return m_meshIntegrator->getMaterialID(geomID);
    }
    
    return 0; // Default material
}