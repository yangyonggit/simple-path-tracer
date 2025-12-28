#include "backends/EmbreeBackend.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>
#include <cmath>

namespace backends {

EmbreeBackend::EmbreeBackend() 
    : device_(nullptr), scene_(nullptr) {
}

EmbreeBackend::~EmbreeBackend() {
    destroy();
}

bool EmbreeBackend::build(const scene::SceneDesc& sceneDesc) {
    // Clean up any previous build
    destroy();
    
    // Create Embree device
    device_ = rtcNewDevice(nullptr);
    if (!device_) {
        std::cerr << "EmbreeBackend: Failed to create RTCDevice\n";
        return false;
    }
    
    // Create scene
    scene_ = rtcNewScene(device_);
    if (!scene_) {
        std::cerr << "EmbreeBackend: Failed to create RTCScene\n";
        rtcReleaseDevice(device_);
        device_ = nullptr;
        return false;
    }
    
    // Clear previous material mapping
    geomMaterialId_.clear();
    
    // Process each instance
    for (const auto& instance : sceneDesc.instances) {
        // Get the mesh data
        if (instance.meshId >= sceneDesc.meshes.size()) {
            std::cerr << "EmbreeBackend: Invalid mesh ID " << instance.meshId << "\n";
            continue;
        }
        
        const auto& meshData = sceneDesc.meshes[instance.meshId];
        
        // Determine material ID
        uint32_t materialId = instance.materialId;
        if (materialId == UINT32_MAX && instance.meshId < sceneDesc.meshes.size()) {
            materialId = sceneDesc.meshes[instance.meshId].materialId;
        }
        if (materialId == UINT32_MAX) {
            materialId = 0; // Default to material 0
        }
        
        // Transform mesh positions and normals
        std::vector<glm::vec3> transformedPositions = meshData.positions;
        std::vector<glm::vec3> transformedNormals = meshData.normals;
        
        // Apply instance transform to positions
        for (auto& pos : transformedPositions) {
            glm::vec4 pos4(pos, 1.0f);
            glm::vec4 transformedPos = instance.worldFromObject * pos4;
            pos = glm::vec3(transformedPos);
        }
        
        // Apply inverse-transpose of 3x3 to normals (if they exist)
        if (!transformedNormals.empty()) {
            glm::mat3 normalMat = glm::mat3(instance.worldFromObject);
            glm::mat3 normalMatInvTrans = glm::transpose(glm::inverse(normalMat));
            
            for (auto& normal : transformedNormals) {
                normal = glm::normalize(normalMatInvTrans * normal);
            }
        }
        
        // Create triangle geometry
        RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (!geometry) {
            std::cerr << "EmbreeBackend: Failed to create geometry for instance " 
                      << instance.meshId << "\n";
            continue;
        }
        
        // Set vertex buffer
        float* vertices = (float*)rtcSetNewGeometryBuffer(geometry,
            RTC_BUFFER_TYPE_VERTEX, 
            0,
            RTC_FORMAT_FLOAT3,
            3 * sizeof(float),
            transformedPositions.size());
        
        if (!vertices) {
            std::cerr << "EmbreeBackend: Failed to allocate vertex buffer\n";
            rtcReleaseGeometry(geometry);
            continue;
        }
        
        // Copy transformed positions to vertex buffer
        std::memcpy(vertices, transformedPositions.data(), 
                   transformedPositions.size() * sizeof(glm::vec3));
        
        // Set index buffer
        unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
            RTC_BUFFER_TYPE_INDEX,
            0,
            RTC_FORMAT_UINT3,
            3 * sizeof(unsigned),
            meshData.indices.size());
        
        if (!indices) {
            std::cerr << "EmbreeBackend: Failed to allocate index buffer\n";
            rtcReleaseGeometry(geometry);
            continue;
        }
        
        // Copy indices to index buffer
        std::memcpy(indices, meshData.indices.data(),
                   meshData.indices.size() * sizeof(glm::uvec3));
        
        // Commit geometry
        rtcCommitGeometry(geometry);
        
        // Attach to scene and get geomID
        unsigned int geomID = rtcAttachGeometry(scene_, geometry);
        
        // Record material mapping
        geomMaterialId_.push_back(materialId);
        
        // Release geometry reference (scene holds a reference)
        rtcReleaseGeometry(geometry);
    }
    
    // Process spheres (user-defined geometry)
    for (size_t sphereIdx = 0; sphereIdx < sceneDesc.spheres.size(); ++sphereIdx) {
        const auto& sphere = sceneDesc.spheres[sphereIdx];
        
        RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_USER);
        if (!geometry) {
            std::cerr << "EmbreeBackend: Failed to create sphere geometry\n";
            continue;
        }
        
        // Create sphere data and store it
        sphereData_.emplace_back();
        SphereUserData& sphereData = sphereData_.back();
        sphereData.center_x = sphere.center.x;
        sphereData.center_y = sphere.center.y;
        sphereData.center_z = sphere.center.z;
        sphereData.radius = sphere.radius;
        sphereData.materialId = sphere.materialId;
        
        // Set user data to point to sphere data
        // Use the stable address of the element we just added
        rtcSetGeometryUserData(geometry, &sphereData_.back());
        
        // Set callback functions
        rtcSetGeometryUserPrimitiveCount(geometry, 1);
        rtcSetGeometryIntersectFunction(geometry, sphereIntersectFunc);
        rtcSetGeometryOccludedFunction(geometry, sphereOccludedFunc);
        rtcSetGeometryBoundsFunction(geometry, sphereBoundsFunc, nullptr);
        
        // Commit geometry
        rtcCommitGeometry(geometry);
        
        // Attach to scene and get geomID
        unsigned int geomID = rtcAttachGeometry(scene_, geometry);
        
        // Record material mapping
        geomMaterialId_.push_back(sphere.materialId);
        
        // Release geometry reference
        rtcReleaseGeometry(geometry);
    }
    
    // Commit scene
    rtcCommitScene(scene_);
    
    std::cout << "EmbreeBackend: Material mapping (geomID -> materialId):\n";
    for (size_t i = 0; i < geomMaterialId_.size(); ++i) {
        std::cout << "  geomID " << i << " -> material " << geomMaterialId_[i] << "\n";
    }
    
    std::cout << "EmbreeBackend: Built scene with " << (geomMaterialId_.size() - sceneDesc.spheres.size()) 
              << " mesh geometries, " << sceneDesc.spheres.size() << " analytical spheres, "
              << sceneDesc.materials.size() << " materials\n";
    
    return true;
}

void EmbreeBackend::destroy() {
    if (scene_) {
        rtcReleaseScene(scene_);
        scene_ = nullptr;
    }
    
    if (device_) {
        rtcReleaseDevice(device_);
        device_ = nullptr;
    }
    
    geomMaterialId_.clear();
}

uint32_t EmbreeBackend::getGeomMaterialId(uint32_t geomID) const {
    if (geomID < geomMaterialId_.size()) {
        return geomMaterialId_[geomID];
    }
    
    // Invalid geomID, return default material 0
    return 0;
}

bool EmbreeBackend::isValidGeomId(uint32_t geomID) const {
    return geomID < geomMaterialId_.size();
}

// Static callback for sphere intersection
void EmbreeBackend::sphereIntersectFunc(const RTCIntersectFunctionNArguments* args) {
    int* valid = args->valid;
    void* ptr = args->geometryUserPtr;
    RTCRayHit* rayhit = (RTCRayHit*)args->rayhit;
    unsigned int primID = args->primID;
    
    if (!valid || !valid[0] || !ptr || !rayhit) return;
    
    const SphereUserData* sphere = (const SphereUserData*)ptr;
    RTCRay* ray = &rayhit->ray;
    RTCHit* hit = &rayhit->hit;
    
    // Ray-sphere intersection
    float oc_x = ray->org_x - sphere->center_x;
    float oc_y = ray->org_y - sphere->center_y;
    float oc_z = ray->org_z - sphere->center_z;
    
    float a = ray->dir_x * ray->dir_x + ray->dir_y * ray->dir_y + ray->dir_z * ray->dir_z;
    float b = 2.0f * (oc_x * ray->dir_x + oc_y * ray->dir_y + oc_z * ray->dir_z);
    float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - sphere->radius * sphere->radius;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant >= 0.0f) {
        float sqrt_disc = std::sqrt(discriminant);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);
        
        float t = -1.0f;
        if (t1 > ray->tnear && t1 < ray->tfar) {
            t = t1;
        } else if (t2 > ray->tnear && t2 < ray->tfar) {
            t = t2;
        }
        
        if (t > 0.0f && t < ray->tfar) {
            ray->tfar = t;
            hit->primID = primID;
            hit->geomID = args->geomID;
            
            static int logged = 0;
            if (logged < 5) {
                std::cout << "[sphereIntersect] Setting geomID=" << args->geomID 
                          << ", center=(" << sphere->center_x << "," << sphere->center_y 
                          << "," << sphere->center_z << ")\n";
                logged++;
            }
            
            // Calculate hit point and normal
            float hit_x = ray->org_x + t * ray->dir_x;
            float hit_y = ray->org_y + t * ray->dir_y;
            float hit_z = ray->org_z + t * ray->dir_z;
            
            // Normal = (hit - center) / radius
            hit->Ng_x = (hit_x - sphere->center_x) / sphere->radius;
            hit->Ng_y = (hit_y - sphere->center_y) / sphere->radius;
            hit->Ng_z = (hit_z - sphere->center_z) / sphere->radius;
        }
    }
}

void EmbreeBackend::sphereOccludedFunc(const RTCOccludedFunctionNArguments* args) {
    const int* valid = args->valid;
    void* ptr = args->geometryUserPtr;
    RTCRay* ray = (RTCRay*)args->ray;
    unsigned int primID = args->primID;
    
    if (!valid || !valid[0] || !ptr || !ray) return;
    
    const SphereUserData* sphere = (const SphereUserData*)ptr;
    
    // Ray-sphere intersection for occlusion test
    float oc_x = ray->org_x - sphere->center_x;
    float oc_y = ray->org_y - sphere->center_y;
    float oc_z = ray->org_z - sphere->center_z;
    
    float a = ray->dir_x * ray->dir_x + ray->dir_y * ray->dir_y + ray->dir_z * ray->dir_z;
    float b = 2.0f * (oc_x * ray->dir_x + oc_y * ray->dir_y + oc_z * ray->dir_z);
    float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - sphere->radius * sphere->radius;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant >= 0.0f) {
        float sqrt_disc = std::sqrt(discriminant);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);
        
        if ((t1 > ray->tnear && t1 < ray->tfar) || (t2 > ray->tnear && t2 < ray->tfar)) {
            ray->tfar = -1.0f;
        }
    }
}

void EmbreeBackend::sphereBoundsFunc(const RTCBoundsFunctionArguments* args) {
    if (!args || !args->geometryUserPtr || !args->bounds_o) return;
    
    const SphereUserData* sphere = (const SphereUserData*)args->geometryUserPtr;
    RTCBounds* bounds = args->bounds_o;
    
    bounds->lower_x = sphere->center_x - sphere->radius;
    bounds->lower_y = sphere->center_y - sphere->radius;
    bounds->lower_z = sphere->center_z - sphere->radius;
    bounds->upper_x = sphere->center_x + sphere->radius;
    bounds->upper_y = sphere->center_y + sphere->radius;
    bounds->upper_z = sphere->center_z + sphere->radius;
}

} // namespace backends
