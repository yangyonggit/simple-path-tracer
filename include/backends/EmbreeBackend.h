#pragma once

#include <embree4/rtcore.h>
#include "scene/SceneDesc.h"
#include <vector>
#include <deque>
#include <cstdint>

namespace backends {

// ========================================
// EmbreeBackend - Build RTCScene from SceneDesc
// ========================================
// ========================================
// SphereUserData - Data for user-defined sphere geometry
// ========================================
struct SphereUserData {
    float center_x, center_y, center_z;
    float radius;
    uint32_t materialId;
};

class EmbreeBackend {
public:
    EmbreeBackend();
    ~EmbreeBackend();
    
    // Build RTCScene from SceneDesc
    // Returns true on success
    bool build(const scene::SceneDesc& sceneDesc);
    
    // Release all Embree resources
    void destroy();
    
    // Get the built RTCScene (valid after build() succeeds)
    RTCScene getScene() const { return scene_; }
    
    // Get RTCDevice (valid after build() succeeds)
    RTCDevice getDevice() const { return device_; }
    
    // Query material ID for a given geometry ID
    uint32_t getGeomMaterialId(uint32_t geomID) const;
    
    // Check if a geometry ID is valid
    bool isValidGeomId(uint32_t geomID) const;
    
    // Get the entire geometry to material ID mapping
    const std::vector<uint32_t>& getGeomMaterialMapping() const { return geomMaterialId_; }

private:
    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    
    // Mapping: geomID -> materialId
    std::vector<uint32_t> geomMaterialId_;
    
    // User-defined geometry data for spheres (using deque to ensure pointer stability)
    std::deque<SphereUserData> sphereData_;
    
    // Static callback functions for sphere geometry
    static void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);
    static void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
    static void sphereBoundsFunc(const RTCBoundsFunctionArguments* args);
};

} // namespace backends
