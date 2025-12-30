#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace optix {

// Material type IDs shared between host/device.
static constexpr int MATERIAL_TYPE_PBR = 0;
static constexpr int MATERIAL_TYPE_DIELECTRIC = 1;

// ========================================
// Wavefront basic data structures (scaffolding)
// ========================================
struct PathState {
    uint32_t pixelIndex;
    uint32_t depth;
    uint32_t rng;
    float3 origin;
    float3 direction;
    float3 throughput;
    float3 radiance;
    uint32_t alive;
};

struct HitRecord {
    uint32_t pathId;
    float t;
    float3 Ng;
    int materialId;
};

struct DeviceMaterial {
    float3 baseColor;   // linear
    float metallic;
    float roughness;
    float ior;          // index of refraction (for dielectrics)
    int type;           // MATERIAL_TYPE_*
    float3 emission;    // unused for now
    float pad;
    float pad2;
};

struct DirectionalLight {
    // Normalized direction pointing FROM the light (i.e., direction of light rays).
    float3 direction;
    // Radiance = color * intensity.
    float3 radiance;
};

// ========================================
// Launch Parameters - Passed to OptiX kernels
// ========================================
struct LaunchParams {
    // Output buffer
    uchar4* output_buffer;  // RGBA output (or RGB packed as RGBA)

    // Accumulation buffer (wavefront shading writes here; progressive-friendly)
    float4* accum;          // size = image_width * image_height
    
    // Image dimensions
    unsigned int image_width;
    unsigned int image_height;

    // Camera (simple pinhole basis)
    float3 cam_pos;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;

    // Scene traversable
    OptixTraversableHandle topHandle;

    // Debug mode
    // 0 = normal (tri=red, sphere=green)
    // 1 = hit/miss (hit=white, miss=black)
    int debug_mode;

    // ========================================
    // Wavefront buffers (not used by device_programs.cu yet)
    // ========================================
    PathState* paths;
    HitRecord* hitRecords;
    uint32_t* rayQueueIn;
    uint32_t* rayQueueOut;
    uint32_t* shadeQueue;
    uint32_t* rayQueueCounter;
    uint32_t* rayQueueOutCounter;
    uint32_t* shadeQueueCounter;
    DeviceMaterial* materials;
    int materialCount;
    int maxDepth;

    // Frame index (for deterministic RNG seeding in wavefront prep passes)
    unsigned int frameIndex;

    // Environment (HDR equirectangular) for miss sampling
    cudaTextureObject_t env_tex;
    int env_enabled;
    float env_intensity;
    float env_max_clamp;

    // Single directional light (for direct lighting; no shadows/MIS).
    DirectionalLight dirLight;
    int hasDirLight;
};

} // namespace optix
