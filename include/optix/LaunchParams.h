#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace optix {

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
    float3 baseColor;
    float roughness;
    float metallic;
    int type;
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
    uint32_t* shadeQueueCounter;
    DeviceMaterial* materials;
    int materialCount;
    int maxDepth;

    // Frame index (for deterministic RNG seeding in wavefront prep passes)
    unsigned int frameIndex;
};

} // namespace optix
