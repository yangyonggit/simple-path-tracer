#pragma once

#include <optix.h>
#include <cuda_runtime.h>

namespace optix {

// ========================================
// Launch Parameters - Passed to OptiX kernels
// ========================================
struct LaunchParams {
    // Output buffer
    uchar4* output_buffer;  // RGBA output (or RGB packed as RGBA)
    
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
};

} // namespace optix
