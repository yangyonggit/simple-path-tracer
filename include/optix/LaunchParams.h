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
    
    // Camera parameters (for future use)
    // float3 camera_eye;
    // float3 camera_u, camera_v, camera_w;
    
    // Scene data (for future use)
    // OptixTraversableHandle traversable;
};

} // namespace optix
