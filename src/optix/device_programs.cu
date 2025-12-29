#include <optix.h>
#include <optix_device.h>
#include "optix/LaunchParams.h"

using namespace optix;

// ========================================
// Launch Parameters (Global in device code)
// ========================================
extern "C" {
    __constant__ LaunchParams params;
}

// ========================================
// Ray Generation Program
// ========================================
extern "C" __global__ void __raygen__rg() {
    // Get pixel coordinates
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    const unsigned int x = idx.x;
    const unsigned int y = idx.y;
    
    // Output fixed color (blue-ish gradient)
    // R: varies with x, G: varies with y, B: 200
    const unsigned int linear_idx = y * params.image_width + x;
    
    unsigned char r = (unsigned char)((float)x / (float)params.image_width * 255.0f);
    unsigned char g = (unsigned char)((float)y / (float)params.image_height * 255.0f);
    unsigned char b = 200;
    unsigned char a = 255;
    
    // Write to output buffer (uchar4 = RGBA)
    params.output_buffer[linear_idx] = make_uchar4(r, g, b, a);
}

// ========================================
// Miss Program
// ========================================
extern "C" __global__ void __miss__ms() {
    // Not used in this minimal version
    // In future: set background color or environment map
}

// ========================================
// Closest Hit Program
// ========================================
extern "C" __global__ void __closesthit__ch() {
    // Not used in this minimal version
    // In future: compute shading, materials, etc.
}
