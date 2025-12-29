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
// Payload helpers
// ========================================
static __forceinline__ __device__ void packColor(const float3& c, unsigned int& p0, unsigned int& p1) {
    unsigned char r = static_cast<unsigned char>(fminf(fmaxf(c.x, 0.0f), 1.0f) * 255.0f);
    unsigned char g = static_cast<unsigned char>(fminf(fmaxf(c.y, 0.0f), 1.0f) * 255.0f);
    unsigned char b = static_cast<unsigned char>(fminf(fmaxf(c.z, 0.0f), 1.0f) * 255.0f);
    p0 = (static_cast<unsigned int>(r) << 16) | (static_cast<unsigned int>(g) << 8) | static_cast<unsigned int>(b);
    p1 = 0u;
}

static __forceinline__ __device__ float3 unpackColor(const unsigned int p0, const unsigned int /*p1*/) {
    const float r = static_cast<float>((p0 >> 16) & 0xFF) * (1.0f / 255.0f);
    const float g = static_cast<float>((p0 >> 8) & 0xFF) * (1.0f / 255.0f);
    const float b = static_cast<float>(p0 & 0xFF) * (1.0f / 255.0f);
    return make_float3(r, g, b);
}

// Minimal float3 helpers to avoid extra math headers
static __forceinline__ __device__ float3 f3_add(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 f3_scale(const float3 a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 f3_normalize(const float3 v) {
    const float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
    const float inv_len = rsqrtf(len2);
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
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

    // Simple pinhole camera ray
    const float2 pixel = make_float2(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
    const float2 ndc = make_float2(
        (pixel.x / static_cast<float>(params.image_width)) * 2.0f - 1.0f,
        1.0f - (pixel.y / params.image_height) * 2.0f);

    float3 origin = params.cam_pos;
    const float3 dir = f3_add(f3_add(f3_scale(params.cam_u, ndc.x), f3_scale(params.cam_v, ndc.y)), params.cam_w);
    float3 direction = f3_normalize(dir);

    unsigned int p0 = 0u;
    unsigned int p1 = 0u;

    optixTrace(
        params.topHandle,
        origin,
        direction,
        0.001f,                // tmin
        1e16f,                  // tmax
        0.0f,                   // time
        0xFF,                   // visibility mask
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                      // SBT offset
        1,                      // SBT stride
        0,                      // miss SBT index
        p0, p1);

    // Unpack payload and write output
    const float3 color = unpackColor(p0, p1);
    const unsigned int linear_idx = y * params.image_width + x;
    params.output_buffer[linear_idx] = make_uchar4(
        static_cast<unsigned char>(color.x * 255.0f),
        static_cast<unsigned char>(color.y * 255.0f),
        static_cast<unsigned char>(color.z * 255.0f),
        255);
}

// ========================================
// Miss Program
// ========================================
extern "C" __global__ void __miss__ms() {
    // Background: dark gray/blue
    const float3 bg = make_float3(0.05f, 0.07f, 0.10f);
    unsigned int p0, p1;
    packColor(bg, p0, p1);
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
}

// ========================================
// Closest Hit Program
// ========================================
extern "C" __global__ void __closesthit__ch() {
    // Hit color: solid red
    const float3 hit = make_float3(1.0f, 0.0f, 0.0f);
    unsigned int p0, p1;
    packColor(hit, p0, p1);
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
}
