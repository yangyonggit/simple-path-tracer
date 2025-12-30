#include <optix.h>
#include <optix_device.h>

#include "optix/LaunchParams.h"

using namespace optix;

// ========================================
// SBT data layout (must match host)
// ========================================
struct HitgroupData {
    int geomType;  // 0=triangles, 1=spheres
};

// ========================================
// Launch Parameters (Global in device code)
// ========================================
extern "C" {
    __constant__ LaunchParams params;
}

// ========================================
// Payload helpers
// ========================================
static __forceinline__ __device__ unsigned int packRGB8(const float3& c) {
    unsigned char r = static_cast<unsigned char>(fminf(fmaxf(c.x, 0.0f), 1.0f) * 255.0f);
    unsigned char g = static_cast<unsigned char>(fminf(fmaxf(c.y, 0.0f), 1.0f) * 255.0f);
    unsigned char b = static_cast<unsigned char>(fminf(fmaxf(c.z, 0.0f), 1.0f) * 255.0f);
    return (static_cast<unsigned int>(r) << 16) | (static_cast<unsigned int>(g) << 8) | static_cast<unsigned int>(b);
}

static __forceinline__ __device__ float3 unpackRGB8(const unsigned int packed) {
    const float r = static_cast<float>((packed >> 16) & 0xFF) * (1.0f / 255.0f);
    const float g = static_cast<float>((packed >> 8) & 0xFF) * (1.0f / 255.0f);
    const float b = static_cast<float>(packed & 0xFF) * (1.0f / 255.0f);
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

static __forceinline__ __device__ uint32_t wang_hash(uint32_t a) {
    a = (a ^ 61u) ^ (a >> 16);
    a *= 9u;
    a = a ^ (a >> 4);
    a *= 0x27d4eb2du;
    a = a ^ (a >> 15);
    return a;
}

static __forceinline__ __device__ void computePrimaryRay(
    const unsigned int x,
    const unsigned int y,
    float3& origin,
    float3& direction)
{
    const float2 pixel = make_float2(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
    const float2 ndc = make_float2(
        (pixel.x / static_cast<float>(params.image_width)) * 2.0f - 1.0f,
        1.0f - (pixel.y / static_cast<float>(params.image_height)) * 2.0f);

    origin = params.cam_pos;
    const float3 dir = f3_add(f3_add(f3_scale(params.cam_u, ndc.x), f3_scale(params.cam_v, ndc.y)), params.cam_w);
    direction = f3_normalize(dir);
}

// ========================================
// Wavefront Primary Initialization Raygen
// ========================================
extern "C" __global__ void __raygen__gen_primary() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int x = idx.x;
    const unsigned int y = idx.y;

    const unsigned int w = params.image_width;
    const unsigned int h = params.image_height;
    if (x >= w || y >= h) {
        return;
    }

    const uint32_t pixel = static_cast<uint32_t>(y * w + x);

    // Initialize path state for this pixel
    PathState& ps = params.paths[pixel];
    ps.pixelIndex = pixel;
    ps.depth = 0u;
    ps.rng = wang_hash((pixel + 1u) ^ (params.frameIndex * 9781u + 1u));

    float3 origin;
    float3 direction;
    computePrimaryRay(x, y, origin, direction);

    ps.origin = origin;
    ps.direction = direction;
    ps.throughput = make_float3(1.0f, 1.0f, 1.0f);
    ps.radiance = make_float3(0.0f, 0.0f, 0.0f);
    ps.alive = 1u;

    // Push into ray queue
    const uint32_t qidx = atomicAdd(params.rayQueueCounter, 1u);
    params.rayQueueIn[qidx] = pixel;
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

    float3 origin;
    float3 direction;
    computePrimaryRay(x, y, origin, direction);

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

    // Payload contract:
    // payload0 = hit flag (0 miss, 1 hit)
    // payload1 = packed RGB8 color for normal mode
    float3 color;
    if (params.debug_mode == 1) {
        color = (p0 != 0u) ? make_float3(1.0f, 1.0f, 1.0f) : make_float3(0.0f, 0.0f, 0.0f);
    } else {
        color = unpackRGB8(p1);
    }
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
    // Miss: payload0=0. payload1 unused.
    optixSetPayload_0(0u);
    optixSetPayload_1(0u);
}

// ========================================
// Closest Hit Program
// ========================================
extern "C" __global__ void __closesthit__ch() {
    // Hit: payload0=1.
    // In debug_mode=1 (hit/miss), raygen ignores payload1, so avoid SBT reads for easier isolation.
    if (params.debug_mode == 1) {
        optixSetPayload_0(1u);
        optixSetPayload_1(0u);
        return;
    }

    // Normal mode: payload1 carries packed RGB8 based on SBT geomType.
    const HitgroupData* hg = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const int geomType = hg ? hg->geomType : 0;
    const float3 hitColor = (geomType == 1) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    optixSetPayload_0(1u);
    optixSetPayload_1(packRGB8(hitColor));
}
