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

static __forceinline__ __device__ float3 f3_neg(const float3 v) {
    return make_float3(-v.x, -v.y, -v.z);
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

    // Push into ray queue (defensive bounds check)
    const uint32_t capacity = static_cast<uint32_t>(params.image_width) * static_cast<uint32_t>(params.image_height);
    const uint32_t qidx = atomicAdd(params.rayQueueCounter, 1u);
    if (qidx < capacity) {
        params.rayQueueIn[qidx] = pixel;
    }
}

// ========================================
// Wavefront Trace Raygen (rayType=1)
// ========================================
extern "C" __global__ void __raygen__trace() {
    const uint32_t tid = optixGetLaunchIndex().x;
    uint32_t count = *params.rayQueueCounter;
    const uint32_t capacity = static_cast<uint32_t>(params.image_width) * static_cast<uint32_t>(params.image_height);
    if (count > capacity) {
        count = capacity;
    }
    if (tid >= count) {
        return;
    }

    const uint32_t pathId = params.rayQueueIn[tid];
    const PathState ps = params.paths[pathId];

    unsigned int p0 = pathId;
    unsigned int p1 = 0u;

    // Use rayType=1 with rayTypeCount=2 so SBT selects wf miss/hit.
    optixTrace(
        params.topHandle,
        ps.origin,
        ps.direction,
        1e-3f,
        1e16f,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_NONE,
        1,  // sbtOffset = rayType
        2,  // sbtStride = rayTypeCount
        1,  // missSBTIndex = rayType
        p0, p1);
}

// ========================================
// Wavefront Shade Raygen
// ========================================
extern "C" __global__ void __raygen__shade() {
    const uint32_t tid = optixGetLaunchIndex().x;
    uint32_t count = *params.shadeQueueCounter;
    const uint32_t capacity = static_cast<uint32_t>(params.image_width) * static_cast<uint32_t>(params.image_height);
    if (count > capacity) {
        count = capacity;
    }
    if (tid >= count) {
        return;
    }

    const uint32_t pathId = params.shadeQueue[tid];
    const PathState ps = params.paths[pathId];
    const HitRecord hr = params.hitRecords[pathId];

    float3 c;
    if (hr.t < 0.0f) {
        c = make_float3(0.1f, 0.1f, 0.1f);
    } else {
        float3 n = hr.Ng;
        const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 > 0.0f) {
            n = f3_scale(n, rsqrtf(len2));
        } else {
            n = make_float3(0.0f, 1.0f, 0.0f);
        }
        c = f3_scale(f3_add(n, make_float3(1.0f, 1.0f, 1.0f)), 0.5f);
    }

    const uint32_t pixel = ps.pixelIndex;
    if (pixel >= capacity) {
        return;
    }

    const float4 old = params.accum[pixel];
    params.accum[pixel] = make_float4(old.x + c.x, old.y + c.y, old.z + c.z, old.w + 1.0f);
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
        0,                      // sbtOffset = rayType (legacy)
        2,                      // sbtStride = rayTypeCount
        0,                      // miss SBT index base
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
// Wavefront Miss Program (rayType=1)
// ========================================
extern "C" __global__ void __miss__ms_wf() {
    const uint32_t pathId = optixGetPayload_0();
    HitRecord& hr = params.hitRecords[pathId];
    hr.pathId = pathId;
    hr.t = -1.0f;
    hr.Ng = make_float3(0.0f, 0.0f, 0.0f);
    hr.materialId = -1;

    const uint32_t idx = atomicAdd(reinterpret_cast<unsigned int*>(params.shadeQueueCounter), 1u);
    params.shadeQueue[idx] = pathId;
}

// ========================================
// Wavefront Closest Hit Program (rayType=1)
// ========================================
extern "C" __global__ void __closesthit__ch_wf() {
    const uint32_t pathId = optixGetPayload_0();
    HitRecord& hr = params.hitRecords[pathId];
    hr.pathId = pathId;
    hr.t = optixGetRayTmax();

    // Placeholder normal: view normal
    const float3 wo = optixGetWorldRayDirection();
    hr.Ng = f3_normalize(f3_neg(wo));
    hr.materialId = 0;

    const uint32_t idx = atomicAdd(reinterpret_cast<unsigned int*>(params.shadeQueueCounter), 1u);
    params.shadeQueue[idx] = pathId;
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
