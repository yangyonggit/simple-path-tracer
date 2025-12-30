#include <optix.h>
#include <optix_device.h>

#include <cuda_runtime.h>

#include "optix/LaunchParams.h"

using namespace optix;

// ========================================
// SBT data layout (must match host)
// ========================================
struct HitgroupData {
    int geomType;  // 0=triangles, 1=spheres

    // Triangles
    const float3* vertices;
    const uint3* indices;
    const float3* normals;  // optional (can be nullptr)

    // Spheres
    const float3* centers;
    const float* radii;

    // Materials
    int materialId;              // per-geometry fallback
    const int* materialIds;      // optional per-primitive material ids
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

static __forceinline__ __device__ float rng_next01(uint32_t& state) {
    state = wang_hash(state);
    // Use 24 high-ish bits to build [0,1)
    return static_cast<float>(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static __forceinline__ __device__ float3 f3_cross(const float3 a, const float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 cosine_sample_hemisphere(const float u1, const float u2) {
    const float r = sqrtf(u1);
    const float phi = 2.0f * 3.14159265358979323846f * u2;
    float s, c;
    sincosf(phi, &s, &c);
    const float x = r * c;
    const float y = r * s;
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return make_float3(x, y, z);
}

static __forceinline__ __device__ void make_onb(const float3& n, float3& t, float3& b) {
    const float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    t = f3_cross(up, n);
    t = f3_normalize(t);
    b = f3_cross(n, t);
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
    PathState ps = params.paths[pathId];
    const HitRecord hr = params.hitRecords[pathId];

    const uint32_t pixel = ps.pixelIndex;
    if (pixel >= capacity) {
        return;
    }

    // Debug must-kill switch: force a visible accum write on the first frame.
    // Flip to 1 temporarily if you suspect resolve/accum wiring.
    constexpr int kForceFirstFrameAccum = 0;
    if (kForceFirstFrameAccum && params.frameIndex == 0u && ps.depth == 0u) {
        atomicAdd(&params.accum[pixel].x, 0.8f);
        atomicAdd(&params.accum[pixel].y, 0.2f);
        atomicAdd(&params.accum[pixel].z, 0.2f);
        atomicAdd(&params.accum[pixel].w, 1.0f);
    }

    // Termination condition 1: miss MUST contribute immediately.
    if (hr.t < 0.0f) {
        float3 dir = ps.direction;
        dir = f3_normalize(dir);

        float3 c = make_float3(0.1f, 0.1f, 0.1f);
        if (params.env_enabled && params.env_tex != 0) {
            // Equirectangular mapping: theta = atan2(z, x), phi = acos(y)
            constexpr float kInvTwoPi = 0.15915494309189535f; // 1 / (2*pi)
            constexpr float kInvPi = 0.3183098861837907f;      // 1 / pi

            const float yclamp = fminf(fmaxf(dir.y, -1.0f), 1.0f);
            const float theta = atan2f(dir.z, dir.x);
            const float phi = acosf(yclamp);
            float u = (theta + 3.14159265358979323846f) * kInvTwoPi;
            float v = phi * kInvPi;

            const float4 t = tex2D<float4>(params.env_tex, u, v);
            c = make_float3(t.x, t.y, t.z);

            // Match CPU-side EnvironmentManager clamp/exposure behavior.
            c.x = fminf(c.x, params.env_max_clamp);
            c.y = fminf(c.y, params.env_max_clamp);
            c.z = fminf(c.z, params.env_max_clamp);
            c = f3_scale(c, params.env_intensity);
        } else {
            // Procedural sky fallback (matches EnvironmentManager::getSkyColor() shape).
            float t = 0.5f * (dir.y + 1.0f);
            // smoothstep(0,1,t)
            t = fminf(fmaxf(t, 0.0f), 1.0f);
            t = t * t * (3.0f - 2.0f * t);

            const float3 horizon = make_float3(0.7f, 0.8f, 0.9f);
            const float3 zenith = make_float3(0.2f, 0.4f, 0.8f);
            c = f3_add(f3_scale(horizon, 1.0f - t), f3_scale(zenith, t));

            const float3 sun_dir = f3_normalize(make_float3(0.3f, 0.6f, -0.8f));
            const float sun_dot = fmaxf(dir.x * sun_dir.x + dir.y * sun_dir.y + dir.z * sun_dir.z, 0.0f);
            const float sun_intensity = powf(sun_dot, 64.0f);
            const float sun_glow = powf(sun_dot, 8.0f) * 0.3f;
            const float3 sun_color = make_float3(1.0f, 0.9f, 0.7f);
            c = f3_add(c, f3_scale(sun_color, sun_intensity + sun_glow));
            c = f3_scale(c, 0.8f);
        }
        atomicAdd(&params.accum[pixel].x, c.x * ps.throughput.x);
        atomicAdd(&params.accum[pixel].y, c.y * ps.throughput.y);
        atomicAdd(&params.accum[pixel].z, c.z * ps.throughput.z);
        atomicAdd(&params.accum[pixel].w, 1.0f);
        return;
    }

    // Termination condition 2: maxDepth reached BEFORE generating the next ray.
    // NOTE: host loops `for depth < maxDepth`, so if we enqueue rays at depth==maxDepth
    // they will be stranded and never contribute. Therefore, terminate when the *next*
    // depth would reach/exceed maxDepth.
    if ((ps.depth + 1u) >= static_cast<uint32_t>(params.maxDepth)) {
        float3 n = hr.Ng;
        const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 > 0.0f) {
            n = f3_scale(n, rsqrtf(len2));
        } else {
            n = make_float3(0.0f, 1.0f, 0.0f);
        }

        float3 albedo = make_float3(1.0f, 1.0f, 1.0f);
        if (params.materials != nullptr && params.materialCount > 0) {
            int mid = hr.materialId;
            if (mid < 0) mid = 0;
            if (mid >= params.materialCount) mid = params.materialCount - 1;
            const DeviceMaterial m = params.materials[mid];
            albedo = m.baseColor;
        }

        const float3 c = f3_scale(f3_add(n, make_float3(1.0f, 1.0f, 1.0f)), 0.5f);
        const float3 shaded = make_float3(albedo.x * c.x, albedo.y * c.y, albedo.z * c.z);
        atomicAdd(&params.accum[pixel].x, shaded.x * ps.throughput.x);
        atomicAdd(&params.accum[pixel].y, shaded.y * ps.throughput.y);
        atomicAdd(&params.accum[pixel].z, shaded.z * ps.throughput.z);
        atomicAdd(&params.accum[pixel].w, 1.0f);
        return;
    }

    // Spawn next diffuse bounce.
    float3 n = hr.Ng;
    const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
    if (len2 > 0.0f) {
        n = f3_scale(n, rsqrtf(len2));
    } else {
        n = make_float3(0.0f, 1.0f, 0.0f);
    }

    const float3 P = f3_add(ps.origin, f3_scale(ps.direction, hr.t));

    uint32_t rng = ps.rng;
    const float u1 = rng_next01(rng);
    const float u2 = rng_next01(rng);
    const float3 local = cosine_sample_hemisphere(u1, u2);
    float3 t, b;
    make_onb(n, t, b);
    float3 newDir = f3_add(f3_add(f3_scale(t, local.x), f3_scale(b, local.y)), f3_scale(n, local.z));
    newDir = f3_normalize(newDir);

    ps.origin = f3_add(P, f3_scale(n, 1e-3f));
    ps.direction = newDir;
    ps.depth = ps.depth + 1u;
    ps.throughput = f3_scale(ps.throughput, 0.8f);
    ps.rng = rng;

    params.paths[pathId] = ps;

    const uint32_t outIdx = atomicAdd(reinterpret_cast<unsigned int*>(params.rayQueueOutCounter), 1u);
    if (outIdx < capacity) {
        params.rayQueueOut[outIdx] = pathId;
    }
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

    const uint32_t capacity = static_cast<uint32_t>(params.image_width) * static_cast<uint32_t>(params.image_height);
    const uint32_t idx = atomicAdd(reinterpret_cast<unsigned int*>(params.shadeQueueCounter), 1u);
    if (idx < capacity) {
        params.shadeQueue[idx] = pathId;
    }
}

// ========================================
// Wavefront Closest Hit Program (rayType=1)
// ========================================
extern "C" __global__ void __closesthit__ch_wf() {
    const HitgroupData* hg = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const uint32_t pathId = optixGetPayload_0();
    HitRecord& hr = params.hitRecords[pathId];
    hr.pathId = pathId;
    const float t = optixGetRayTmax();
    hr.t = t;

    const float3 O = optixGetWorldRayOrigin();
    const float3 D = optixGetWorldRayDirection();
    const float3 P = f3_add(O, f3_scale(D, t));

    float3 Ng = make_float3(0.0f, 1.0f, 0.0f);
    int materialId = 0;

    const int geomType = hg ? hg->geomType : 0;
    if (geomType == 0) {
        // Triangles: compute geometric normal from vertex positions.
        const int primId = optixGetPrimitiveIndex();
        const uint3 tri = hg->indices[primId];
        const float3 v0 = hg->vertices[tri.x];
        const float3 v1 = hg->vertices[tri.y];
        const float3 v2 = hg->vertices[tri.z];

        const float3 e1 = f3_add(v1, f3_neg(v0));
        const float3 e2 = f3_add(v2, f3_neg(v0));
        float3 Ng_obj = f3_cross(e1, e2);
        Ng_obj = f3_normalize(Ng_obj);

        // Transform to world space (handles instancing transforms correctly).
        Ng = optixTransformNormalFromObjectToWorldSpace(Ng_obj);
        Ng = f3_normalize(Ng);

        if (Ng.x * D.x + Ng.y * D.y + Ng.z * D.z > 0.0f) {
            Ng = f3_neg(Ng);
        }

        materialId = hg ? hg->materialId : 0;
    } else {
        // Spheres: compute normal from hit point and center.
        const int primId = optixGetPrimitiveIndex();
        const float3 P_obj = optixTransformPointFromWorldToObjectSpace(P);
        const float3 C_obj = hg->centers[primId];
        float3 Ng_obj = f3_add(P_obj, f3_neg(C_obj));
        Ng_obj = f3_normalize(Ng_obj);

        Ng = optixTransformNormalFromObjectToWorldSpace(Ng_obj);
        Ng = f3_normalize(Ng);
        if (Ng.x * D.x + Ng.y * D.y + Ng.z * D.z > 0.0f) {
            Ng = f3_neg(Ng);
        }

        if (hg && hg->materialIds) {
            materialId = hg->materialIds[primId];
        } else {
            materialId = hg ? hg->materialId : 0;
        }
    }

    hr.Ng = Ng;
    hr.materialId = materialId;

    const uint32_t capacity = static_cast<uint32_t>(params.image_width) * static_cast<uint32_t>(params.image_height);
    const uint32_t idx = atomicAdd(reinterpret_cast<unsigned int*>(params.shadeQueueCounter), 1u);
    if (idx < capacity) {
        params.shadeQueue[idx] = pathId;
    }
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

// ========================================
// Wavefront Resolve Raygen
// ========================================
extern "C" __global__ void __raygen__resolve() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int x = idx.x;
    const unsigned int y = idx.y;
    const unsigned int w = params.image_width;
    const unsigned int h = params.image_height;
    if (x >= w || y >= h) {
        return;
    }

    const uint32_t pixel = static_cast<uint32_t>(y * w + x);
    const float4 a = params.accum[pixel];
    const float inv = (a.w > 0.0f) ? (1.0f / a.w) : 0.0f;
    float3 c = make_float3(a.x * inv, a.y * inv, a.z * inv);

    c.x = fminf(fmaxf(c.x, 0.0f), 1.0f);
    c.y = fminf(fmaxf(c.y, 0.0f), 1.0f);
    c.z = fminf(fmaxf(c.z, 0.0f), 1.0f);

    params.output_buffer[pixel] = make_uchar4(
        static_cast<unsigned char>(c.x * 255.0f),
        static_cast<unsigned char>(c.y * 255.0f),
        static_cast<unsigned char>(c.z * 255.0f),
        255);
}
