#pragma once
#include <cstdint>

// Keep everything POD-friendly for future CUDA port.
// Avoid std::vector / glm types inside these structs.

namespace wf {

struct float3 {
  float x, y, z;
};

inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }

// Stage (queued kernel)
enum class Stage : uint8_t {
  Intersect = 0,
  ShadeSurface = 1,
  ShadowIntersect = 2,
  ShadowShade = 3,
  Done = 255
};

// Per-path state (SoA-friendly POD)
struct PathState {
  float3 ray_o;
  float3 ray_d;

  float3 throughput;   // multiplicative path weight
  float3 radiance;     // accumulated result

  uint32_t pixel_index;
  uint16_t bounce;
  uint8_t  stage;
  uint8_t  alive;      // 1 = active, 0 = terminated

  uint32_t rng;        // simple RNG state
};

// Primary hit result (from INTERSECT stage)
struct Hit {
  float t;
  float3 Ng;           // geometric normal
  uint32_t prim_id;    // optional
  uint32_t mat_id;     // material index
  uint8_t  hit;        // 1 hit, 0 miss
  uint8_t  _pad[3];
};

// Shadow ray task (spawned by SHADE, consumed by SHADOW_INTERSECT/SHADOW_SHADE)
struct ShadowTask {
  float3 ray_o;
  float3 ray_d;
  float  tmax;

  float3 contrib;      // already includes BSDF * Li * cos / pdf etc.
  uint32_t path_index; // which PathState to add to
};

struct ShadowHit {
  uint8_t visible;     // 1 visible, 0 occluded
  uint8_t _pad[3];
};

} // namespace wf
