#include "wavefront/wf_integrator_cpu.h"
#include "wavefront/wf_math.h"

#include <algorithm>
#include <cmath>

// Boundary uses glm (CPU only)
#include <glm/glm.hpp>

namespace wf {

// ---- boundary helpers (glm <-> wf::float3) ----
static inline wf::float3 to_wf(const glm::vec3& v) { return {v.x, v.y, v.z}; }
static inline glm::vec3 to_glm(const wf::float3& v) { return {v.x, v.y, v.z}; }

// Simple hash RNG (placeholder; we'll replace with a better per-path RNG later)
static inline float rand01(uint32_t& rng) {
  rng = wf::wang_hash(rng);
  // Use 24 bits
  return float(rng & 0x00FFFFFFu) / float(0x01000000u);
}

// Minimal pinhole camera (temporary). Step 2 will wire to your real Camera.
static inline void generate_primary_ray(uint32_t x, uint32_t y, const RenderParams& p,
                                        uint32_t& rng, wf::float3& o, wf::float3& d)
{
  // Camera at origin looking -Z, NDC in [-1,1]
  const float fx = (float(x) + 0.5f) / float(p.width);
  const float fy = (float(y) + 0.5f) / float(p.height);
  const float ndc_x = 2.0f * fx - 1.0f;
  const float ndc_y = 1.0f - 2.0f * fy;

  const float aspect = (p.height > 0) ? float(p.width) / float(p.height) : 1.0f;
  const float fov = 45.0f * 3.14159265f / 180.0f;
  const float tan_half = std::tan(0.5f * fov);

  glm::vec3 origin(0.0f, 0.0f, 0.0f);
  glm::vec3 dir(ndc_x * aspect * tan_half, ndc_y * tan_half, -1.0f);

  // tiny jitter can be added later with rng
  dir = glm::normalize(dir);

  o = to_wf(origin);
  d = to_wf(dir);
}

// Temporary environment (miss shader)
static inline wf::float3 eval_environment(const wf::float3& dir)
{
  // map dir.y from [-1,1] to [0,1]
  const float t = 0.5f * (dir.y + 1.0f);
  // horizon warm, zenith blue
  const wf::float3 a = wf::make_float3(1.0f, 0.6f, 0.2f);
  const wf::float3 b = wf::make_float3(0.2f, 0.5f, 1.0f);
  return wf::add(wf::mul(a, 1.0f - t), wf::mul(b, t));
}

void WavefrontIntegratorCPU::render(const RenderParams& params, std::vector<float>& out_rgb)
{
  const uint32_t W = params.width;
  const uint32_t H = params.height;
  const uint32_t N = W * H;  // Step 1: 1 path per pixel (spp=1)

  out_rgb.assign(size_t(N) * 3u, 0.0f);

  // Allocate buffers
  paths_.resize(N);
  hits_.resize(N);

  shadow_tasks_.clear();
  shadow_hits_.clear();

  q_intersect_.clear();
  q_shade_.clear();
  q_shadow_intersect_.clear();
  q_shadow_shade_.clear();

  // Init primary rays
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      const uint32_t idx = y * W + x;

      PathState& ps = paths_[idx];
      ps.pixel_index = idx;
      ps.bounce = 0;
      ps.stage = uint8_t(Stage::Intersect);
      ps.alive = 1;
      ps.rng = wang_hash(idx + 1u);

      ps.throughput = make_float3(1.0f, 1.0f, 1.0f);
      ps.radiance = make_float3(0.0f, 0.0f, 0.0f);

      generate_primary_ray(x, y, params, ps.rng, ps.ray_o, ps.ray_d);

      q_intersect_.push(idx);
    }
  }

  // Wavefront loop (Step 1: intersect = always miss)
  while (!q_intersect_.empty() || !q_shade_.empty() ||
         !q_shadow_intersect_.empty() || !q_shadow_shade_.empty())
  {
    // Shadow queues (unused in Step 1)
    q_shadow_intersect_.clear();
    q_shadow_shade_.clear();

    // INTERSECT stage
    if (!q_intersect_.empty()) {
      q_shade_.clear();

      for (uint32_t i = 0; i < (uint32_t)q_intersect_.items.size(); ++i) {
        const uint32_t path_index = q_intersect_.items[i];
        PathState& ps = paths_[path_index];

        Hit& hit = hits_[path_index];
        hit.hit = 0;
        hit.t = 0.0f;
        hit.Ng = make_float3(0,0,0);
        hit.mat_id = 0;
        hit.prim_id = 0;

        // Always miss in Step 1
        ps.stage = uint8_t(Stage::ShadeSurface);
        q_shade_.push(path_index);
      }
      q_intersect_.clear();
    }

    // SHADE_SURFACE stage (miss shader only)
    if (!q_shade_.empty()) {
      for (uint32_t i = 0; i < (uint32_t)q_shade_.items.size(); ++i) {
        const uint32_t path_index = q_shade_.items[i];
        PathState& ps = paths_[path_index];
        const Hit& hit = hits_[path_index];

        if (!ps.alive) continue;

        if (hit.hit == 0) {
          // miss: accumulate env and terminate
          const wf::float3 env = eval_environment(ps.ray_d);
          ps.radiance = add(ps.radiance, mul3(ps.throughput, env));
          ps.alive = 0;
          ps.stage = uint8_t(Stage::Done);
        }
      }
      q_shade_.clear();
    }
  }

  // Write to output
  for (uint32_t i = 0; i < N; ++i) {
    const wf::float3 c = paths_[i].radiance;
    out_rgb[size_t(i) * 3u + 0u] = c.x;
    out_rgb[size_t(i) * 3u + 1u] = c.y;
    out_rgb[size_t(i) * 3u + 2u] = c.z;
  }
}

} // namespace wf
