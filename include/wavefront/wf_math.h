#pragma once
#include <cmath>
#include <algorithm>
#include "wf_types.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace wf {

inline float3 add(const float3& a, const float3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline float3 sub(const float3& a, const float3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float3 mul(const float3& a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline float3 mul3(const float3& a, const float3& b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }

inline float dot(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

inline float3 normalize(const float3& v) {
  const float len2 = dot(v, v);
  if (len2 <= 0.0f) return {0,0,0};
  const float inv = 1.0f / std::sqrt(len2);
  return {v.x*inv, v.y*inv, v.z*inv};
}


// -----------------------------
// Small math helpers
// -----------------------------
static inline glm::vec3 safe_normalize(const glm::vec3& v)
{
  const float len2 = glm::dot(v, v);
  if (len2 <= 0.0f) return glm::vec3(0.0f);
  return v * (1.0f / std::sqrt(len2));
}

static inline uint32_t wang_hash(uint32_t a)
{
  a = (a ^ 61u) ^ (a >> 16u);
  a *= 9u;
  a = a ^ (a >> 4u);
  a *= 0x27d4eb2du;
  a = a ^ (a >> 15u);
  return a;
}

static inline float default_rand01(uint32_t& rng)
{
  rng = wang_hash(rng);
  return float(rng & 0x00FFFFFFu) / float(0x01000000u);
}

static inline glm::vec3 default_cosine_sample(const glm::vec3& n, uint32_t& rng)
{
  // Cosine-weighted hemisphere sampling around n
  const float r1 = default_rand01(rng);
  const float r2 = default_rand01(rng);

  const float phi = 2.0f * glm::pi<float>() * r1;
  const float r = std::sqrt(r2);

  const float x = r * std::cos(phi);
  const float y = r * std::sin(phi);
  const float z = std::sqrt(std::max(0.0f, 1.0f - r2));

  // Build an orthonormal basis (t, b, n)
  glm::vec3 nn = safe_normalize(n);
  glm::vec3 t = (std::abs(nn.z) < 0.999f) ? glm::normalize(glm::cross(nn, glm::vec3(0, 0, 1)))
                                          : glm::normalize(glm::cross(nn, glm::vec3(0, 1, 0)));
  glm::vec3 b = glm::cross(t, nn);

  // Local -> world
  return safe_normalize(t * x + b * y + nn * z);
}

static inline glm::vec3 default_safe_origin(const glm::vec3& p, const glm::vec3& n, bool front)
{
  // Small offset to avoid self-intersection
  const float eps = 1e-4f;
  return front ? (p + n * eps) : (p - n * eps);
}

// Snell refraction helper; returns (0,0,0) on TIR
static inline glm::vec3 refract_dir(const glm::vec3& I, const glm::vec3& N, float eta)
{
  // I and N assumed normalized; I points *towards* surface
  const float cosi = -glm::dot(N, I);
  const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
  if (k < 0.0f) {
    return glm::vec3(0.0f);
  }
  return eta * I + (eta * cosi - std::sqrt(k)) * N;
}

// A simple Schlick Fresnel; used as fallback if you don't have shouldTransmit()
static inline float fresnel_schlick(float cos_theta, float ior)
{
  float r0 = (1.0f - ior) / (1.0f + ior);
  r0 = r0 * r0;
  const float x = 1.0f - std::clamp(cos_theta, 0.0f, 1.0f);
  return r0 + (1.0f - r0) * x * x * x * x * x;
}

} // namespace wf
