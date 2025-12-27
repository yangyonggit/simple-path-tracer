#pragma once
#include <cmath>
#include "wf_types.h"

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

} // namespace wf
