#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <embree4/rtcore.h>

#include "wavefront/wf_types.h"
#include "wavefront/wf_queues.h"

class MaterialManager;
class LightManager;
class EnvironmentManager;
struct Material;

namespace wf {

struct WFCallbacks {
  // Use std::function to support lambdas with captures
  std::function<glm::vec3(const glm::vec3& hit_point, const glm::vec3& normal, bool front)> safe_origin;
  std::function<glm::vec3(const glm::vec3& normal, uint32_t& rng)> cosine_sample;
  std::function<float(uint32_t& rng)> rand01;
};

struct WFContext {
  RTCScene scene = nullptr;
  const MaterialManager* materials = nullptr;
  const LightManager* lights = nullptr;
  const EnvironmentManager* env = nullptr;
  WFCallbacks cb;
};

struct WFParams {
  uint32_t max_depth = 5;
  uint32_t spp = 1;
};

class WavefrontPathTracerCPU {
 public:
  glm::vec3 traceRay(const WFContext& ctx,
                     const WFParams& p,
                     const glm::vec3& origin,
                     const glm::vec3& direction,
                     uint32_t pixel_seed) const;

 private:
  bool intersect(const WFContext& ctx,
                 const glm::vec3& o,
                 const glm::vec3& d,
                 RTCRayHit& out_hit) const;
};

} // namespace wf
