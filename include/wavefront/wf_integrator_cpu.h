#pragma once
#include <cstdint>
#include <vector>
#include "wf_types.h"
#include "wf_queues.h"

namespace wf {

// Forward declare your existing Scene/Camera types later.
// For Step 0, keep it generic.
struct RenderParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t spp = 1;
  uint32_t max_bounce = 5;
};

class WavefrontIntegratorCPU {
 public:
  WavefrontIntegratorCPU() = default;

  // Temporary entrypoint: fills out_rgb (size = width*height*3).
  // We'll wire this to your existing render pipeline in later steps.
  void render(const RenderParams& params, std::vector<float>& out_rgb);

 private:
  // buffers
  std::vector<PathState> paths_;
  std::vector<Hit> hits_;

  std::vector<ShadowTask> shadow_tasks_;
  std::vector<ShadowHit> shadow_hits_;

  IndexQueue q_intersect_;
  IndexQueue q_shade_;
  IndexQueue q_shadow_intersect_;
  IndexQueue q_shadow_shade_;
};

} // namespace wf
