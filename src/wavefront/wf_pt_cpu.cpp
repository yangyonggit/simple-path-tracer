#include "wavefront/wf_pt_cpu.h"
#include "wavefront/wf_math.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <algorithm>
#include <cmath>
#include <limits>

#include "MaterialManager.h"
#include "Light.h"
#include "EnvironmentManager.h"
#include "Material.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>

// Embree
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

namespace wf {

// -----------------------------
// Embree intersect (you may want to reuse your intersectRay() here)
// -----------------------------
bool WavefrontPathTracerCPU::intersect(const WFContext& ctx,
                                       const glm::vec3& o,
                                       const glm::vec3& d,
                                       RTCRayHit& out_hit) const
{
  // Initialize ray/hit
  out_hit = RTCRayHit{};
  out_hit.ray.org_x = o.x;
  out_hit.ray.org_y = o.y;
  out_hit.ray.org_z = o.z;

  out_hit.ray.dir_x = d.x;
  out_hit.ray.dir_y = d.y;
  out_hit.ray.dir_z = d.z;

  out_hit.ray.tnear = 0.0f;
  out_hit.ray.tfar  = std::numeric_limits<float>::infinity();
  out_hit.ray.time  = 0.0f;
  out_hit.ray.mask  = 0xFFFFFFFFu;
  out_hit.ray.id    = 0;
  out_hit.ray.flags = 0;

  out_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  out_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(ctx.scene, &out_hit);

  return out_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

// -----------------------------
// Wavefront-style (single-ray) path tracing loop
// -----------------------------
glm::vec3 WavefrontPathTracerCPU::traceRay(const WFContext& ctx,
                                           const WFParams& p,
                                           const glm::vec3& origin,
                                           const glm::vec3& direction,
                                           uint32_t pixel_seed) const
{
  // Validate context
  if (!ctx.scene || !ctx.materials || !ctx.lights || !ctx.env) {
    return glm::vec3(0.0f);
  }

  // Set up callbacks (use defaults if caller didn't provide)
  auto safe_origin_fn = ctx.cb.safe_origin ? ctx.cb.safe_origin : 
      std::function<glm::vec3(const glm::vec3&, const glm::vec3&, bool)>(wf::default_safe_origin);
  auto cosine_sample_fn = ctx.cb.cosine_sample ? ctx.cb.cosine_sample : 
      std::function<glm::vec3(const glm::vec3&, uint32_t&)>(wf::default_cosine_sample);
  auto rand01_fn = ctx.cb.rand01 ? ctx.cb.rand01 : 
      std::function<float(uint32_t&)>(wf::default_rand01);

  glm::vec3 color(0.0f);

  // Multiple samples per pixel (keep this at this layer for now; later you can batch across pixels)
  for (uint32_t s = 0; s < p.spp; ++s) {
    // Path state (explicit, GPU-friendly shape)
    glm::vec3 ray_o = origin;
    glm::vec3 ray_d = wf::safe_normalize(direction);

    glm::vec3 throughput(1.0f);
    glm::vec3 radiance(0.0f);

    uint32_t rng = wf::wang_hash(pixel_seed ^ (s * 9781u + 1u));

    // Wavefront-like state machine (Intersect -> Shade -> Intersect -> ...)
    for (uint32_t bounce = 0; bounce < p.max_depth; ++bounce) {
      RTCRayHit rayhit;
      const bool hit = intersect(ctx, ray_o, ray_d, rayhit);

      if (!hit) {
        // Miss: environment
        const glm::vec3 env = ctx.env->getEnvironmentColor(wf::safe_normalize(ray_d));
        radiance += throughput * env;
        break;
      }

      // Hit point
      const float t = rayhit.ray.tfar;
      glm::vec3 hit_point = ray_o + t * ray_d;

      // Embree geometric normal
      glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
      normal = safe_normalize(normal);

      // Face-forward normal
      if (glm::dot(normal, ray_d) > 0.0f) {
        normal = -normal;
      }

      // Material
      const Material& material = ctx.materials->getMaterialFromHit(ctx.scene, rayhit);

      // Emission
      if (glm::length2(material.emission) > 0.0f) {
        radiance += throughput * material.emission;
      }

      // Direct lighting (synchronous occlusion for now; Step 3 will split into shadow queue)
      {
        const glm::vec3 view_dir = -ray_d;
        const auto& lights = *ctx.lights;

        for (size_t i = 0; i < lights.getLightCount(); ++i) {
          const Light& light = lights.getLight(i);

          glm::vec3 light_dir;
          float light_dist = 0.0f;
          const glm::vec3 Li = light.getRadiance(hit_point, normal, light_dir, light_dist);

          const float cos_theta = std::max(glm::dot(normal, light_dir), 0.0f);
          if (cos_theta <= 0.0f) continue;

          const bool occluded = light.isOccluded(hit_point, normal, light_dir, light_dist, ctx.scene);
          if (occluded) continue;

          const glm::vec3 brdf = material.evaluateBRDF(normal, view_dir, light_dir);
          radiance += throughput * (brdf * Li * cos_theta);
        }
      }

      // Indirect bounce based on material type
      // Note: This matches your recursive logic structure.
      if (material.metallic > 0.5f) {
        // Metallic reflection
        const glm::vec3 reflect_dir = glm::reflect(ray_d, normal);
        const glm::vec3 next_o = safe_origin_fn(hit_point, normal, true);
        const glm::vec3 next_d = safe_normalize(reflect_dir);

        // Update throughput similarly to your recursive: color += albedo * indirect * metallic
        // In path formulation: throughput *= albedo * metallic
        throughput *= (material.albedo * material.metallic);

        ray_o = next_o;
        ray_d = next_d;
        continue;
      }

      if (material.isTransparent()) {
        // Glass / transparent material
        const float ior = material.ior;
        const float cosine = -glm::dot(ray_d, normal);

        // Choose eta based on whether we are entering or exiting
        // We face-forwarded normal, so cosine should be >= 0 most of the time.
        // Still keep this logic robust.
        const bool entering = (cosine >= 0.0f);
        const float eta = entering ? (1.0f / ior) : ior;

        const float transparency = material.getTransparency();

        // Fresnel-based decision as fallback (if you had shouldTransmit(), plug it in here)
        const float F = wf::fresnel_schlick(std::abs(cosine), ior);

        // Probabilistic reflect/refract
        const float xi = rand01_fn(rng);
        if (xi < F) {
          // Reflect
          const glm::vec3 reflect_dir = glm::reflect(ray_d, normal);
          const glm::vec3 next_o = safe_origin_fn(hit_point, normal, true);
          const glm::vec3 next_d = safe_normalize(reflect_dir);

          // Throughput: reflection path gets (1 - transparency) portion (approx)
          throughput *= glm::vec3(1.0f - transparency);

          ray_o = next_o;
          ray_d = next_d;
          continue;
        }
        else {
          // Refract (or TIR fallback to reflect)
          glm::vec3 refr_dir = wf::refract_dir(ray_d, normal, eta);

          if (glm::length2(refr_dir) > 0.0f) {
            const glm::vec3 next_o = safe_origin_fn(hit_point, normal, false);
            const glm::vec3 next_d = wf::safe_normalize(refr_dir);

            throughput *= glm::vec3(transparency);

            ray_o = next_o;
            ray_d = next_d;
            continue;
          }
          else {
            // TIR -> reflect
            const glm::vec3 reflect_dir = glm::reflect(ray_d, normal);
            const glm::vec3 next_o = safe_origin_fn(hit_point, normal, true);
            const glm::vec3 next_d = safe_normalize(reflect_dir);

            ray_o = next_o;
            ray_d = next_d;
            continue;
          }
        }
      }

      // Diffuse material
      {
        const glm::vec3 next_d = cosine_sample_fn(normal, rng);
        const glm::vec3 next_o = safe_origin_fn(hit_point, normal, true);

        // Russian roulette (same idea as your recursive code)
        const float survival = glm::max(glm::max(material.albedo.r, material.albedo.g), material.albedo.b);
        const float xi = rand01_fn(rng);

        if (bounce > 2) { // optional: start RR after a few bounces
          if (xi >= survival) {
            break;
          }
          // Compensate
          throughput *= (material.albedo / std::max(survival, 1e-6f));
        }
        else {
          throughput *= material.albedo;
        }

        ray_o = next_o;
        ray_d = safe_normalize(next_d);
        continue;
      }
    } // bounce loop

    color += radiance;
  } // spp loop

  color /= float(std::max(1u, p.spp));
  return color;
}

} // namespace wf
