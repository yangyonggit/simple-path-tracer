#include "Material.h"
#include <algorithm>
#include <cmath>

#include <glm/gtc/constants.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static glm::vec3 build_tangent(const glm::vec3& n) {
    const glm::vec3 up = (std::fabs(n.z) < 0.999f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 t = glm::cross(up, n);
    const float len2 = glm::dot(t, t);
    if (len2 > 0.0f) {
        return t * (1.0f / std::sqrt(len2));
    }
    return glm::vec3(1.0f, 0.0f, 0.0f);
}

static void make_onb(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    t = build_tangent(n);
    b = glm::cross(n, t);
}

static float G1_SchlickGGX(float NdotX, float k) {
    return NdotX / (NdotX * (1.0f - k) + k);
}

// GGX/Trowbridge-Reitz Normal Distribution Function
// Parameter semantic: alpha (not perceptual roughness). Do NOT square twice.
float Material::distributionGGX(const glm::vec3& N, const glm::vec3& H, float alpha) {
    float a2 = alpha * alpha;
    float NdotH = glm::max(glm::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PI * denom * denom;

    return num / denom;
}

// Schlick-GGX Geometry Function (for a single direction)
float Material::geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float num = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / denom;
}

// Smith Geometry Function (combines light and view directions)
// Parameter semantic: alpha (not perceptual roughness). k must be based on r=sqrt(alpha).
float Material::geometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float alpha) {
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);

    const float r = glm::clamp(std::sqrt(std::max(alpha, 0.0f)), 0.02f, 1.0f);
    float ggx2 = geometrySchlickGGX(NdotV, r);
    float ggx1 = geometrySchlickGGX(NdotL, r);

    return ggx1 * ggx2;
}

// Improved Schlick's Fresnel approximation with roughness consideration
glm::vec3 Material::fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    // Enhanced Fresnel with smoother falloff
    float fresnel_power = 5.0f;
    return F0 + (1.0f - F0) * glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), fresnel_power);
}

// Enhanced Fresnel with roughness consideration for more realistic edges
glm::vec3 Material::fresnelSchlickRoughness(float cosTheta, const glm::vec3& F0, float roughness) {
    float fresnel_power = 5.0f;
    glm::vec3 F = F0 + (glm::max(glm::vec3(1.0f - roughness), F0) - F0) * 
                  glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), fresnel_power);
    return F;
}

// Complete Cook-Torrance BRDF evaluation
glm::vec3 Material::evaluateBRDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L) const {
    glm::vec3 H = glm::normalize(V + L);
    
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);
    float HdotV = glm::max(glm::dot(H, V), 0.0f);
    
    // Unify roughness/alpha semantics with GPU:
    // - perceptual roughness r in [0.02, 1]
    // - alpha = r^2
    const float r = glm::clamp(roughness, 0.02f, 1.0f);
    const float alpha = r * r;

    // Calculate Cook-Torrance BRDF components (GGX math consumes alpha).
    float D = distributionGGX(N, H, alpha);
    float G = geometrySmith(N, V, L, alpha);
    glm::vec3 F = fresnelSchlick(HdotV, getF0());
    
    // Calculate specular contribution
    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + 0.0001f; // Add small epsilon to prevent division by zero
    glm::vec3 specular = numerator / denominator;
    
    // Energy conservation: what's not reflected is refracted
    glm::vec3 kS = F; // Specular contribution
    glm::vec3 kD = glm::vec3(1.0f) - kS; // Diffuse contribution
    // NOTE: metallic suppression is already applied in getDiffuseColor().
    
    // Lambertian diffuse
    glm::vec3 diffuse = getDiffuseColor() / float(M_PI);
    
    // Combine diffuse and specular
    return (kD * diffuse + specular) * NdotL;
}

BSDFSample Material::evaluateSample(const glm::vec3& N_in, const glm::vec3& V_in, float u1, float u2) const {
    BSDFSample out{};
    out.wi = glm::vec3(0.0f);
    out.f = glm::vec3(0.0f);
    out.pdf = 0.0f;
    out.valid = false;

    // Use existing material members (baseColor == albedo in this codebase).
    const glm::vec3 baseColor = albedo;

    // Match GPU semantics: perceptual roughness r in [0.02, 1], alpha = r^2.
    const float r = glm::clamp(roughness, 0.02f, 1.0f);
    const float alpha = r * r;

    // Clamp randoms to avoid edge NaNs.
    u1 = glm::clamp(u1, 0.0f, 1.0f);
    u2 = glm::clamp(u2, 0.0f, 1.0f);

    const glm::vec3 N = glm::normalize(N_in);
    const glm::vec3 V = glm::normalize(V_in);

    const float NdotV_raw = glm::dot(N, V);
    if (NdotV_raw <= 0.0f) {
        return out;
    }

    // (a) Sample GGX VNDF half-vector H (Heitz 2014) in local space.
    // Important: alpha semantics must match GPU (alpha=r^2). No extra squaring of roughness.
    glm::vec3 T, B;
    make_onb(N, T, B);

    // Transform view direction to local frame around N.
    glm::vec3 Vh(glm::dot(V, T), glm::dot(V, B), glm::dot(V, N));
    Vh = glm::normalize(Vh);

    // Stretch view.
    glm::vec3 Vh_stretched(alpha * Vh.x, alpha * Vh.y, Vh.z);
    Vh_stretched = glm::normalize(Vh_stretched);

    // Orthonormal basis around stretched view.
    glm::vec3 T1;
    if (Vh_stretched.z < 0.9999f) {
        T1 = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), Vh_stretched));
    } else {
        T1 = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    const glm::vec3 T2 = glm::cross(Vh_stretched, T1);

    // Sample point on disk.
    const float r_disk = std::sqrt(u1);
    const float phi = 2.0f * glm::pi<float>() * u2;
    float t1 = r_disk * std::cos(phi);
    float t2 = r_disk * std::sin(phi);

    // Projected area correction (Heitz).
    const float s = 0.5f * (1.0f + Vh_stretched.z);
    t2 = (1.0f - s) * std::sqrt(std::max(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Reconstruct visible normal in stretched space.
    const float t3 = std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2));
    const glm::vec3 Nh = t1 * T1 + t2 * T2 + t3 * Vh_stretched;

    // Unstretch and normalize -> microfacet half-vector in local space.
    glm::vec3 H_local(alpha * Nh.x, alpha * Nh.y, std::max(0.0f, Nh.z));
    H_local = glm::normalize(H_local);

    // (b) Transform H to world.
    glm::vec3 H = glm::normalize(T * H_local.x + B * H_local.y + N * H_local.z);

    // (c) wi = reflect(-V, H)
    const glm::vec3 wi = glm::normalize(glm::reflect(-V, H));
    if (glm::dot(N, wi) <= 0.0f) {
        return out;
    }

    const float NdotL = std::max(glm::dot(N, wi), 0.0f);
    const float NdotH = std::max(glm::dot(N, H), 0.0f);
    const float VdotH = std::max(glm::dot(V, H), 0.0f);

    constexpr float kEps = 1e-6f;
    if (NdotL <= 0.0f || NdotH <= 0.0f || VdotH <= kEps) {
        return out;
    }

    // D: uses alpha (do not square roughness again)
    const float a2 = alpha * alpha;
    const float NdotH2 = NdotH * NdotH;
    const float d_denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    const float D = a2 / (glm::pi<float>() * d_denom * d_denom);

    // G: Schlick-GGX with k computed from r (NOT alpha)
    const float k = ((r + 1.0f) * (r + 1.0f)) / 8.0f;
    const float G = G1_SchlickGGX(std::max(NdotV_raw, 0.0f), k) * G1_SchlickGGX(NdotL, k);

    // F: Fresnel Schlick with F0 = mix(0.04, baseColor, metallic)
    const glm::vec3 F0 = glm::mix(glm::vec3(0.04f), baseColor, metallic);
    const glm::vec3 F = fresnelSchlick(VdotH, F0);

    // Cook-Torrance specular BRDF
    const float denom_f = 4.0f * std::max(NdotV_raw, kEps) * std::max(NdotL, kEps);
    const glm::vec3 f = (D * G) * F / denom_f;

    // pdf: map from half-vector to direction
    const float G1V = G1_SchlickGGX(std::max(NdotV_raw, 0.0f), kEps);
    const float pdf = (D * G1V * NdotH) /
                    (4.0f * std::max(VdotH, kEps) * std::max(NdotV_raw, kEps));
    if (!(pdf > 0.0f) || !std::isfinite(pdf)) {
        return out;
    }

    out.wi = wi;
    out.f = f;
    out.pdf = pdf;
    out.valid = true;
    return out;
}

optix::DeviceMaterial Material::toDevice() const {
    optix::DeviceMaterial dm{};
    // IMPORTANT: baseColor is the raw PBR baseColor/albedo (NOT multiplied by (1-metallic)).
    // Device code derives diffuseColor = baseColor * (1-metallic).
    const glm::vec3 bc = albedo;
    dm.baseColor = make_float3(bc.x, bc.y, bc.z);
    dm.metallic = metallic;
    dm.roughness = roughness;
    dm.ior = ior;
    dm.type = static_cast<int>(materialType);
    dm.emission = make_float3(0.0f, 0.0f, 0.0f);
    dm.pad = 0.0f;
    dm.pad2 = 0.0f;
    return dm;
}