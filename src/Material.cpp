#include "Material.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GGX/Trowbridge-Reitz Normal Distribution Function
float Material::distributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
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
float Material::geometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness) {
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Schlick's Fresnel approximation
glm::vec3 Material::fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    return F0 + (1.0f - F0) * glm::pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

// Complete Cook-Torrance BRDF evaluation
glm::vec3 Material::evaluateBRDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L) const {
    glm::vec3 H = glm::normalize(V + L);
    
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);
    float HdotV = glm::max(glm::dot(H, V), 0.0f);
    
    // Calculate Cook-Torrance BRDF components
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    glm::vec3 F = fresnelSchlick(HdotV, getF0());
    
    // Calculate specular contribution
    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + 0.0001f; // Add small epsilon to prevent division by zero
    glm::vec3 specular = numerator / denominator;
    
    // Energy conservation: what's not reflected is refracted
    glm::vec3 kS = F; // Specular contribution
    glm::vec3 kD = glm::vec3(1.0f) - kS; // Diffuse contribution
    kD *= 1.0f - metallic; // Metals don't have diffuse
    
    // Lambertian diffuse
    glm::vec3 diffuse = getDiffuseColor() / float(M_PI);
    
    // Combine diffuse and specular
    return (kD * diffuse + specular) * NdotL;
}