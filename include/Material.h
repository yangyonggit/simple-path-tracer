#pragma once

#include <glm/glm.hpp>

#include "optix/LaunchParams.h"

struct BSDFSample {
    glm::vec3 wi;   // sampled incident direction (world space)
    glm::vec3 f;    // BSDF value
    float pdf;      // pdf(wi)
    bool valid;
};

enum class MaterialType : int {
    PBR = optix::MATERIAL_TYPE_PBR,
    DIELECTRIC = optix::MATERIAL_TYPE_DIELECTRIC,
};

struct Material {
    glm::vec3 albedo;       // Base color (diffuse reflectance)
    float metallic;         // Metallic factor [0.0, 1.0]
    float roughness;        // Roughness factor [0.0, 1.0] 
    glm::vec3 emission;     // Emissive color
    float ior;              // Index of refraction (for dielectrics)
    MaterialType materialType = MaterialType::PBR;
    
    // Constructor with default values
    Material(const glm::vec3& albedo_ = glm::vec3(0.5f, 0.5f, 0.5f),
             float metallic_ = 0.0f,
             float roughness_ = 0.5f,
             const glm::vec3& emission_ = glm::vec3(0.0f, 0.0f, 0.0f),
                         float ior_ = 1.5f,
                         MaterialType type_ = MaterialType::PBR)
        : albedo(albedo_), metallic(metallic_), roughness(roughness_), 
                    emission(emission_), ior(ior_), materialType(type_) {
        // Clamp values to valid ranges
        metallic = glm::clamp(metallic, 0.0f, 1.0f);
        roughness = glm::clamp(roughness, 0.01f, 1.0f); // Avoid zero roughness
    }
    
    // Get base reflectivity (F0) for Cook-Torrance
    glm::vec3 getF0() const {
        // For metals: F0 = albedo
        // For dielectrics: F0 = ((ior - 1) / (ior + 1))^2
        float dielectric_f0 = (ior - 1.0f) / (ior + 1.0f);
        dielectric_f0 *= dielectric_f0;
        
        return glm::mix(glm::vec3(dielectric_f0), albedo, metallic);
    }
    
    // Get diffuse color (affected by metallic)
    glm::vec3 getDiffuseColor() const {
        return albedo * (1.0f - metallic);
    }
    
    // Check if material is emissive
    bool isEmissive() const {
        return glm::length(emission) > 0.0f;
    }
    
    // Check if material is transparent (glass-like)
    bool isTransparent() const {
        // Material is transparent if it's not metallic and has high IOR
        return metallic < 0.1f && ior > 1.3f;
    }
    
    // Get transparency factor
    float getTransparency() const {
        if (isTransparent()) {
            // Higher IOR = more transparent for glass-like materials
            return glm::clamp((ior - 1.0f) / 0.7f, 0.0f, 0.95f);
        }
        return 0.0f;
    }
    
    // PBR Cook-Torrance BRDF calculations
    static float distributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness);
    static float geometrySchlickGGX(float NdotV, float roughness);
    static float geometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness);
    static glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0);
    static glm::vec3 fresnelSchlickRoughness(float cosTheta, const glm::vec3& F0, float roughness);
    
    // Main BRDF evaluation function
    glm::vec3 evaluateBRDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L) const;

    // Sampling-based GGX interface for CPU path tracing (specular only; no diffuse).
    BSDFSample evaluateSample(
        const glm::vec3& N,
        const glm::vec3& V,
        float u1,
        float u2
    ) const;

    // Export minimal GPU material (no BRDF logic on GPU yet)
    optix::DeviceMaterial toDevice() const;
};

// Material library with predefined materials
namespace Materials {
    // Metals
    inline Material Gold() {
        return Material(glm::vec3(1.0f, 0.71f, 0.29f), 1.0f, 0.05f); // Smoother for more reflections
    }
    
    inline Material Silver() {
        return Material(glm::vec3(0.95f, 0.93f, 0.88f), 1.0f, 0.02f); // Very smooth
    }
    
    inline Material Copper() {
        return Material(glm::vec3(0.95f, 0.64f, 0.54f), 1.0f, 0.08f); // Slightly smoother
    }
    
    inline Material Iron() {
        return Material(glm::vec3(0.56f, 0.57f, 0.58f), 1.0f, 0.3f);
    }
    
    // Dielectrics
    inline Material Plastic() {
        return Material(glm::vec3(0.8f, 0.2f, 0.2f), 0.0f, 0.4f, glm::vec3(0.0f), 1.2f); // Lower IOR
    }
    
    inline Material Rubber() {
        return Material(glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 0.8f, glm::vec3(0.0f), 1.1f); // Lower IOR
    }
    
    inline Material Glass() {
        return Material(glm::vec3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f), 1.5f, MaterialType::DIELECTRIC);
    }
    
    // High-quality clear glass with slight blue tint
    inline Material ClearGlass() {
        return Material(glm::vec3(0.95f, 0.98f, 1.0f), 0.0f, 0.02f, glm::vec3(0.0f), 1.5f, MaterialType::DIELECTRIC);
    }
    
    // Mixed materials
    inline Material Wood() {
        return Material(glm::vec3(0.4f, 0.25f, 0.1f), 0.0f, 0.7f, glm::vec3(0.0f), 1.0f); // No refraction
    }
    
    inline Material Concrete() {
        return Material(glm::vec3(0.6f, 0.6f, 0.6f), 0.0f, 0.9f, glm::vec3(0.0f), 1.0f); // No refraction
    }
    
    // Emissive material
    inline Material Light(const glm::vec3& color = glm::vec3(1.0f), float intensity = 5.0f) {
        return Material(glm::vec3(0.0f), 0.0f, 1.0f, color * intensity);
    }
}