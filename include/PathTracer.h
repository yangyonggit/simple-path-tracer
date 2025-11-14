#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <random>
#include <vector>
#include <atomic>
#include "Light.h"
#include "MaterialManager.h"
#include "EnvironmentManager.h"

// Forward declaration
class Camera;

class PathTracer {
public:
    // Path tracing parameters
    struct Settings {
        int samples_per_pixel = 4;
        int max_depth = 4;
        float min_contribution = 0.001f;
        bool enable_path_tracing = true;
    };

private:
    Settings m_settings;
    LightManager m_light_manager;
    MaterialManager m_material_manager;
    EnvironmentManager m_environment_manager;
    
    // Random number generation (thread local)
    thread_local static std::mt19937 s_rng;
    thread_local static std::uniform_real_distribution<float> s_uniform_dist;

    // Helper functions
    static float randomFloat();
    static float randomFloat(float min, float max);
    static glm::vec3 randomUnitSphere();
    static glm::vec3 randomUnitHemisphere(const glm::vec3& normal);
    static glm::vec3 cosineHemisphereSample(const glm::vec3& normal);
    
    // Core ray intersection and tracing helpers
    bool intersectRay(RTCScene scene, const glm::vec3& origin, const glm::vec3& direction,
                     RTCRayHit& rayhit) const;
    glm::vec3 calculateSafeRayOrigin(const glm::vec3& hit_point, const glm::vec3& normal, 
                                    bool offset_forward = true) const;
    
    // Core tracing functions
    glm::vec3 tracePathMonteCarlo(RTCScene scene, const glm::vec3& origin, 
                                 const glm::vec3& direction, int depth) const;
    glm::vec3 traceRaySimple(RTCScene scene, const glm::vec3& origin, 
                            const glm::vec3& direction) const;
    
    // Path tracing helper functions
    glm::vec3 calculateDirectLighting(RTCScene scene, const glm::vec3& hit_point,
                                     const glm::vec3& normal, const glm::vec3& view_dir,
                                     const Material& material, int depth) const;
    glm::vec3 calculateMetallicIndirectLighting(RTCScene scene, const glm::vec3& hit_point,
                                               const glm::vec3& normal, const glm::vec3& view_dir,
                                               const Material& material, int depth) const;
    glm::vec3 calculateTransparentIndirectLighting(RTCScene scene, const glm::vec3& hit_point,
                                                  const glm::vec3& normal, const glm::vec3& view_dir,
                                                  const Material& material, int depth) const;
    glm::vec3 calculateDielectricIndirectLighting(RTCScene scene, const glm::vec3& hit_point,
                                                 const glm::vec3& normal, const glm::vec3& view_dir,
                                                 const Material& material, int depth) const;

public:
    PathTracer(const Settings& settings = Settings{});
    
    // Main tracing interface
    glm::vec3 traceRay(RTCScene scene, const glm::vec3& origin, 
                      const glm::vec3& direction) const;
    
    // Render entire image using parallel tile processing
    void renderImage(std::vector<unsigned char>& pixels, int width, int height,
                    const Camera& camera, RTCScene scene,
                    std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                    bool camera_moved, std::atomic<int>& tiles_completed) const;
    
    // Settings management
    void setSettings(const Settings& settings) { m_settings = settings; }
    const Settings& getSettings() const { return m_settings; }
    
    // Light management
    LightManager& getLightManager() { return m_light_manager; }
    const LightManager& getLightManager() const { return m_light_manager; }
    
    // Material management
    MaterialManager& getMaterialManager() { return m_material_manager; }
    const MaterialManager& getMaterialManager() const { return m_material_manager; }
    
    // Environment management
    EnvironmentManager& getEnvironmentManager() { return m_environment_manager; }
    const EnvironmentManager& getEnvironmentManager() const { return m_environment_manager; }
    
    // Utility functions
    static void initializeRandomSeed();
    
    // Helper functions for transparency and refraction
    float schlickFresnel(float cosine, float ior) const;
    glm::vec3 refract(const glm::vec3& incident, const glm::vec3& normal, float eta) const;
    bool shouldTransmit(const Material& material, float cosine) const;
    
    // Tile rendering function
    static void renderTileTask(int tileIndex, int threadIndex, std::vector<unsigned char>& pixels,
                              int width, int height, const class Camera& camera, RTCScene scene,
                              const PathTracer& pathTracer, int numTilesX, int numTilesY,
                              std::vector<glm::vec3>& accumulation_buffer, int accumulated_samples,
                              bool camera_moved, std::atomic<int>& tiles_completed);
};