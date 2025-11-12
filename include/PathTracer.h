#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <random>

class PathTracer {
public:
    // Path tracing parameters
    struct Settings {
        int samples_per_pixel = 16;
        int max_depth = 8;
        float min_contribution = 0.001f;
        bool enable_path_tracing = true;
    };

private:
    Settings m_settings;
    
    // Random number generation (thread local)
    thread_local static std::mt19937 s_rng;
    thread_local static std::uniform_real_distribution<float> s_uniform_dist;

    // Helper functions
    static float randomFloat();
    static float randomFloat(float min, float max);
    static glm::vec3 randomUnitSphere();
    static glm::vec3 randomUnitHemisphere(const glm::vec3& normal);
    static glm::vec3 cosineHemisphereSample(const glm::vec3& normal);
    
    // Material functions
    static glm::vec3 getColorFromGeometryID(int geomID);
    static glm::vec3 getMaterialAlbedo(int geomID);
    
    // Core tracing functions
    glm::vec3 tracePathMonteCarlo(RTCScene scene, const glm::vec3& origin, 
                                 const glm::vec3& direction, int depth) const;
    glm::vec3 traceRaySimple(RTCScene scene, const glm::vec3& origin, 
                            const glm::vec3& direction) const;

public:
    PathTracer(const Settings& settings = Settings{});
    
    // Main tracing interface
    glm::vec3 traceRay(RTCScene scene, const glm::vec3& origin, 
                      const glm::vec3& direction) const;
    
    // Settings management
    void setSettings(const Settings& settings) { m_settings = settings; }
    const Settings& getSettings() const { return m_settings; }
    
    // Utility functions
    static void initializeRandomSeed();
};