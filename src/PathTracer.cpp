#include "PathTracer.h"
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Thread local static member definitions
thread_local std::mt19937 PathTracer::s_rng(std::random_device{}());
thread_local std::uniform_real_distribution<float> PathTracer::s_uniform_dist(0.0f, 1.0f);

PathTracer::PathTracer(const Settings& settings) : m_settings(settings) {
}

void PathTracer::initializeRandomSeed() {
    s_rng.seed(std::random_device{}());
}

float PathTracer::randomFloat() {
    return s_uniform_dist(s_rng);
}

float PathTracer::randomFloat(float min, float max) {
    return min + (max - min) * randomFloat();
}

glm::vec3 PathTracer::randomUnitSphere() {
    glm::vec3 p;
    do {
        p = 2.0f * glm::vec3(randomFloat(), randomFloat(), randomFloat()) - glm::vec3(1.0f);
    } while (glm::dot(p, p) >= 1.0f);
    return p;
}

glm::vec3 PathTracer::randomUnitHemisphere(const glm::vec3& normal) {
    glm::vec3 in_unit_sphere = randomUnitSphere();
    if (glm::dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

glm::vec3 PathTracer::cosineHemisphereSample(const glm::vec3& normal) {
    // Generate random numbers
    float u1 = randomFloat();
    float u2 = randomFloat();
    
    // Convert to spherical coordinates with cosine weighting
    float phi = 2.0f * M_PI * u1;
    float cos_theta = sqrt(u2);
    float sin_theta = sqrt(1.0f - u2);
    
    // Local coordinates (z-up)
    glm::vec3 local_dir(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    
    // Create orthonormal basis around normal
    glm::vec3 nt = (abs(normal.x) > 0.1f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 u = glm::normalize(glm::cross(normal, nt));
    glm::vec3 v = glm::cross(normal, u);
    
    // Transform to world space
    return local_dir.x * u + local_dir.y * v + local_dir.z * normal;
}

glm::vec3 PathTracer::getColorFromGeometryID(int geomID) {
    switch(geomID) {
        case 0: return glm::vec3(0.8f, 0.8f, 0.8f); // Ground plane - Light gray
        case 1: return glm::vec3(1.0f, 0.2f, 0.2f); // Test box - Red
        case 2: return glm::vec3(0.2f, 1.0f, 0.2f); // Cube - Green
        case 3: return glm::vec3(0.2f, 0.2f, 1.0f); // Sphere - Blue
        default: return glm::vec3(0.0f, 0.0f, 0.0f); // Background - Black
    }
}

glm::vec3 PathTracer::getMaterialAlbedo(int geomID) {
    switch(geomID) {
        case 0: return glm::vec3(0.7f, 0.7f, 0.7f); // Ground plane - Light gray diffuse
        case 1: return glm::vec3(0.8f, 0.3f, 0.3f); // Test box - Red diffuse
        case 2: return glm::vec3(0.3f, 0.8f, 0.3f); // Cube - Green diffuse
        case 3: return glm::vec3(0.3f, 0.3f, 0.8f); // Sphere - Blue diffuse
        default: return glm::vec3(0.0f, 0.0f, 0.0f); // Background - Black
    }
}

glm::vec3 PathTracer::tracePathMonteCarlo(RTCScene scene, const glm::vec3& origin, 
                                         const glm::vec3& direction, int depth) const {
    // Stop if we've reached max depth
    if (depth <= 0) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    RTCRayHit rayhit;
    
    // Set ray origin and direction
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    
    // Set ray parameters
    rayhit.ray.tnear = 0.001f; // Small offset to avoid self-intersection
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray intersection
    rtcIntersect1(scene, &rayhit);

    // If no hit, return background color (sky)
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        // Simple sky gradient
        float t = 0.5f * (glm::normalize(direction).y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }

    // Calculate hit point
    glm::vec3 hit_point = origin + rayhit.ray.tfar * direction;
    
    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);
    
    // Make sure normal faces the incoming ray
    if (glm::dot(normal, direction) > 0.0f) {
        normal = -normal;
    }

    // Get material properties
    glm::vec3 albedo = getMaterialAlbedo(rayhit.hit.geomID);
    
    // Simple emission for light sources (none in this scene)
    glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // Generate random direction using cosine-weighted hemisphere sampling
    glm::vec3 scatter_direction = cosineHemisphereSample(normal);
    
    // Recursive ray tracing for indirect lighting
    glm::vec3 indirect_light = tracePathMonteCarlo(scene, hit_point, scatter_direction, depth - 1);
    
    // BRDF for Lambertian diffuse: albedo / π
    // The cosine term is already accounted for in the cosine-weighted sampling
    // The π factor cancels out with the π in the Monte Carlo estimator
    return emission + albedo * indirect_light;
}

glm::vec3 PathTracer::traceRaySimple(RTCScene scene, const glm::vec3& origin, 
                                    const glm::vec3& direction) const {
    RTCRayHit rayhit;
    
    // Set ray origin and direction
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    
    // Set ray parameters
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray intersection
    rtcIntersect1(scene, &rayhit);

    // If no hit, return background color
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        float t = 0.5f * (glm::normalize(direction).y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }

    // Get hit normal from Embree (geometric normal)
    glm::vec3 normal(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    normal = glm::normalize(normal);

    // Simple directional light
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
    float diffuseIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

    // Get material color
    glm::vec3 baseColor = getMaterialAlbedo(rayhit.hit.geomID);
    return baseColor * diffuseIntensity;
}

glm::vec3 PathTracer::traceRay(RTCScene scene, const glm::vec3& origin, 
                              const glm::vec3& direction) const {
    if (!m_settings.enable_path_tracing) {
        return traceRaySimple(scene, origin, direction);
    }
    
    glm::vec3 color(0.0f, 0.0f, 0.0f);
    
    // Shoot multiple samples and average
    for (int s = 0; s < m_settings.samples_per_pixel; ++s) {
        color += tracePathMonteCarlo(scene, origin, direction, m_settings.max_depth);
    }
    
    // Average the samples
    color /= float(m_settings.samples_per_pixel);
    
    // Simple tone mapping (gamma correction)
    color = glm::sqrt(color);
    
    return color;
}