#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

class Cubemap {
public:
    Cubemap();
    ~Cubemap();

    // Load a single image and treat it as a cross-layout cubemap
    // or load 6 separate face images
    bool loadFromFile(const std::string& filename);
    bool loadFromFiles(const std::string faces[6]); // +X, -X, +Y, -Y, +Z, -Z
    
    // Sample the cubemap using a 3D direction vector
    glm::vec3 sample(const glm::vec3& direction) const;
    
    // Check if cubemap is loaded
    bool isLoaded() const { return m_loaded; }

    // Equirectangular source (for GPU sampling)
    bool hasEquirectangular() const { return m_equirect_loaded; }
    int getEquirectWidth() const { return m_equirect_width; }
    int getEquirectHeight() const { return m_equirect_height; }
    const float* getEquirectRGBA() const { return m_equirect_rgba.empty() ? nullptr : m_equirect_rgba.data(); }
    uint64_t getEquirectRevision() const { return m_equirect_revision; }
    
    // Get cubemap dimensions
    int getSize() const { return m_size; }

private:
    struct Face {
        std::vector<glm::vec3> data;
        int width, height;
    };
    
    Face m_faces[6]; // +X, -X, +Y, -Y, +Z, -Z
    int m_size;
    bool m_loaded;

    // Retained equirectangular HDR pixels in RGBA float32 (size = w*h*4)
    bool m_equirect_loaded = false;
    int m_equirect_width = 0;
    int m_equirect_height = 0;
    std::vector<float> m_equirect_rgba;
    uint64_t m_equirect_revision = 0;
    
    // Helper functions
    glm::vec3 sampleFace(int face, float u, float v) const;
    void directionToUV(const glm::vec3& dir, int& face, float& u, float& v) const;
    glm::vec3 bilinearSample(const Face& face, float u, float v) const;
    glm::vec3 faceCoordToDirection(int face, int x, int y, int size) const;
    
    // Cross layout and equirectangular loading
    bool loadCrossLayout(float* data, int width, int height);
    bool loadEquirectangular(float* data, int width, int height);
};