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
    
    // Helper functions
    glm::vec3 sampleFace(int face, float u, float v) const;
    void directionToUV(const glm::vec3& dir, int& face, float& u, float& v) const;
    glm::vec3 bilinearSample(const Face& face, float u, float v) const;
    glm::vec3 faceCoordToDirection(int face, int x, int y, int size) const;
    
    // Cross layout and equirectangular loading
    bool loadCrossLayout(float* data, int width, int height);
    bool loadEquirectangular(float* data, int width, int height);
};