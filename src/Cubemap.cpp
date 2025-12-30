#include "Cubemap.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Cubemap::Cubemap() : m_size(0), m_loaded(false) {
}

Cubemap::~Cubemap() {
    // Data will be automatically cleaned up by vectors
}

bool Cubemap::loadFromFile(const std::string& filename) {
    // Load image using stb_image
    int width, height, channels;
    float* data = stbi_loadf(filename.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        std::cerr << "Failed to load cubemap: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Loaded environment map: " << filename << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;
    
    // Auto-detect format based on aspect ratio and file extension
    float aspect = (float)width / (float)height;
    std::string ext = filename.substr(filename.find_last_of('.'));
    
    // HDR files are typically equirectangular (2:1 ratio)
    // Cross layout is typically 4:3 ratio and used for LDR cubemaps
    if (ext == ".hdr" || ext == ".exr" || abs(aspect - 2.0f) < 0.1f) {
        std::cout << "Detected HDR equirectangular environment map (aspect: " << aspect << ")" << std::endl;
        return loadEquirectangular(data, width, height);
    } else if (abs(aspect - 4.0f/3.0f) < 0.1f) {
        std::cout << "Detected cross layout cubemap (aspect: " << aspect << ")" << std::endl;
        return loadCrossLayout(data, width, height);
    } else {
        std::cout << "Unknown format, trying equirectangular (aspect: " << aspect << ")" << std::endl;
        return loadEquirectangular(data, width, height);
    }
}

bool Cubemap::loadFromFiles(const std::string faces[6]) {
    // Load 6 separate face files
    for (int i = 0; i < 6; i++) {
        int width, height, channels;
        float* data = stbi_loadf(faces[i].c_str(), &width, &height, &channels, 3);
        
        if (!data) {
            std::cerr << "Failed to load cubemap face " << i << ": " << faces[i] << std::endl;
            return false;
        }
        
        m_faces[i].width = width;
        m_faces[i].height = height;
        m_faces[i].data.resize(width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                m_faces[i].data[y * width + x] = glm::vec3(
                    data[idx + 0],
                    data[idx + 1],
                    data[idx + 2]
                );
            }
        }
        
        stbi_image_free(data);
    }
    
    m_size = m_faces[0].width;
    m_loaded = true;
    return true;
}

glm::vec3 Cubemap::sample(const glm::vec3& direction) const {
    if (!m_loaded) {
        return glm::vec3(0.5f, 0.7f, 1.0f); // Default sky color
    }
    
    int face;
    float u, v;
    directionToUV(direction, face, u, v);
    
    return bilinearSample(m_faces[face], u, v);
}

void Cubemap::directionToUV(const glm::vec3& dir, int& face, float& u, float& v) const {
    glm::vec3 d = glm::normalize(dir);
    float absX = abs(d.x);
    float absY = abs(d.y);
    float absZ = abs(d.z);
    
    float maxAxis, uc, vc;
    
    if (absX >= absY && absX >= absZ) {
        // X face
        maxAxis = absX;
        if (d.x > 0) {
            // +X face
            face = 0;
            uc = -d.z;
            vc = -d.y;
        } else {
            // -X face  
            face = 1;
            uc = d.z;
            vc = -d.y;
        }
    } else if (absY >= absX && absY >= absZ) {
        // Y face
        maxAxis = absY;
        if (d.y > 0) {
            // +Y face
            face = 2;
            uc = d.x;
            vc = d.z;
        } else {
            // -Y face
            face = 3; 
            uc = d.x;
            vc = -d.z;
        }
    } else {
        // Z face
        maxAxis = absZ;
        if (d.z > 0) {
            // +Z face
            face = 4;
            uc = d.x;
            vc = -d.y;
        } else {
            // -Z face
            face = 5;
            uc = -d.x;
            vc = -d.y;
        }
    }
    
    // Convert range from [-1, 1] to [0, 1]
    u = (uc / maxAxis + 1.0f) * 0.5f;
    v = (vc / maxAxis + 1.0f) * 0.5f;
    
    // Clamp to [0, 1]
    u = glm::clamp(u, 0.0f, 1.0f);
    v = glm::clamp(v, 0.0f, 1.0f);
}

glm::vec3 Cubemap::bilinearSample(const Face& face, float u, float v) const {
    // Convert UV to pixel coordinates
    float x = u * (face.width - 1);
    float y = v * (face.height - 1);
    
    // Get integer parts
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = std::min(x0 + 1, face.width - 1);
    int y1 = std::min(y0 + 1, face.height - 1);
    
    // Get fractional parts
    float fx = x - x0;
    float fy = y - y0;
    
    // Sample four pixels
    glm::vec3 c00 = face.data[y0 * face.width + x0];
    glm::vec3 c10 = face.data[y0 * face.width + x1];
    glm::vec3 c01 = face.data[y1 * face.width + x0];
    glm::vec3 c11 = face.data[y1 * face.width + x1];
    
    // Bilinear interpolation
    glm::vec3 c0 = glm::mix(c00, c10, fx);
    glm::vec3 c1 = glm::mix(c01, c11, fx);
    return glm::mix(c0, c1, fy);
}

bool Cubemap::loadCrossLayout(float* data, int width, int height) {
    // No equirect source for cross-layout cubemaps.
    m_equirect_loaded = false;
    m_equirect_width = 0;
    m_equirect_height = 0;
    m_equirect_rgba.clear();

    // Cross layout:
    //     [+Y]
    // [+X][+Z][-X][-Z]  
    //     [-Y]
    
    int faceSize = width / 4; // Each face is width/4 x height/3
    if (faceSize != height / 3) {
        std::cerr << "Invalid cross layout proportions" << std::endl;
        return false;
    }
    
    m_size = faceSize;
    
    // Define face positions in the cross layout
    struct FaceOffset {
        int x, y;
    };
    
    FaceOffset offsets[6] = {
        {2, 1}, // +X (right)  - position (2,1)
        {0, 1}, // -X (left)   - position (0,1)
        {1, 0}, // +Y (top)    - position (1,0)
        {1, 2}, // -Y (bottom) - position (1,2)
        {1, 1}, // +Z (front)  - position (1,1)
        {3, 1}  // -Z (back)   - position (3,1)
    };
    
    // Extract each face from the cross layout
    for (int face = 0; face < 6; face++) {
        m_faces[face].width = faceSize;
        m_faces[face].height = faceSize;
        m_faces[face].data.resize(faceSize * faceSize);
        
        int startX = offsets[face].x * faceSize;
        int startY = offsets[face].y * faceSize;
        
        for (int y = 0; y < faceSize; y++) {
            for (int x = 0; x < faceSize; x++) {
                int srcX = startX + x;
                int srcY = startY + y;
                
                // Check bounds to prevent reading outside image
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    int srcIdx = (srcY * width + srcX) * 3;
                    
                    m_faces[face].data[y * faceSize + x] = glm::vec3(
                        data[srcIdx + 0],
                        data[srcIdx + 1],
                        data[srcIdx + 2]
                    );
                } else {
                    // Fill with magenta to spot errors
                    m_faces[face].data[y * faceSize + x] = glm::vec3(1.0f, 0.0f, 1.0f);
                }
            }
        }
    }
    
    stbi_image_free(data);
    m_loaded = true;
    return true;
}

bool Cubemap::loadEquirectangular(float* data, int width, int height) {
    // Retain the original equirectangular pixels for GPU sampling.
    // stb_image loads RGB float32 (3 channels) due to stbi_loadf(..., 3).
    m_equirect_loaded = true;
    m_equirect_width = width;
    m_equirect_height = height;
    m_equirect_rgba.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int srcIdx = (y * width + x) * 3;
            const size_t dstIdx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
            m_equirect_rgba[dstIdx + 0] = data[srcIdx + 0];
            m_equirect_rgba[dstIdx + 1] = data[srcIdx + 1];
            m_equirect_rgba[dstIdx + 2] = data[srcIdx + 2];
            m_equirect_rgba[dstIdx + 3] = 1.0f;
        }
    }
    m_equirect_revision++;

    // Convert equirectangular to cubemap faces
    m_size = 512; // Fixed size for generated cube faces
    
    for (int face = 0; face < 6; face++) {
        m_faces[face].width = m_size;
        m_faces[face].height = m_size;
        m_faces[face].data.resize(m_size * m_size);
        
        // Generate each face by sampling the equirectangular image
        for (int y = 0; y < m_size; y++) {
            for (int x = 0; x < m_size; x++) {
                // Convert face pixel to 3D direction
                glm::vec3 direction = faceCoordToDirection(face, x, y, m_size);
                
                // Convert 3D direction to equirectangular UV coordinates
                float theta = atan2(direction.z, direction.x); // Azimuth
                float phi = acos(direction.y); // Elevation
                
                // Convert to UV coordinates [0,1]
                float u = (theta + M_PI) / (2.0f * M_PI);
                float v = phi / M_PI;
                
                // Sample from equirectangular image with bounds checking
                int srcX = glm::clamp((int)(u * width), 0, width - 1);
                int srcY = glm::clamp((int)(v * height), 0, height - 1);
                int srcIdx = (srcY * width + srcX) * 3;
                
                m_faces[face].data[y * m_size + x] = glm::vec3(
                    data[srcIdx + 0],
                    data[srcIdx + 1],
                    data[srcIdx + 2]
                );
            }
        }
    }
    
    stbi_image_free(data);
    m_loaded = true;
    return true;
}

// Helper function to convert face coordinates to 3D direction
glm::vec3 Cubemap::faceCoordToDirection(int face, int x, int y, int size) const {
    // Convert pixel coordinates to [-1, 1] range
    float u = (2.0f * x / (size - 1)) - 1.0f;
    float v = (2.0f * y / (size - 1)) - 1.0f;
    
    glm::vec3 direction;
    
    switch (face) {
        case 0: // +X
            direction = glm::vec3(1.0f, -v, -u);
            break;
        case 1: // -X
            direction = glm::vec3(-1.0f, -v, u);
            break;
        case 2: // +Y
            direction = glm::vec3(u, 1.0f, v);
            break;
        case 3: // -Y
            direction = glm::vec3(u, -1.0f, -v);
            break;
        case 4: // +Z
            direction = glm::vec3(u, -v, 1.0f);
            break;
        case 5: // -Z
            direction = glm::vec3(-u, -v, -1.0f);
            break;
        default:
            direction = glm::vec3(0.0f, 1.0f, 0.0f);
            break;
    }
    
    return glm::normalize(direction);
}