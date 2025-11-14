#include "MaterialManager.h"
#include <iostream>
#include <algorithm>

MaterialManager::MaterialManager() {
    setupDefaultMaterials();
}

void MaterialManager::setupDefaultMaterials() {
    // Clear existing materials
    m_materials.clear();
    m_materials.reserve(10); // Reserve space for common materials
    
    // Material 0: Gold (metallic, warm color)
    m_materials.push_back(Materials::Gold());
    
    // Material 1: Silver (metallic, cool color) 
    m_materials.push_back(Materials::Silver());
    
    // Material 2: Copper (metallic, orange-red color)
    m_materials.push_back(Materials::Copper());
    
    // Material 3: Iron (metallic, dark gray)
    m_materials.push_back(Materials::Iron());
    
    // Material 4: Glass (transparent dielectric)
    m_materials.push_back(Materials::Glass());
    
    // Material 5: Plastic (smooth dielectric, various colors)
    m_materials.push_back(Materials::Plastic());
    
    // Material 6: Rubber (rough dielectric)
    m_materials.push_back(Materials::Rubber());
    
    // Material 7: Wood (diffuse, natural color)
    m_materials.push_back(Materials::Wood());
    
    // Material 8: Concrete (rough diffuse, gray)
    m_materials.push_back(Materials::Concrete());
}

void MaterialManager::setMaterial(int index, const Material& material) {
    if (index >= 0 && index < static_cast<int>(m_materials.size())) {
        m_materials[index] = material;
    } else {
        std::cerr << "Material index " << index << " out of range" << std::endl;
    }
}

void MaterialManager::setMaterial(int index, Material&& material) {
    if (index >= 0 && index < static_cast<int>(m_materials.size())) {
        m_materials[index] = std::move(material);
    } else {
        std::cerr << "Material index " << index << " out of range" << std::endl;
    }
}

const Material& MaterialManager::getMaterial(int index) const {
    if (index >= 0 && index < static_cast<int>(m_materials.size())) {
        return m_materials[index];
    } else {
        std::cerr << "Material index " << index << " out of range, using material 0" << std::endl;
        return m_materials[0]; // Return default material
    }
}

const Material& MaterialManager::getMaterialByID(int geomID) const {
    // Map geometry ID to material index
    // For spheres: geomID 0-8 maps to materials 0-8
    if (geomID >= 0 && geomID < static_cast<int>(m_materials.size())) {
        return m_materials[geomID];
    }
    
    // For other geometry, use modulo to wrap around available materials
    int materialIndex = geomID % static_cast<int>(m_materials.size());
    return m_materials[materialIndex];
}

const Material& MaterialManager::getMaterialFromHit(RTCScene scene, const RTCRayHit& rayhit) const {
    // Try to get material from geometry user data first
    RTCGeometry geometry = rtcGetGeometry(scene, rayhit.hit.geomID);
    if (geometry) {
        void* userData = rtcGetGeometryUserData(geometry);
        if (userData) {
            // For spheres, userData points to SphereData which contains materialID
            // For meshes, userData contains the material ID as uintptr_t
            // This is a simplified approach - in a real system you'd have a more robust method
            uintptr_t materialID = reinterpret_cast<uintptr_t>(userData);
            if (materialID < m_materials.size()) {
                return m_materials[materialID];
            }
        }
    }
    
    // Fallback: use geometry ID to determine material
    return getMaterialByID(rayhit.hit.geomID);
}

glm::vec3 MaterialManager::getColorFromGeometryID(int geomID) {
    // Map geometry IDs to distinct colors for visualization
    switch (geomID) {
        case 0: return glm::vec3(1.0f, 0.843f, 0.0f);   // Gold
        case 1: return glm::vec3(0.753f, 0.753f, 0.753f); // Silver
        case 2: return glm::vec3(0.722f, 0.451f, 0.200f); // Copper
        case 3: return glm::vec3(0.329f, 0.329f, 0.329f); // Iron
        case 4: return glm::vec3(0.9f, 0.9f, 1.0f);     // Glass (slight blue tint)
        case 5: return glm::vec3(0.8f, 0.2f, 0.2f);     // Red plastic
        case 6: return glm::vec3(0.1f, 0.1f, 0.1f);     // Black rubber
        case 7: return glm::vec3(0.545f, 0.271f, 0.075f); // Wood (saddle brown)
        case 8: return glm::vec3(0.5f, 0.5f, 0.5f);     // Concrete
        default: 
            // Generate a color based on ID for unknown geometries
            float hue = (geomID * 137.508f) / 360.0f; // Use golden angle for distribution
            hue = hue - floorf(hue); // Keep fractional part
            // Simple HSV to RGB conversion for hue with full saturation and value
            float c = 1.0f;
            float x = c * (1.0f - abs(fmod(hue * 6.0f, 2.0f) - 1.0f));
            float m = 0.0f;
            
            if (hue < 1.0f/6.0f) return glm::vec3(c, x, 0.0f);
            else if (hue < 2.0f/6.0f) return glm::vec3(x, c, 0.0f);
            else if (hue < 3.0f/6.0f) return glm::vec3(0.0f, c, x);
            else if (hue < 4.0f/6.0f) return glm::vec3(0.0f, x, c);
            else if (hue < 5.0f/6.0f) return glm::vec3(x, 0.0f, c);
            else return glm::vec3(c, 0.0f, x);
    }
}

glm::vec3 MaterialManager::getMaterialAlbedo(int geomID) {
    // This is a simplified version - in practice you'd want to get this from the actual material
    return getColorFromGeometryID(geomID);
}