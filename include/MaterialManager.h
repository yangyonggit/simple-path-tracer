#pragma once

#include "Material.h"
#include <vector>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>

class MaterialManager {
private:
    std::vector<Material> m_materials;
    std::vector<uint32_t> m_geom_material_id;  // geomID -> materialId mapping

public:
    MaterialManager();
    ~MaterialManager() = default;
    
    // Non-copyable but movable
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
    MaterialManager(MaterialManager&&) = default;
    MaterialManager& operator=(MaterialManager&&) = default;
    
    // Material management
    void addMaterial(const Material& material) { m_materials.push_back(material); }
    void addMaterial(Material&& material) { m_materials.push_back(std::move(material)); }
    void setMaterial(int index, const Material& material);
    void setMaterial(int index, Material&& material);
    const Material& getMaterial(int index) const;
    size_t getMaterialCount() const { return m_materials.size(); }
    
    // Geometry to material ID mapping
    void setGeomMaterialMapping(const std::vector<uint32_t>& geomMaterialId) {
        m_geom_material_id = geomMaterialId;
    }
    
    // Material lookup by geometry ID
    const Material& getMaterialByID(int geomID) const;
    const Material& getMaterialFromHit(RTCScene scene, const RTCRayHit& rayhit) const;
    
    // Utility functions for geometry color mapping
    static glm::vec3 getColorFromGeometryID(int geomID);
    static glm::vec3 getMaterialAlbedo(int geomID);
    
private:
    void setupDefaultMaterials();
};