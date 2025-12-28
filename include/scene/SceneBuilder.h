#pragma once

#include "SceneDesc.h"

namespace scene {

// ========================================
// Build a default test scene
// ========================================
// Creates a scene with ground plane, cubes, spheres, and at least one emissive object
// This scene mimics the layout from EmbreeScene but uses the backend-agnostic SceneDesc
SceneDesc BuildDefaultScene();

} // namespace scene
