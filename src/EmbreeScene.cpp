#include "EmbreeScene.h"
#include <iostream>
#include <cmath>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

EmbreeScene::EmbreeScene() : m_device(nullptr), m_scene(nullptr) {
    initialize();
}

EmbreeScene::~EmbreeScene() {
    cleanup();
}

void EmbreeScene::initialize() {
    // Create Embree device
    m_device = rtcNewDevice(nullptr);
    if (!m_device) {
        std::cerr << "Failed to create Embree device!\n";
        return;
    }
    
    // Create scene
    m_scene = rtcNewScene(m_device);
    if (!m_scene) {
        std::cerr << "Failed to create Embree scene!\n";
        rtcReleaseDevice(m_device);
        m_device = nullptr;
        return;
    }
    
    // Add geometries to scene
    addGroundPlane();
    addTestBox();
    addCube();
    addSphere();
    
    // Commit scene for ray tracing
    rtcCommitScene(m_scene);
    
    std::cout << "EmbreeScene initialized successfully with ground plane, box, cube, and sphere.\n";
}

void EmbreeScene::addGroundPlane() {
    // Create a quad (ground plane) using two triangles
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Define quad vertices (4 vertices for ground plane)
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 4);
    
    // Ground plane vertices (Y = -1.0, large quad)
    vertices[0] = -3.0f; vertices[1] = -1.0f; vertices[2] = -3.0f;  // vertex 0
    vertices[3] = 3.0f;  vertices[4] = -1.0f; vertices[5] = -3.0f;  // vertex 1
    vertices[6] = 3.0f;  vertices[7] = -1.0f; vertices[8] = 3.0f;   // vertex 2
    vertices[9] = -3.0f; vertices[10] = -1.0f; vertices[11] = 3.0f; // vertex 3

    // Define triangle indices (2 triangles forming a quad)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 2);
    
    // Triangle 1: vertices 0, 1, 2
    indices[0] = 0; indices[1] = 1; indices[2] = 2;
    // Triangle 2: vertices 0, 2, 3
    indices[3] = 0; indices[4] = 2; indices[5] = 3;

    // Commit geometry and attach to scene
    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::addTestBox() {
    // Create a simple box using 12 triangles (6 faces * 2 triangles each)
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Define box vertices (8 vertices for a cube)
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 8);
    
    float box_size = 0.3f;
    float box_height = 0.15f;
    float pos_x = -1.0f; // Position to the left
    
    // Bottom face vertices
    vertices[0] = pos_x - box_size; vertices[1] = -1.0f + box_height; vertices[2] = -box_size;  // 0
    vertices[3] = pos_x + box_size; vertices[4] = -1.0f + box_height; vertices[5] = -box_size;  // 1
    vertices[6] = pos_x + box_size; vertices[7] = -1.0f + box_height; vertices[8] = box_size;   // 2
    vertices[9] = pos_x - box_size; vertices[10] = -1.0f + box_height; vertices[11] = box_size; // 3
    
    // Top face vertices
    vertices[12] = pos_x - box_size; vertices[13] = -1.0f + box_height * 2; vertices[14] = -box_size; // 4
    vertices[15] = pos_x + box_size; vertices[16] = -1.0f + box_height * 2; vertices[17] = -box_size; // 5
    vertices[18] = pos_x + box_size; vertices[19] = -1.0f + box_height * 2; vertices[20] = box_size;  // 6
    vertices[21] = pos_x - box_size; vertices[22] = -1.0f + box_height * 2; vertices[23] = box_size;  // 7

    // Define triangle indices (12 triangles for 6 faces)
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 12);
    
    int idx = 0;
    // Bottom face
    indices[idx++] = 0; indices[idx++] = 2; indices[idx++] = 1;
    indices[idx++] = 0; indices[idx++] = 3; indices[idx++] = 2;
    // Top face
    indices[idx++] = 4; indices[idx++] = 5; indices[idx++] = 6;
    indices[idx++] = 4; indices[idx++] = 6; indices[idx++] = 7;
    // Front face
    indices[idx++] = 0; indices[idx++] = 1; indices[idx++] = 5;
    indices[idx++] = 0; indices[idx++] = 5; indices[idx++] = 4;
    // Back face
    indices[idx++] = 2; indices[idx++] = 3; indices[idx++] = 7;
    indices[idx++] = 2; indices[idx++] = 7; indices[idx++] = 6;
    // Left face
    indices[idx++] = 3; indices[idx++] = 0; indices[idx++] = 4;
    indices[idx++] = 3; indices[idx++] = 4; indices[idx++] = 7;
    // Right face
    indices[idx++] = 1; indices[idx++] = 2; indices[idx++] = 6;
    indices[idx++] = 1; indices[idx++] = 6; indices[idx++] = 5;

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::addCube() {
    // Create a larger cube positioned differently
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 8);
    
    float cube_size = 0.4f;
    float cube_height = 0.8f;
    float pos_x = 1.0f; // Position to the right
    
    // Bottom face vertices  
    vertices[0] = pos_x - cube_size; vertices[1] = -1.0f; vertices[2] = -cube_size;  // 0
    vertices[3] = pos_x + cube_size; vertices[4] = -1.0f; vertices[5] = -cube_size;  // 1
    vertices[6] = pos_x + cube_size; vertices[7] = -1.0f; vertices[8] = cube_size;   // 2
    vertices[9] = pos_x - cube_size; vertices[10] = -1.0f; vertices[11] = cube_size; // 3
    
    // Top face vertices
    vertices[12] = pos_x - cube_size; vertices[13] = -1.0f + cube_height; vertices[14] = -cube_size; // 4
    vertices[15] = pos_x + cube_size; vertices[16] = -1.0f + cube_height; vertices[17] = -cube_size; // 5
    vertices[18] = pos_x + cube_size; vertices[19] = -1.0f + cube_height; vertices[20] = cube_size;  // 6
    vertices[21] = pos_x - cube_size; vertices[22] = -1.0f + cube_height; vertices[23] = cube_size;  // 7

    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 12);
    
    int idx = 0;
    // Bottom face
    indices[idx++] = 0; indices[idx++] = 2; indices[idx++] = 1;
    indices[idx++] = 0; indices[idx++] = 3; indices[idx++] = 2;
    // Top face  
    indices[idx++] = 4; indices[idx++] = 5; indices[idx++] = 6;
    indices[idx++] = 4; indices[idx++] = 6; indices[idx++] = 7;
    // Front face
    indices[idx++] = 0; indices[idx++] = 1; indices[idx++] = 5;
    indices[idx++] = 0; indices[idx++] = 5; indices[idx++] = 4;
    // Back face
    indices[idx++] = 2; indices[idx++] = 3; indices[idx++] = 7;
    indices[idx++] = 2; indices[idx++] = 7; indices[idx++] = 6;
    // Left face
    indices[idx++] = 3; indices[idx++] = 0; indices[idx++] = 4;
    indices[idx++] = 3; indices[idx++] = 4; indices[idx++] = 7;
    // Right face
    indices[idx++] = 1; indices[idx++] = 2; indices[idx++] = 6;
    indices[idx++] = 1; indices[idx++] = 6; indices[idx++] = 5;

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::addSphere() {
    // Create a sphere using triangulated mesh
    RTCGeometry geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    const int segments = 20; // Number of horizontal and vertical segments
    const int num_vertices = (segments + 1) * (segments + 1);
    const int num_triangles = segments * segments * 2;
    
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), num_vertices);
    
    float sphere_radius = 0.3f;
    float pos_x = 0.0f;
    float pos_y = -1.0f + sphere_radius; // Sitting on ground
    float pos_z = 1.5f; // Position in front
    
    // Generate sphere vertices
    int vertex_idx = 0;
    for (int i = 0; i <= segments; ++i) {
        float phi = M_PI * float(i) / float(segments); // Vertical angle
        for (int j = 0; j <= segments; ++j) {
            float theta = 2.0f * M_PI * float(j) / float(segments); // Horizontal angle
            
            float x = sphere_radius * std::sin(phi) * std::cos(theta);
            float y = sphere_radius * std::cos(phi);
            float z = sphere_radius * std::sin(phi) * std::sin(theta);
            
            vertices[vertex_idx * 3 + 0] = pos_x + x;
            vertices[vertex_idx * 3 + 1] = pos_y + y;
            vertices[vertex_idx * 3 + 2] = pos_z + z;
            vertex_idx++;
        }
    }
    
    // Generate sphere indices
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), num_triangles);
    
    int triangle_idx = 0;
    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < segments; ++j) {
            int curr_row = i * (segments + 1);
            int next_row = (i + 1) * (segments + 1);
            
            // First triangle
            indices[triangle_idx * 3 + 0] = curr_row + j;
            indices[triangle_idx * 3 + 1] = next_row + j;
            indices[triangle_idx * 3 + 2] = curr_row + j + 1;
            triangle_idx++;
            
            // Second triangle
            indices[triangle_idx * 3 + 0] = curr_row + j + 1;
            indices[triangle_idx * 3 + 1] = next_row + j;
            indices[triangle_idx * 3 + 2] = next_row + j + 1;
            triangle_idx++;
        }
    }

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(m_scene, geometry);
    rtcReleaseGeometry(geometry);
}

void EmbreeScene::cleanup() {
    if (m_scene) {
        rtcReleaseScene(m_scene);
        m_scene = nullptr;
    }
    if (m_device) {
        rtcReleaseDevice(m_device);
        m_device = nullptr;
    }
}