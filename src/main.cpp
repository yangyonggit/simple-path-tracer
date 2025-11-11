#include <embree4/rtcore.h>
#include <iostream>
#include <vector>

// Simple test to verify Embree4 is working
int main() {
    std::cout << "Testing Embree4 integration...\n";

    // Initialize Embree device
    RTCDevice device = rtcNewDevice(nullptr);
    if (!device) {
        std::cerr << "Failed to create Embree device!\n";
        return -1;
    }

    std::cout << "Embree device created successfully!\n";

    // Create a scene
    RTCScene scene = rtcNewScene(device);
    if (!scene) {
        std::cerr << "Failed to create Embree scene!\n";
        rtcReleaseDevice(device);
        return -1;
    }

    std::cout << "Embree scene created successfully!\n";

    // Create a simple triangle mesh (a single triangle)
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    
    // Define triangle vertices
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, 
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 3);
    
    vertices[0] = 0.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;  // vertex 0
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;  // vertex 1
    vertices[6] = 0.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;  // vertex 2

    // Define triangle indices
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geometry,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 1);
    
    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    // Commit geometry
    rtcCommitGeometry(geometry);

    // Attach geometry to scene
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);

    // Commit scene
    rtcCommitScene(scene);

    std::cout << "Triangle mesh added to scene successfully!\n";

    // Test ray casting
    RTCRayHit rayhit;
    rayhit.ray.org_x = 0.25f; rayhit.ray.org_y = 0.25f; rayhit.ray.org_z = 1.0f;  // ray origin
    rayhit.ray.dir_x = 0.0f;  rayhit.ray.dir_y = 0.0f;  rayhit.ray.dir_z = -1.0f; // ray direction
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Perform ray-triangle intersection
    rtcIntersect1(scene, &rayhit);

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        std::cout << "Ray intersection found!\n";
        std::cout << "Hit distance: " << rayhit.ray.tfar << "\n";
        std::cout << "Hit coordinates (u,v): (" << rayhit.hit.u << ", " << rayhit.hit.v << ")\n";
    } else {
        std::cout << "No ray intersection found.\n";
    }

    // Cleanup
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    std::cout << "Embree4 test completed successfully!\n";
    return 0;
}