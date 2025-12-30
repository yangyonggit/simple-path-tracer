#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>

// Forward declarations to avoid full header inclusion
struct OptixDeviceContext_t;
typedef OptixDeviceContext_t* OptixDeviceContext;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

struct OptixModule_t;
typedef OptixModule_t* OptixModule;
struct OptixPipeline_t;
typedef OptixPipeline_t* OptixPipeline;
struct OptixProgramGroup_t;
typedef OptixProgramGroup_t* OptixProgramGroup;

struct OptixShaderBindingTable;
struct OptixPipelineCompileOptions;
typedef unsigned long long OptixTraversableHandle;

namespace scene {
    struct SceneDesc;
}

class Camera;
class EnvironmentManager;
class MaterialManager;
class LightManager;

namespace backends {

// ========================================
// OptixBackend - OptiX 7.7 Ray Tracing Backend
// ========================================
class OptixBackend {
public:
    OptixBackend();
    ~OptixBackend();
    
    // Build OptiX pipeline from SceneDesc
    // Returns true on success
    bool build(const scene::SceneDesc& sceneDesc);
    
    // Render image using OptiX ray tracing
    // pixels: output buffer (width * height * 3 RGB bytes)
    // width, height: image dimensions
    void render(unsigned char* pixels, int width, int height, const Camera& camera);

    // Provide access to environment data (equirectangular HDR) for GPU sampling.
    // Pointer must remain valid for the duration of rendering.
    void setEnvironment(const EnvironmentManager* env) { env_ = env; }

    // Provide access to CPU materials so we can upload them to the GPU.
    // Pointer must remain valid for the duration of rendering.
    void setMaterialManager(const MaterialManager* mm) { material_manager_ = mm; }

    // Provide access to lights (directional light for now) so GPU direct lighting matches CPU.
    // Pointer must remain valid for the duration of rendering.
    void setLightManager(const LightManager* lm) { light_manager_ = lm; }
    
    // Release all OptiX resources
    void destroy();
    
    // Get OptiX context (valid after build() succeeds)
    OptixDeviceContext getContext() const { return context_; }
    void setTopHandle(OptixTraversableHandle handle) { top_handle_ = handle; }
    void setDebugMode(int mode) { debug_mode_ = mode; }

private:
    // OptiX core objects
    OptixDeviceContext context_ = nullptr;
    CUstream stream_ = nullptr;
    
    // Pipeline components
    OptixModule module_ = nullptr;
    OptixModule sphere_is_module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup raygen_prog_group_ = nullptr;
    OptixProgramGroup raygen_primary_prog_group_ = nullptr;
    OptixProgramGroup raygen_trace_prog_group_ = nullptr;
    OptixProgramGroup raygen_shade_prog_group_ = nullptr;
    OptixProgramGroup raygen_resolve_prog_group_ = nullptr;
    OptixProgramGroup miss_prog_group_ = nullptr;
    OptixProgramGroup miss_wf_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_wf_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_sphere_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_sphere_wf_prog_group_ = nullptr;
    
    // Shader Binding Table
    OptixShaderBindingTable* sbt_ = nullptr;
    CUdeviceptr raygen_record_ = 0;
    CUdeviceptr raygen_primary_record_ = 0;
    CUdeviceptr raygen_trace_record_ = 0;
    CUdeviceptr raygen_shade_record_ = 0;
    CUdeviceptr raygen_resolve_record_ = 0;
    CUdeviceptr miss_record_ = 0;
    CUdeviceptr hitgroup_record_ = 0;
    
    // Output buffer on device
    CUdeviceptr d_output_buffer_ = 0;
    size_t output_buffer_size_ = 0;
    
    // Launch parameters buffer on device
    CUdeviceptr d_launch_params_ = 0;

    // Wavefront scaffolding buffers (lifetime managed by OptixBackend)
    CUdeviceptr d_paths_ = 0;
    CUdeviceptr d_hit_records_ = 0;
    CUdeviceptr d_accum_ = 0;
    CUdeviceptr d_ray_queue_in_ = 0;
    CUdeviceptr d_ray_queue_out_ = 0;
    CUdeviceptr d_shade_queue_ = 0;
    CUdeviceptr d_ray_queue_counter_ = 0;
    CUdeviceptr d_ray_queue_out_counter_ = 0;
    CUdeviceptr d_shade_queue_counter_ = 0;
    CUdeviceptr d_materials_ = 0;
    uint32_t wavefront_capacity_ = 0; // == width*height
    bool wavefront_buffers_logged_ = false;
    bool materials_logged_ = false;
    int material_count_ = 0;
    uint32_t frame_index_ = 0;
    bool gen_primary_validated_ = false;
    bool resolve_logged_ = false;

    // Progressive accumulation reset when camera moves
    bool has_last_camera_ = false;
    std::array<float, 3> last_cam_pos_{};
    std::array<float, 3> last_cam_front_{};
    std::array<float, 3> last_cam_up_{};

    // Geometry buffers and handles
    CUdeviceptr d_vertices_ = 0;
    CUdeviceptr d_indices_ = 0;
    CUdeviceptr d_gas_buffer_ = 0;
    OptixTraversableHandle gas_handle_ = 0;

    // Persistent pointer arrays for OptiX build inputs (avoid short-lived stack arrays)
    std::array<CUdeviceptr, 1> triangle_vertex_buffers_{};

    // Sphere geometry (OPTIX_BUILD_INPUT_TYPE_SPHERES)
    CUdeviceptr d_sphere_centers_ = 0;  // float3[1]
    CUdeviceptr d_sphere_radii_ = 0;    // float[1]
    CUdeviceptr d_sphere_material_ids_ = 0; // int[sphereCount]
    std::array<CUdeviceptr, 1> sphere_center_buffers_{};
    std::array<CUdeviceptr, 1> sphere_radius_buffers_{};
    CUdeviceptr d_gas_sphere_buffer_ = 0;
    OptixTraversableHandle gas_sphere_handle_ = 0;

    int triangle_material_id_ = 0;
    int sphere_default_material_id_ = 0;

    // Instance acceleration structure (IAS/TLAS)
    CUdeviceptr d_instances_ = 0;
    CUdeviceptr d_ias_buffer_ = 0;
    OptixTraversableHandle ias_handle_ = 0;
    OptixTraversableHandle top_handle_ = 0;

    // Debug mode for device shading (host-controlled)
    int debug_mode_ = 0;          // 0=normal, 1=hit/miss

    // Environment map (CUDA texture object) for GPU miss sampling
    const EnvironmentManager* env_ = nullptr;
    const MaterialManager* material_manager_ = nullptr;
    const LightManager* light_manager_ = nullptr;
    void* d_env_array_ = nullptr; // cudaArray_t (kept opaque in header)
    uint64_t env_tex_ = 0;        // cudaTextureObject_t
    int env_width_ = 0;
    int env_height_ = 0;
    uint64_t env_revision_ = 0;
    
    // PTX code
    std::string ptx_code_;
    
    // Helper functions
    bool initializeOptix();
    bool createModule();
    bool createProgramGroups();
    bool createPipeline();
    bool createSBT();
    bool buildTriangleGAS(const scene::SceneDesc& sceneDesc);
    bool buildSphereGAS(const scene::SceneDesc& sceneDesc);
    bool buildIAS(const scene::SceneDesc& sceneDesc);
    void allocateOutputBuffer(int width, int height);
    void allocateWavefrontBuffers(int width, int height);

    bool updateEnvironmentTextureIfNeeded();
    void destroyEnvironmentTexture();

    bool uploadMaterialsIfNeeded();
    
    // Logging callback
    static void contextLogCallback(unsigned int level, const char* tag, 
                                   const char* message, void* cbdata);
};

} // namespace backends
