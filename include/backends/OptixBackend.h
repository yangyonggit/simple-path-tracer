#pragma once

#include <string>
#include <vector>
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
    void render(unsigned char* pixels, int width, int height);
    
    // Release all OptiX resources
    void destroy();
    
    // Get OptiX context (valid after build() succeeds)
    OptixDeviceContext getContext() const { return context_; }
    void setTopHandle(OptixTraversableHandle handle) { top_handle_ = handle; }

private:
    // OptiX core objects
    OptixDeviceContext context_ = nullptr;
    CUstream stream_ = nullptr;
    
    // Pipeline components
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup raygen_prog_group_ = nullptr;
    OptixProgramGroup miss_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_prog_group_ = nullptr;
    
    // Shader Binding Table
    OptixShaderBindingTable* sbt_ = nullptr;
    CUdeviceptr raygen_record_ = 0;
    CUdeviceptr miss_record_ = 0;
    CUdeviceptr hitgroup_record_ = 0;
    
    // Output buffer on device
    CUdeviceptr d_output_buffer_ = 0;
    size_t output_buffer_size_ = 0;
    
    // Launch parameters buffer on device
    CUdeviceptr d_launch_params_ = 0;

    // Geometry buffers and handles
    CUdeviceptr d_vertices_ = 0;
    CUdeviceptr d_indices_ = 0;
    CUdeviceptr d_gas_buffer_ = 0;
    OptixTraversableHandle gas_handle_ = 0;
    OptixTraversableHandle top_handle_ = 0;
    
    // PTX code
    std::string ptx_code_;
    
    // Helper functions
    bool initializeOptix();
    bool createModule();
    bool createProgramGroups();
    bool createPipeline();
    bool createSBT();
    bool buildTriangleGAS();
    void allocateOutputBuffer(int width, int height);
    
    // Logging callback
    static void contextLogCallback(unsigned int level, const char* tag, 
                                   const char* message, void* cbdata);
};

} // namespace backends
