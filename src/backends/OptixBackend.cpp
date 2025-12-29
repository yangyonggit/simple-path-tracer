#include "backends/OptixBackend.h"
#include "optix/LaunchParams.h"
#include "scene/SceneDesc.h"

#define NOMINMAX  // Prevent Windows min/max macros from conflicting with std::numeric_limits

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace backends {

namespace {
OptixPipelineCompileOptions makePipelineCompileOptions() {
    OptixPipelineCompileOptions opts = {};
    opts.usesMotionBlur = false;
    opts.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    opts.numPayloadValues = 2;
    opts.numAttributeValues = 2;
    opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    opts.pipelineLaunchParamsVariableName = "params";
    opts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
    return opts;
}
} // namespace

// ========================================
// SBT Record Types
// ========================================
template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord = SbtRecord<int>;    // Empty data for now
using MissRecord = SbtRecord<int>;      // Empty data for now
struct HitgroupData {
    int geomType;  // 0=triangles, 1=spheres
};
using HitgroupRecord = SbtRecord<HitgroupData>;

// ========================================
// Helper Macros
// ========================================
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX Error: " << optixGetErrorName(res)             \
                      << " (" << optixGetErrorString(res) << ") at "           \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ========================================
// Constructor / Destructor
// ========================================
OptixBackend::OptixBackend() {
    sbt_ = new OptixShaderBindingTable();
    memset(sbt_, 0, sizeof(OptixShaderBindingTable));
}

OptixBackend::~OptixBackend() {
    destroy();
    if (sbt_) {
        delete sbt_;
        sbt_ = nullptr;
    }
}

// ========================================
// Logging Callback
// ========================================
void OptixBackend::contextLogCallback(unsigned int level, const char* tag,
                                      const char* message, void* cbdata) {
    std::cerr << "[OptiX][" << tag << "][Level " << level << "]: " 
              << message << std::endl;
}

// ========================================
// Initialize OptiX
// ========================================
bool OptixBackend::initializeOptix() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    
    // Initialize OptiX API
    OPTIX_CHECK(optixInit());
    
    // Create OptiX device context
    CUcontext cuContext = 0;  // Use current CUDA context (NULL)
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &OptixBackend::contextLogCallback;
    options.logCallbackLevel = 4;  // Log all messages
    
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context_));
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    std::cout << "[OptixBackend] OptiX initialized successfully" << std::endl;
    return true;
}

// ========================================
// Read PTX file (generated from device_programs.cu)
// ========================================
static std::string readPTXFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        return "";
    }
    
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

// ========================================
// Create OptiX Module
// ========================================
bool OptixBackend::createModule() {
    // Read PTX code from file (generated by CMake in same directory as executable)
    // Try multiple possible locations
    std::vector<std::string> ptx_paths = {
        "device_programs.ptx",           // Same directory as exe
        "./device_programs.ptx",
        "bin/Release/device_programs.ptx",
        "bin/Debug/device_programs.ptx",
        "../bin/Release/device_programs.ptx",
        "../bin/Debug/device_programs.ptx",
        "../device_programs.ptx",
        "bin/device_programs.ptx",
        "../bin/device_programs.ptx"
    };
    
    for (const auto& path : ptx_paths) {
        ptx_code_ = readPTXFile(path);
        if (!ptx_code_.empty()) {
            std::cout << "[OptixBackend] Loaded PTX from: " << path << std::endl;
            break;
        }
    }
    
    if (ptx_code_.empty()) {
        std::cerr << "[OptixBackend] Failed to load PTX code from any location" << std::endl;
        return false;
    }
    
    // Module compile options
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    
    // Pipeline compile options (shared with pipeline creation)
    OptixPipelineCompileOptions pipeline_compile_options = makePipelineCompileOptions();
    
    // Create module
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        context_,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code_.c_str(),
        ptx_code_.size(),
        log,
        &sizeof_log,
        &module_
    ));
    
    if (sizeof_log > 1) {
        std::cout << "[OptixBackend] Module creation log:\n" << log << std::endl;
    }
    
    std::cout << "[OptixBackend] Module created successfully" << std::endl;

    // Built-in intersection module for spheres (required for OPTIX_BUILD_INPUT_TYPE_SPHERES)
    {
        OptixBuiltinISOptions builtin_is_options = {};
        builtin_is_options.usesMotionBlur = false;
        builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixBuiltinISModuleGet(
            context_,
            &module_compile_options,
            &pipeline_compile_options,
            &builtin_is_options,
            &sphere_is_module_
        ));
    }

    return true;
}

// ========================================
// Create Program Groups
// ========================================
bool OptixBackend::createProgramGroups() {
    char log[2048];
    size_t sizeof_log;
    
    // Raygen program group
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module_;
        pg_desc.raygen.entryFunctionName = "__raygen__rg";
        
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,  // num program groups
            &pg_options,
            log,
            &sizeof_log,
            &raygen_prog_group_
        ));
        
        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Raygen program group log:\n" << log << std::endl;
        }
    }
    
    // Miss program group
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = module_;
        pg_desc.miss.entryFunctionName = "__miss__ms";
        
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &miss_prog_group_
        ));
        
        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Miss program group log:\n" << log << std::endl;
        }
    }
    
    // Hitgroup program group for triangles (closest hit only)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        pg_desc.hitgroup.moduleAH = nullptr;
        pg_desc.hitgroup.entryFunctionNameAH = nullptr;
        pg_desc.hitgroup.moduleIS = nullptr;
        pg_desc.hitgroup.entryFunctionNameIS = nullptr;
        
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group_
        ));
        
        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Hitgroup program group log:\n" << log << std::endl;
        }
    }

    // Hitgroup program group for spheres (bind built-in sphere IS module)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        pg_desc.hitgroup.moduleAH = nullptr;
        pg_desc.hitgroup.entryFunctionNameAH = nullptr;
        pg_desc.hitgroup.moduleIS = sphere_is_module_;
        pg_desc.hitgroup.entryFunctionNameIS = nullptr;

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &hitgroup_sphere_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Sphere hitgroup program group log:\n" << log << std::endl;
        }
    }
    
    std::cout << "[OptixBackend] Program groups created successfully" << std::endl;
    return true;
}

// ========================================
// Create Pipeline
// ========================================
bool OptixBackend::createPipeline() {
        // Pipeline compile options (must match those used in createModule)
        OptixPipelineCompileOptions pipeline_compile_options = makePipelineCompileOptions();

    // IMPORTANT: The pipeline must include ALL program groups referenced by the SBT.
    // We have two hitgroup program groups (triangles + spheres).
    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back(raygen_prog_group_);
    program_groups.push_back(miss_prog_group_);
    program_groups.push_back(hitgroup_prog_group_);
    if (hitgroup_sphere_prog_group_ != nullptr) {
        program_groups.push_back(hitgroup_sphere_prog_group_);
    }
    
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;  // Simple for now
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixPipelineCreate(
        context_,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>(program_groups.size()),
        log,
        &sizeof_log,
        &pipeline_
    ));
    
    if (sizeof_log > 1) {
        std::cout << "[OptixBackend] Pipeline creation log:\n" << log << std::endl;
    }
    
    // Set pipeline stack size (optional, but good practice)
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline_));
    }
    
    uint32_t max_trace_depth = 1;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));
    
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        2  // maxTraversableGraphDepth
    ));
    
    std::cout << "[OptixBackend] Pipeline created successfully" << std::endl;
    return true;
}

// ========================================
// Create Shader Binding Table (SBT)
// ========================================
bool OptixBackend::createSBT() {
    // Raygen record
    {
        const size_t raygen_record_size = sizeof(RaygenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record_), raygen_record_size));
        
        RaygenRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &rg_sbt));
        
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_record_),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Miss record
    {
        const size_t miss_record_size = sizeof(MissRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record_), miss_record_size));
        
        MissRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_, &ms_sbt));
        
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(miss_record_),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Hitgroup records (1=triangle only, 2=triangle+sphere)
    const uint32_t hitgroup_record_count = (gas_sphere_handle_ != 0) ? 2u : 1u;
    {
        const size_t hitgroup_record_size = sizeof(HitgroupRecord);
        const size_t total_size = hitgroup_record_size * hitgroup_record_count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record_), total_size));

        std::vector<HitgroupRecord> records(hitgroup_record_count);

        // Record 0: triangles
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &records[0]));
        records[0].data.geomType = 0;

        if (hitgroup_record_count == 2u) {
            // Record 1: spheres
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_sphere_prog_group_, &records[1]));
            records[1].data.geomType = 1;
        }

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record_),
            records.data(),
            total_size,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Setup SBT
    sbt_->raygenRecord = raygen_record_;
    sbt_->missRecordBase = miss_record_;
    sbt_->missRecordStrideInBytes = sizeof(MissRecord);
    sbt_->missRecordCount = 1;
    sbt_->hitgroupRecordBase = hitgroup_record_;
    sbt_->hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_->hitgroupRecordCount = hitgroup_record_count;

    std::cout << "[OptixBackend] SBT created successfully (hitgroupRecordCount=" << sbt_->hitgroupRecordCount << ")" << std::endl;
    return true;
}

// ========================================
// Build minimal triangle GAS
// ========================================
bool OptixBackend::buildTriangleGAS() {
    // Hardcoded single triangle
    float3 vertices[3] = {
        make_float3(-1.0f, 0.0f, -3.0f),
        make_float3( 1.0f, 0.0f, -3.0f),
        make_float3( 0.0f, 1.0f, -3.0f)
    };
    uint3 indices[1] = { make_uint3(0u, 1u, 2u) };

    // Upload buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), sizeof(vertices)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices_), vertices, sizeof(vertices), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), sizeof(indices)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices_), indices, sizeof(indices), cudaMemcpyHostToDevice));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = 3;
    triangle_vertex_buffers_[0] = d_vertices_;
    build_input.triangleArray.vertexBuffers = triangle_vertex_buffers_.data();

    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = 1;
    build_input.triangleArray.indexBuffer = d_indices_;

    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));

    // Scratch buffer
    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_buffer_), gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context_,
        stream_,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_buffer_,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        nullptr,
        0
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    // Default top-level handle uses this GAS
    top_handle_ = gas_handle_;
    std::cout << "[OptixBackend] Built triangle GAS" << std::endl;
    return true;
}

// ========================================
// Build minimal sphere GAS (OPTIX_BUILD_INPUT_TYPE_SPHERES)
// ========================================
bool OptixBackend::buildSphereGAS(const scene::SceneDesc& sceneDesc) {
    if (sceneDesc.spheres.empty()) {
        std::cout << "[OptixBackend] No spheres in SceneDesc; skipping sphere GAS." << std::endl;
        gas_sphere_handle_ = 0;
        return true;
    }

    // Minimal: build one sphere array GAS from SceneDesc analytical spheres.
    std::vector<float3> centers;
    std::vector<float> radii;
    centers.reserve(sceneDesc.spheres.size());
    radii.reserve(sceneDesc.spheres.size());
    for (const auto& s : sceneDesc.spheres) {
        centers.push_back(make_float3(s.center.x, s.center.y, s.center.z));
        radii.push_back(s.radius);
    }

    std::cout << "[OptixBackend] Building sphere GAS from SceneDesc: count=" << centers.size() << std::endl;
    std::cout << "  sphere[0].center: (" << centers[0].x << ", " << centers[0].y << ", " << centers[0].z << ")" << std::endl;
    std::cout << "  sphere[0].radius: " << radii[0] << std::endl;

    auto sync_and_check = [&](const char* where) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[OptixBackend] CUDA error after " << where << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            std::cerr << "[OptixBackend] CUDA stream sync failed after " << where << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    };

    // Allocate persistent sphere buffers (must outlive traversal)
    if (d_sphere_centers_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sphere_centers_)));
        d_sphere_centers_ = 0;
    }
    if (d_sphere_radii_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sphere_radii_)));
        d_sphere_radii_ = 0;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphere_centers_), sizeof(float3) * centers.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sphere_centers_),
        centers.data(),
        sizeof(float3) * centers.size(),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphere_radii_), sizeof(float) * radii.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sphere_radii_),
        radii.data(),
        sizeof(float) * radii.size(),
        cudaMemcpyHostToDevice
    ));

    sphere_center_buffers_[0] = d_sphere_centers_;
    sphere_radius_buffers_[0] = d_sphere_radii_;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    build_input.sphereArray.vertexBuffers = sphere_center_buffers_.data();
    build_input.sphereArray.numVertices = static_cast<unsigned int>(centers.size());
    build_input.sphereArray.radiusBuffers = sphere_radius_buffers_.data();

    // One build input, one SBT record for this primitive array.
    // Instance.sbtOffset selects the correct SBT record at traversal time.
    uint32_t sphere_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    build_input.sphereArray.flags = sphere_input_flags;
    build_input.sphereArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));
    if (!sync_and_check("optixAccelComputeMemoryUsage(sphere)")) {
        return false;
    }

    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    if (d_gas_sphere_buffer_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_sphere_buffer_)));
        d_gas_sphere_buffer_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_sphere_buffer_), gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context_,
        stream_,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_sphere_buffer_,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_sphere_handle_,
        nullptr,
        0
    ));
    if (!sync_and_check("optixAccelBuild(sphere)")) {
        return false;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    d_temp_buffer = 0;

    std::cout << "[OptixBackend] Built sphere GAS (handle=" << gas_sphere_handle_ << ")" << std::endl;
    return gas_sphere_handle_ != 0;
}

// ========================================
// Build Instance Acceleration Structure (IAS/TLAS)
// ========================================
bool OptixBackend::buildIAS() {
    std::cout << "[OptixBackend] Building IAS..." << std::endl;

    auto sync_and_check = [&](const char* where) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[OptixBackend] CUDA error after " << where << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            std::cerr << "[OptixBackend] CUDA stream sync failed after " << where << ": " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    };

    const bool has_sphere = (gas_sphere_handle_ != 0);
    const uint32_t num_instances = has_sphere ? 2u : 1u;

    OptixInstance instances[2] = {};
    auto set_identity_transform = [](float m[12]) {
        m[0] = 1.0f; m[1] = 0.0f; m[2] = 0.0f; m[3] = 0.0f;
        m[4] = 0.0f; m[5] = 1.0f; m[6] = 0.0f; m[7] = 0.0f;
        m[8] = 0.0f; m[9] = 0.0f; m[10] = 1.0f; m[11] = 0.0f;
    };

    // Instance 0: triangle GAS, sbtOffset=0
    set_identity_transform(instances[0].transform);
    instances[0].instanceId = 0;
    instances[0].sbtOffset = 0;
    instances[0].visibilityMask = 0xFF;
    instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
    instances[0].traversableHandle = gas_handle_;

    if (has_sphere) {
        // Instance 1: sphere GAS, sbtOffset=1
        set_identity_transform(instances[1].transform);
        instances[1].instanceId = 1;
        instances[1].sbtOffset = 1;
        instances[1].visibilityMask = 0xFF;
        instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[1].traversableHandle = gas_sphere_handle_;
    }

    if (d_instances_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances_)));
        d_instances_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances_), sizeof(OptixInstance) * num_instances));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances_),
        instances,
        sizeof(OptixInstance) * num_instances,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = d_instances_;
    build_input.instanceArray.numInstances = num_instances;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &build_input,
        1,
        &ias_buffer_sizes
    ));
    if (!sync_and_check("optixAccelComputeMemoryUsage(IAS)")) {
        return false;
    }

    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));

    if (d_ias_buffer_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_buffer_)));
        d_ias_buffer_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ias_buffer_), ias_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context_,
        stream_,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias_buffer_,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle_,
        nullptr,
        0
    ));
    if (!sync_and_check("optixAccelBuild(IAS)")) {
        return false;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    d_temp_buffer = 0;

    top_handle_ = ias_handle_;
    std::cout << "[OptixBackend] Built IAS successfully" << std::endl;
    std::cout << "[OptixBackend]   numInstances: " << num_instances << std::endl;
    std::cout << "[OptixBackend]   ias_handle_: " << ias_handle_ << std::endl;
    return ias_handle_ != 0;
}

// ========================================
// Build
// ========================================
bool OptixBackend::build(const scene::SceneDesc& sceneDesc) {
    std::cout << "[OptixBackend] Building OptiX backend..." << std::endl;
    
    // Initialize OptiX
    if (!initializeOptix()) {
        return false;
    }
    
    // Create module from PTX
    if (!createModule()) {
        return false;
    }
    
    // Create program groups
    if (!createProgramGroups()) {
        return false;
    }
    
    // Create pipeline
    if (!createPipeline()) {
        return false;
    }

    // Build minimal triangle GAS
    if (!buildTriangleGAS()) {
        return false;
    }

    // Build minimal sphere GAS from SceneDesc
    if (!buildSphereGAS(sceneDesc)) {
        std::cerr << "[OptixBackend] WARNING: Sphere GAS build failed; continuing with triangle-only IAS." << std::endl;
        gas_sphere_handle_ = 0;
    }

    // Build IAS combining triangle + optional sphere
    if (!buildIAS()) {
        return false;
    }

    // Create SBT (1 or 2 hitgroup records)
    if (!createSBT()) {
        return false;
    }

    std::cout << "[OptixBackend] === Build Validation ===" << std::endl;
    std::cout << "[OptixBackend] gas_handle_: " << gas_handle_ << std::endl;
    std::cout << "[OptixBackend] gas_sphere_handle_: " << gas_sphere_handle_ << std::endl;
    std::cout << "[OptixBackend] ias_handle_: " << ias_handle_ << std::endl;
    std::cout << "[OptixBackend] top_handle_: " << top_handle_ << std::endl;
    std::cout << "[OptixBackend] SBT hitgroupRecordCount: " << sbt_->hitgroupRecordCount << std::endl;

    if (gas_handle_ == 0 || ias_handle_ == 0) {
        std::cerr << "[OptixBackend] ERROR: Invalid traversable handles after build." << std::endl;
        return false;
    }
    
    std::cout << "[OptixBackend] Build completed successfully" << std::endl;
    return true;
}

// ========================================
// Allocate Output Buffer
// ========================================
void OptixBackend::allocateOutputBuffer(int width, int height) {
    size_t required_size = width * height * sizeof(uchar4);
    
    if (d_output_buffer_ != 0 && output_buffer_size_ >= required_size) {
        return;  // Already allocated with sufficient size
    }
    
    // Free old buffer if exists
    if (d_output_buffer_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_output_buffer_)));
    }
    
    // Allocate new buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer_), required_size));
    output_buffer_size_ = required_size;
    
    std::cout << "[OptixBackend] Allocated output buffer: " << width << "x" << height << std::endl;
}

// ========================================
// Render
// ========================================
void OptixBackend::render(unsigned char* pixels, int width, int height) {
    // Allocate output buffer on device
    allocateOutputBuffer(width, height);
    
    // Setup launch parameters
    optix::LaunchParams launch_params;
    launch_params.output_buffer = reinterpret_cast<uchar4*>(d_output_buffer_);
    launch_params.image_width = width;
    launch_params.image_height = height;
    launch_params.topHandle = ias_handle_;
    launch_params.debug_mode = debug_mode_;

    // Simple pinhole camera looking down -Z with 60 deg vertical fov
    const float fov_deg = 60.0f;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tan_half_fov = tanf(glm::radians(fov_deg * 0.5f));

    const glm::vec3 forward(0.0f, 0.0f, -1.0f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 right = glm::normalize(glm::cross(forward, up));

    const glm::vec3 cam_u = right * (tan_half_fov * aspect);
    const glm::vec3 cam_v = glm::normalize(glm::cross(cam_u, forward)) * tan_half_fov;
    const glm::vec3 cam_w = forward;

    launch_params.cam_pos = make_float3(0.0f, 0.0f, 0.0f);
    launch_params.cam_u   = make_float3(cam_u.x, cam_u.y, cam_u.z);
    launch_params.cam_v   = make_float3(cam_v.x, cam_v.y, cam_v.z);
    launch_params.cam_w   = make_float3(cam_w.x, cam_w.y, cam_w.z);
    
    // Allocate launch params on device if not already allocated
    if (d_launch_params_ == 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(optix::LaunchParams)));
    }
    
    // Copy launch params to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_launch_params_),
        &launch_params,
        sizeof(optix::LaunchParams),
        cudaMemcpyHostToDevice
    ));
    
    // Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_launch_params_,
        sizeof(optix::LaunchParams),
        sbt_,
        width,
        height,
        1  // depth
    ));
    
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Copy result back to host
    // Convert uchar4 (RGBA) to 3-channel RGB
    std::vector<uchar4> temp_buffer(width * height);
    CUDA_CHECK(cudaMemcpy(
        temp_buffer.data(),
        reinterpret_cast<void*>(d_output_buffer_),
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));
    
    // Convert RGBA to RGB
    for (int i = 0; i < width * height; ++i) {
        pixels[i * 3 + 0] = temp_buffer[i].x;  // R
        pixels[i * 3 + 1] = temp_buffer[i].y;  // G
        pixels[i * 3 + 2] = temp_buffer[i].z;  // B
    }
    
}

// ========================================
// Destroy
// ========================================
void OptixBackend::destroy() {
    // Free SBT records
    if (raygen_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(raygen_record_));
        raygen_record_ = 0;
    }
    if (miss_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(miss_record_));
        miss_record_ = 0;
    }
    if (hitgroup_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(hitgroup_record_));
        hitgroup_record_ = 0;
    }
    
    // Free output buffer
    if (d_output_buffer_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_output_buffer_));
        d_output_buffer_ = 0;
    }

    // Free geometry buffers
    if (d_vertices_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_vertices_));
        d_vertices_ = 0;
    }
    if (d_indices_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_indices_));
        d_indices_ = 0;
    }
    if (d_gas_buffer_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_gas_buffer_));
        d_gas_buffer_ = 0;
    }

    // Free sphere buffers
    if (d_sphere_centers_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_sphere_centers_));
        d_sphere_centers_ = 0;
    }
    if (d_sphere_radii_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_sphere_radii_));
        d_sphere_radii_ = 0;
    }
    if (d_gas_sphere_buffer_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_gas_sphere_buffer_));
        d_gas_sphere_buffer_ = 0;
    }

    // Free IAS buffers
    if (d_instances_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_instances_));
        d_instances_ = 0;
    }
    if (d_ias_buffer_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_ias_buffer_));
        d_ias_buffer_ = 0;
    }
    
    // Free launch params
    if (d_launch_params_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_launch_params_));
        d_launch_params_ = 0;
    }
    
    // Destroy program groups
    if (raygen_prog_group_ != nullptr) {
        optixProgramGroupDestroy(raygen_prog_group_);
        raygen_prog_group_ = nullptr;
    }
    if (miss_prog_group_ != nullptr) {
        optixProgramGroupDestroy(miss_prog_group_);
        miss_prog_group_ = nullptr;
    }
    if (hitgroup_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_prog_group_);
        hitgroup_prog_group_ = nullptr;
    }
    if (hitgroup_sphere_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_sphere_prog_group_);
        hitgroup_sphere_prog_group_ = nullptr;
    }
    
    // Destroy pipeline
    if (pipeline_ != nullptr) {
        optixPipelineDestroy(pipeline_);
        pipeline_ = nullptr;
    }
    
    // Destroy module
    if (module_ != nullptr) {
        optixModuleDestroy(module_);
        module_ = nullptr;
    }
    if (sphere_is_module_ != nullptr) {
        optixModuleDestroy(sphere_is_module_);
        sphere_is_module_ = nullptr;
    }
    
    // Destroy context
    if (context_ != nullptr) {
        optixDeviceContextDestroy(context_);
        context_ = nullptr;
    }
    
    // Destroy stream
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    std::cout << "[OptixBackend] Destroyed" << std::endl;
}

} // namespace backends
