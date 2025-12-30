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

#include "Camera.h"

namespace backends {

namespace {
inline void glmMat4ToOptixTransformRowMajor3x4(const glm::mat4& m, float out[12]) {
    // GLM is column-major: m[col][row]
    // OptixInstance expects row-major 3x4: rows 0..2, cols 0..3
    out[0]  = m[0][0]; out[1]  = m[1][0]; out[2]  = m[2][0]; out[3]  = m[3][0];
    out[4]  = m[0][1]; out[5]  = m[1][1]; out[6]  = m[2][1]; out[7]  = m[3][1];
    out[8]  = m[0][2]; out[9]  = m[1][2]; out[10] = m[2][2]; out[11] = m[3][2];
}

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

    // Wavefront primary init raygen program group (no tracing, no fb writes)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module_;
        pg_desc.raygen.entryFunctionName = "__raygen__gen_primary";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &raygen_primary_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] gen_primary raygen program group log:\n" << log << std::endl;
        }
    }

    // Wavefront trace raygen program group (rayType=1)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module_;
        pg_desc.raygen.entryFunctionName = "__raygen__trace";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &raygen_trace_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] trace raygen program group log:\n" << log << std::endl;
        }
    }

    // Wavefront shade raygen program group
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module_;
        pg_desc.raygen.entryFunctionName = "__raygen__shade";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &raygen_shade_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] shade raygen program group log:\n" << log << std::endl;
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

    // Wavefront miss program group (rayType=1)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_desc.miss.module = module_;
        pg_desc.miss.entryFunctionName = "__miss__ms_wf";

        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &pg_desc,
            1,
            &pg_options,
            log,
            &sizeof_log,
            &miss_wf_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Wavefront miss program group log:\n" << log << std::endl;
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

    // Wavefront hitgroup program group for triangles (closest hit only)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_wf";
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
            &hitgroup_wf_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Wavefront hitgroup program group log:\n" << log << std::endl;
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

    // Wavefront hitgroup program group for spheres (bind built-in sphere IS module)
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};
        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_desc.hitgroup.moduleCH = module_;
        pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_wf";
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
            &hitgroup_sphere_wf_prog_group_
        ));

        if (sizeof_log > 1) {
            std::cout << "[OptixBackend] Wavefront sphere hitgroup program group log:\n" << log << std::endl;
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
    if (raygen_primary_prog_group_ != nullptr) {
        program_groups.push_back(raygen_primary_prog_group_);
    }
    if (raygen_trace_prog_group_ != nullptr) {
        program_groups.push_back(raygen_trace_prog_group_);
    }
    if (raygen_shade_prog_group_ != nullptr) {
        program_groups.push_back(raygen_shade_prog_group_);
    }
    program_groups.push_back(miss_prog_group_);
    if (miss_wf_prog_group_ != nullptr) {
        program_groups.push_back(miss_wf_prog_group_);
    }
    program_groups.push_back(hitgroup_prog_group_);
    if (hitgroup_wf_prog_group_ != nullptr) {
        program_groups.push_back(hitgroup_wf_prog_group_);
    }
    if (hitgroup_sphere_prog_group_ != nullptr) {
        program_groups.push_back(hitgroup_sphere_prog_group_);
    }
    if (hitgroup_sphere_wf_prog_group_ != nullptr) {
        program_groups.push_back(hitgroup_sphere_wf_prog_group_);
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

    // gen_primary raygen record
    {
        const size_t raygen_record_size = sizeof(RaygenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_primary_record_), raygen_record_size));

        RaygenRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_primary_prog_group_, &rg_sbt));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_primary_record_),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));
    }

    // trace raygen record
    {
        const size_t raygen_record_size = sizeof(RaygenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_trace_record_), raygen_record_size));

        RaygenRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_trace_prog_group_, &rg_sbt));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_trace_record_),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));
    }

    // shade raygen record
    {
        const size_t raygen_record_size = sizeof(RaygenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_shade_record_), raygen_record_size));

        RaygenRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_shade_prog_group_, &rg_sbt));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(raygen_shade_record_),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Miss records (rayTypeCount=2): [0]=legacy, [1]=wavefront
    {
        const size_t miss_record_size = sizeof(MissRecord);
        const size_t total_size = miss_record_size * 2ull;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record_), total_size));

        std::vector<MissRecord> records(2);
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_, &records[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_wf_prog_group_, &records[1]));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(miss_record_),
            records.data(),
            total_size,
            cudaMemcpyHostToDevice
        ));
    }
    
    // Hitgroup records: 2 ray types (legacy + wavefront) per geometry
    const uint32_t sbt_geom_count = (gas_sphere_handle_ != 0) ? 2u : 1u;
    const uint32_t hitgroup_record_count = sbt_geom_count * 2u;
    {
        const size_t hitgroup_record_size = sizeof(HitgroupRecord);
        const size_t total_size = hitgroup_record_size * hitgroup_record_count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record_), total_size));

        std::vector<HitgroupRecord> records(hitgroup_record_count);

        // Geometry 0: triangles
        //   idx 0: rayType 0 (legacy)
        //   idx 1: rayType 1 (wavefront)
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &records[0]));
        records[0].data.geomType = 0;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_wf_prog_group_, &records[1]));
        records[1].data.geomType = 0;

        if (sbt_geom_count == 2u) {
            // Geometry 1: spheres
            //   idx 2: rayType 0 (legacy)
            //   idx 3: rayType 1 (wavefront)
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_sphere_prog_group_, &records[2]));
            records[2].data.geomType = 1;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_sphere_wf_prog_group_, &records[3]));
            records[3].data.geomType = 1;
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
    sbt_->missRecordCount = 2;
    sbt_->hitgroupRecordBase = hitgroup_record_;
    sbt_->hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_->hitgroupRecordCount = hitgroup_record_count;

    std::cout << "[OptixBackend] SBT created successfully (hitgroupRecordCount=" << sbt_->hitgroupRecordCount << ")" << std::endl;
    return true;
}

// ========================================
// Build minimal triangle GAS
// ========================================
bool OptixBackend::buildTriangleGAS(const scene::SceneDesc& sceneDesc) {
    if (sceneDesc.meshes.empty() || !sceneDesc.meshes[0].isValid()) {
        std::cerr << "[OptixBackend] ERROR: SceneDesc.meshes[0] is missing or invalid (need positions + indices)." << std::endl;
        return false;
    }

    const scene::MeshData& mesh = sceneDesc.meshes[0];

    std::vector<float3> vertices;
    vertices.reserve(mesh.positions.size());
    for (const glm::vec3& p : mesh.positions) {
        vertices.push_back(make_float3(p.x, p.y, p.z));
    }

    std::vector<uint3> indices;
    indices.reserve(mesh.indices.size());
    for (const glm::uvec3& tri : mesh.indices) {
        indices.push_back(make_uint3(tri.x, tri.y, tri.z));
    }

    // (Re)upload buffers
    if (d_vertices_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices_)));
        d_vertices_ = 0;
    }
    if (d_indices_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices_)));
        d_indices_ = 0;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), sizeof(float3) * vertices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices_),
        vertices.data(),
        sizeof(float3) * vertices.size(),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), sizeof(uint3) * indices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices_),
        indices.data(),
        sizeof(uint3) * indices.size(),
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
    triangle_vertex_buffers_[0] = d_vertices_;
    build_input.triangleArray.vertexBuffers = triangle_vertex_buffers_.data();

    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size());
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

    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    if (d_gas_buffer_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_buffer_)));
        d_gas_buffer_ = 0;
    }
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

    // Required post-build checks
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    d_temp_buffer = 0;

    // Default top-level handle uses this GAS (later overridden by IAS build)
    top_handle_ = gas_handle_;
    std::cout << "[OptixBackend] Built triangle GAS from SceneDesc (V=" << vertices.size()
              << ", T=" << indices.size() << ")" << std::endl;
    return gas_handle_ != 0;
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
bool OptixBackend::buildIAS(const scene::SceneDesc& sceneDesc) {
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

    std::vector<OptixInstance> instances;
    instances.reserve(sceneDesc.instances.size());

    for (uint32_t i = 0; i < static_cast<uint32_t>(sceneDesc.instances.size()); ++i) {
        const auto& inst = sceneDesc.instances[i];

        OptixInstance oi = {};
        glmMat4ToOptixTransformRowMajor3x4(inst.worldFromObject, oi.transform);
        oi.instanceId = i;         // fill for easier debug
        oi.sbtOffset = 0;          // triangles hitgroup record
        oi.visibilityMask = 0xFF;
        oi.flags = OPTIX_INSTANCE_FLAG_NONE;
        oi.traversableHandle = gas_handle_;  // single mesh GAS for now
        instances.push_back(oi);
    }

    // Optional: one sphere instance referencing a single GAS that contains N sphere primitives.
    // Sphere positions are stored in the sphere center buffer, so we keep identity instance transform.
    if (!sceneDesc.spheres.empty() && gas_sphere_handle_ != 0) {
        OptixInstance oi = {};
        oi.transform[0] = 1.0f;  oi.transform[1] = 0.0f;  oi.transform[2] = 0.0f;  oi.transform[3] = 0.0f;
        oi.transform[4] = 0.0f;  oi.transform[5] = 1.0f;  oi.transform[6] = 0.0f;  oi.transform[7] = 0.0f;
        oi.transform[8] = 0.0f;  oi.transform[9] = 0.0f;  oi.transform[10] = 1.0f; oi.transform[11] = 0.0f;
        oi.instanceId = static_cast<uint32_t>(instances.size());
        oi.sbtOffset = 2;  // sphere geometry base (rayTypeCount=2)
        oi.visibilityMask = 0xFF;
        oi.flags = OPTIX_INSTANCE_FLAG_NONE;
        oi.traversableHandle = gas_sphere_handle_;
        instances.push_back(oi);
    }

    if (instances.empty()) {
        std::cerr << "[OptixBackend] ERROR: No instances to build IAS (no triangle instances and no sphere instance)." << std::endl;
        return false;
    }

    const uint32_t num_instances = static_cast<uint32_t>(instances.size());

    if (d_instances_ != 0) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances_)));
        d_instances_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances_), sizeof(OptixInstance) * num_instances));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances_),
        instances.data(),
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

    // Build triangle GAS from SceneDesc
    if (!buildTriangleGAS(sceneDesc)) {
        return false;
    }

    // Build sphere GAS (optional): one GAS containing N sphere primitives
    if (!buildSphereGAS(sceneDesc)) {
        std::cerr << "[OptixBackend] WARNING: Sphere GAS build failed; continuing with triangle-only IAS." << std::endl;
        gas_sphere_handle_ = 0;
    }

    // Build IAS from SceneDesc.instances (+ optional sphere instance)
    if (!buildIAS(sceneDesc)) {
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
// Allocate Wavefront Buffers (Scaffolding)
// ========================================
void OptixBackend::allocateWavefrontBuffers(int width, int height) {
    const uint32_t capacity = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);
    if (capacity == 0u) {
        return;
    }

    if (wavefront_capacity_ == capacity &&
        d_paths_ != 0 && d_hit_records_ != 0 &&
        d_accum_ != 0 &&
        d_ray_queue_in_ != 0 && d_ray_queue_out_ != 0 && d_shade_queue_ != 0 &&
        d_ray_queue_counter_ != 0 && d_shade_queue_counter_ != 0 &&
        d_materials_ != 0) {
        return;
    }

    auto freeIf = [&](CUdeviceptr& p) {
        if (p != 0) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(p)));
            p = 0;
        }
    };

    freeIf(d_paths_);
    freeIf(d_hit_records_);
    freeIf(d_accum_);
    freeIf(d_ray_queue_in_);
    freeIf(d_ray_queue_out_);
    freeIf(d_shade_queue_);
    freeIf(d_ray_queue_counter_);
    freeIf(d_shade_queue_counter_);
    freeIf(d_materials_);

    const size_t paths_bytes = sizeof(optix::PathState) * static_cast<size_t>(capacity);
    const size_t hit_bytes = sizeof(optix::HitRecord) * static_cast<size_t>(capacity);
    const size_t accum_bytes = sizeof(float4) * static_cast<size_t>(capacity);
    const size_t queue_bytes = sizeof(uint32_t) * static_cast<size_t>(capacity);
    const size_t counter_bytes = sizeof(uint32_t);
    const size_t materials_bytes = sizeof(optix::DeviceMaterial) * static_cast<size_t>(material_count_);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_paths_), paths_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_records_), hit_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accum_), accum_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_queue_in_), queue_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_queue_out_), queue_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shade_queue_), queue_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_queue_counter_), counter_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shade_queue_counter_), counter_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materials_), materials_bytes));

    // Upload one default material (scaffolding only; device programs don't use it yet)
    {
        optix::DeviceMaterial m{};
        m.baseColor = make_float3(1.0f, 1.0f, 1.0f);
        m.roughness = 1.0f;
        m.metallic = 0.0f;
        m.type = 0;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_materials_),
            &m,
            sizeof(optix::DeviceMaterial),
            cudaMemcpyHostToDevice
        ));
    }

    wavefront_capacity_ = capacity;

    if (!wavefront_buffers_logged_) {
        wavefront_buffers_logged_ = true;
        std::cout << "[OptixBackend] Wavefront buffers allocated (scaffolding):" << std::endl;
        std::cout << "  paths: " << paths_bytes << " bytes" << std::endl;
        std::cout << "  hitRecords: " << hit_bytes << " bytes" << std::endl;
        std::cout << "  accum: " << accum_bytes << " bytes" << std::endl;
        std::cout << "  rayQueueIn: " << queue_bytes << " bytes" << std::endl;
        std::cout << "  rayQueueOut: " << queue_bytes << " bytes" << std::endl;
        std::cout << "  shadeQueue: " << queue_bytes << " bytes" << std::endl;
        std::cout << "  rayQueueCounter: " << counter_bytes << " bytes" << std::endl;
        std::cout << "  shadeQueueCounter: " << counter_bytes << " bytes" << std::endl;
        std::cout << "  materials: " << materials_bytes << " bytes (count=" << material_count_ << ")" << std::endl;
    }
}

// ========================================
// Render
// ========================================
void OptixBackend::render(unsigned char* pixels, int width, int height, const Camera& camera) {
    // Allocate output buffer on device
    allocateOutputBuffer(width, height);

    // Allocate wavefront scaffolding buffers on first use / resize
    const uint32_t prev_wavefront_capacity = wavefront_capacity_;
    allocateWavefrontBuffers(width, height);
    const bool resized = (prev_wavefront_capacity != wavefront_capacity_);
    
    // Setup launch parameters
    optix::LaunchParams launch_params;
    launch_params.output_buffer = reinterpret_cast<uchar4*>(d_output_buffer_);
    launch_params.accum = reinterpret_cast<float4*>(d_accum_);
    launch_params.image_width = width;
    launch_params.image_height = height;
    launch_params.topHandle = ias_handle_;
    launch_params.debug_mode = debug_mode_;

    // Wavefront scaffolding pointers (unused by current device programs)
    launch_params.paths = reinterpret_cast<optix::PathState*>(d_paths_);
    launch_params.hitRecords = reinterpret_cast<optix::HitRecord*>(d_hit_records_);
    launch_params.rayQueueIn = reinterpret_cast<uint32_t*>(d_ray_queue_in_);
    launch_params.rayQueueOut = reinterpret_cast<uint32_t*>(d_ray_queue_out_);
    launch_params.shadeQueue = reinterpret_cast<uint32_t*>(d_shade_queue_);
    launch_params.rayQueueCounter = reinterpret_cast<uint32_t*>(d_ray_queue_counter_);
    launch_params.shadeQueueCounter = reinterpret_cast<uint32_t*>(d_shade_queue_counter_);
    launch_params.materials = reinterpret_cast<optix::DeviceMaterial*>(d_materials_);
    launch_params.materialCount = material_count_;
    launch_params.maxDepth = 6;
    launch_params.frameIndex = frame_index_++;

    if ((launch_params.frameIndex == 0u || resized) && d_accum_ != 0) {
        const size_t accum_bytes = sizeof(float4) * static_cast<size_t>(width) * static_cast<size_t>(height);
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_accum_), 0, accum_bytes, stream_));
    }

    // Camera: match CPU/Embree camera rays.
    // Device raygen expects: dir = normalize(cam_w + ndc.x*cam_u + ndc.y*cam_v), with ndc in [-1,1] and Y flipped.
    // We derive cam_u/cam_v scales from Camera::getRayDirection() so we don't depend on Camera internals.
    const glm::vec3 cam_pos = camera.getPosition();
    const glm::vec3 forward = glm::normalize(camera.getFront());
    const glm::vec3 right = glm::normalize(camera.getRight());
    const glm::vec3 up = glm::normalize(camera.getUp());

    const glm::vec3 dir_x = camera.getRayDirection(1.0f, 0.5f);  // ndc.x = +1, ndc.y = 0
    const float denom_x = glm::dot(dir_x, forward);
    const float half_width = (denom_x != 0.0f) ? (glm::dot(dir_x, right) / denom_x) : 0.0f;

    const glm::vec3 dir_y = camera.getRayDirection(0.5f, 0.0f);  // ndc.x = 0, ndc.y = +1
    const float denom_y = glm::dot(dir_y, forward);
    const float half_height = (denom_y != 0.0f) ? (glm::dot(dir_y, up) / denom_y) : 0.0f;

    const glm::vec3 cam_u = right * half_width;
    const glm::vec3 cam_v = up * half_height;
    const glm::vec3 cam_w = forward;

    launch_params.cam_pos = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);
    launch_params.cam_u   = make_float3(cam_u.x, cam_u.y, cam_u.z);
    launch_params.cam_v   = make_float3(cam_v.x, cam_v.y, cam_v.z);
    launch_params.cam_w   = make_float3(cam_w.x, cam_w.y, cam_w.z);
    
    // Allocate launch params on device if not already allocated
    if (d_launch_params_ == 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(optix::LaunchParams)));
    }

    // Per-frame: reset wavefront counters (required by scaffolding)
    if (d_ray_queue_counter_ != 0) {
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_ray_queue_counter_), 0, sizeof(uint32_t), stream_));
    }
    if (d_shade_queue_counter_ != 0) {
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_shade_queue_counter_), 0, sizeof(uint32_t), stream_));
    }

    const uint32_t capacity = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);

    // 1) gen_primary: initialize PathState + fill rayQueueIn. Does NOT change framebuffer.
    uint32_t ray_queue_count = 0u;
    {
        // Copy launch params to device
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_launch_params_),
            &launch_params,
            sizeof(optix::LaunchParams),
            cudaMemcpyHostToDevice
        ));

        // Switch raygen record
        sbt_->raygenRecord = raygen_primary_record_;

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            stream_,
            d_launch_params_,
            sizeof(optix::LaunchParams),
            sbt_,
            width,
            height,
            1
        ));
    }

    // Read the true queued ray count every frame (for trace launch dimension).
    CUDA_CHECK(cudaMemcpyAsync(
        &ray_queue_count,
        reinterpret_cast<void*>(d_ray_queue_counter_),
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream_
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (ray_queue_count > capacity) {
        ray_queue_count = capacity;
    }

    // 2) trace: consume rayQueueIn[0..N) and write hitRecords + shadeQueue (no fb writes)
    if (d_shade_queue_counter_ != 0) {
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(d_shade_queue_counter_), 0, sizeof(uint32_t), stream_));
    }
    {
        if (ray_queue_count > 0u) {
            // Copy launch params to device (unchanged)
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_launch_params_),
                &launch_params,
                sizeof(optix::LaunchParams),
                cudaMemcpyHostToDevice
            ));

            sbt_->raygenRecord = raygen_trace_record_;
            OPTIX_CHECK(optixLaunch(
                pipeline_,
                stream_,
                d_launch_params_,
                sizeof(optix::LaunchParams),
                sbt_,
                ray_queue_count,
                1,
                1
            ));
        }
    }

    // Read shade queue count every frame (for shade launch dimension).
    uint32_t shade_queue_count = 0u;
    CUDA_CHECK(cudaMemcpyAsync(
        &shade_queue_count,
        reinterpret_cast<void*>(d_shade_queue_counter_),
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream_
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (shade_queue_count > capacity) {
        shade_queue_count = capacity;
    }

    // 3) shade: consume shadeQueue[0..M) and write accum (no fb writes)
    {
        if (shade_queue_count > 0u) {
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_launch_params_),
                &launch_params,
                sizeof(optix::LaunchParams),
                cudaMemcpyHostToDevice
            ));

            sbt_->raygenRecord = raygen_shade_record_;
            OPTIX_CHECK(optixLaunch(
                pipeline_,
                stream_,
                d_launch_params_,
                sizeof(optix::LaunchParams),
                sbt_,
                shade_queue_count,
                1,
                1
            ));
        }
    }

    // Frame-0 validation logs (must print once)
    if (launch_params.frameIndex == 0u) {
        std::cout << "[Wavefront] rayQueueCount = " << ray_queue_count
                  << " (expect " << capacity << ")" << std::endl;
        std::cout << "[Wavefront] shadeQueueCount = " << shade_queue_count
                  << " (expect " << ray_queue_count << ")" << std::endl;
        std::cout << "[Wavefront] shadeLaunchCount = " << shade_queue_count << std::endl;
    }
    
    // 4) Original raygen: renders debug color / geom coloring into framebuffer (must remain unchanged)
    sbt_->raygenRecord = raygen_record_;

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
    if (raygen_primary_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(raygen_primary_record_));
        raygen_primary_record_ = 0;
    }
    if (raygen_trace_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(raygen_trace_record_));
        raygen_trace_record_ = 0;
    }
    if (raygen_shade_record_ != 0) {
        cudaFree(reinterpret_cast<void*>(raygen_shade_record_));
        raygen_shade_record_ = 0;
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

    // Free wavefront scaffolding buffers
    if (d_paths_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_paths_));
        d_paths_ = 0;
    }
    if (d_hit_records_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_hit_records_));
        d_hit_records_ = 0;
    }
    if (d_accum_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_accum_));
        d_accum_ = 0;
    }
    if (d_ray_queue_in_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_ray_queue_in_));
        d_ray_queue_in_ = 0;
    }
    if (d_ray_queue_out_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_ray_queue_out_));
        d_ray_queue_out_ = 0;
    }
    if (d_shade_queue_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_shade_queue_));
        d_shade_queue_ = 0;
    }
    if (d_ray_queue_counter_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_ray_queue_counter_));
        d_ray_queue_counter_ = 0;
    }
    if (d_shade_queue_counter_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_shade_queue_counter_));
        d_shade_queue_counter_ = 0;
    }
    if (d_materials_ != 0) {
        cudaFree(reinterpret_cast<void*>(d_materials_));
        d_materials_ = 0;
    }
    wavefront_capacity_ = 0;

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
    if (raygen_primary_prog_group_ != nullptr) {
        optixProgramGroupDestroy(raygen_primary_prog_group_);
        raygen_primary_prog_group_ = nullptr;
    }
    if (raygen_trace_prog_group_ != nullptr) {
        optixProgramGroupDestroy(raygen_trace_prog_group_);
        raygen_trace_prog_group_ = nullptr;
    }
    if (raygen_shade_prog_group_ != nullptr) {
        optixProgramGroupDestroy(raygen_shade_prog_group_);
        raygen_shade_prog_group_ = nullptr;
    }
    if (miss_prog_group_ != nullptr) {
        optixProgramGroupDestroy(miss_prog_group_);
        miss_prog_group_ = nullptr;
    }
    if (miss_wf_prog_group_ != nullptr) {
        optixProgramGroupDestroy(miss_wf_prog_group_);
        miss_wf_prog_group_ = nullptr;
    }
    if (hitgroup_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_prog_group_);
        hitgroup_prog_group_ = nullptr;
    }
    if (hitgroup_wf_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_wf_prog_group_);
        hitgroup_wf_prog_group_ = nullptr;
    }
    if (hitgroup_sphere_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_sphere_prog_group_);
        hitgroup_sphere_prog_group_ = nullptr;
    }
    if (hitgroup_sphere_wf_prog_group_ != nullptr) {
        optixProgramGroupDestroy(hitgroup_sphere_wf_prog_group_);
        hitgroup_sphere_wf_prog_group_ = nullptr;
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
