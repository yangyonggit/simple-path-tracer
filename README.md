# Simple Path Tracer

A real-time path tracing renderer built with Intel Embree 4, featuring physically-based rendering, parallel processing, and interactive camera controls.

## Features
![](./images/DefaultScene.png)
### Core Rendering
- **Monte Carlo Path Tracing** - Physically accurate light transport simulation
- **Intel Embree 4** - High-performance ray tracing acceleration
- **Intel TBB Parallelization** - Multi-threaded rendering with automatic CPU core management
- **Real-time Accumulation** - Progressive sample accumulation for noise reduction
- **Tile-based Rendering** - 32x32 pixel tiles for optimal CPU cache usage

### Material System
- **Lambertian (Diffuse) Materials** - Perfect diffuse reflection
- **Metallic Materials** - Adjustable roughness and metallic properties
- **Dielectric (Glass) Materials** - Refraction, reflection, and Fresnel effects
- **Emissive Materials** - Self-illuminating surfaces for area lights
- **Physically-Based Parameters** - Albedo, roughness, metallic, and IOR properties

### Lighting & Environment
- **HDR Environment Mapping** - Equirectangular HDRI backgrounds with proper sampling
- **Procedural Sky** - Atmospheric scattering simulation for outdoor scenes
- **Directional Lights** - Sun-like directional illumination
- **Point Lights** - Omnidirectional point light sources
- **ACES Tone Mapping** - Industry-standard tone mapping for HDR-to-LDR conversion

### Asset Support
- **glTF 2.0 Loading** - Complete 3D scene import with materials and transforms
- **HDR Environment Maps** - Support for .hdr equirectangular environment textures
- **Automatic Material Assignment** - Smart material mapping for imported geometry

### Interactive Features
- **Real-time Camera Controls** - WASD movement and mouse look
- **Performance Monitoring** - Live MRay/s (Million Rays per Second) and FPS display
- **Adaptive Quality** - Camera movement detection for responsive interaction

## Technical Architecture

### Core Components
- **EmbreeScene** - Ray tracing acceleration structure management
- **PathTracer** - Monte Carlo path tracing implementation
- **MaterialManager** - PBR material system with various BRDF models
- **EnvironmentManager** - HDR environment and procedural sky handling
- **GLRenderer** - OpenGL display and real-time interaction
- **GLTFLoader** - glTF 2.0 asset pipeline integration

### Rendering Pipeline
1. **Scene Setup** - Geometry and material initialization
2. **Acceleration Structure** - Embree BVH construction
3. **Tile Distribution** - TBB parallel tile assignment
4. **Path Tracing** - Monte Carlo ray sampling per pixel
5. **Accumulation** - Progressive sample integration
6. **Tone Mapping** - HDR to display conversion
7. **Display** - OpenGL texture upload and presentation

## Building

### Prerequisites
- **Visual Studio 2019/2022** (Windows)
- **CMake 3.20+**
- **vcpkg** package manager

### Dependencies (via vcpkg)
```bash
vcpkg install embree:x64-windows
vcpkg install glm:x64-windows
vcpkg install glfw3:x64-windows
vcpkg install glew:x64-windows
vcpkg install tbb:x64-windows
vcpkg install stb:x64-windows
```

### Build Steps
```bash
git clone <repository-url>
cd simple-path-tracer
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

## Usage

### Basic Execution
```bash
# Run with default scene (spheres and test objects)
./simple-path-tracer.exe

# Display help information
./simple-path-tracer.exe --help
```

### Command Line Options
```bash
# Load custom glTF model (replaces default scene)
./simple-path-tracer.exe --i "path/to/model.gltf"
./simple-path-tracer.exe -i "assets/models/chair/scene.gltf"

# Load custom HDR environment map
./simple-path-tracer.exe --s "path/to/environment.hdr"
./simple-path-tracer.exe -s "assets/hdri/sunset.hdr"

# Combined custom scene and environment
./simple-path-tracer.exe -i "model.gltf" -s "environment.hdr"
```

### Interactive Controls
- **W/A/S/D** - Camera movement (forward/left/back/right)
- **Mouse** - Look around (first-person camera)
- **ESC** - Exit application

### Performance Tuning
- **Samples per Pixel**: 4 (optimized for real-time performance)
- **Maximum Ray Depth**: 6 bounces
- **Thread Usage**: Automatically uses N-1 CPU cores (reserves 1 for system)
- **Tile Size**: 32x32 pixels for optimal cache performance

## Sample Scenes

### Default Scene
- Collection of spheres with various materials (diffuse, metallic, glass)
- Glass cube demonstrating refraction
- Sample glTF model (rattan dining chair)
- HDR environment lighting

### Material Showcase
- **Material ID 0**: Diffuse red
- **Material ID 1**: Diffuse green  
- **Material ID 2**: Metallic copper
- **Material ID 3**: Diffuse blue
- **Material ID 5**: Metallic gold
- **Material ID 6**: Glass (clear dielectric)
- **Material ID 7**: Metallic steel
- **Material ID 8**: Glass (green tinted)

## Performance Metrics

### Typical Performance (RTX 3060 Laptop)
- **Resolution**: 800x600
- **Frame Rate**: 25-30 FPS
- **Ray Throughput**: 12-15 MRay/s
- **Sample Accumulation**: Progressive up to 500+ samples when stationary

### Optimization Features
- **Motion Detection** - Resets accumulation when camera moves
- **Adaptive Sampling** - Higher quality when stationary
- **CPU Threading** - Efficient TBB parallel processing
- **Memory Management** - Optimized data structures for cache efficiency

## Technical Details

### Path Tracing Implementation
- **Importance Sampling** - Cosine-weighted hemisphere sampling
- **Multiple Importance Sampling** - Balanced light and BRDF sampling
- **Russian Roulette** - Path termination for efficiency
- **Fresnel Calculations** - Accurate dielectric reflection/transmission

### Ray-Scene Intersection
- **Embree BVH** - Hardware-accelerated traversal
- **Custom Primitives** - Spheres with analytical intersection
- **Triangle Meshes** - Embree-optimized mesh handling
- **Instancing Support** - Efficient geometry replication

### Memory Layout
- **Structure of Arrays** - Cache-friendly data organization
- **Tile-based Processing** - Reduced memory bandwidth
- **Accumulation Buffers** - High-precision floating-point storage

## Asset Pipeline

### Supported Formats
- **Geometry**: glTF 2.0 (.gltf, .glb)
- **Textures**: HDR (.hdr), PNG, JPEG (via stb_image)
- **Environment**: Equirectangular HDR images

### Material Mapping
- glTF materials automatically converted to PBR parameters
- Support for metallic-roughness workflow
- Texture coordinate handling for UV mapping

## Future Enhancements

- [ ] **glTF Material Support** - Complete PBR material pipeline with texture mapping
- [ ] **glTF Animation** - Keyframe and skeletal animation playback
- [ ] **Denoising** - AI-based or temporal denoising
- [ ] **Volumetric Rendering** - Fog, clouds, and participating media
- [ ] **Advanced Materials** - Subsurface scattering, clearcoat
- [ ] **Temporal Accumulation** - Frame-to-frame reprojection
- [ ] **GPU Acceleration** - CUDA or OptiX backend

## License

This project is built using the following libraries:
- **Intel Embree** - Apache 2.0 License
- **Intel TBB** - Apache 2.0 License  
- **GLFW** - zlib/libpng License
- **GLM** - MIT License
- **stb_image** - MIT License

