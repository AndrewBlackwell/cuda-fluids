# CUDA Fluid Simulation

This is a CUDA port of my previous [OpenMP CPU-based stable fluid simulator](https://github.com/AndrewBlackwell/fluid-sim). I wanted to see how much faster it could run on a GPU, so I rented a Linux desktop + NVIDIA A5000 via RunPod and rewrote the whole thing.

I was happy to find that after some optimization, it's way faster. At 4096×4096 resolution, I'm getting 136 FPS while fully saturating the GPU. Full benchmarks in [benchmarks/BENCHMARK.md](./benchmarks/BENCHMARKS.md).

**Quick summary:** Near-linear scaling until you hit memory bandwidth limits. The 4096² grid runs at 75-100% GPU utilization with 100% memory bandwidth saturation. I would consider this xcellent efficiency over the 1024 or 2048 grids; 16× more cells, but only 13× slower.

## Build

You need CUDA toolkit, CMake, OpenGL/GLEW, and X11 libs:

```bash
sudo apt install build-essential cmake libgl1-mesa-dev libglew-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

Then build:

```bash
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

## Some Pertinent Optimizations

- **Persistent GPU buffers** - no repeated allocation/deallocation (saves ~3ms/frame)
- **Pinned host memory** - 2-3× faster CPU↔GPU transfers
- **Async memory ops** - cudaMemsetAsync instead of blocking calls
- **Red-black Gauss-Seidel solver** - parallelized pressure projection on GPU
- **8 optimized kernels** - add_source, advect, project (div/grad), decay, splat, toRGBA
- **Direct OpenGL interop** - minimal CPU involvement in rendering

The big wins came from keeping data on the device and only transferring the final RGBA texture back for display.
