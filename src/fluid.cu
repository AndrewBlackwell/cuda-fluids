#include "fluid.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

#define PROFILE_KERNELS

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifdef PROFILE_KERNELS
#include <fstream>
#include <string>
static cudaEvent_t prof_start, prof_stop;
static bool prof_initialized = false;
static FILE *prof_file = nullptr;
static int prof_frame_count = 0;

void init_profiling()
{
    if (!prof_initialized)
    {
        cudaEventCreate(&prof_start);
        cudaEventCreate(&prof_stop);
        prof_file = fopen("kernel_profile.csv", "w");
        fprintf(prof_file, "frame,kernel,time_ms\n");
        prof_initialized = true;
    }
}

void cleanup_profiling()
{
    if (prof_initialized)
    {
        if (prof_file)
            fclose(prof_file);
        cudaEventDestroy(prof_start);
        cudaEventDestroy(prof_stop);
    }
}

#define PROFILE_KERNEL(kernel_name, kernel_call)                                   \
    do                                                                             \
    {                                                                              \
        cudaEventRecord(prof_start);                                               \
        kernel_call;                                                               \
        cudaEventRecord(prof_stop);                                                \
        cudaEventSynchronize(prof_stop);                                           \
        float ms = 0;                                                              \
        cudaEventElapsedTime(&ms, prof_start, prof_stop);                          \
        if (prof_file)                                                             \
            fprintf(prof_file, "%d,%s,%.6f\n", prof_frame_count, kernel_name, ms); \
    } while (0)
#else
#define PROFILE_KERNEL(kernel_name, kernel_call) kernel_call
void init_profiling() {}
void cleanup_profiling() {}
#endif

#define BLOCK_SIZE 16

__device__ inline int IX(int i, int j, int N)
{
    return i + (N + 2) * j;
}

__device__ inline float clampf(float x, float a, float b)
{
    return x < a ? a : (x > b ? b : x);
}

/**
 * @brief kernel to add source field scaled by dt
 */
__global__ void add_source_kernel(float *x, const float *s, float dt, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        x[idx] += dt * s[idx];
    }
}

/**
 * @brief kernel to set boundary conditions
 */
__global__ void set_bnd_edges_kernel(int b, float *x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N)
    {
        // left and right boundaries
        x[IX(0, i, N)] = (b == 1) ? -x[IX(1, i, N)] : x[IX(1, i, N)];
        x[IX(N + 1, i, N)] = (b == 1) ? -x[IX(N, i, N)] : x[IX(N, i, N)];
        // top and bottom boundaries
        x[IX(i, 0, N)] = (b == 2) ? -x[IX(i, 1, N)] : x[IX(i, 1, N)];
        x[IX(i, N + 1, N)] = (b == 2) ? -x[IX(i, N, N)] : x[IX(i, N, N)];
    }
}

__global__ void set_bnd_corners_kernel(float *x, int N)
{
    x[IX(0, 0, N)] = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    x[IX(0, N + 1, N)] = 0.5f * (x[IX(1, N + 1, N)] + x[IX(0, N, N)]);
    x[IX(N + 1, 0, N)] = 0.5f * (x[IX(N, 0, N)] + x[IX(N + 1, 1, N)]);
    x[IX(N + 1, N + 1, N)] = 0.5f * (x[IX(N, N + 1, N)] + x[IX(N + 1, N, N)]);
}

/**
 * @brief kernel for Gauss-Seidel iteration
 */
__global__ void lin_solve_kernel(int b, float *x, const float *x0,
                                 float a, float invC, int N, int parity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N)
    {
        // red-black ordering to avoid race conditions
        if ((i + j) % 2 == parity)
        {
            int idx = IX(i, j, N);
            x[idx] = (x0[idx] + a * (x[IX(i - 1, j, N)] + x[IX(i + 1, j, N)] +
                                     x[IX(i, j - 1, N)] + x[IX(i, j + 1, N)])) *
                     invC;
        }
    }
}

/**
 * @brief kernel for advection using semi-Lagrangian method
 */
__global__ void advect_kernel(int b, float *d, const float *d0,
                              const float *u, const float *v,
                              float dt0, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N)
    {
        // backtrace particle position
        float x = i - dt0 * u[IX(i, j, N)];
        float y = j - dt0 * v[IX(i, j, N)];

        x = clampf(x, 0.5f, N + 0.5f);
        y = clampf(y, 0.5f, N + 0.5f);

        // get integer coords
        int i0 = (int)floorf(x);
        int i1 = i0 + 1;
        int j0 = (int)floorf(y);
        int j1 = j0 + 1;

        float s1 = x - i0, s0 = 1.0f - s1;
        float t1 = y - j0, t0 = 1.0f - t1;

        // bilinear interpolation
        d[IX(i, j, N)] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                         s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
    }
}

/**
 * @brief kernel to compute divergence for projection
 */
__global__ void project_divergence_kernel(float *div, float *p,
                                          const float *u, const float *v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N)
    {
        div[IX(i, j, N)] = -0.5f * ((u[IX(i + 1, j, N)] - u[IX(i - 1, j, N)] +
                                     v[IX(i, j + 1, N)] - v[IX(i, j - 1, N)]) /
                                    N);
        p[IX(i, j, N)] = 0.0f;
    }
}

/**
 * @brief kernel to subtract pressure gradient
 */
__global__ void project_gradient_kernel(float *u, float *v, const float *p, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N)
    {
        u[IX(i, j, N)] -= 0.5f * N * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]);
        v[IX(i, j, N)] -= 0.5f * N * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]);
    }
}

/**
 * @brief kernel to apply decay/dissipation
 */
__global__ void apply_decay_kernel(float *u, float *v,
                                   float *rD, float *gD, float *bD,
                                   float vel_decay, float dye_decay, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= N && j <= N)
    {
        int idx = IX(i, j, N);
        u[idx] *= vel_decay;
        v[idx] *= vel_decay;
        rD[idx] *= dye_decay;
        gD[idx] *= dye_decay;
        bD[idx] *= dye_decay;
    }
}

/**
 * @brief kernel to clear source fields
 */
__global__ void clear_sources_kernel(float *u0, float *v0,
                                     float *r0, float *g0, float *b0, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        u0[idx] = 0.0f;
        v0[idx] = 0.0f;
        r0[idx] = 0.0f;
        g0[idx] = 0.0f;
        b0[idx] = 0.0f;
    }
}

/**
 * @brief kernel to add Gaussian splat
 */
__global__ void add_splat_kernel(float *u0, float *v0,
                                 float *r0, float *g0, float *b0,
                                 int cx, int cy, float rad, int rInt,
                                 float fx, float fy,
                                 float r, float g, float b,
                                 float dye_amount, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + cx - rInt;
    int j = blockIdx.y * blockDim.y + threadIdx.y + cy - rInt;

    if (i >= 1 && i <= N && j >= 1 && j <= N)
    {
        float dx = (float)(i - cx);
        float dy = (float)(j - cy);
        float d2 = dx * dx + dy * dy;

        if (d2 <= (rInt * rInt))
        {
            float w = expf(-d2 / (2.0f * rad * rad));

            int idx = IX(i, j, N);
            atomicAdd(&u0[idx], fx * w);
            atomicAdd(&v0[idx], fy * w);

            float add = dye_amount * w;
            atomicAdd(&r0[idx], add * r);
            atomicAdd(&g0[idx], add * g);
            atomicAdd(&b0[idx], add * b);
        }
    }
}

/**
 * @brief kernel to convert dye to RGBA
 */
__global__ void toRGBA_kernel(const float *rD, const float *gD, const float *bD,
                              uint8_t *rgba, float gain, float invGamma, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N)
    {
        int gridIdx = IX(x + 1, y + 1, N);
        int imgIdx = (y * N + x) * 4;

        float rVal = rD[gridIdx] * gain;
        float gVal = gD[gridIdx] * gain;
        float bVal = bD[gridIdx] * gain;

        rVal = rVal / (1.0f + rVal);
        gVal = gVal / (1.0f + gVal);
        bVal = bVal / (1.0f + bVal);

        // gamma correction
        rVal = powf(fmaxf(rVal, 0.0f), invGamma);
        gVal = powf(fmaxf(gVal, 0.0f), invGamma);
        bVal = powf(fmaxf(bVal, 0.0f), invGamma);

        // convert to uint8
        rgba[imgIdx + 0] = (uint8_t)(fminf(rVal * 255.0f, 255.0f));
        rgba[imgIdx + 1] = (uint8_t)(fminf(gVal * 255.0f, 255.0f));
        rgba[imgIdx + 2] = (uint8_t)(fminf(bVal * 255.0f, 255.0f));
        rgba[imgIdx + 3] = 255;
    }
}

Fluid2D::Fluid2D(int N) : mN(N), mSize((N + 2) * (N + 2)),
                          d_rgba(nullptr), d_rgba_size(0),
                          h_rgba_pinned(nullptr), h_rgba_pinned_size(0)
{
    // temp: print gpu info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    fprintf(stderr, "using GPU: %s\n", prop.name);
    fprintf(stderr, "capability: %d.%d\n", prop.major, prop.minor);
    fprintf(stderr, "global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "mem clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);

    // allocate device memory
    CUDA_CHECK(cudaMalloc(&d_u, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v0, mSize * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_rD, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gD, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bD, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b0, mSize * sizeof(float)));

    // init to 0, async
    CUDA_CHECK(cudaMemsetAsync(d_u, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_v, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_u0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_v0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_rD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_gD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_bD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_r0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_g0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_b0, 0, mSize * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    init_profiling();

    fprintf(stderr, "[CUDA] Fluid simulation initialized (%dx%d grid, %.2f MB VRAM)\n",
            N, N, (10 * mSize * sizeof(float)) / (1024.0 * 1024.0));
}

Fluid2D::~Fluid2D()
{
    cleanup_profiling();

    cudaError_t err;

    if ((err = cudaFree(d_u)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_u) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_v)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_v) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_u0)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_u0) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_v0)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_v0) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_rD)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_rD) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_gD)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_gD) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_bD)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_bD) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_r0)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_r0) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_g0)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_g0) failed: %s\n", cudaGetErrorString(err));
    if ((err = cudaFree(d_b0)) != cudaSuccess)
        fprintf(stderr, "cudaFree(d_b0) failed: %s\n", cudaGetErrorString(err));

    if (d_rgba)
    {
        if ((err = cudaFree(d_rgba)) != cudaSuccess)
            fprintf(stderr, "cudaFree(d_rgba) failed: %s\n", cudaGetErrorString(err));
    }

    if (h_rgba_pinned)
    {
        if ((err = cudaFreeHost(h_rgba_pinned)) != cudaSuccess)
            fprintf(stderr, "cudaFreeHost(h_rgba_pinned) failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceReset();
}

void Fluid2D::clear()
{
    CUDA_CHECK(cudaMemsetAsync(d_u, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_v, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_u0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_v0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_rD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_gD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_bD, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_r0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_g0, 0, mSize * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_b0, 0, mSize * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void Fluid2D::add_source(float *x, const float *s, float dt)
{
    int blockSize = 256;
    int gridSize = (mSize + blockSize - 1) / blockSize;
    PROFILE_KERNEL("add_source",
                   (add_source_kernel<<<gridSize, blockSize>>>(x, s, dt, mSize)));
    // cudaGetLastError() doesn't synchronize, just checks for launch errors
    // actual execution errors will be caught by next synchronizing call
}

void Fluid2D::set_bnd(int b, float *x)
{
    int blockSize = 256;
    int gridSize = (mN + blockSize - 1) / blockSize;
    set_bnd_edges_kernel<<<gridSize, blockSize>>>(b, x, mN);
    set_bnd_corners_kernel<<<1, 1>>>(x, mN);
}

void Fluid2D::lin_solve(int b, float *x, const float *x0, float a, float c, int iters)
{
    float invC = 1.0f / c;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((mN + BLOCK_SIZE - 1) / BLOCK_SIZE, (mN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int k = 0; k < iters; ++k)
    {
        // red-black Gauss-Seidel
        lin_solve_kernel<<<gridSize, blockSize>>>(b, x, x0, a, invC, mN, 0);
        set_bnd(b, x);

        lin_solve_kernel<<<gridSize, blockSize>>>(b, x, x0, a, invC, mN, 1);
        set_bnd(b, x);
    }
    // this boundary checks for errors, too
    CUDA_CHECK(cudaGetLastError());
}

void Fluid2D::diffuse(int b, float *x, const float *x0, float diff, float dt, int iters)
{
    float a = dt * diff * mN * mN;
    lin_solve(b, x, x0, a, 1.0f + 4.0f * a, iters);
}

void Fluid2D::advect(int b, float *d, const float *d0, const float *u, const float *v, float dt)
{
    float dt0 = dt * mN;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((mN + BLOCK_SIZE - 1) / BLOCK_SIZE, (mN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    PROFILE_KERNEL("advect",
                   (advect_kernel<<<gridSize, blockSize>>>(b, d, d0, u, v, dt0, mN)));
    set_bnd(b, d);
}

void Fluid2D::project(float *u, float *v, float *p, float *div, int iters)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((mN + BLOCK_SIZE - 1) / BLOCK_SIZE, (mN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    PROFILE_KERNEL("project_div",
                   (project_divergence_kernel<<<gridSize, blockSize>>>(div, p, u, v, mN)));
    set_bnd(0, div);
    set_bnd(0, p);

    lin_solve(0, p, div, 1.0f, 4.0f, iters);

    PROFILE_KERNEL("project_grad",
                   (project_gradient_kernel<<<gridSize, blockSize>>>(u, v, p, mN)));
    set_bnd(1, u);
    set_bnd(2, v);
}

void Fluid2D::vel_step(float visc, float dt, int iters)
{
    add_source(d_u, d_u0, dt);
    add_source(d_v, d_v0, dt);

    // Diffuse velocity
    std::swap(d_u0, d_u);
    diffuse(1, d_u, d_u0, visc, dt, iters);
    std::swap(d_v0, d_v);
    diffuse(2, d_v, d_v0, visc, dt, iters);

    project(d_u, d_v, d_u0, d_v0, iters);

    // Advect velocity
    std::swap(d_u0, d_u);
    std::swap(d_v0, d_v);
    advect(1, d_u, d_u0, d_u0, d_v0, dt);
    advect(2, d_v, d_v0, d_u0, d_v0, dt);

    project(d_u, d_v, d_u0, d_v0, iters);
}

void Fluid2D::dens_step(float *x, float *x0, float diff, float dt, int iters)
{
    add_source(x, x0, dt);
    std::swap(x0, x);
    diffuse(0, x, x0, diff, dt, iters);
    std::swap(x0, x);
    advect(0, x, x0, d_u, d_v, dt);
}

void Fluid2D::addSplat(float xN, float yN, float dxN, float dyN,
                       float r, float g, float b, const FluidParams &p)
{
    int cx = (int)std::floor(1.0f + xN * (mN - 1));
    int cy = (int)std::floor(1.0f + yN * (mN - 1));

    float rad = std::max(1.0f, p.splat_radius);
    int rInt = (int)std::ceil(rad);

    float fx = dxN * p.force;
    float fy = dyN * p.force;

    dim3 blockSize(8, 8);
    dim3 gridSize((2 * rInt + blockSize.x - 1) / blockSize.x,
                  (2 * rInt + blockSize.y - 1) / blockSize.y);

    PROFILE_KERNEL("splat",
                   (add_splat_kernel<<<gridSize, blockSize>>>(
                       d_u0, d_v0, d_r0, d_g0, d_b0,
                       cx, cy, rad, rInt, fx, fy, r, g, b, p.dye_amount, mN)));
}

void Fluid2D::step(const FluidParams &p)
{
    vel_step(p.visc, p.dt, p.iters);

    dens_step(d_rD, d_r0, p.diff, p.dt, p.iters);
    dens_step(d_gD, d_g0, p.diff, p.dt, p.iters);
    dens_step(d_bD, d_b0, p.diff, p.dt, p.iters);

#ifdef PROFILE_KERNELS
    prof_frame_count++;
#endif

    // apply dissipation
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((mN + BLOCK_SIZE - 1) / BLOCK_SIZE, (mN + BLOCK_SIZE - 1) / BLOCK_SIZE);
    PROFILE_KERNEL("decay",
                   (apply_decay_kernel<<<gridSize, blockSize>>>(
                       d_u, d_v, d_rD, d_gD, d_bD, p.vel_decay, p.dye_decay, mN)));

    // clear
    int blockSize1D = 256;
    int gridSize1D = (mSize + blockSize1D - 1) / blockSize1D;
    PROFILE_KERNEL("clear_sources",
                   (clear_sources_kernel<<<gridSize1D, blockSize1D>>>(
                       d_u0, d_v0, d_r0, d_g0, d_b0, mSize)));

    CUDA_CHECK(cudaGetLastError());
}

void Fluid2D::toRGBA(std::vector<std::uint8_t> &outRGBA, float gain, float gamma) const
{
    const size_t required_size = (size_t)mN * (size_t)mN * 4;
    outRGBA.resize(required_size);

    // allocate device buffer on first call
    if (d_rgba_size != required_size)
    {
        if (d_rgba)
        {
            CUDA_CHECK(cudaFree(d_rgba));
        }
        CUDA_CHECK(cudaMalloc(&d_rgba, required_size));
        d_rgba_size = required_size;
    }

    if (required_size > 256 * 1024)
    {
        if (h_rgba_pinned_size != required_size)
        {
            if (h_rgba_pinned)
            {
                CUDA_CHECK(cudaFreeHost(h_rgba_pinned));
            }
            CUDA_CHECK(cudaMallocHost(&h_rgba_pinned, required_size));
            h_rgba_pinned_size = required_size;
        }
    }

    float invGamma = 1.0f / std::max(0.0001f, gamma);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((mN + BLOCK_SIZE - 1) / BLOCK_SIZE, (mN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    PROFILE_KERNEL("toRGBA",
                   (toRGBA_kernel<<<gridSize, blockSize>>>(d_rD, d_gD, d_bD, d_rgba, gain, invGamma, mN)));

    // copy result back to host; pinned mem if large, direct if small enough
    if (h_rgba_pinned && required_size > 256 * 1024)
    {
        CUDA_CHECK(cudaMemcpy(h_rgba_pinned, d_rgba, required_size, cudaMemcpyDeviceToHost));
        memcpy(outRGBA.data(), h_rgba_pinned, required_size);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(outRGBA.data(), d_rgba, required_size, cudaMemcpyDeviceToHost));
    }
}
