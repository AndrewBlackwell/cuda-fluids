#pragma once

#include <vector>
#include <cstdint>

struct FluidParams
{
    int N = 512; // grid interior size (N x N) cells
    float dt = 0.0325f;
    float visc = 0.00000f; // velocity diffusion (viscosity)
    float diff = 0.00000f; // dye diffusion (often 0 for crisp filaments)
    int iters = 10;        // solver iterations (diffusion + pressure)
    float vel_decay = 1.0f;
    float dye_decay = 0.950f;
    float splat_radius = 3.0f; // measured in cells
    float force = 1500.0f;     // velocity injection scale
    float dye_amount = 200.0f; // dye injection scale
};

class Fluid2D
{
public:
    explicit Fluid2D(int N);
    ~Fluid2D();

    int N() const { return mN; }

    void addSplat(float x, float y, float dx, float dy,
                  float r, float g, float b,
                  const FluidParams &p);

    void step(const FluidParams &p);

    void toRGBA(std::vector<std::uint8_t> &outRGBA,
                float gain = 1.2f, float gamma = 2.2f) const;

    void clear();

private:
    int mN;
    int mSize; // (N+2)*(N+2) including boundary padding

    // devie memory pointers
    float *d_u, *d_v, *d_u0, *d_v0;
    float *d_rD, *d_gD, *d_bD, *d_r0, *d_g0, *d_b0;
    // Persistent device buffer for RGBA output (avoid repeated alloc/free)
    mutable uint8_t *d_rgba;
    mutable size_t d_rgba_size;

    // pinned host memory for faster transfers
    mutable uint8_t *h_rgba_pinned;
    mutable size_t h_rgba_pinned_size;
    // stable fluids core
    void add_source(float *x, const float *s, float dt);
    void set_bnd(int b, float *x);
    void lin_solve(int b, float *x, const float *x0, float a, float c, int iters);
    void diffuse(int b, float *x, const float *x0, float diff, float dt, int iters);
    void advect(int b, float *d, const float *d0, const float *u, const float *v, float dt);
    void project(float *u, float *v, float *p, float *div, int iters);

    // velocity and density step functions
    void vel_step(float visc, float dt, int iters);
    void dens_step(float *x, float *x0, float diff, float dt, int iters);
};
