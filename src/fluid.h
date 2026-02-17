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

    std::vector<float> u, v, u0, v0;

    std::vector<float> rD, gD, bD, r0, g0, b0;

    inline int IX(int i, int j) const { return i + (mN + 2) * j; }

    // stable fluids core
    void add_source(std::vector<float> &x, const std::vector<float> &s, float dt);
    void set_bnd(int b, std::vector<float> &x);
    void lin_solve(int b, std::vector<float> &x, const std::vector<float> &x0, float a, float c, int iters);
    void diffuse(int b, std::vector<float> &x, const std::vector<float> &x0, float diff, float dt, int iters);
    void advect(int b, std::vector<float> &d, const std::vector<float> &d0, const std::vector<float> &u, const std::vector<float> &v, float dt);
    void project(std::vector<float> &u, std::vector<float> &v, std::vector<float> &p, std::vector<float> &div, int iters);

    // velocity and density step functions
    void vel_step(float visc, float dt, int iters);
    void dens_step(std::vector<float> &x, std::vector<float> &x0, float diff, float dt, int iters);

    static float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
};
