#include "fluid.h"
#include <cmath>
#include <algorithm>

#if defined(FLUID_USE_OPENMP)
#include <omp.h>
#endif

/**
 * @brief Constructs a 2D fluid simulator with grid resolution N×N.
 * @param N Grid resolution (actual internal grid is (N+2)×(N+2) with boundary padding)
 */
Fluid2D::Fluid2D(int N) : mN(N), mSize((N + 2) * (N + 2))
{
    u.assign(mSize, 0.0f);
    v.assign(mSize, 0.0f);
    u0.assign(mSize, 0.0f);
    v0.assign(mSize, 0.0f);

    rD.assign(mSize, 0.0f);
    gD.assign(mSize, 0.0f);
    bD.assign(mSize, 0.0f);
    r0.assign(mSize, 0.0f);
    g0.assign(mSize, 0.0f);
    b0.assign(mSize, 0.0f);
}

/**
 * @brief Clears all velocity and dye fields to zero.
 */
void Fluid2D::clear()
{
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(u0.begin(), u0.end(), 0.0f);
    std::fill(v0.begin(), v0.end(), 0.0f);
    std::fill(rD.begin(), rD.end(), 0.0f);
    std::fill(gD.begin(), gD.end(), 0.0f);
    std::fill(bD.begin(), bD.end(), 0.0f);
    std::fill(r0.begin(), r0.end(), 0.0f);
    std::fill(g0.begin(), g0.end(), 0.0f);
    std::fill(b0.begin(), b0.end(), 0.0f);
}

/**
 * @brief Adds a source field scaled by timestep to a field.
 * @param x Output field
 * @param s Source field
 * @param dt Timestep
 */
void Fluid2D::add_source(std::vector<float> &x, const std::vector<float> &s, float dt)
{
#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < mSize; ++i)
        x[i] += dt * s[i];
}

/**
 * @brief Sets boundary conditions for a field.
 * @param b Boundary type: 0=scalar, 1=u-velocity, 2=v-velocity
 * @param x Field to apply boundaries to
 */
void Fluid2D::set_bnd(int b, std::vector<float> &x)
{
    // b=0 scalar, b=1 u, b=2 v
    for (int i = 1; i <= mN; ++i)
    {
        // Set boundary conditions
        x[IX(0, i)] = (b == 1) ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(mN + 1, i)] = (b == 1) ? -x[IX(mN, i)] : x[IX(mN, i)];
        x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, mN + 1)] = (b == 2) ? -x[IX(i, mN)] : x[IX(i, mN)];
    }
    // corners
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, mN + 1)] = 0.5f * (x[IX(1, mN + 1)] + x[IX(0, mN)]);
    x[IX(mN + 1, 0)] = 0.5f * (x[IX(mN, 0)] + x[IX(mN + 1, 1)]);
    x[IX(mN + 1, mN + 1)] = 0.5f * (x[IX(mN, mN + 1)] + x[IX(mN + 1, mN)]);
}

/**
 * @brief Solves a linear system using Gauss-Seidel iteration.
 * @param b Boundary condition type
 * @param x Output solution field
 * @param x0 Input field
 * @param a Diffusion coefficient
 * @param c Inverse diagonal coefficient
 * @param iters Number of solver iterations
 */
void Fluid2D::lin_solve(int b, std::vector<float> &x, const std::vector<float> &x0, float a, float c, int iters)
{
    const float invC = 1.0f / c;
    for (int k = 0; k < iters; ++k)
    {
#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
        for (int j = 1; j <= mN; ++j)
        {
            for (int i = 1; i <= mN; ++i)
            {
                // Gauss-Seidel relaxation
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) * invC;
            }
        }
        set_bnd(b, x);
    }
}

/**
 * @brief Applies diffusion to a field.
 * @param b Boundary condition type
 * @param x Output diffused field
 * @param x0 Input field
 * @param diff Diffusion coefficient
 * @param dt Timestep
 * @param iters Solver iterations
 */
void Fluid2D::diffuse(int b, std::vector<float> &x, const std::vector<float> &x0, float diff, float dt, int iters)
{
    float a = dt * diff * mN * mN;
    // solve the resulting linear system
    lin_solve(b, x, x0, a, 1.0f + 4.0f * a, iters);
}

/**
 * @brief Advects a field along a velocity field using semi-Lagrangian method.
 * @param b Boundary condition type
 * @param d Output advected field
 * @param d0 Input field
 * @param uF Horizontal velocity field
 * @param vF Vertical velocity field
 * @param dt Timestep
 */
void Fluid2D::advect(int b, std::vector<float> &d, const std::vector<float> &d0,
                     const std::vector<float> &uF, const std::vector<float> &vF, float dt)
{
    float dt0 = dt * mN;

#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 1; j <= mN; ++j)
    {
        for (int i = 1; i <= mN; ++i)
        {
            // backtrace particle position
            float x = i - dt0 * uF[IX(i, j)];
            float y = j - dt0 * vF[IX(i, j)];

            // clamp to grid
            x = clampf(x, 0.5f, mN + 0.5f);
            y = clampf(y, 0.5f, mN + 0.5f);

            // get integer coords and interpolation weights
            int i0 = (int)std::floor(x);
            int i1 = i0 + 1;
            int j0 = (int)std::floor(y);
            int j1 = j0 + 1;

            // bilinear weights
            float s1 = x - i0, s0 = 1.0f - s1;
            float t1 = y - j0, t0 = 1.0f - t1;

            // bilinear interpolation
            d[IX(i, j)] =
                s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(b, d);
}

/**
 * @brief Projects velocity field to enforce incompressibility constraint.
 * @param uF Horizontal velocity (modified in-place)
 * @param vF Vertical velocity (modified in-place)
 * @param p Pressure field (temporary)
 * @param div Divergence field (temporary)
 * @param iters Solver iterations
 */
void Fluid2D::project(std::vector<float> &uF, std::vector<float> &vF,
                      std::vector<float> &p, std::vector<float> &div, int iters)
{
#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 1; j <= mN; ++j)
    {
        for (int i = 1; i <= mN; ++i)
        {
            // this is the negative divergence of the velocity field
            div[IX(i, j)] = -0.5f * ((uF[IX(i + 1, j)] - uF[IX(i - 1, j)] + vF[IX(i, j + 1)] - vF[IX(i, j - 1)]) / mN);
            p[IX(i, j)] = 0.0f;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);

    lin_solve(0, p, div, 1.0f, 4.0f, iters);

#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 1; j <= mN; ++j)
    {
        for (int i = 1; i <= mN; ++i)
        {
            // subtract pressure gradient from velocity field
            uF[IX(i, j)] -= 0.5f * mN * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            vF[IX(i, j)] -= 0.5f * mN * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    set_bnd(1, uF);
    set_bnd(2, vF);
}

static inline void SWAP(std::vector<float> &a, std::vector<float> &b) { a.swap(b); }

/**
 * @brief Advances velocity field by one timestep.
 * @param visc Viscosity coefficient
 * @param dt Timestep
 * @param iters Solver iterations
 */
void Fluid2D::vel_step(float visc, float dt, int iters)
{
    add_source(u, u0, dt);
    add_source(v, v0, dt);

    // diffuse velocity
    SWAP(u0, u);
    diffuse(1, u, u0, visc, dt, iters);
    SWAP(v0, v);
    diffuse(2, v, v0, visc, dt, iters);

    project(u, v, u0, v0, iters);

    // advect velocity
    SWAP(u0, u);
    SWAP(v0, v);
    advect(1, u, u0, u0, v0, dt);
    advect(2, v, v0, u0, v0, dt);

    project(u, v, u0, v0, iters);
}

/**
 * @brief Advances density field by one timestep.
 * @param x Output density field
 * @param x0 Input source density
 * @param diff Diffusion coefficient
 * @param dt Timestep
 * @param iters Solver iterations
 */
void Fluid2D::dens_step(std::vector<float> &x, std::vector<float> &x0, float diff, float dt, int iters)
{
    // simply diffuse and advect density
    add_source(x, x0, dt);
    SWAP(x0, x);
    diffuse(0, x, x0, diff, dt, iters);
    SWAP(x0, x);
    advect(0, x, x0, u, v, dt);
}

/**
 * @brief Adds a Gaussian splat of velocity and dye to the simulation.
 * @param xN Normalized x-coordinate [0,1]
 * @param yN Normalized y-coordinate [0,1]
 * @param dxN Normalized x-velocity delta
 * @param dyN Normalized y-velocity delta
 * @param r Red dye component [0,1]
 * @param g Green dye component [0,1]
 * @param b Blue dye component [0,1]
 * @param p Simulation parameters
 */
void Fluid2D::addSplat(float xN, float yN, float dxN, float dyN,
                       float r, float g, float b,
                       const FluidParams &p)
{
    // convert normalized [0,1] window coords to grid coords (1..N)
    int cx = (int)std::floor(1.0f + xN * (mN - 1));
    int cy = (int)std::floor(1.0f + yN * (mN - 1));

    float rad = std::max(1.0f, p.splat_radius);
    int rInt = (int)std::ceil(rad);

    // velocity injection in grid units
    float fx = dxN * p.force;
    float fy = dyN * p.force;

    // gaussian splat
    for (int j = cy - rInt; j <= cy + rInt; ++j)
    {
        if (j < 1 || j > mN)
            continue;
        for (int i = cx - rInt; i <= cx + rInt; ++i)
        {
            // this is inside the grid?
            if (i < 1 || i > mN)
                continue;

            // compute gaussian weight
            float dx = (float)(i - cx);
            float dy = (float)(j - cy);
            float d2 = dx * dx + dy * dy;
            float w = std::exp(-d2 / (2.0f * rad * rad));

            // apply splat to velocity and dye source fields
            int id = IX(i, j);
            u0[id] += fx * w;
            v0[id] += fy * w;

            // dye injection
            float add = p.dye_amount * w;
            r0[id] += add * r;
            g0[id] += add * g;
            b0[id] += add * b;
        }
    }
}

/**
 * @brief Performs one complete simulation step.
 * @param p Simulation parameters (viscosity, diffusion, decay, etc.)
 *
 * Updates velocity and dye fields, applies dissipation, and clears source terms.
 */
void Fluid2D::step(const FluidParams &p)
{
    // run stable fluids using whatever sources were accumulated into u0/v0/r0/g0/b0
    vel_step(p.visc, p.dt, p.iters);

    dens_step(rD, r0, p.diff, p.dt, p.iters);
    dens_step(gD, g0, p.diff, p.dt, p.iters);
    dens_step(bD, b0, p.diff, p.dt, p.iters);

    // dissipation
#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 1; j <= mN; ++j)
    {
        for (int i = 1; i <= mN; ++i)
        {
            // apply decay to velocity and dye
            int id = IX(i, j);
            u[id] *= p.vel_decay;
            v[id] *= p.vel_decay;
            rD[id] *= p.dye_decay;
            gD[id] *= p.dye_decay;
            bD[id] *= p.dye_decay;
        }
    }

    // clear the sources for next frame
    std::fill(u0.begin(), u0.end(), 0.0f);
    std::fill(v0.begin(), v0.end(), 0.0f);
    std::fill(r0.begin(), r0.end(), 0.0f);
    std::fill(g0.begin(), g0.end(), 0.0f);
    std::fill(b0.begin(), b0.end(), 0.0f);
}

/**
 * @brief Converts internal dye densities to RGBA image data.
 * @param outRGBA Output RGBA pixel buffer (8-bit per channel)
 * @param gain Brightness multiplier for dye values
 * @param gamma Gamma correction exponent
 *
 * Applies tone mapping (soft clamp), gamma correction, and converts to uint8.
 */
void Fluid2D::toRGBA(std::vector<std::uint8_t> &outRGBA, float gain, float gamma) const
{
    outRGBA.resize((size_t)mN * (size_t)mN * 4);

    // the inverse gamma value
    const float invGamma = 1.0f / std::max(0.0001f, gamma);

#if defined(FLUID_USE_OPENMP)
#pragma omp parallel for
#endif
    for (int y = 0; y < mN; ++y)
    {
        for (int x = 0; x < mN; ++x)
        {
            int i = x + 1;
            int j = y + 1;
            int id = IX(i, j);

            float rr = std::max(0.0f, rD[id] * gain);
            float gg = std::max(0.0f, gD[id] * gain);
            float bb = std::max(0.0f, bD[id] * gain);

            // soft clamp (prevents hard clipping of bright colors)
            rr = rr / (1.0f + rr);
            gg = gg / (1.0f + gg);
            bb = bb / (1.0f + bb);

            // gamma-ish :)
            rr = std::pow(rr, invGamma);
            gg = std::pow(gg, invGamma);
            bb = std::pow(bb, invGamma);

            // convert to uint8
            std::uint8_t R = (std::uint8_t)std::clamp(rr * 255.0f, 0.0f, 255.0f);
            std::uint8_t G = (std::uint8_t)std::clamp(gg * 255.0f, 0.0f, 255.0f);
            std::uint8_t B = (std::uint8_t)std::clamp(bb * 255.0f, 0.0f, 255.0f);

            // write to output RGBA buffer
            size_t o = ((size_t)y * (size_t)mN + (size_t)x) * 4;
            outRGBA[o + 0] = R;
            outRGBA[o + 1] = G;
            outRGBA[o + 2] = B;
            outRGBA[o + 3] = 255;
        }
    }
}