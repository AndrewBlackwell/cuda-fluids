#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "fluid.h"

// very minimal shader compilation utility
static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetShaderInfoLog(s, (GLsizei)sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}

// make program from vertex and fragment shader sources
static GLuint makeProgram(const char *vs, const char *fs)
{
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "aPos");
    glBindAttribLocation(p, 1, "aUV");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetProgramInfoLog(p, (GLsizei)sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// HSV->RGB for pretty dye
static void hsv2rgb(float h, float s, float v, float &r, float &g, float &b)
{
    h = std::fmod(std::fmax(h, 0.0f), 1.0f) * 6.0f;
    int i = (int)std::floor(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    default:
        r = v;
        g = p;
        b = q;
        break;
    }
}

int main()
{
    if (!glfwInit())
        return 1;

    // requesting OpenGL 4.5 core profile context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const int winW = 1024, winH = 1024;
    GLFWwindow *window = glfwCreateWindow(winW, winH, "CUDA Fluid Simulation (OpenGL 4.5)", nullptr, nullptr);
    if (!window)
        return 2;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize glew, after creating gl context
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        return 3;
    }

    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450 core");

    // shaders
    const char *VS = R"GLSL(
    #version 450 core
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec2 aUV;
    out vec2 vUV;
    void main() {
      vUV = aUV;
      gl_Position = vec4(aPos, 0.0, 1.0);
    }
  )GLSL";

    const char *FS = R"GLSL(
    #version 450 core
    layout(location = 0) out vec4 FragColor;
    layout(binding = 0) uniform sampler2D uTex;
    in vec2 vUV;
    void main() {
      vec3 c = texture(uTex, vUV).rgb;
      FragColor = vec4(c, 1.0);
    }
  )GLSL";

    GLuint prog = makeProgram(VS, FS);
    glUseProgram(prog);

    // fullscreen quad
    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    float verts[] = {
        // first 2 are pos, last 2 are UV
        -1.f, -1.f, 0.f, 0.f,
        1.f, -1.f, 1.f, 0.f,
        1.f, 1.f, 1.f, 1.f,
        -1.f, 1.f, 0.f, 1.f};
    unsigned idx[] = {0, 1, 2, 2, 3, 0};

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

    // texture
    FluidParams P;
    Fluid2D sim(P.N);
    std::vector<std::uint8_t> rgba;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, P.N, P.N, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // interaction state
    double lastX = 0, lastY = 0;
    bool haveLast = false;
    float hue = 0.0f;

    bool paused = false;

    auto lastTime = std::chrono::high_resolution_clock::now();

    // main loop for simulation and rendering
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        int winW, winH;
        glfwGetWindowSize(window, &winW, &winH);
        glViewport(0, 0, fbW, fbH);

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI
        ImGui::Begin("Controls");
        ImGui::Checkbox("Pause", &paused);
        ImGui::SliderFloat("dt", &P.dt, 0.001f, 0.033f, "%.4f");
        ImGui::SliderInt("iters", &P.iters, 5, 120);
        ImGui::SliderFloat("viscosity", &P.visc, 0.0f, 0.001f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("dye diffusion", &P.diff, 0.0f, 0.001f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("vel decay", &P.vel_decay, 0.90f, 1.0f, "%.5f");
        ImGui::SliderFloat("dye decay", &P.dye_decay, 0.90f, 1.0f, "%.5f");
        ImGui::SliderFloat("splat radius (cells)", &P.splat_radius, 2.0f, 40.0f, "%.1f");
        ImGui::SliderFloat("force", &P.force, 0.0f, 1500.0f, "%.0f");
        ImGui::SliderFloat("dye amount", &P.dye_amount, 0.0f, 300.0f, "%.0f");

        if (ImGui::Button("Reset"))
            sim.clear();
        ImGui::Text("Drag LMB: inject dye + velocity");
        ImGui::End();

        // time (optionally keep dt fixed via UI; still compute for hue animation)
        auto now = std::chrono::high_resolution_clock::now();
        float realDt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        // "inject" dye into the simulation based on mouse input/movement
        const bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);

        if (lmb && !ImGui::GetIO().WantCaptureMouse)
        {
            if (!haveLast)
            {
                lastX = mx;
                lastY = my;
                haveLast = true;
            }

            float nx = (float)(mx / (double)winW);
            float ny = (float)(my / (double)winH);
            // flip Y so bottom is 0
            ny = 1.0f - ny;

            float dx = (float)((mx - lastX) / (double)winW);
            float dy = (float)((my - lastY) / (double)winH);
            dy = -dy; // because we flipped ny

            hue = std::fmod(hue + realDt * 0.10f, 1.0f);
            float rr, gg, bb;
            hsv2rgb(hue, 1.0f, 1.0f, rr, gg, bb);

            // staging sources by writing directly into sim's source buffers:
            // we do that by temporarily calling addSplat after clearing sources inside a manual step
            // the easiest way is to replicate the stable-fluids "sources" approach here by calling
            // addSplat() into internal u0/v0/r0/g0/b0, then running the pipeline

            // access via public method; it writes into internal source buffers (u0,v0,r0,g0,b0).
            sim.addSplat(nx, ny, dx, dy, rr, gg, bb, P);

            lastX = mx;
            lastY = my;
        }
        else
        {
            haveLast = false;
        }

        // run simulation pipeline (manual, clear sources once per frame)
        if (!paused)
        {
            sim.step(P);
        }

        // rendering
        sim.toRGBA(rgba, 1.3f, 2.2f);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, P.N, P.N, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // ImGui render
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
