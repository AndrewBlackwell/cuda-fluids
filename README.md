# WIP cuda fluid sim

#### Linux

```bash
sudo apt update
sudo apt install build-essential cmake git ninja-build
sudo apt install libgl1-mesa-dev libglu1-mesa-dev libglew-dev
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

### Build Tools

- CMake 3.20+
- C++17 compatible compiler (GCC 9+ on Linux, Clang on macOS)
- CUDA Toolkit 11.8+
- Git (for fetching dependencies)

```bash
./build.sh
```

or

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```
