## Benchmarks & Results

This is my port of my previous fluid simulator, tested on an NVIDIA RTX A5000 (24GB, Ampere) running on Ubuntu 20.04 VM with CUDA 10.1.

In a sentence: the 4096² grid saturates the A5000 completely, and scaling is near-linear until memory bandwidth limit. I'm happy with these results.

### Methodology

- Grid sizes: 512², 2048², 4096²
- Profiling: CUDA events for kernel timing, nvidia-smi for GPU utilization
- 8 CUDA kernels: add_source, advect, project_div, project_grad, splat, decay, clear_sources, toRGBA

### Performance

| Grid Size | Frame Time | FPS Limit | GPU Util | Mem Util | Takeaway         |
| --------- | ---------- | --------- | -------- | -------- | ---------------- |
| 512²      | ~0.07ms    | ~14000    | 20-30%   | ~30%     | Underutilized    |
| 2048²     | 0.558ms    | 1793      | 60-85%   | 73-88%   | Good utilization |
| 4096²     | 7.346ms    | 136       | 75-100%  | 80-100%  | Fully saturated  |

#### Kernel Breakdown (4096×4096)

| Kernel        | Time (ms) | % of Frame |
| ------------- | --------- | ---------- |
| decay         | 1.493     | 20.3%      |
| project_grad  | 0.588     | 8.0%       |
| clear_sources | 0.450     | 6.1%       |
| project_div   | 0.374     | 5.1%       |
| toRGBA        | 0.369     | 5.0%       |
| advect        | 0.338     | 4.6%       |
| add_source    | 0.283     | 3.9%       |
| splat         | 0.023     | 0.3%       |

### A Note On Scaling...

2048² to 4096²: 16× more cells, 13× slower (which shows good efficiency)
At 4096², memory bandwidth is the bottleneck (100% utilization)

### System Specs

- GPU: NVIDIA RTX A5000, 24GB GDDR6, 768 GB/s bandwidth
- Memory: 936 MB allocated at 4096²
- Power: Peak 226W, Temp 66°C
- PCIe: 3.6-3.7 GB/s transfer bursts to display
