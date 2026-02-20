## Benchmarks & Results

This is my port of my previous fluid simulator, tested on an NVIDIA RTX A5000 (24GB, Ampere) running on Ubuntu 20.04 VM with CUDA 10.1.

In a sentence: the 4096² grid saturates the A5000 completely, and scaling is near-linear until memory bandwidth limit. I'm happy with these results.

### Methodology

- Grid sizes: 512², 2048², 4096²
- Profiling: CUDA events for kernel timing, nvidia-smi for GPU utilization
- 8 CUDA kernels: add_source, advect, project_div, project_grad, splat, decay, clear_sources, toRGBA
- Numbers below are mean values; GPU/Mem util use active samples (sm > 0)
- Latency stats are per-frame GPU kernel time (sum of kernels)

### Performance

| Grid Size | Frame Time | FPS Limit | avg. GPU Util | avg. Mem Util | Takeaway         |
| --------- | ---------- | --------- | ------------- | ------------- | ---------------- |
| 512²      | n/a        | n/a       | 25.07%        | 0.82%         | Underutilized    |
| 2048²     | 0.558ms    | 1793.2    | 61.23%        | 49.95%        | Good utilization |
| 4096²     | 7.346ms    | 176.1     | 88.23%        | 93.32%        | Fully saturated  |

#### Latency (Frame Time, GPU Kernels)

| Grid Size | mean    | p50     | p95     | p99     | min     | max     |
| --------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 2048²     | 0.558ms | 0.556ms | 0.608ms | 0.622ms | 0.154ms | 0.678ms |
| 4096²     | 7.346ms | 7.366ms | 7.390ms | 7.399ms | 2.328ms | 7.470ms |

Tail latency at 4096²: p99 is 7.399ms (0.72% over mean)

#### Kernel Breakdown (4096×4096)

| Kernel        | Time (ms) | % of Frame |
| ------------- | --------- | ---------- |
| decay         | 1.493     | 20.33%     |
| project_grad  | 0.588     | 8.00%      |
| clear_sources | 0.450     | 6.13%      |
| project_div   | 0.374     | 5.10%      |
| toRGBA        | 0.369     | 5.03%      |
| advect        | 0.338     | 4.60%      |
| add_source    | 0.283     | 3.85%      |
| splat         | 0.023     | 0.32%      |

### A Note On Scaling...

2048² to 4096²: 16× more cells, 13.16× slower (0.558ms -> 7.346ms)
At 4096², memory bandwidth is the bottleneck (100% utilization)

### System Specs

- GPU: NVIDIA RTX A5000, 24GB GDDR6, 768 GB/s bandwidth
- Memory: 936 MB allocated at 4096²
- Power: Peak 226W, Temp 66°C
- PCIe: 3.6-3.7 GB/s transfer bursts to display
