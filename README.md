# Benchmarking CUDA Kernels

In this mini-project I used Nvidia's Nsight Compute to write and profile 5 fundamental CUDA kernels on an RTX 4070S:

1) Vector add
2) Naive matmul
3) Tiled matmul
4) Fused softmax
5) Fused LayerNorm

This is an educational project that I did to get accustomed to `ncu` and understanding concepts like memory bandwidth utilization, Welford's algorithm, warp-level `__shfl_down_sync reduction` etc.

The benchmark scripts and profiling data are provided in each folder. No AI was used in this project.
