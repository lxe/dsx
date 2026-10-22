# Hardware microbenchmarks

Reference machine: i7-12700K, 128 GB DDR4-3200 (dual channel), RTX 3090 on PCIe 4.0 x16 + RTX 3090 on PCIe 3.0 x4.
Sources: `membw.c`, `qdot.c` (synthetic weights, real kernels), `pcie.cu`, and `nvidia/src/bin/cpubench.rs`.

## Memory and PCIe

| Test | Result | Takeaway |
|---|---:|---|
| RAM read bandwidth | **39.5 GB/s** (2 threads already 38.5) | the decode bottleneck |
| Host-to-device, GPU0, pinned 4 KB pages | 13 GB/s | the IOMMU halves DMA from small pages |
| Host-to-device, GPU0, hugepage memory + `cudaHostRegister` | **24.7 GB/s** | the RAM arena must be hugepage-backed |
| Host-to-device, GPU1 (PCIe 3.0 x4) | 3.3 GB/s | GPU1 holds a fixed expert set |
| Host-to-device, both GPUs on PCIe 3.0 x16 CPU lanes (i9-10900X / X299), each alone and both at once | 12.4 + 12.4 GB/s | GPU1 swaps experts and takes half of prefill's stream (FINDINGS 8c) |
| Kernel launch: back-to-back / in a CUDA graph / launch + sync | 1.17 / 0.58 / 3.8-5.2 µs | |
| GPU → CPU → GPU flag round trip (mapped memory, `cuStreamWaitValue32`) | **5.3 µs** GPU0, 8.5 µs GPU1 | a per-layer CPU handoff is cheap |

## CPU expert kernels

Synthetic 2-bit matvec on the P-cores keeps up with RAM; the E-cores slow it down.

| Format | 16 P-core threads |
|---|---:|
| 2-bit (2.5 bpw) | 38.9 GB/s |
| 2-bit symmetric (2.31 bpw) | 38.6 GB/s |
| Ternary (1.70 bpw) | 35.6 GB/s (8 threads: 19.9) |
| Adding the 4 E-cores | 33-35 GB/s |

Engine kernels (`cpubench`), weights streamed from RAM:

| Format | 1 token, 16 threads | 6 tokens, 16 threads |
|---|---:|---:|
| Q2_0 | 42.4 GB/s | 39.6 GB/s |
| PTQ1_0 | 39.8 GB/s | 25.3 GB/s |
| Q3_K | 39.5 GB/s | 36.2 GB/s |
| Q8_0 | 40.1 GB/s | 39.3 GB/s |

One layer of 6 mixed-format experts from RAM: 1.58 ms for 1 token, 1.88 ms for 6. The thread pool must keep spinning
between layers: a parked pool takes ~470 µs to wake.
