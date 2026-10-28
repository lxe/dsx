#!/usr/bin/env python3
"""Back-of-envelope speed model for DeepSeek-V4.1-Flash(-1017) on a two-RTX-3090, 128 GB DDR4 machine.

Every number is a bandwidth or FLOP ceiling, not a measurement. Edit BOX / scenarios and rerun.
Model shape is from the official config.json; the measured anchors are Xolotl's RTX 4090 numbers.
"""

# ---- model (config.json) ----
EXPERTS = 40 * 384                      # routed experts in the backbone
EXPERT_PARAMS = 3 * 5120 * 2304         # w1, w2, w3 per expert = 35.4M
ROUTED_ACTIVE = 40 * 6 * EXPERT_PARAMS  # 8.49B routed params read per token
DENSE_ACTIVE = 16e9 - ROUTED_ACTIVE     # DeepSeek says 16B active in decode -> ~7.5B dense
KV_BYTES_PER_TOKEN = 890                # FP4 global KV (model card)

# ---- box (2x RTX 3090, 128 GB DDR4) ----
VRAM_FOR_EXPERTS = {1: 11e9, 2: 11e9 + 21e9}  # GPU0 also holds dense + KV + DSpark (~13 GB)
RAM_BW = 40e9            # DDR4-3200 dual channel: 51.2 GB/s peak, ~40 achievable
NVME_BW = 6e9            # SN850X large reads; striping both drives could ~double this
PCIE_GPU0 = 25e9         # PCIe 4.0 x16, real
DENSE_MS = 20.0          # Xolotl measured the dense half at ~19 ms/token on a 4090 (Q8_0); 3090 ~same bandwidth
GPU_PREFILL_FLOPS = 60e12  # one 3090, realistic mixed int8 MMQ / fp16 on these shapes
RAM_ARENA = 95e9         # pinned arena with Strata stopped (123 GB visible, leave ~25 for OS + page cache)


def pool_bytes(bpw):
    return EXPERTS * EXPERT_PARAMS * bpw / 8


def decode(bpw, gpus, vram_share=None, nvme_bw=NVME_BW):
    """ms per single-token pass. vram_share: fraction of routed traffic served from VRAM.
    None = uniform routing (traffic share = capacity share; DeepSeek's balanced router makes this the honest default)."""
    pool = pool_bytes(bpw)
    vram = min(VRAM_FOR_EXPERTS[gpus], pool)
    ram = min(RAM_ARENA, pool - vram)
    disk = pool - vram - ram
    per_tok = ROUTED_ACTIVE * bpw / 8
    if vram_share is None:
        f_v, f_r, f_d = vram / pool, ram / pool, disk / pool
    else:  # extra VRAM share comes out of RAM first, then disk, in proportion
        rest = 1 - vram_share
        f_v = vram_share
        f_r = rest * ram / (ram + disk) if ram + disk else 0
        f_d = rest - f_r
    t_ram = per_tok * f_r / RAM_BW * 1e3
    t_disk = per_tok * f_d / nvme_bw * 1e3
    # CPU computes RAM experts while disk reads land; GPU experts are ~free next to the dense part
    return DENSE_MS + max(t_ram, t_disk) + 0.3 * min(t_ram, t_disk), pool, disk


def prefill(bpw, gpus, chunk):
    """tok/s ceiling for a long prompt: weight streaming over PCIe vs GPU FLOPs."""
    pool = pool_bytes(bpw)
    stream_s = max(pool - VRAM_FOR_EXPERTS[gpus], 0) / PCIE_GPU0  # every chunk touches ~every expert
    flops_s = chunk * 2 * 16e9 / (GPU_PREFILL_FLOPS * gpus)
    return chunk / max(stream_s, flops_s)


if __name__ == "__main__":
    print(f"routed params/token {ROUTED_ACTIVE/1e9:.2f}B, dense ~{DENSE_ACTIVE/1e9:.1f}B")
    print(f"KV: 128K ctx = {131072*KV_BYTES_PER_TOKEN/1e6:.0f} MB, 1M = {2**20*KV_BYTES_PER_TOKEN/1e9:.2f} GB\n")
    print("decode, 1 token per pass (no speculative decoding)")
    print(f"{'experts bpw':>11} {'pool GB':>8} {'GPUs':>4} {'routing':>14} {'on NVMe GB':>10} {'ms/pass':>8} {'tok/s':>6}")
    for bpw in (2.4, 2.1, 1.9, 1.75):
        for gpus in (1, 2):
            for label, share, nv in (("uniform", None, NVME_BW), ("measured skew", 0.58 if gpus == 2 else 0.35, NVME_BW),
                                     ("biased 60% VRAM", 0.6, NVME_BW)):
                ms, pool, disk = decode(bpw, gpus, share, nv)
                print(f"{bpw:>11} {pool/1e9:>8.0f} {gpus:>4} {label:>14} {disk/1e9:>10.0f} {ms:>8.0f} {1e3/ms:>6.1f}")
    print("\nprefill ceiling, tok/s (PCIe streaming vs FLOPs)")
    for bpw in (2.4, 1.9):
        for gpus in (1, 2):
            row = "  ".join(f"chunk {c//1024}K: {prefill(bpw, gpus, c):>5.0f}" for c in (4096, 8192, 16384, 32768))
            print(f"bpw {bpw} GPUs {gpus}: {row}")
