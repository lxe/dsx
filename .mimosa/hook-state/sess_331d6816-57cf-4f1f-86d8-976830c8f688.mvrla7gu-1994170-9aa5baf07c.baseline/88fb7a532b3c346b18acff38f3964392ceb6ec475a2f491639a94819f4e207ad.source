"""exllamav3 v1.6.4 (stock, mul1) GPU decode speed on DeepSeek-V4.1-Flash's expert shapes, as the outside reference
for dsx's T16 kernels: 1 token, 6 experts picked from a pool of 64 in VRAM (hidden 5120, intermediate 2304), random
trellises, K = 1.5 / 2 / 2.5.
  coop: ext.exl3_moe_coop, the fused decode MoE kernels of BlockSparseMLP's bsz <= MAX_BSZN path (run_bszN): input
        rotation + gate/up GEMVs + activation, then down GEMVs + the weighted sum, two launches per layer (mul1)
  gemm: ext.exl3_gemm with one row (m = 1), which takes exllamav3's int8-activation GEMV for mul1 up to
        exl3_gemv_int8_max_k bits: the 18 matrices of a layer call (6 x gate, up, down), one launch each
Timing: CUDA events over 400 calls, order moe, gemm, gemm, moe, 2 s pauses."""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/src/t16q/exl3-stock/tests"))
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts
from testlib.moe import expert_ptr_tables, experts_to, moe_buffers

H, I, E, CALLS, CAP = 5120, 2304, 64, 400, 128
dev = torch.device("cuda:0")
print(torch.cuda.get_device_name(0), "| int8 GEMV max K:", ext.exl3_gemv_int8_max_k(0) if hasattr(ext, "exl3_gemv_int8_max_k") else "?")
for K in (1.5, 2.0, 2.5):
    gen = torch.Generator().manual_seed(7)
    exd = experts_to(rand_experts(E, H, I, K, gen), dev)
    tabs = expert_ptr_tables(exd, dev)
    bufs = moe_buffers(H, I, CAP, dev)
    y = (torch.randn((1, H), generator = gen) * 0.05).half().to(dev)
    weight_sorted = torch.full((6,), 1 / 6, dtype = torch.half, device = dev)
    token_sorted = torch.zeros((6,), dtype = torch.long, device = dev)
    sels = [[(i * 7 + j * 11) % E for j in range(6)] for i in range(CALLS)]
    counts = []
    for s in sels:
        c = torch.zeros((E + 1,), dtype = torch.long)
        for e in s: c[e] += 1
        counts.append(c.to(dev))
    out = torch.zeros((1, H), dtype = torch.float, device = dev)

    sel_d = [torch.tensor([s], dtype = torch.long, device = dev) for s in sels]
    rw = torch.full((1, 6), 1 / 6, dtype = torch.half, device = dev)
    smax = 4 * 6
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = dev); gu_u = torch.empty_like(gu_g)
    act_out = torch.empty((smax, 1, I), dtype = torch.half, device = dev)
    d_out = torch.empty((smax, 1, H), dtype = torch.float, device = dev)
    cout = torch.empty((8, H), dtype = torch.float, device = dev)
    ctr = torch.zeros((smax * (I // 128) + 8 * (H // 128) + 2 * smax + 3,), dtype = torch.int, device = dev)
    had_g = torch.empty((6, H), dtype = torch.half, device = dev); had_u = torch.empty_like(had_g)

    def moe(i):
        ext.exl3_moe_coop(y, sel_d[i], rw, -1, -1, H, *tabs["g"], *tabs["u"], *tabs["d"], None, None, None,
                          K, K, K, False, True, 0, 0.0, True, had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, cout[:1],
                          None, None)

    C1 = torch.empty((1, I), dtype = torch.half, device = dev); C2 = torch.empty((1, H), dtype = torch.half, device = dev)
    y2 = (torch.randn((1, I), generator = gen) * 0.05).half().to(dev)
    ah1, ah2 = torch.empty_like(y), torch.empty_like(y2)

    def gemm(i):
        for e in sels[i]:
            for p in ("g", "u"):
                t, su, sv = exd[p][e]
                ext.exl3_gemm(y, t, C1, su, ah1, sv, 0, False, True, 0)
            t, su, sv = exd["d"][e]
            ext.exl3_gemm(y2, t, C2, su, ah2, sv, 0, False, True, 0)

    res = {"moe": [], "gemm": []}  # moe = exl3_moe_coop
    for name in ("moe", "gemm", "gemm", "moe"):
        f = moe if name == "moe" else gemm
        time.sleep(2)
        f(0); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        for i in range(CALLS): f(i)
        b.record(); torch.cuda.synchronize()
        res[name].append(a.elapsed_time(b) * 1e3 / CALLS)
    tb = 6 * 3 * H * I * K / 8
    for name, v in res.items():
        us = sum(v) / len(v)
        print(f"K {K}: exllamav3 {"coop (decode MoE)" if name == "moe" else "gemm m=1 x18 (int8 GEMV)"}: {us:7.1f} us per layer call ({', '.join(f'{x:.1f}' for x in v)}), {tb / us / 1e3:.0f} GB/s of trellis")
    del exd, tabs, bufs; torch.cuda.empty_cache()
