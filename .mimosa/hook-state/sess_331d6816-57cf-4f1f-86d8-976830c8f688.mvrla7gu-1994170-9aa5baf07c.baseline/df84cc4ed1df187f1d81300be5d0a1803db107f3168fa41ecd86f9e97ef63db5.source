"""EXL3 CPU expert decode speed on this CPU, shaped like DeepSeek-V4.1-Flash: hidden 5120, intermediate 2304,
gate/up/down per expert, single-token decode, calls of 6 experts drawn from a pool larger than the caches.
Random trellis bytes (decode cost does not depend on the values). Prints GB/s of trellis read and ms per call,
the same quantities dsx's cpubench reports for its own formats.

usage: python exl3bench.py [--rates 1.5,2,2.5,3] [--experts 64] [--threads 8,14,16] [--calls 300]
"""
import argparse, glob, importlib.util, os, sys, time
import torch

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.join(here, "exllamav3")
so = os.path.join(here, "build", "exl3cpu.so")   # CPU-only build of exllamav3_ext/cpu/moe_mul1.cpp (build_cpu.py)
spec = importlib.util.spec_from_file_location("exl3cpu", so)
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
sys.path.insert(0, os.path.join(repo, "tests"))
from testlib.moe import repack_trellis   # noqa: E402

HID, INT = 5120, 2304
p = argparse.ArgumentParser()
p.add_argument("--rates", default="1.5,2,2.5,3")
p.add_argument("--experts", type=int, default=64)
p.add_argument("--threads", default="8,14,16")
p.add_argument("--calls", type=int, default=300)
a = p.parse_args()
tier = "vbmi" if ext.exl3_moe_cpu_has_avx512_vbmi() else "vnni" if ext.exl3_moe_cpu_has_avx512_vnni() else \
       "bw" if ext.exl3_moe_cpu_has_avx512_bw() else "avx2" if ext.exl3_moe_cpu_has_avx2() else "scalar"
print(f"tier {tier}; hidden {HID}, intermediate {INT}; {a.experts} experts per rate; calls of 6 experts, 1 token")
torch.set_num_threads(1)
gen = torch.Generator().manual_seed(1)


def trellis(k, n, K):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, int(16 * K)), dtype=torch.int16, generator=gen)
    g = ext.exl3_moe_cpu_swizzle_group(K)
    return repack_trellis(t, g, bool(ext.exl3_moe_cpu_planar_layout(K))) if g else t.contiguous()


for K in [float(r) for r in a.rates.split(",")]:
    mats = {}
    for name, (k, n) in (("g", (HID, INT)), ("u", (HID, INT)), ("d", (INT, HID))):
        mats[name] = ([trellis(k, n, K) for _ in range(a.experts)],
                      [(torch.randn(k, generator=gen) * 0.015).half() for _ in range(a.experts)],
                      [torch.randn(n, generator=gen).half() for _ in range(a.experts)])
    swz = int(ext.exl3_moe_cpu_swizzle_group(K) != 0)
    h = ext.exl3_moe_cpu_make_layer(*mats["g"], *mats["u"], *mats["d"], [], [], [], 0, 1.0, swz)
    per_expert = sum(mats[x][0][0].numel() * 2 for x in "gud")
    x = torch.randn(1, HID, generator=gen).half()
    w = torch.full((1, 6), 1 / 6, dtype=torch.half)
    out = torch.empty(1, HID)
    line = f"K={K:<4} {per_expert / 1e6:5.1f} MB/expert ({per_expert * 8 / (3 * HID * INT):.2f} bpw):"
    for th in [int(t) for t in a.threads.split(",")]:
        sels = [torch.randperm(a.experts, generator=gen)[:6].view(1, 6) for _ in range(a.calls)]
        for s in sels[:20]:
            ext.exl3_moe_cpu_forward(h, x, s, w, out, th)
        t0 = time.perf_counter()
        for s in sels:
            ext.exl3_moe_cpu_forward(h, x, s, w, out, th)
        dt = (time.perf_counter() - t0) / a.calls
        line += f"  {th:>2} thr {6 * per_expert / dt / 1e9:5.1f} GB/s ({dt * 1e3:.2f} ms)"
    print(line, flush=True)
    ext.exl3_moe_cpu_free_layer(h)
