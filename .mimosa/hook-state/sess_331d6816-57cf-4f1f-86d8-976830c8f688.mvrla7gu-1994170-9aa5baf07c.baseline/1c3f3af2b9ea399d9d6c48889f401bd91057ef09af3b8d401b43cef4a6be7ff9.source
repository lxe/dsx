"""qtf_check.py TAG [L E,E,...]: run exllamav3's quantizer (T16 build, column-major tiles, damp 2) and save the indices,
so two runs can be compared bit for bit: EXL3_QTF_OPT=0 (generic fractional kernel) vs 1 (qt_frac_opt.cuh).
- 46,080 Gaussian tiles (one 2304 x 5120 matrix) through quantize_tiles_frac at K = 1.5 and 2.5, timed;
- with L and experts: quantize_exl3 on those experts' gate / up / down (dumps/udump, rates of the planned mix:
  layers 0-19 1.5/1.5/2, 20-39 2/2/2.5), timed per matrix; saves every tensor exllamav3 returns.
Writes qtf_{TAG}.pt. qtf_check.py cmp TAG1 TAG2 compares two of them."""
import os, sys, time
from functools import lru_cache
import torch

if sys.argv[1] == "cmp":
    a, b = torch.load(f"qtf_{sys.argv[2]}.pt"), torch.load(f"qtf_{sys.argv[3]}.pt")
    for k in a["data"]:
        same = all(torch.equal(x, y) for x, y in zip(a["data"][k], b["data"][k]))
        print(f"{k:28s} {'identical' if same else 'DIFFERENT'}   {a['time'][k]:7.2f} s -> {b['time'][k]:7.2f} s")
    sys.exit()

import exllamav3.modules.quant.exl3_lib.quantize as QZ
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


@lru_cache
def cm_perm(device):
    return torch.tensor([(t % 16) * 16 + t // 16 for t in range(256)], dtype=torch.long, device=device)


@lru_cache
def cm_perm_i(device):
    return torch.argsort(cm_perm(device))


QZ.tensor_core_perm, QZ.tensor_core_perm_i = cm_perm, cm_perm_i
dev = torch.device("cuda:0")
out = {"data": {}, "time": {}}


def timed(key, fn):
    torch.cuda.synchronize()
    t = time.time()
    r = fn()
    torch.cuda.synchronize()
    out["time"][key] = time.time() - t
    return r


torch.manual_seed(0)
tiles = torch.randn(46080, 256, device=dev)
QZ.quantize_tiles_frac(tiles[:256], {"K": 1.5, "mul1": True})  # warm-up
for K in (1.5, 2.5):
    q, qi = timed(f"gauss K{K}", lambda: QZ.quantize_tiles_frac(tiles, {"K": K, "mul1": True}))
    out["data"][f"gauss K{K}"] = [qi.cpu()]
q, qi = timed("gauss K2 (integer kernel)", lambda: QZ.quantize_tiles(tiles, {"K": 2, "mul1": True, "devices": [dev]}))
out["data"]["gauss K2 (integer kernel)"] = [qi.cpu()]

if len(sys.argv) > 2:
    L = int(sys.argv[2])
    rates = (1.5, 1.5, 2) if L < 20 else (2, 2, 2.5)
    for e in (int(x) for x in sys.argv[3].split(",")):
        d = torch.load(f"dumps/udump/L{L:02d}_e{e:03d}.pt")
        for name, W, H, K in (("gate", d["W1"], d["H1"], rates[0]), ("up", d["W3"], d["H1"], rates[1]), ("down", d["W2"], d["H2"], rates[2])):
            H_data = {"H": H.to(dev, torch.float32).clone(), "first_key": name, "count": 1, "finalized": False, "num_total": 0,
                      "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
            qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": 2.0}
            Wt = W.float().to(dev).t().contiguous()
            _, _, ts = timed(f"L{L} e{e} {name} K{K}", lambda: quantize_exl3(Wt, H_data, qa, return_weight_q=False))
            out["data"][f"L{L} e{e} {name} K{K}"] = [ts[k].cpu() for k in sorted(ts)]
            if "keys" not in out:
                out["keys"] = {k: (tuple(ts[k].shape), str(ts[k].dtype)) for k in sorted(ts)}
torch.save(out, f"qtf_{sys.argv[1]}.pt")
print(sys.argv[1], {k: round(v, 2) for k, v in out["time"].items()}, out.get("keys"))
