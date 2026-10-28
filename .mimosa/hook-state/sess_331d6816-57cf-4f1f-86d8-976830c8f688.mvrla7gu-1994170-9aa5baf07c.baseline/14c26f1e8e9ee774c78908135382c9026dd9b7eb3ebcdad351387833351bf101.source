"""tile_order_eval.py L RATES [MAX_EXPERTS]: quality of T16 with exllamav3's tensor-core position order inside each 16x16
tile ("tc", stock) against column-major order ("cm": position t = row t % 16, column t // 16; set T16_PERM=cm), same
quantizer otherwise (exllamav3 v1.6.4 + make_mul16_exllamav3.sh on PYTHONPATH; LDLQ, Viterbi, Hadamard unchanged; only
tensor_core_perm / tensor_core_perm_i are replaced). Damping: sigma_reg = 2 x mean diag (V2's "damp2").
Experts: dumps/udump/LNN_eNNN.pt (W1/W3/W2, H1/H2, calib inputs) and dumps/edump/ (Lucebox eval inputs), from R2
dsx/t16/dumps/. Scores as quant/t16/q_eval.py: per expert err = sum (y_hat - y)^2, ye = sum y^2 on both token sets.
Writes rows_L{L}_{perm}.json (resumable). RATES e.g. 1.5,1.5,2 (gate, up, down)."""
import glob, json, os, sys, time
from functools import lru_cache
import torch
import torch.nn.functional as F
import exllamav3.modules.quant.exl3_lib.quantize as QZ
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

L, rates = int(sys.argv[1]), [float(x) if "." in x else int(x) for x in sys.argv[2].split(",")]
max_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 999
perm = os.environ.get("T16_PERM", "tc")
dev = torch.device("cuda:0")
ROWS = 4096

if perm == "cm":
    @lru_cache
    def cm_perm(device):
        return torch.tensor([(t % 16) * 16 + t // 16 for t in range(256)], dtype=torch.long, device=device)

    @lru_cache
    def cm_perm_i(device):
        return torch.argsort(cm_perm(device))

    QZ.tensor_core_perm, QZ.tensor_core_perm_i = cm_perm, cm_perm_i


def expert_f(x, W1, W3, W2, g, lim):
    """fp32 SwiGLU expert with the training clamps and the routing weight (quant/experiments/quant_variants.py)"""
    out = []
    for c in range(0, len(x), ROWS):
        xx, gg = x[c:c + ROWS].float(), g[c:c + ROWS].float()
        gate, up = xx @ W1.T, xx @ W3.T
        if lim > 0:
            up = up.clamp(-lim, lim)
            gate = gate.clamp(max=lim)
        out.append((F.silu(gate) * up * gg) @ W2.T)
    return torch.cat(out)


def exl3(W, H, K, key):
    """W [out, in] -> quantized [out, in] (exllamav3 takes (in, out) and the input Hessian), sigma_reg 2"""
    H_data = {"H": H.to(dev, torch.float32).clone(), "first_key": key, "count": 1, "finalized": False, "num_total": 0,
              "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": 2.0}
    wq, _, _ = quantize_exl3(W.t().contiguous().float(), H_data, qa, return_weight_q=True)
    return wq.t().contiguous().float().to(dev)


out_path = f"rows_L{L}_{perm}.json"
rows = json.load(open(out_path)) if os.path.exists(out_path) else []
done = {r["ex"] for r in rows}
files = sorted(glob.glob(f"dumps/udump/L{L:02d}_e*.pt"))[:max_ex]
for f in files:
    e = int(f[-6:-3])
    fe = f"dumps/edump/L{L:02d}_e{e:03d}.pt"
    if e in done or not os.path.exists(fe):
        continue
    d, de = torch.load(f), torch.load(fe)
    if not len(d["xv"]) or not len(de["xv"]):
        continue
    W1, W3, W2 = (d[k].float().to(dev) for k in ("W1", "W3", "W2"))
    H1, H2, lim = d["H1"].to(dev), d["H2"].to(dev), d["lim"]
    torch.cuda.synchronize()
    tq = time.time()
    a, b, c = exl3(W1, H1, rates[0], "gate"), exl3(W3, H1, rates[1], "up"), exl3(W2, H2, rates[2], "down")
    torch.cuda.synchronize()
    r = {"ex": e, "perm": perm, "rates": rates, "quant_s": round(time.time() - tq, 2)}
    for k, (x, g) in {"calib": (d["xv"], d["gv"]), "eval": (de["xv"], de["gv"])}.items():
        x, g = x.to(dev), g.to(dev)
        y, yq = expert_f(x, W1, W3, W2, g, lim), expert_f(x, a, b, c, g, lim)
        r[k] = {"n": len(x), "err": float((yq - y).square().sum()), "ye": float(y.square().sum())}
    rows.append(r)
    json.dump(rows, open(out_path, "w"))
    print(f"L{L} e{e} {perm}: calib {r['calib']['err'] / r['calib']['ye']:.4f}  eval {r['eval']['err'] / r['eval']['ye']:.4f}  quantize {r['quant_s']} s", flush=True)
