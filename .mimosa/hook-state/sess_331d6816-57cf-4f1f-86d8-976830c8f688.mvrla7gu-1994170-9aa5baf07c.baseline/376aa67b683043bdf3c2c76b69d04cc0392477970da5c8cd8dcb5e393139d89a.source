"""t16_golden.py OUTDIR: golden T16 files from exllamav3 itself (T16-patched build on PYTHONPATH), for checking a T16
decoder. For K = 1.5, 2, 2.5 and each tile order (cm = column-major positions, what the full quant writes; tc =
exllamav3's tensor-core order), quantize a Gaussian W (in 512 x out 256, seed per K) with a random SPD Hessian
(sigma_reg 2, as the full run) and write OUTDIR/t16_golden_K{K}_{order}.safetensors:
  trellis int16 [in/16][out/16][16K], suh fp16 [in], svh fp16 [out], mul1 int32 [] (0x3A2F): as quantize_exl3 returns them
  w_inner fp16 (in, out): the rotated-basis weight, decoded by exllamav3's reconstruct kernel
  w_eff   fp16 (in, out): LinearEXL3.get_weight_tensor(), i.e. had128 / suh / had128 / svh applied: y = x @ w_eff
  weight_q fp32 (in, out): the quantizer's own effective weight (fp32 scales, before fp16 rounding of suh / svh)
  x fp32 [in]; y fp32 [out] = x @ w_eff (fp32); y_exl3 fp16 [out] (tc only): LinearEXL3.forward on the reconstruct path
exllamav3's reconstruct kernel places position t at tile element tensor_core_perm[t]; for cm files its output is
re-permuted in Python (value at position t moved to element (t % 16) * 16 + t // 16) before get_weight_tensor's
Hadamards and scales, which are unchanged. Orientation of every weight here: (in, out), row k = input k."""
import os, sys
import torch
from safetensors.torch import save_file
import exllamav3.modules.quant.exl3_lib.quantize as QZ
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from exllamav3.modules.quant.exl3 import LinearEXL3

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda:0")
IN, OUT = 512, 256
tc_perm_f, tc_perm_i_f = QZ.tensor_core_perm, QZ.tensor_core_perm_i
cm = torch.tensor([(t % 16) * 16 + t // 16 for t in range(256)], dtype=torch.long, device=dev)


def tiles(w):   # (in, out) -> [tile][256], tile element r * 16 + c, tiles in row-major tile order
    return w.view(IN // 16, 16, OUT // 16, 16).permute(0, 2, 1, 3).reshape(-1, 256)


def untiles(t):
    return t.view(IN // 16, OUT // 16, 16, 16).permute(0, 2, 1, 3).reshape(IN, OUT)


for order in ("cm", "tc"):
    if order == "cm":
        QZ.tensor_core_perm, QZ.tensor_core_perm_i = (lambda d: cm.to(d)), (lambda d: torch.argsort(cm).to(d))
    else:
        QZ.tensor_core_perm, QZ.tensor_core_perm_i = tc_perm_f, tc_perm_i_f
    for K in (1.5, 2, 2.5):
        g = torch.Generator(device="cpu").manual_seed(int(K * 10))
        W = torch.randn(IN, OUT, generator=g)
        A = torch.randn(2048, IN, generator=g)
        H = (A.T @ A / 2048).to(dev)
        x = torch.randn(IN, generator=g)
        H_data = {"H": H.clone(), "first_key": "golden", "count": 1, "finalized": False, "num_total": 0,
                  "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
        qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": 2.0}
        weight_q, proxy, ts = quantize_exl3(W.to(dev), H_data, qa, return_weight_q=True)
        lin = LinearEXL3(None, IN, OUT, suh=ts["suh"], svh=ts["svh"], trellis=ts["trellis"], mul1=ts["mul1"])
        inner = lin.get_inner_weight_tensor()
        if order == "cm":   # position t's value sits at tc_perm[t]; move it to cm[t]
            t_tc = tiles(inner)
            t_cm = torch.empty_like(t_tc)
            t_cm[:, cm] = t_tc[:, tc_perm_f(dev).long()]
            inner = untiles(t_cm).contiguous()
            lin.get_inner_weight_tensor = lambda inner=inner: inner.clone()
        w_eff = lin.get_weight_tensor()
        y = x.to(dev) @ w_eff.float()
        rec = {"trellis": ts["trellis"], "suh": ts["suh"], "svh": ts["svh"], "mul1": ts["mul1"].reshape(()),
               "w_inner": inner, "w_eff": w_eff, "weight_q": weight_q.float(), "x": x, "y": y}
        if order == "tc":
            rec["y_exl3"] = lin.forward(x.to(dev).half().view(1, IN).contiguous(), {"reconstruct": True}).view(OUT)
        rec = {k: v.detach().contiguous().cpu() for k, v in rec.items()}
        f = os.path.join(out_dir, f"t16_golden_K{K}_{order}.safetensors")
        save_file(rec, f, metadata={"K": str(K), "order": order, "orientation": "(in, out); y = x @ w_eff",
                                    "in": str(IN), "out": str(OUT), "sigma_reg": "2", "seed": str(int(K * 10))})
        dq = (w_eff.float() - weight_q.float()).abs().max() / weight_q.float().abs().max()
        msg = f"{os.path.basename(f)}: trellis {tuple(ts['trellis'].shape)}  |w_eff - weight_q| / max {dq:.2e}"
        if order == "tc":
            msg += f"  |y_exl3 - y| / max {((rec['y_exl3'].float() - rec['y']).abs().max() / rec['y'].abs().max()):.2e}"
        print(msg, flush=True)
