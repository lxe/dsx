"""t16_qselftest.py (PYTHONPATH: the T16 exllamav3 tree): checks a T16 build on this GPU before a full quant.
- the build is T16 (codebook marker 0x3A2F) with column-major tile positions;
- the dense fractional-K kernel gives the same indices as exllamav3's generic one (EXL3_QTF_OPT=0) at K 1.5 and 2.5;
- quantize_exl3 on a Gaussian 2304 x 5120 matrix (H = I) at K 1.5 / 2 / 2.5: expected error, and its time.
Exit 0 when everything passes."""
import os, sys, time
import torch
import exllamav3.modules.quant.exl3_lib.quantize as QZ
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

dev = torch.device("cuda:0")
ok = True
cm = [(t % 16) * 16 + t // 16 for t in range(256)]
order = QZ.tensor_core_perm(dev).tolist() == cm
print(f"{torch.cuda.get_device_name(0)}; column-major tile order {order}, T16 codebook {QZ.codebook_mul1_mult == 0x3A2F}", flush=True)
ok &= order and QZ.codebook_mul1_mult == 0x3A2F

torch.manual_seed(0)
tiles = torch.randn(4096, 256, device=dev)
for K in (1.5, 2.5):
    res = {}
    for opt in ("0", "1"):
        os.environ["EXL3_QTF_OPT"] = opt
        torch.cuda.synchronize(); t = time.time()
        _, qi = QZ.quantize_tiles_frac(tiles, {"K": K, "mul1": True})
        torch.cuda.synchronize()
        res[opt] = (qi.clone(), time.time() - t)
    same = torch.equal(res["0"][0], res["1"][0])
    print(f"frac kernel K {K}: indices identical to exllamav3's generic kernel {same}  ({res['0'][1]:.2f} s -> {res['1'][1]:.2f} s, 4096 tiles)", flush=True)
    ok &= same
os.environ["EXL3_QTF_OPT"] = "1"

W = torch.randn(2304, 5120, device=dev)
for K in (1.5, 2, 2.5):
    H_data = {"H": torch.eye(5120, device=dev), "first_key": "gauss", "count": 1, "finalized": False, "num_total": 0,
              "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": 2.0}
    torch.cuda.synchronize(); t = time.time()
    wq, proxy, packed = quantize_exl3(W.t().contiguous(), H_data, qa, return_weight_q=True)
    torch.cuda.synchronize(); dt = time.time() - t
    mse = float((wq.float() - W.t()).square().mean())
    good = mse < {1.5: 0.15, 2: 0.085, 2.5: 0.055}[K] and tuple(packed["trellis"].shape) == (320, 144, int(16 * K))
    print(f"T16 K {K}: gaussian mse {mse:.4f}, trellis {tuple(packed['trellis'].shape)}, {dt:.2f} s  {'ok' if good else 'FAIL'}", flush=True)
    ok &= good
sys.exit(0 if ok else 1)
