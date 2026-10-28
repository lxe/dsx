"""test_iq2_native.py - native IQ2_XXS quantizer (v41_iq2.py, native/qkern.cu) against the torch reference
(quant_variants.IQ2XXS) on dumped experts: scales, GPTQ values, GGUF encode -> decode, time.

usage: python test_iq2_native.py DUMPDIR [--limit N]"""
import argparse, glob, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "v41"), os.path.join(HERE, "..")]
import v41_gptq as G  # noqa: E402
import v41_iq2 as I  # noqa: E402
import v41_iqgrid  # noqa: E402
import quant_variants as QV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--limit", type=int, default=2)
    a = ap.parse_args()
    ref = QV.IQ2XXS({"iq2xxs_grid": [v - (1 << 64) if v >= 1 << 63 else v for v in v41_iqgrid.IQ2XXS_GRID],
                     "ksigns_iq2xs": v41_iqgrid.KSIGNS_IQ2XS})
    for f in sorted(glob.glob(os.path.join(a.dump, "*.pt")))[:a.limit]:
        dmp = torch.load(f)
        for name, W, H in (("gate+up", torch.cat([dmp["W1"], dmp["W3"]]).float().cuda(), dmp["H1"].cuda()), ("down", dmp["W2"].float().cuda(), dmp["H2"].cuda())):
            R, C = W.shape
            d = torch.diag(H).clamp_min(0) + 1e-8
            torch.cuda.synchronize(); t0 = time.time()
            dsup, ls, db = I.search_scales(W, d)
            Hinv, dead = G.hinv_chol(H, 1.0)
            Q, IDX, SGN = I.gptq_batched(W[None], db[None], d[None], Hinv[None], dead[None])
            torch.cuda.synchronize(); tn = time.time() - t0
            db_ref = ref.static_scales(W, d)
            t0 = time.time()
            Q_ref = ref.gptq(W, H, 1.0)
            torch.cuda.synchronize(); tr = time.time() - t0
            raw = I.encode(IDX[0].cpu().numpy(), SGN[0].cpu().numpy(), ls.cpu().numpy(), dsup.cpu().numpy())
            dec = torch.from_numpy(I.decode(raw, C)).cuda()
            err = lambda q: float(((q - W) @ H * (q - W)).sum() / (W @ H * W).sum())
            print(f"{os.path.basename(f)} {name} {R}x{C}: scales equal {float((db == db_ref).float().mean()):.4f}, "
                  f"values equal {float((Q[0] == Q_ref).float().mean()):.4f}, decode == Q {bool(torch.equal(dec, Q[0]))}, "
                  f"H-error native {err(Q[0]):.5f} ref {err(Q_ref):.5f}, time native {tn:.2f} s ref {tr:.2f} s", flush=True)


if __name__ == "__main__":
    main()
