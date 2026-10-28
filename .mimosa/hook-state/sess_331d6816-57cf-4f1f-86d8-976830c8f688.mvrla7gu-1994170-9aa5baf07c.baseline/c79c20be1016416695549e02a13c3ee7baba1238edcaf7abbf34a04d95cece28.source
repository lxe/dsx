"""t16_qworker.py WID DUMPDIR PACKDIR: one worker of the single-GPU full-model T16 quant.

The forward chain (v41_calib.py layer with V41_DUMP_ALL=DUMPDIR) writes every routed expert's weights and fit Hessians
to DUMPDIR/LNN/eNNN.pt and DUMPDIR/LNN/DONE when a layer is complete. Workers claim experts by renaming the file
(atomic), quantize gate / up / down with exllamav3's quantize_exl3 on the T16 build (make_mul16_exllamav3.sh: T16
codebook, column-major tile positions, H kept on the GPU, the dense fractional-K kernel) and write
PACKDIR/LNN/eNNN.safetensors (atomic rename; its presence marks the expert done), the layout dsx's converter
(quant/t16/t16_gguf.py) reads:
  L{LL}.e{EEE}.{gate,up,down}.{trellis,suh,svh,mul1}  as quantize_exl3 returns them: trellis I16 [in/16][out/16][16 K]
  for W in (in, out) orientation (gate / up 5120 -> 2304, down 2304 -> 5120), suh F16 [in], svh F16 [out], mul1 I32
  (0x3A2F), plus the expert's stats as JSON in the file's metadata ("stats"): per matrix proxy error, relative weight
  error wrel, gain <Wq, W> / <W, W> and H-weighted gain_h.
A restart skips finished experts and first takes back its own claims. Layout (bits gate, up, down): layers 0-19 1.5,
1.5, 2; layers 20-39 2, 2, 2.5 (gate and up share K). LDLQ damping sigma_reg = T16_SIGMA (default 2). Experts without a
Hessian (below the chain's --min-fit tokens, or unrouted) use H = I. CUDA_VISIBLE_DEVICES picks the GPU; Q_THREADS
(default 2) caps torch's CPU threads (pods cap CPUs by cgroup).
"""
import glob, json, os, sys, time
import torch

torch.set_num_threads(int(os.environ.get("Q_THREADS", "2")))
import exllamav3.modules.quant.exl3_lib.quantize as QZ  # noqa: E402
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

wid, dumpdir, packdir = sys.argv[1], sys.argv[2], sys.argv[3]
SIGMA = float(os.environ.get("T16_SIGMA", "2"))
dev = torch.device("cuda:0")
CM = [(t % 16) * 16 + t // 16 for t in range(256)]
assert QZ.tensor_core_perm(dev).tolist() == CM, "exllamav3 build without the column-major tile order (make_mul16_exllamav3.sh step 4)"
assert QZ.codebook_mul1_mult == 0x3A2F, "exllamav3 build without the T16 codebook"


def layout(L):
    return (1.5, 1.5, 2) if L < 20 else (2, 2, 2.5)


def exl3(W, H, K, key):
    """W [out, in] -> exllamav3's tensors for W^T (in, out) and stats: proxy error, relative weight error, and the gains
    <Wq, W> / <W, W> and (H-weighted, H = the input Hessian) tr(Wq H W^T) / tr(W H W^T): the MSE-optimal Viterbi
    shrinks the weights toward 0 by ~(1 - wrel), which per-expert output scales can undo later"""
    k = W.shape[1]
    Hm = torch.eye(k, device=dev) if H is None else H.to(dev, torch.float32)
    H_data = {"H": Hm.clone(), "first_key": key, "count": 1, "finalized": False, "num_total": 0,
              "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": SIGMA}
    wq, proxy, packed = quantize_exl3(W.t().contiguous(), H_data, qa, return_weight_q=True)   # it rewrites its input
    Wt, Wq = W.t(), wq.float()
    HW = Hm @ Wt                                   # (in, out)
    st = {"proxy": float(proxy), "wrel": float((Wq - Wt).square().sum() / W.square().sum()),
          "gain": float((Wq * Wt).sum() / Wt.square().sum()), "gain_h": float((Wq * HW).sum() / (Wt * HW).sum())}
    return packed, st


def process(L, f):
    t0 = time.time()
    d = torch.load(f, map_location="cpu")
    e = d["ex"]
    kg, ku, kd = layout(L)
    st = {"L": L, "ex": e, "n_fit": d["n_fit"], "bits": [kg, ku, kd], "sigma_reg": SIGMA, "hessian": d["H1"] is not None,
          "order": "column-major"}
    out = {}
    for role, Wk, Hk, K in (("gate", "W1", "H1", kg), ("up", "W3", "H1", ku), ("down", "W2", "H2", kd)):
        packed, ms = exl3(d[Wk].to(dev).float(), d[Hk], K, role)
        st.update({f"{n}_{role}": v for n, v in ms.items()})
        for n, t in packed.items():
            out[f"L{L:02d}.e{e:03d}.{role}.{n}"] = t.detach().reshape(()) if n == "mul1" else t.detach()
    out = {k: v.contiguous().cpu() for k, v in out.items()}
    st["seconds"] = round(time.time() - t0, 2)
    pd = os.path.join(packdir, f"L{L:02d}")
    os.makedirs(pd, exist_ok=True)
    save_file(out, os.path.join(pd, f"e{e:03d}.safetensors.tmp"), metadata={"stats": json.dumps(st)})
    os.replace(os.path.join(pd, f"e{e:03d}.safetensors.tmp"), os.path.join(pd, f"e{e:03d}.safetensors"))
    return e, st


def main():
    tag = f".claim{wid}"
    L = int(os.environ.get("T16_FIRST_LAYER", "0"))
    last = int(os.environ.get("T16_LAST_LAYER", "39"))
    while L <= last:
        ld, pd = os.path.join(dumpdir, f"L{L:02d}"), os.path.join(packdir, f"L{L:02d}")
        mine = sorted(glob.glob(os.path.join(ld, f"e*.pt{tag}")))   # my claims from before a restart first
        free = sorted(glob.glob(os.path.join(ld, "e*.pt")))
        f = None
        for c in mine + free:
            e = int(os.path.basename(c)[1:4])
            if os.path.exists(os.path.join(pd, f"e{e:03d}.safetensors")):   # done before (restarted after the write)
                try: os.remove(c)
                except FileNotFoundError: pass
                continue
            if c.endswith(tag):
                f = c; break
            try:
                os.rename(c, c + tag); f = c + tag; break
            except FileNotFoundError:   # another worker took it
                continue
        if f is None:
            if os.path.exists(os.path.join(ld, "DONE")) and not glob.glob(os.path.join(ld, "e*.pt*")):
                L += 1   # layer finished (every file claimed and removed)
            elif not os.path.isdir(ld) and any(os.path.isdir(os.path.join(dumpdir, f"L{M:02d}")) for M in range(L + 1, 40)):
                L += 1   # uploaded and removed by t16_pod_up.sh (the chain dumps layers in order)
            else:
                time.sleep(3)
            continue
        e, st = process(L, f)
        os.remove(f)
        print(f"w{wid} L{L} e{e} {st['seconds']} s  proxy {st['proxy_gate']:.4f} {st['proxy_up']:.4f} {st['proxy_down']:.4f}", flush=True)


if __name__ == "__main__":
    main()
