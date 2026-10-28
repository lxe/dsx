"""t16_worker.py WID GGUF DUMPDIR STATSDIR: one GPU worker of the full-model T16 quant.

The forward chain (v41_calib.py layer with V41_DUMP_ALL=DUMPDIR) writes every routed expert's weights and fit Hessians
to DUMPDIR/LNN/eNNN.pt and DUMPDIR/LNN/DONE when a layer is complete. Workers claim experts by renaming the file
(atomic), quantize gate / up / down with exllamav3's quantize_exl3 on the patched T16 build (on PYTHONPATH), and write
the dequantized result as Q8_0 in place into GGUF (a v41_gguf.py --rtn q8_0 --experts-in-place file, offsets from
GGUF.experts.json), and with T16_PACKDIR exllamav3's packed tensors (trellis, suh, svh, mul1) per expert. STATSDIR/LNN/eNNN.json (proxy errors, bits) marks an expert done; a restart skips those and first
reclaims its own claimed files. CUDA_VISIBLE_DEVICES picks the GPU; Q_THREADS (default 2) caps torch's CPU threads
(pods cap CPUs by cgroup).

Layout (T16 bits gate, up, down): layers 0-19 1.5, 1.5, 2; layers 20-39 2, 2, 2.5 (UD-IQ1_M's bits within ~2%).
LDLQ damping sigma_reg = T16_SIGMA (default 2). Experts without a Hessian (below --min-fit tokens or unrouted) use H = I.
"""
import glob, json, os, sys, time
import torch

torch.set_num_threads(int(os.environ.get("Q_THREADS", "2")))
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3  # noqa: E402

wid, gguf, dumpdir, statsdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SIGMA = float(os.environ.get("T16_SIGMA", "2"))
PACKDIR = os.environ.get("T16_PACKDIR")   # exllamav3's packed tensors per expert (LNN/eNNN.safetensors), for dsx's T16 reader
dev = torch.device("cuda:0")
off = json.load(open(gguf + ".experts.json"))["layers"]


def layout(L):
    return (1.5, 1.5, 2) if L < 20 else (2, 2, 2.5)


def exl3(W, H, K, key):
    """W [out, in] -> T16-quantized [out, in] (exllamav3 takes (in, out) and the input Hessian); proxy error"""
    k = W.shape[1]
    Hm = torch.eye(k, device=dev) if H is None else H.to(dev, torch.float32)
    H_data = {"H": Hm.clone(), "first_key": key, "count": 1, "finalized": False, "num_total": 0,
              "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [dev], "mul1": True, "sigma_reg": SIGMA}
    wq, proxy, packed = quantize_exl3(W.t().contiguous().float().to(dev), H_data, qa, return_weight_q=True)
    return wq.t().contiguous().float(), float(proxy), packed


def q8_0(W):
    """W [R, C] (on the GPU) -> Q8_0 bytes, as ggml quantize_row_q8_0_ref / v41_formats.encode_q8_0: d = amax / 127
    (stored fp16), q = roundf(x / d) with ties away from zero"""
    R, C = W.shape
    g = W.float().view(R, C // 32, 32)
    d = g.abs().amax(-1) / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    x = g * idv.unsqueeze(-1)
    q = torch.trunc(x + torch.copysign(torch.full_like(x, 0.5), x)).to(torch.int8)
    out = torch.empty((R, C // 32, 34), dtype=torch.uint8, device=W.device)
    out[..., :2] = d.half().contiguous().view(torch.uint8).view(R, C // 32, 2)
    out[..., 2:] = q.view(torch.uint8)
    return out.cpu().numpy().tobytes()


def process(L, f, fd):
    t0 = time.time()
    d = torch.load(f, map_location="cpu")
    e = d["ex"]
    kg, ku, kd = layout(L)
    o = off[str(L)]["off"][str(e)]
    st = {"L": L, "ex": e, "n_fit": d["n_fit"], "bits": [kg, ku, kd], "sigma_reg": SIGMA, "hessian": d["H1"] is not None}
    packed = {}
    for role, Wk, Hk, K in (("gate", "W1", "H1", kg), ("up", "W3", "H1", ku), ("down", "W2", "H2", kd)):
        W = d[Wk].to(dev).float()
        Q, proxy, pk = exl3(W, d[Hk], K, role)
        for n, t in pk.items():   # exllamav3's packed T16 (trellis, suh, svh, mul1 marker) for a native dsx reader
            packed[f"blk.{L}.ffn_{role}_exps.e{e:03d}.{n}"] = t.detach().contiguous().cpu()
        st[f"proxy_{role}"] = proxy
        st[f"wrel_{role}"] = float((Q - W).square().sum() / W.square().sum())
        b = q8_0(Q)
        os.pwrite(fd, b, o[role])
    if PACKDIR:
        from safetensors.torch import save_file
        pd = os.path.join(PACKDIR, f"L{L:02d}")
        os.makedirs(pd, exist_ok=True)
        save_file(packed, os.path.join(pd, f"e{e:03d}.safetensors.tmp"))
        os.replace(os.path.join(pd, f"e{e:03d}.safetensors.tmp"), os.path.join(pd, f"e{e:03d}.safetensors"))
        if e == 0:
            st["packed_shapes"] = {k.split(".", 4)[-1]: [str(v.dtype), list(v.shape)] for k, v in packed.items()}
    st["seconds"] = round(time.time() - t0, 2)
    return e, st


def main():
    fd = os.open(gguf, os.O_RDWR)
    tag = f".claim{wid}"
    L = 0
    while L < 40:
        ld, sd = os.path.join(dumpdir, f"L{L:02d}"), os.path.join(statsdir, f"L{L:02d}")
        os.makedirs(sd, exist_ok=True)
        mine = sorted(glob.glob(os.path.join(ld, f"e*.pt{tag}")))   # my claims from before a restart first
        free = sorted(glob.glob(os.path.join(ld, "e*.pt")))
        f = None
        for c in mine + free:
            e = int(os.path.basename(c)[1:4])
            if os.path.exists(os.path.join(sd, f"e{e:03d}.json")):   # done before (e.g. restarted after the write)
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
            else:
                time.sleep(3)
            continue
        e, st = process(L, f, fd)
        json.dump(st, open(os.path.join(sd, f"e{e:03d}.json.tmp"), "w"))
        os.replace(os.path.join(sd, f"e{e:03d}.json.tmp"), os.path.join(sd, f"e{e:03d}.json"))
        os.remove(f)
        print(f"w{wid} L{L} e{e} {st['seconds']} s", flush=True)
    os.fsync(fd)
    os.close(fd)


if __name__ == "__main__":
    main()
