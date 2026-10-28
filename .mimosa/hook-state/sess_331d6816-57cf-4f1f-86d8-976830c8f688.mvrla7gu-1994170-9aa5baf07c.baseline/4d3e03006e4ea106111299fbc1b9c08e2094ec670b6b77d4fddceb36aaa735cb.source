"""t16_selftest.py (PYTHONPATH: the T16 exllamav3 tree, quant/v41): the worker's Q8_0 encoder is bit-exact with
v41_formats.encode_q8_0, and T16 quantizes a Gaussian matrix at K 1.5 / 2 / 2.5 with the expected error."""
import os, sys
sys.argv = ["t16_worker.py", "0", "/dev/null", "/dev/null", "/dev/null"]
import torch
import v41_formats as F
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t16_worker.py")).read()
ns = {"__name__": "selftest"}
exec(src.replace('off = json.load(open(gguf + ".experts.json"))["layers"]', "off = {}"), ns)
torch.manual_seed(0)
ok = True
for R, C in ((2304, 5120), (5120, 2304)):
    W = torch.randn(R, C, device="cuda") * 0.02
    W[0, :32] = 0; W[1, 5] = 0.5
    a, b = ns["q8_0"](W), F.encode_q8_0(W.cpu().numpy()).tobytes()
    same = a == b
    print(f"q8_0 {R}x{C}: bit-exact {same}", flush=True)
    ok &= same
W = torch.randn(2304, 5120, device="cuda")
for K in (1.5, 2, 2.5):
    Q, proxy, pk = ns["exl3"](W, None, K, "gauss")
    print(f"  packed: {[(n, str(t.dtype), list(t.shape)) for n, t in pk.items()]}", flush=True)
    mse = float((Q.cuda() - W).square().mean())
    print(f"T16 K{K} gaussian mse {mse:.4f} proxy {proxy:.4f}", flush=True)
    ok &= mse < {1.5: 0.15, 2: 0.085, 2.5: 0.055}[K]
sys.exit(0 if ok else 1)
