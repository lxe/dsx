# frac_k_time.py: time exllamav3's tile quantizer on one 2304 x 5120 matrix of Gaussian tiles (46,080 tiles): the
# fractional path (K = 1.5 / 2.5) at several batch sizes against the integer path at K = 2, interleaved A-B-B-A.
# Run with the T16 exllamav3 build on PYTHONPATH (as tile_order_eval.py).
import time, torch
import exllamav3.modules.quant.exl3_lib.quantize as QZ
dev = torch.device("cuda:0")
torch.manual_seed(0)
tiles = torch.randn(46080, 256, device=dev)

def frac(K, B):
    ka, mask = QZ.frac_k(K)
    edges = 65536 >> ka
    tc = torch.zeros((B, 2, edges), dtype=torch.half, device=dev)
    te = torch.zeros((B, 256, edges), dtype=torch.short, device=dev)
    q, qi = torch.zeros_like(tiles), torch.zeros_like(tiles, dtype=torch.short)
    torch.cuda.synchronize(); t = time.time()
    QZ.ext.quantize_tiles_frac(tiles, q, qi, tc, te, ka, mask)
    torch.cuda.synchronize(); dt = time.time() - t
    del tc, te
    return dt, float((q - tiles).square().mean()), qi

def integer(K):
    torch.cuda.synchronize(); t = time.time()
    q, qi = QZ.quantize_tiles(tiles, {"K": K, "mul1": True, "devices": [dev]})
    torch.cuda.synchronize()
    return time.time() - t, float((q - tiles).square().mean()), qi

frac(1.5, 128); integer(2)  # warm-up (allocations, first launches)
ref = None
for K, B in [("int", 2), (1.5, 128), (1.5, 256), (1.5, 512), (2.5, 128), (2.5, 256), (1.5, 512), (1.5, 256), (1.5, 128), ("int", 2)]:
    dt, mse, qi = integer(B) if K == "int" else frac(K, B)
    same = ""
    if K == 1.5:
        if ref is None: ref = qi
        same = " same idx as B=128" if torch.equal(qi, ref) else " DIFFERENT idx"
    print(f"K {'2 (int path, B 256)' if K == 'int' else f'{K} B {B}'}: {dt:.2f} s, mse {mse:.4f}{same}", flush=True)
