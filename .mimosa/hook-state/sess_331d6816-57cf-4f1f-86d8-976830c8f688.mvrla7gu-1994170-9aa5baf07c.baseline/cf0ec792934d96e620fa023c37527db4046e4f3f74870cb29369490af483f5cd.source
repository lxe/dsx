#!/usr/bin/env bash
# make_mul16_exllamav3.sh SRC DST - copy an exllamav3 checkout (v1.6.4) and turn its mul1 codebook slot into the
# "mul16" (T16) codebook, so exllamav3's own quantizer (Viterbi + LDLQ + Hadamard) produces T16 trellises.
# Only the quantize and reconstruct paths are meant to be used from the patched build; its inference kernels and
# file markers still say "mul1". Build each tree separately (stock = EXL3 mul1, patched = T16) and run the same
# quantization script against each.
#
# T16 decode of a 16-bit trellis state s:  p = (s * 0x3A2F) mod 2^16;  w = ((p & 0xFF) + (p >> 8) - 255) / 104.51
# (EXL3 mul1:                              p = (s * 0x83DCD12D) mod 2^32; w = (bytesum4(p) - 510) / 147.7)
set -euo pipefail
SRC=$1; DST=$2
rm -rf "$DST"; cp -r "$SRC" "$DST"
Q="$DST/exllamav3/exllamav3_ext/quant"
# 1. the multiplier everywhere the quantizer / reconstruct paths compute products (incl. the Viterbi step constants)
sed -i 's/0x83DCD12Du/0x3A2Fu/g; s/0x83DCD12D/0x3A2F/g' "$Q"/*.cu "$Q"/*.cuh
# 2. decode: keep the low 16 bits of the product before the byte sum, and the 2-byte-sum normalization
#    (mean 255, std 104.511: k_inv = fp16 0x20e6 = 0.009567, k_bias = fp16 0xca1e = (-1024 - 255) * k_inv)
C="$Q/codebook.cuh"
python3 - "$C" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
s = s.replace("uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);", "uint32_t sum0 = __dp4a(x0 & 0xFFFFu, 0x01010101u, acc);")
s = s.replace("uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);", "uint32_t sum1 = __dp4a(x1 & 0xFFFFu, 0x01010101u, acc);")
s = s.replace("uint32_t sum = __dp4a(x, 0x01010101u, acc);", "uint32_t sum = __dp4a(x & 0xFFFFu, 0x01010101u, acc);")
n1 = s.count("__ushort_as_half(0x1eee)"); n2 = s.count("__ushort_as_half(0xc931)")
s = s.replace("__ushort_as_half(0x1eee)", "__ushort_as_half(0x20e6)").replace("__ushort_as_half(0xc931)", "__ushort_as_half(0xca1e)")
assert n1 >= 2 and n2 >= 2 and s.count("& 0xFFFFu, 0x01010101u") == 3, (n1, n2)
open(p, "w").write(s)
print("codebook.cuh patched:", n1, n2)
PY
# 3. the Python-side constant (marker value only)
sed -i 's/^codebook_mul1_mult = 0x83DCD12D/codebook_mul1_mult = 0x3A2F/' "$DST/exllamav3/modules/quant/exl3_lib/quantize.py"
grep -n "codebook_mul1_mult" "$DST/exllamav3/modules/quant/exl3_lib/quantize.py"
grep -c "0x3A2F" "$Q"/quantize_tiles_kernel.cuh
# 4. column-major positions inside each tile (position t = row t % 16, column t // 16), the order dsx's T16 kernels
#    read; EXL3_T16_ORDER=tc keeps exllamav3's tensor-core order. Only the quantizer uses this tree, so its own
#    tensor-core reconstruct / inference kernels (which assume the stock order) are not used on these trellises
python3 - "$DST/exllamav3/modules/quant/exl3_lib/quantize.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    The cached index maps row-major tile elements into the lane/interleave order used by the CUDA encoder.\n    \"\"\"\n"
new = old + ("    if os.environ.get(\"EXL3_T16_ORDER\", \"cm\") != \"tc\":   # T16 patch: column-major positions\n"
             "        return torch.tensor([(t % 16) * 16 + t // 16 for t in range(256)], dtype = torch.int, device = device)\n")
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("tile order: column-major unless EXL3_T16_ORDER=tc")
PY
# 5. H stays on the GPU after finalize: exllamav3 moves it to the CPU for the proxy error and the scale refit, which
#    under a pod's cgroup CPU cap (2 threads per worker) ran ~20x slower than on the GPU
python3 - "$DST/exllamav3/modules/quant/exl3_lib/quantize.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = '        H = H.cpu()\n        H_data["H"] = H.cpu()\n'
assert s.count(old) == 1
open(p, "w").write(s.replace(old, '        H_data["H"] = H   # T16 patch: kept on the GPU\n'))
print("H kept on the GPU")
PY
# 6. the dense fractional-K quantizer (K = 1.5 / 2.5 / 3.5): same indices as exllamav3's generic kernel, ~5x faster
python3 "$(dirname "$0")/patch_qtf_opt.py" "$DST"
echo "T16 tree ready: $DST"
