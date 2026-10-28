#!/bin/bash
# T16 transfer scoring again: H kept on the GPU in exllamav3's finalize (its CPU un-rotation under a 13.6-CPU cgroup
# quota was the bottleneck), 2 torch threads per process, results saved per expert (resumable)
set -x
J=/root/t16job
F=/root/x3t16/exllamav3/modules/quant/exl3_lib/quantize.py
python3 - "$F" <<'PY'
import sys; p = sys.argv[1]; s = open(p).read()
old = '        H = H.cpu()\n        H_data["H"] = H.cpu()\n'
if old in s:
    s = s.replace(old, '        H_data["H"] = H   # kept on the device (dsx: the CPU un-rotation was the bottleneck)\n'); open(p, "w").write(s); print("patched")
else:
    print("already patched" if 'kept on the device (dsx' in s else "PATTERN NOT FOUND")
PY
cd /tmp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 Q_THREADS=2
q() { Q_SUFFIX=_$1 PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 python3 $J/q_eval.py t16 "${@:2}"; }
for m in full diag damp0.1 damp1 eye; do
  ( extra=""; [ $m = full ] && extra=ud
    for L in 5 20; do q $m $L 2,2,2@$m 2,2,3@$m $extra >> /root/q_${m}_$L.log 2>&1; echo "Q2-$m-$L $?" >> /root/t16.flag; done ) &
done
( q half 5 1.5,1.5,2@full 1.5,1.5,2@diag >> /root/q_half_5.log 2>&1; echo "Q2-half-5 $?" >> /root/t16.flag
  q half 20 2,2,2.5@full 2,2,2.5@diag >> /root/q_half_20.log 2>&1; echo "Q2-half-20 $?" >> /root/t16.flag ) &
wait
python3 $J/table.py > /root/t16_table.txt 2>&1
echo T16B-ALL-DONE >> /root/t16.flag
