#!/bin/bash
# After the quant chain: assemble the 1017 GGUF from qwork/q (dense from the FP8 checkpoint, experts copied in), then
# the hotness ranking from the full calibration. Engram stays in the checkpoint shards (--no-engram).
cd ~/ai/ds41/quant
while [ ! -f ~/ai/ds41/qwork/q/L39.json ]; do sleep 60; done
export V41_WORK=~/ai/ds41/calib
../venv/bin/python v41_gguf.py --out ~/ai/ds41/gguf/1017-GSQ.gguf --experts ~/ai/ds41/qwork/q --no-engram --threads 4 || { echo "GGUF FAILED"; exit 1; }
../venv/bin/python make_rank.py ~/ai/ds41/calib
echo "finish done"
