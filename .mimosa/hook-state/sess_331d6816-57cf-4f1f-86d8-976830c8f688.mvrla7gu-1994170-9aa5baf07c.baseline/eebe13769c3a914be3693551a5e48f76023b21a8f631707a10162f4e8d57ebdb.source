#!/bin/bash
# Calibration chain on GPU1, layer by layer (each layer reads the previous layer's output); resumable.
cd ~/ai/ds41/quant
export CUDA_VISIBLE_DEVICES=1 V41_WORK=~/ai/ds41/calib V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for L in $(seq ${1:-0} 39); do
  if [ -f ~/ai/ds41/calib/stats/L$(printf %02d $L).json ]; then echo "layer $L done"; continue; fi
  # every layer after 0 may need the Engram shards (layers 1 and 14 read them)
  while [ $L -ge 1 ] && { [ ! -f ../fp8-1017/model-00047-of-00048.safetensors.ok ] || [ ! -f ../fp8-1017/model-00048-of-00048.safetensors.ok ]; }; do sleep 30; done
  ../venv/bin/python v41_calib.py layer $L --device cuda --gc 2>&1 | grep --line-buffered -v "UserWarning\|from_numpy"
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "layer $L FAILED"; exit 1; }
done
echo "calibration chain done"
