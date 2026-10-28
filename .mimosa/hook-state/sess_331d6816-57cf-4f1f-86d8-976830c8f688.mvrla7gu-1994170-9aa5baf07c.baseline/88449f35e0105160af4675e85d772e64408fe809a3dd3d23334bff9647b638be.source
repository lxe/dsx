#!/bin/bash
# Downloads to /data/models: the FP8 checkpoint, dsx 2-bit quant, Lucebox eval set. Run after PY-DONE.
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
HF=/root/venv/bin/hf
cd /data/models
$HF download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF --include 'eval/*' --local-dir lucebox > /root/fetch-eval.log 2>&1 && touch lucebox/.eval-done &
$HF download lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit --local-dir dsx-2bit > /root/fetch-dsx.log 2>&1 && touch dsx-2bit/.done &
$HF download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir fp8 > /root/fetch-fp8.log 2>&1 && touch fp8/.done &
wait
echo FETCH-DONE
