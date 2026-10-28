#!/bin/bash
# hydrate.sh MANIFEST - on a rented pod: Python deps, our artifacts from R2 (MANIFEST lines: URL<TAB>PATH, presigned),
# the FP8 checkpoint from Hugging Face, the native GPTQ library. Everything under /work/ds41.
set -eo pipefail
M=$1; W=/work/ds41; mkdir -p $W && cd $W
t0=$(date +%s)
pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1 || true   # the checkpoint loader needs torch.float4_e2m1fn_x2
pip install -q numpy safetensors gguf tokenizers "huggingface_hub[hf_xet]" pillow 2>&1 | tail -1 || true
# our artifacts: 32 at a time, retried
awk -F'\t' '{print $1"\t"$2}' "$M" | while IFS=$'\t' read -r url path; do echo "$url $path"; done | \
  xargs -P 32 -n 2 sh -c 'mkdir -p "$(dirname "$1")" && curl -sf --retry 5 -o "$1" "$0"'
echo "artifacts: $(du -sh $W | cut -f1) in $(( $(date +%s) - t0 )) s"
tar -xzf quant2.tar.gz && rm quant2.tar.gz
# the checkpoint, many connections at once: from R2 when a presigned aria2 list sits next to the manifest (ckpt.aria2),
# else from Hugging Face
t1=$(date +%s)
if [ -s "$(dirname "$M")/ckpt.aria2" ]; then
  command -v aria2c >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2 >/dev/null 2>&1)
  aria2c -q -i "$(dirname "$M")/ckpt.aria2" -j 16 -x 16 -s 16 -k 64M --file-allocation=none --auto-file-renaming=false --allow-overwrite=true --max-tries=10 \
    || echo "aria2 from R2 failed"
fi
[ -f fp8-1017/model-00048-of-00048.safetensors ] && [ ! -f fp8-1017/model-00048-of-00048.safetensors.aria2 ] || \
  HF_XET_HIGH_PERFORMANCE=1 hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir fp8-1017 --max-workers 32 >/dev/null 2>&1
[ -f ds41-engram.json ] && mv ds41-engram.json fp8-1017/
for f in fp8-1017/model-*.safetensors; do touch "$f.ok"; done   # v41_calib reads only shards marked verified (hf checks every file)
echo "checkpoint: $(du -sh fp8-1017 | cut -f1) in $(( $(date +%s) - t1 )) s"
PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -shared -Xcompiler -fPIC -gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_86,code=sm_86 -o native/libqkern.so native/qkern.cu
echo "HYDRATED in $(( $(date +%s) - t0 )) s"
