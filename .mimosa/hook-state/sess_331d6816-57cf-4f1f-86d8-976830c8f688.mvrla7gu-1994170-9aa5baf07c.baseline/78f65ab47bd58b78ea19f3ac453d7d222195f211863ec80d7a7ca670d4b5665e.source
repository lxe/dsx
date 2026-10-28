#!/bin/bash
# t16_pod.sh on a 1-GPU pod (files in /root/t16job, from t16_pod_prep.sh): the full-model T16 quant, uploaded to R2 per
# layer. NW (default 2) workers share the GPU with the chain, under MPS when it starts.
#   1. hydrate.sh (torch 2.11, deps, our R2 artifacts incl. the calibration set in calib/, the FP8 checkpoint from HF)
#      and, in parallel, exllamav3 da68d57 + make_mul16_exllamav3.sh (T16 codebook, column-major tile positions, H on
#      the GPU, dense fractional-K kernel) and its extension build
#   2. t16_qselftest.py on this GPU; abort on failure
#   3. chain: v41_calib.py init + layer 0..39 with V41_DUMP_ALL (every expert's W and fit Hessians), at most ~2 layers
#      of dumps on disk; t16_qworker.py x NW quantize them (layers 0-19 K 1.5/1.5/2, 20-39 2/2/2.5, damp 2)
#   4. t16_pod_up.sh: each finished layer -> R2 dsx/t16/full/LNN.tar (presigned PUTs in puts.tsv)
# Flags in /root/t16.flag; logs in /root/*.log. Restartable: finished experts and uploaded layers are skipped.
set -x
J=/root/t16job; W=/work/ds41; T=/work/t16
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
flag() { echo "$* $(date +%T)" >> /root/t16.flag; }
NW=${NW:-2}
mkdir -p $T/packed /work/dump
cat /sys/fs/cgroup/cpu.max > /root/cgroup.txt; nproc >> /root/cgroup.txt; free -g >> /root/cgroup.txt; nvidia-smi -L >> /root/cgroup.txt
CPUS=$(awk '{ if ($1 == "max") print 16; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max)
# 1. hydrate + build
if ! grep -q "HYDRATE 0" /root/t16.flag 2> /dev/null; then
  bash $J/hydrate.sh $J/manifest.tsv > /root/hydrate.log 2>&1 &
  HPID=$!
  ( until python3 -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.11') else 1)" 2> /dev/null; do sleep 15; done
    pip install -q ninja rich marisa_trie pydantic pyyaml llguidance transformers pyarrow safetensors > /dev/null 2>&1
    rm -rf /root/x3 && mkdir -p /root/x3 && cd /root/x3 && git init -q && git remote add origin https://github.com/turboderp-org/exllamav3
    git fetch -q --depth 1 origin da68d57de61781e5b74ad68698ac4caf36408d7a && git checkout -q FETCH_HEAD
    bash $J/make_mul16_exllamav3.sh /root/x3 /root/x3t16 > /root/t16patch.log 2>&1 || { flag "BUILD-t16 patch-failed"; exit 1; }
    cd /tmp && TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=$CPUS PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 \
      python3 -c "import exllamav3.ext" > /root/build_t16.log 2>&1
    flag "BUILD-t16 $?" ) &
  wait $HPID; flag "HYDRATE $?"; tail -3 /root/hydrate.log
fi
until grep -q "BUILD-t16" /root/t16.flag; do sleep 10; done
grep -q "BUILD-t16 0" /root/t16.flag || { flag ABORT-build; exit 1; }
# 2. selftest
if ! grep -q "SELFTEST 0" /root/t16.flag; then
  ( cd /tmp && PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 python3 $J/t16_qselftest.py > /root/selftest.log 2>&1; flag "SELFTEST $?" )
  grep -q "SELFTEST 0" /root/t16.flag || { flag ABORT-selftest; exit 1; }
fi
# 3. chain + workers + uploader
if [ ! -d /work/q/quant ]; then
  mkdir -p /root/v41 /work/q/quant /work/q/w && tar xzf $J/v41.tgz -C /root/v41
  cp -r $W/quant2/* /work/q/quant/ && cp -r /root/v41/* /work/q/quant/ && cp $W/calib/* /work/q/w/
fi
nvidia-cuda-mps-control -d > /root/mps.log 2>&1 && flag MPS-ON || flag MPS-OFF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( cd /work/q/quant
  export V41_MODEL=$W/fp8-1017 V41_REF=$W/fp8-1017/inference V41_ROWS=8192 V41_SCREEN_TIERS="" V41_SCREEN_IQ2=0 \
    V41_WORK=/work/q/w V41_DUMP_ALL=/work/dump OMP_NUM_THREADS=$(( CPUS > 2 * NW + 4 ? CPUS - 2 * NW : 4 ))
  [ -f /work/q/w/h/L00.bin ] || ls /work/q/w/h/L*.bin > /dev/null 2>&1 || python v41_calib.py init 2>&1 | tail -2
  for L in $(seq 0 39); do
    LL=$(printf %02d $L)
    grep -q "UP-L$LL " /root/t16.flag && continue                          # uploaded before a restart
    [ -f /work/dump/L$LL/DONE ] && continue                                 # dumped before a restart
    while [ $(find /work/dump -name 'e*.pt*' | wc -l) -gt 770 ]; do sleep 10; done   # <= ~2 layers of dumps (~150 GB)
    python v41_calib.py layer $L --device cuda --gc 2>&1 | grep --line-buffered -E "done in|Error|Traceback|rror"
    [ -f /work/dump/L$LL/DONE ] || { echo "CHAIN-FAILED L$LL"; break; }
  done
  echo CHAIN-END ) > /root/chain.log 2>&1 &
CPID=$!
for k in $(seq 1 $NW); do
  ( cd /tmp && Q_THREADS=2 PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 \
      python3 -u $J/t16_qworker.py $k /work/dump $T/packed >> /root/worker_$k.log 2>&1 ) &
done
bash $J/t16_pod_up.sh /work/dump $T/packed $J/puts.tsv > /root/up.log 2>&1 &
UPID=$!
wait $CPID; flag "CHAIN-END $(tail -1 /root/chain.log)"
wait $UPID
flag ALL-DONE
