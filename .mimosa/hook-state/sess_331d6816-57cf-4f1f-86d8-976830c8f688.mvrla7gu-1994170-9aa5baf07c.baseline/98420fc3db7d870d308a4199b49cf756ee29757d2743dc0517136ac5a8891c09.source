#!/bin/bash
# t16_pod_resume.sh: restarts the GPU side of t16_pod.sh after a fault took down the MPS clients (chain + workers):
# MPS, the chain from the first layer that has neither an upload nor a DONE (two attempts per layer), and workers
# 1..NW, each restarted up to 5 times (a worker takes back its own claims). t16_pod_up.sh keeps running untouched.
# Never kill an MPS client mid-kernel: on 2026-10-22 stopping one corrupted the server and every client faulted.
set -x
J=/root/t16job; W=/work/ds41; T=/work/t16; NW=${NW:-2}
flag() { echo "$* $(date +%T)" >> /root/t16.flag; }
echo quit | nvidia-cuda-mps-control; sleep 3
nvidia-cuda-mps-control -d && flag MPS-RESTART || flag MPS-OFF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPUS=$(awk '{ if ($1 == "max") print 16; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max)
( cd /work/q/quant
  export V41_MODEL=$W/fp8-1017 V41_REF=$W/fp8-1017/inference V41_ROWS=8192 V41_SCREEN_TIERS="" V41_SCREEN_IQ2=0 \
    V41_WORK=/work/q/w V41_DUMP_ALL=/work/dump OMP_NUM_THREADS=$(( CPUS > 2 * NW + 4 ? CPUS - 2 * NW : 4 ))
  for L in $(seq 0 39); do
    LL=$(printf %02d $L)
    grep -q "UP-L$LL " /root/t16.flag && continue
    [ -f /work/dump/L$LL/DONE ] && continue
    while [ $(find /work/dump -name 'e*.pt*' | wc -l) -gt 770 ]; do sleep 10; done
    for attempt in 1 2; do
      python v41_calib.py layer $L --device cuda --gc 2>&1 | grep --line-buffered -E "done in|Error|Traceback|rror"
      [ -f /work/dump/L$LL/DONE ] && break
      sleep 20
    done
    [ -f /work/dump/L$LL/DONE ] || { echo "CHAIN-FAILED L$LL"; break; }
  done
  echo CHAIN-END ) >> /root/chain.log 2>&1 &
CPID=$!
for k in $(seq 1 $NW); do
  ( cd /tmp
    for attempt in 1 2 3 4 5; do
      Q_THREADS=2 PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 python3 -u $J/t16_qworker.py $k /work/dump $T/packed >> /root/worker_$k.log 2>&1 && break
      echo "worker $k exited, restart $attempt" >> /root/worker_$k.log; sleep 20
    done ) &
done
wait $CPID; flag "CHAIN-END $(tail -1 /root/chain.log)"
wait
flag WORKERS-END
