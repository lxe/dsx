#!/bin/bash
# usage: pfbench.sh NAME BIN [prof] [cold]  -- 8K prompt, 3 rounds, fixed placement (same env as the 10-25 checkpoint)
set -u
cd ~/ai/ds41
NAME=$1; BIN=$2; PROF=${3:-}; COLD=${4:-}; IDS=${5:-prompt8000.ids}
[ "$(cat GPU_LOCK 2>/dev/null | cut -c1-16)" = "DSX .102 thread:" ] || { echo "lock not ours: $(cat GPU_LOCK 2>/dev/null)"; exit 1; }
[ -n "$COLD" ] && { sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
mkdir -p engram-pf/out
env -u DSX_PROF DSX_ADAPT=0 DSX_CPU_THREADS=14 DSX_AUX_STREAM=0 DSX_AUX_ADAPTIVE=0 DSX_T16_WARP_HAD=1 DSX_T16_SPECIALIZED_DQ=1 \
  DSX_PREFILL_REUSE_BUFFERS=1 DSX_T16_FUSED_POST_GU=1 ${PROF:+DSX_PROF=1} \
  timeout -s KILL 900 ./guard.sh pf-$NAME 118 "$BIN" --model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank calib/rank.json --ids $IDS --ctx 262144 --chunk 8192 --rounds 3 \
  --dump-logits engram-pf/out/$NAME.f32 > engram-pf/out/$NAME.log 2>&1
echo "RC $?" >> engram-pf/out/$NAME.log
grep -E "prefill_bench|summary|engram L|RC" engram-pf/out/$NAME.log | cut -c1-220
