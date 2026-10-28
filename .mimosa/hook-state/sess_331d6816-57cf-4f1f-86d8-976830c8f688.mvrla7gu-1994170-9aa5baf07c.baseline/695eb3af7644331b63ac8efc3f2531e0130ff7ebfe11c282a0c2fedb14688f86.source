#!/bin/bash
# Dev4's engram E1 (FP8 wkv, kernel fixed for sm_86) + E2 (persistent gather threads): teacher-forced decode A/B
# (e2_102.sh: kernel microbench, 2 interleaved rounds of f16 / fp8 / fp8e2, DSX_PROF evals). Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: Dev4 engram E1+E2 teacher-forced A/B (e2_102), serve stopped (~30 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
cd ~/src/dsx-efp8 && EVAL_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --eval $HOME/ai/ds41/eval --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1" MAIN_GB=-1.5 bash bench/e2_102.sh > ~/ai/ds41/e2_102.out 2>&1
grep -v "^\s*$" ~/ai/ds41/e2_102.out | tail -45
cd ~/ai/ds41
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN24-DONE $(date)"
