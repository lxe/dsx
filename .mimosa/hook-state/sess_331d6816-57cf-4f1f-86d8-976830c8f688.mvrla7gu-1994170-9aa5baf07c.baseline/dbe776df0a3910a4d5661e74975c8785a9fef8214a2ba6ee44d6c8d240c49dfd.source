#!/bin/bash
# Dev 4's window (~40 min): bench/w4_102.sh on window-dev4 4e5791f (~/src/dsx-w4): engram GEMVs and the T16 decode layer
# call DSX_T16_GEMV 0 vs 1 (lever 5) with bit hashes, sudo ncu on sm_86, cpubench t16 at 2 / 14 threads, teacher-forced
# eval --decode base / e1 / t16 / both (KL must be equal). Serve stopped, restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1 for Dev 4: w4_102.sh (T16 GEMV restructure, E1), serve stopped (~40 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
cd ~/src/dsx-w4 && EVAL_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --eval $HOME/ai/ds41/eval --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1 --main-gb=-1.5" CPU_THREADS=8 bash bench/w4_102.sh > ~/ai/ds41/w4.out 2>&1
tail -60 ~/ai/ds41/w4.out
cd ~/ai/ds41
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN29-DONE $(date)"
