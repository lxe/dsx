#!/bin/bash
# Dev4's E1 (engram wkv as FP8) A/B: ef102.sh (kernel microbench first: kill if gemv_fp8b32 < 650 GB/s; interleaved
# f16 / fp8 same placement / fp8 with the freed VRAM; DSX_PROF incl. engram join wait; decode evals). Serve restarted.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: Dev4 engram-FP8 A/B (ef102), serve stopped (~30 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 5
cd ~/src/dsx-efp8 && GEN_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --ids @$HOME/ai/ds41/prompt3000.ids -n 400 --ctx 32768 --chunk 4096 --aux-gb=-1.0 --arena-gb 110 --rank $HOME/ai/ds41/calib/rank.json" EVAL_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --eval $HOME/ai/ds41/eval --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --main-gb=-1.5" MAIN_GB=-1.5 bash bench/ef102.sh > ~/ai/ds41/ef102.out 2>&1
tail -40 ~/ai/ds41/ef102.out
cd ~/ai/ds41
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN21-DONE $(date)"
