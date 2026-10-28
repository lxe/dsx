#!/bin/bash
# Dev4's DSX_PREFETCH A/B (pf102.sh: builds, DSX_PREFETCH 0/1/2 x DSX_DEFER none/3 interleaved, eval --decode 0 vs 2),
# serve stopped; restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: Dev4 prefetch A/B (pf102), serve stopped (~25 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 3
cd ~/src/dsx-prefetch && GEN_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --ids @$HOME/ai/ds41/prompt3000.ids -n 400 --ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank $HOME/ai/ds41/calib/rank.json" EVAL_ARGS="--model $HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --eval $HOME/ai/ds41/eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json" bash bench/pf102.sh > ~/ai/ds41/pf102.out 2>&1
tail -45 ~/ai/ds41/pf102.out
cd ~/ai/ds41
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN17-DONE $(date)"
