#!/bin/bash
# Verify-pass cost on the current build (speculative decoding's price): gen --bench-verify T for T = 2, 3, 4 after a
# 3000-token prompt, next to the plain decode step; DSX_DEFER off. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: verify-pass cost, serve stopped (~8 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 3
B=/tmp/dsx-cur; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"; P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
for t in 2 3 4; do sleep 3
  echo "== verify T=$t"; ./guard.sh bv$RANDOM 118 $B gen $M --ids @prompt3000.ids -n 64 --bench-verify $t $P 2>&1 | grep -E -i "^decode|verify" | cut -c1-200
done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN15-DONE $(date)"
