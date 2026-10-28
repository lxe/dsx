#!/bin/bash
# hc_dots_t (tiled prefill hc dots): bit-identity check vs the previous build (same top5 logits) and timing,
# 3000 tokens (CPU share off) and 300 tokens (on), old / new alternating. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: hc_dots_t check, serve stopped (~12 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 3
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"; P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
for f in "prompt3000 0" "prompt300 1"; do set -- $f; for b in old new new old; do X=/tmp/dsx-cur; [ $b = new ] && X=/tmp/dsx-hcd; sleep 3
  echo "$1 cpu=$2 $b: $(DSX_PREFILL_CPU=$2 ./guard.sh hd$RANDOM 118 $X gen $M --ids @$1.ids -n 2 $P 2>&1 | grep -E "^prefill [0-9]" | cut -c1-170)"
done; done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN16-DONE $(date)"
