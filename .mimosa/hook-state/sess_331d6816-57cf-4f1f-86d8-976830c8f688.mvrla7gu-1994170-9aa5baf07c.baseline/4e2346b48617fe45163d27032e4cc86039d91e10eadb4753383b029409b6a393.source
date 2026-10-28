#!/bin/bash
# Server-like turns: a 3000-token prompt then 60 / 300 new tokens as their own chunk (--chunk 3000: warm kernels, 3000
# cached), CPU share on vs off; DSX_PROF of the 60-token turn; Q8_0 GEMV bench (q8r vs q8r4) on idle GPU0 and GPU1.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: server-like turn prefill + GEMV bench, serve stopped (~20 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; sleep 3
echo "== GEMV bench GPU0"; CUDA_VISIBLE_DEVICES=0 /tmp/t16dq/q8g; echo "== GEMV bench GPU1"; CUDA_VISIBLE_DEVICES=1 /tmp/t16dq/q8g
B=~/src/dsx-dev/target/release/dsx; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
P="--ctx 32768 --chunk 3000 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
g() { systemctl --user reset-failed ds41-w7.scope 2>/dev/null; sleep 2; env $2 ./guard.sh w7 118 $B gen $M --ids @$1.ids -n 2 $P 2>&1 | grep -E "^prefill chunk" | tr "\n" " "; }
for f in turn60 turn300; do for v in 0 1 1 0; do echo "$f cpu=$v: $(g $f DSX_PREFILL_CPU=$v)"; done; done
systemctl --user reset-failed ds41-w7.scope 2>/dev/null; sleep 2
echo "== DSX_PROF turn60"; DSX_PROF=1 ./guard.sh w7 118 $B gen $M --ids @turn60.ids -n 2 $P 2>&1 | grep -E "^prefill chunk|  prefill" | cut -c1-110
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN7-DONE $(date)"
