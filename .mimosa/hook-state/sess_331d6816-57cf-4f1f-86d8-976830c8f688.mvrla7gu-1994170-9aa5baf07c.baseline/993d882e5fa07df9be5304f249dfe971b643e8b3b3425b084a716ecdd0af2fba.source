#!/bin/bash
# Prefill: few-token resident experts on decode's T16 kernels (DSX_PRE_DEV_NT, default 6) on vs off (=0), with the CPU
# share on; 60/300/1024 alternating; Lucebox KL with it on; Dev4's multi-token CPU benchmark (serve stopped).
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: prefill device path test + Dev4 CPU bench, serve stopped (~30 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; sleep 3
B=~/src/dsx-dev/target/release/dsx; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
g() { systemctl --user reset-failed ds41-w6.scope 2>/dev/null; sleep 2; env $2 ./guard.sh w6 118 $B gen $M --ids @$1.ids -n 2 $P 2>&1 | grep -E "^prefill [0-9]" | sed -E "s/next top5 //" | cut -c1-140; }
for f in prompt60 prompt300 prompt3k_1024; do for v in 0 6 6 0; do echo "$f dev_nt=$v: $(g $f DSX_PRE_DEV_NT=$v)"; done; done
systemctl --user reset-failed ds41-w6.scope 2>/dev/null; sleep 2
echo "== DSX_PROF 60 dev_nt=6"; DSX_PROF=1 ./guard.sh w6 118 $B gen $M --ids @prompt60.ids -n 2 $P 2>&1 | grep -E "^prefill [0-9]|  prefill" | cut -c1-120
systemctl --user reset-failed ds41-w6.scope 2>/dev/null; sleep 2
echo -n "eval dev_nt=6 (CPU share on): "; DSX_ADAPT=0 ./guard.sh w6 118 $B eval $M --eval eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json 2>&1 | grep -E "^\{|rror|panic"
echo "== Dev4 mt102 (serve stopped)"; ./mt102_nolock.sh 2>&1 | tail -40
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN6-DONE $(date)"
