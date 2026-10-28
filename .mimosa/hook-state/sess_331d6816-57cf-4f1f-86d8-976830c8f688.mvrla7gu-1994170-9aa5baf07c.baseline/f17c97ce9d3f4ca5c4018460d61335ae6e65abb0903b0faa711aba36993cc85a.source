#!/bin/bash
# nsys of a 3000-token prefill on the current build (link idle time per layer: attention vs MoE phases), plus 8000-token
# prefill timing for scaling. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: long-prefill timeline, serve stopped (~8 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 3
B=/tmp/dsx-cur; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"; P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
./guard.sh lp$RANDOM 118 nsys profile -o /tmp/lp3000 --force-overwrite true --trace=cuda --sample=none --cpuctxsw=none $B gen $M --ids @prompt3000.ids -n 2 $P 2>&1 | grep -E "^prefill [0-9]" | cut -c1-80
nsys export --type sqlite -o /tmp/lp3000.sqlite --force-overwrite true /tmp/lp3000.nsys-rep > /dev/null 2>&1; ls -la /tmp/lp3000.sqlite
sleep 3; ./guard.sh lp$RANDOM 118 $B gen $M --ids @prompt8000.ids -n 2 $P 2>&1 | grep -E "^prefill" | cut -c1-80
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN14-DONE $(date)"
