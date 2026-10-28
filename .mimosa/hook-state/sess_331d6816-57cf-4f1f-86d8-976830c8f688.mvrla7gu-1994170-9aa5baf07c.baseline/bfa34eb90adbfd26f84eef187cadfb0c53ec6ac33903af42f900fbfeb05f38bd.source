#!/bin/bash
# Decode-eval KL for DSX_DEFER 3 / 2 / 1 (kill for 1 and 2: KL up > 0.01 vs 3), DSX_CPU_THREADS=8 (bit-exact, cooler:
# 14-thread evals hit the 95 C watchdog), 60 s cool-downs, temperatures logged. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: DSX_DEFER decode evals (8 CPU threads), serve stopped (~20 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
for d in 3 2 1; do sleep 60
  echo "== eval --decode defer=$d (start $(sensors | grep Package | awk '{print $4}'))"
  DSX_CPU_THREADS=8 DSX_DEFER=$d DSX_ADAPT=0 ./guard.sh ev$RANDOM 118 /tmp/dsx-cur eval $M --eval eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json --decode --seqs 2 > /tmp/ev_$d.out 2>&1
  grep -E "^\{" /tmp/ev_$d.out || tail -3 /tmp/ev_$d.out; echo "   end $(sensors | grep Package | awk '{print $4}'); $(grep STOP watchdog.log | tail -1)"
done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN22-DONE $(date)"
