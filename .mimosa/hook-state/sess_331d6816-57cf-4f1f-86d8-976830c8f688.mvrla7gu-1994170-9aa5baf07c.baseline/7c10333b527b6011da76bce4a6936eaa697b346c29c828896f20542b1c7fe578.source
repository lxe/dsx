#!/bin/bash
# Budget measurements on the production build: DRAM read/write GB/s (perf IMC counters, 500 ms intervals) and
# DSX_EVPROF stage times during a 600-token decode after a 3000-token prompt, without and with DSX_DEFER=3; idle DRAM.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: budget measurements (DRAM counters, EVPROF), serve stopped (~12 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 5
EV="-e uncore_imc_free_running/data_read/ -e uncore_imc_free_running/data_write/"
echo "== idle"; sudo -n perf stat -a $EV -I 500 -x, -o /tmp/dram_idle.csv sleep 5 2>/dev/null; python3 dram.py /tmp/dram_idle.csv frac 0.1 0.9
for d in off 3; do E=""; [ $d = 3 ] && E="DSX_DEFER=3"; sleep 10
  echo "== decode defer=$d"
  sudo -n perf stat -a $EV -I 500 -x, -o /tmp/dram_$d.csv sleep 400 2>/dev/null & pp=$!
  env DSX_EVPROF=1 $E ./guard.sh bd$RANDOM 118 /tmp/dsx-cur gen --model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --ids @prompt3000.ids -n 600 --ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json > /tmp/bd_$d.out 2>&1
  sudo -n pkill -INT -x perf 2>/dev/null; sleep 2
  grep -E "^prefill [0-9]|^decode" /tmp/bd_$d.out | cut -c1-200
  grep -E "^  [0-9] |^  0[a-h] |CPU worker" /tmp/bd_$d.out
  # decode is the last part of the run (600 tokens, ~16-19 s): the 12 s before exit
  python3 dram.py /tmp/dram_$d.csv last 12
done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN18-DONE $(date)"
