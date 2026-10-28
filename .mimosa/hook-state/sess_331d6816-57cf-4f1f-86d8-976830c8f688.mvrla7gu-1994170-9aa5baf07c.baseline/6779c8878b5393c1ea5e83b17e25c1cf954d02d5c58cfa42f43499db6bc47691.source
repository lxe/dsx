#!/bin/bash
# Lever 7a + lever 4 A/B (branch small-kernels 8433e12 = /tmp/dsx-new1 vs its parent f1f53e4 = /tmp/dsx-base1, same
# build dir layout): teacher-forced eval --decode (1 seq, 1023 steps, defer 3, adapt 8) interleaved 2 rounds; one
# DSX_ADAPT=0 pair (no residency changes: the kernels are bit-identical, so KL must match exactly); gen 120 tokens
# after 300 untraced for both; nsys of the new build (between-token gap should drop from 0.65 ms). Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: small-kernels A/B (adapt before sync, hc_pre_norm, rope split), serve stopped (~25 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M=$HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf
EV="--model $M --eval $HOME/ai/ds41/eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1 --decode"
temps() { echo "  $(date +%T) $(sensors | grep Package | awk '{print $4}') GPU $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ')"; }
run() { # label env binary
  sleep 60; temps; echo "== $1"
  env $2 DSX_DEFER=3 $3 eval $EV 2>&1 | grep -E "^teacher-forced|kl_top|rror|panic"
}
for r in 1 2; do
  run "round $r base" "" /tmp/dsx-base1
  run "round $r new" "" /tmp/dsx-new1
done
run "adapt 0 base" "DSX_ADAPT=0" /tmp/dsx-base1
run "adapt 0 new" "DSX_ADAPT=0" /tmp/dsx-new1
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base1 new1 base1 new1; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic" | sed -E 's/GPU0 expert swaps.*//')"
done
sleep 40; rm -f /tmp/dec8.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec8 /tmp/dsx-new1 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec8.out 2>&1
echo "traced new: $(grep -E "^decode" /tmp/dec8.out | sed -E 's/GPU0 expert swaps.*//')"
/usr/local/bin/nsys export --type sqlite -o /tmp/dec8.sqlite /tmp/dec8.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec8.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN26-DONE $(date)"
