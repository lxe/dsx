#!/bin/bash
# Window 26 failed (binaries built without --features cuda). Lever 7a + 4 (8433e12 + the 6fc3f62 type fix = /tmp/dsx-new1)
# and + lever 4b (hc mixes on a third stream, 6fc3f62 = /tmp/dsx-new2) against their base f1f53e4 (/tmp/dsx-base), each
# built with CUDA_ARCHS=86 --features cuda in its own target dir: teacher-forced eval --decode (1 seq, 1023 steps, defer 3,
# adapt 8) interleaved 2 rounds; one
# DSX_ADAPT=0 pair (no residency changes: the kernels are bit-identical, so KL must match exactly); gen 120 tokens
# after 300 untraced for both; nsys of the new build (between-token gap should drop from 0.65 ms). Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: small-kernels A/B (adapt before sync, hc_pre_norm, rope split), serve stopped (~30 min) $(date)" > GPU_LOCK
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
  run "round $r base" "" /tmp/dsx-base
  run "round $r new1" "" /tmp/dsx-new1
  run "round $r new2" "" /tmp/dsx-new2
done
run "adapt 0 base" "DSX_ADAPT=0" /tmp/dsx-base
run "adapt 0 new2" "DSX_ADAPT=0" /tmp/dsx-new2
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base new2 base new2; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic" | sed -E 's/GPU0 expert swaps.*//')"
done
sleep 40; rm -f /tmp/dec9.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec9 /tmp/dsx-new2 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec9.out 2>&1
echo "traced new: $(grep -E "^decode" /tmp/dec9.out | sed -E 's/GPU0 expert swaps.*//')"
/usr/local/bin/nsys export --type sqlite -o /tmp/dec9.sqlite /tmp/dec9.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec9.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN27-DONE $(date)"
