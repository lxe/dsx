#!/bin/bash
# Teacher-forced speed A/B of the small-kernels levers, each arm cumulative and built with CUDA_ARCHS=86 --features cuda
# (all with the eval --decode tok/s print from 25c822b): base f1f53e4; A = 7a adapt step before the logits sync + 4 fused
# hc_pre/rmsnorm + rope split (8433e12 + type fix); B = A + 4b hc mixes on a third stream (6fc3f62); C = B + 7b GPU1
# enqueued after layer-0 attention + 4c router_publish (f854697). eval --decode, 1 seq x 1023 steps, defer 3, adapt 8,
# 2 interleaved rounds; then DSX_ADAPT=0 base vs C (no residency changes: KL must be equal); nsys of C. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: small-kernels teacher-forced A/B (4 arms), serve stopped (~35 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M=$HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf
EV="--model $M --eval $HOME/ai/ds41/eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1 --decode"
temps() { echo "  $(date +%T) $(sensors | grep Package | awk '{print $4}') GPU $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ')"; }
run() { # label env arm
  sleep 45; temps; echo "== $1"
  env $2 DSX_DEFER=3 /tmp/dsx-sk-$3 eval $EV > /tmp/sk_eval.out 2>&1
  grep -E "^teacher-forced|kl_top" /tmp/sk_eval.out || tail -5 /tmp/sk_eval.out
}
for r in 1 2; do for a in base A B C; do run "round $r $a" "" $a; done; done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 C" "DSX_ADAPT=0" C
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
sleep 40; rm -f /tmp/dec10.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec10 /tmp/dsx-sk-C gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec10.out 2>&1
echo "traced C: $(grep -E "^decode" /tmp/dec10.out | sed -E 's/GPU0 expert swaps.*//')"; grep -iE "rror|panic" /tmp/dec10.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec10.sqlite /tmp/dec10.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec10.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN28-DONE $(date)"
