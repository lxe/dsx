#!/bin/bash
# small-kernels teacher-forced A/B. Binaries built with CUDA_ARCHS=86 --features cuda, one target dir each, both with
# the eval --decode tok/s print: base = f1f53e4 (/tmp/dsx-sk-base); head = 3bdc255 (/tmp/dsx-sk-head).
# Arms: base; K = head with DSX_ADAPT_EARLY=0 DSX_AUX_LATE=0 (hc_pre_norm, rope split, hc mixes on a third stream,
# router_publish: bit-identical kernels only); F = head defaults (+ adapt scan/swap starts before the sync with commits
# after it, GPU1 enqueued after layer-0 attention). eval --decode 1 seq x 1023 steps, defer 3, adapt 8, 2 interleaved
# rounds; DSX_ADAPT=0 base vs F (KL must be equal); gen 120 after 300 untraced base / F x2 (full lines: CPU picks,
# swaps); nsys of F. Predicted: K -1.1 ms/token (+4%), F a further -0.45 ms with base's CPU share. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: small-kernels teacher-forced A/B (base / K / F), serve stopped (~35 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M=$HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf
EV="--model $M --eval $HOME/ai/ds41/eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1 --decode"
temps() { echo "  $(date +%T) $(sensors | grep Package | awk '{print $4}') GPU $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ')"; }
run() { # label env binary
  sleep 45; temps; echo "== $1"
  env $2 DSX_DEFER=3 /tmp/dsx-sk-$3 eval $EV > /tmp/sk_eval.out 2>&1
  grep -E "^teacher-forced|kl_top" /tmp/sk_eval.out || tail -5 /tmp/sk_eval.out
}
for r in 1 2; do
  run "round $r base" "" base
  run "round $r K" "DSX_ADAPT_EARLY=0 DSX_AUX_LATE=0" head
  run "round $r F" "" head
done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 F" "DSX_ADAPT=0" head
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base head base head; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-sk-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic")"
done
sleep 40; rm -f /tmp/dec11.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec11 /tmp/dsx-sk-head gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec11.out 2>&1
echo "traced F: $(grep -E "^decode" /tmp/dec11.out)"; grep -iE "rror|panic" /tmp/dec11.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec11.sqlite /tmp/dec11.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec11.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN30-DONE $(date)"
