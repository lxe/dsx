#!/bin/bash
# split-k one-token kernels (DSX_SK1) and combine_post (DSX_COMBINE_FUSED) A/B. Binaries (CUDA_ARCHS=86 --features cuda,
# own target dirs, eval tok/s print): base = f1f53e4 (/tmp/dsx-sk-base); h3 = 33e6535 (/tmp/dsx-sk-h3: window 31's QA
# plus vector-load split-k GEMVs, *_sk1, combine_post, late GPU1 enqueue default). Arms: base; P = h3 DSX_SK1=0
# DSX_COMBINE_FUSED=0; S = h3 DSX_COMBINE_FUSED=0; C = h3. eval --decode 1 seq x 1023 steps, defer 3, adapt 8,
# 2 interleaved rounds; DSX_ADAPT=0 base vs C (KL must be equal); gen 120 after 300 base / C x2; nsys of C.
# Predicted: S over P -0.3 ms/token (router 11 -> ~6 us, q_a/kv, x 40-80), C over S -0.3 ms (kill each < 0.1 ms).
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: split-k sk1 / combine_post teacher-forced A/B (base / P / S / C), serve stopped (~40 min) $(date)" > GPU_LOCK
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
  run "round $r P" "DSX_SK1=0 DSX_COMBINE_FUSED=0" h3
  run "round $r S" "DSX_COMBINE_FUSED=0" h3
  run "round $r C" "" h3
done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 C" "DSX_ADAPT=0" h3
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base h3 base h3; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-sk-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic")"
done
sleep 40; rm -f /tmp/dec13.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec13 /tmp/dsx-sk-h3 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec13.out 2>&1
echo "traced C: $(grep -E "^decode" /tmp/dec13.out)"; grep -iE "rror|panic" /tmp/dec13.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec13.sqlite /tmp/dec13.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec13.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN32-DONE $(date)"
