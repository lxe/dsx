#!/bin/bash
# Window 32 folded into this one (never ran). Split-k one-token kernels (DSX_SK1), combine_post (DSX_COMBINE_FUSED) and
# wo_a rounding + shared gate/up/swiglu fusion (DSX_FUSE2), cumulative arms from one binary. Binaries (CUDA_ARCHS=86
# --features cuda, own target dirs, eval tok/s print): base = f1f53e4 (/tmp/dsx-sk-base); h4 = 795d236 (/tmp/dsx-sk-h4).
# Arms: base; P = h4 DSX_SK1=0 DSX_COMBINE_FUSED=0 DSX_FUSE2=0 (window 31's QA); S = + sk1; C = + combine_post;
# F2 = h4 (+ fuse2). eval --decode 1 seq x 1023 steps, defer 3, adapt 8, 2 interleaved rounds; DSX_ADAPT=0 base vs F2
# (KL must be equal); gen 120 after 300 base / F2 x2; nsys of F2. Predicted per step: S -0.3 ms/token, C -0.3 ms,
# F2 -0.2 ms (kill each < 0.1 ms).
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: sk1 / combine_post / fuse2 teacher-forced A/B (base / P / S / C / F2), serve stopped (~50 min) $(date)" > GPU_LOCK
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
  run "round $r P" "DSX_SK1=0 DSX_COMBINE_FUSED=0 DSX_FUSE2=0" h4
  run "round $r S" "DSX_COMBINE_FUSED=0 DSX_FUSE2=0" h4
  run "round $r C" "DSX_FUSE2=0" h4
  run "round $r F2" "" h4
done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 F2" "DSX_ADAPT=0" h4
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base h4 base h4; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-sk-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic")"
done
sleep 40; rm -f /tmp/dec14.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec14 /tmp/dsx-sk-h4 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec14.out 2>&1
echo "traced F2: $(grep -E "^decode" /tmp/dec14.out)"; grep -iE "rror|panic" /tmp/dec14.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec14.sqlite /tmp/dec14.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec14.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN33-DONE $(date)"
