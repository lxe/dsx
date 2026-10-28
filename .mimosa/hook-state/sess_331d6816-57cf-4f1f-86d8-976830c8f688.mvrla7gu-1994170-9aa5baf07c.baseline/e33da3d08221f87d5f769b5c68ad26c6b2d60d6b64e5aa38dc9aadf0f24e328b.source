#!/bin/bash
# The integration branch (integrate b0f4d07: small-kernels + main + Dev 4's window-dev4 + budget-102 docs), one binary
# with switches, cumulative arms. Binaries (CUDA_ARCHS=86 --features cuda, own target dirs, eval tok/s print): base =
# f1f53e4 (/tmp/dsx-sk-base); h5 = b0f4d07 (/tmp/dsx-sk-h5). Arms: base; X = window 33's F2 (DSX_FUSE3=0
# DSX_PAR_ATTN=0 DSX_FUSE4=0 DSX_T16_GEMV=0); Y = + attention epilogue / q_b RoPE fusion; Z = + kv / compressor / indexer
# branch on the par stream; W = + closing hc_post fusions; T = + Dev 4's cp.async T16 decode GEMV (h5 defaults);
# E = + FP8 engram (DSX_ENGRAM_FP8=1, gemv_fp8b32_o). eval --decode 1 seq x 1023 steps, defer 3, adapt 8, 2 interleaved
# rounds; DSX_ADAPT=0 base vs E (KL must be equal: every change is bit-identical); gen 120 after 300 base / T x2; nsys of
# T. Predicted per step: Y -0.3, Z -0.4..-0.6, W -0.2, T 0..-0.5, E -0.4 ms/token (kill each < 0.1 ms).
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: integrate b0f4d07 teacher-forced A/B (base / X / Y / Z / W / T / E), serve stopped (~65 min) $(date)" > GPU_LOCK
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
  run "round $r X" "DSX_FUSE3=0 DSX_PAR_ATTN=0 DSX_FUSE4=0 DSX_T16_GEMV=0" h5
  run "round $r Y" "DSX_PAR_ATTN=0 DSX_FUSE4=0 DSX_T16_GEMV=0" h5
  run "round $r Z" "DSX_FUSE4=0 DSX_T16_GEMV=0" h5
  run "round $r W" "DSX_T16_GEMV=0" h5
  run "round $r T" "" h5
  run "round $r E" "DSX_ENGRAM_FP8=1" h5
done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 E" "DSX_ADAPT=0 DSX_ENGRAM_FP8=1" h5
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base h5 base h5; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-sk-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic")"
done
sleep 40; rm -f /tmp/dec15.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec15 /tmp/dsx-sk-h5 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec15.out 2>&1
echo "traced T: $(grep -E "^decode" /tmp/dec15.out)"; grep -iE "rror|panic" /tmp/dec15.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec15.sqlite /tmp/dec15.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec15.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN34-DONE $(date)"
