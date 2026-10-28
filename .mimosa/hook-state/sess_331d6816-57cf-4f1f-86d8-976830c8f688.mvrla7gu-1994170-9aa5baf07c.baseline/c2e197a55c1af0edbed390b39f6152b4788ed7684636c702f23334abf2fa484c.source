#!/bin/bash
# gemv_q8r1 and DSX_AUX_LATE A/B. Binaries (CUDA_ARCHS=86 --features cuda, own target dirs, eval tok/s print):
# base = f1f53e4 (/tmp/dsx-sk-base); h2 = c2c3d70 (/tmp/dsx-sk-h2: bit-identical kernels from window 30's K, gemv_q8r1
# for one-token Q8_0 GEMVs, early adapt off). Arms: base; K = h2 DSX_Q8R1=0; Q = h2; QA = h2 DSX_AUX_LATE=1.
# eval --decode 1 seq x 1023 steps, defer 3, adapt 8, 2 interleaved rounds; DSX_ADAPT=0 base vs Q (KL must be equal);
# gen 120 after 300 base / Q x2; nsys of Q. Predicted: Q over K -0.4..-1.0 ms/token (kill < 0.3 ms); K over base +3.4%
# as in window 30. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: gemv_q8r1 / AUX_LATE teacher-forced A/B (base / K / Q / QA), serve stopped (~40 min) $(date)" > GPU_LOCK
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
  run "round $r K" "DSX_Q8R1=0" h2
  run "round $r Q" "" h2
  run "round $r QA" "DSX_AUX_LATE=1" h2
done
run "adapt 0 base" "DSX_ADAPT=0" base
run "adapt 0 Q" "DSX_ADAPT=0" h2
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
for b in base h2 base h2; do sleep 40
  echo "gen $b: $(DSX_DEFER=3 /tmp/dsx-sk-$b gen --model $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode|rror|panic")"
done
sleep 40; rm -f /tmp/dec12.*
DSX_DEFER=3 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec12 /tmp/dsx-sk-h2 gen --model $M --ids @prompt300.ids -n 120 $P > /tmp/dec12.out 2>&1
echo "traced Q: $(grep -E "^decode" /tmp/dec12.out)"; grep -iE "rror|panic" /tmp/dec12.out | head -3
/usr/local/bin/nsys export --type sqlite -o /tmp/dec12.sqlite /tmp/dec12.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec12.sqlite
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN31-DONE $(date)"
