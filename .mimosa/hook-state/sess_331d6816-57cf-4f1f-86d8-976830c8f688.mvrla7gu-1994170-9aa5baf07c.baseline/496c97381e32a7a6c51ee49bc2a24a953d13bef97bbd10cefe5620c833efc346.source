#!/bin/bash
# Merged build (CPU share + multi-token CPU kernel + new cost model): A) prefill 60/300/1024/2048 off vs on (2048 forced),
# decode 300 tokens after 3000 off vs on; B) decode profiling (dprof.sh's EVPROF and nsys) on this build.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: merged-build prefill/decode measurements + decode profiling, serve stopped (~35 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; sleep 3
B=/tmp/dsx-m1; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
g() { systemctl --user reset-failed ds41-w5.scope 2>/dev/null; sleep 2; env $2 ./guard.sh w5 118 $B gen $M --ids @$1.ids -n $3 $P 2>&1 | grep -E "^prefill [0-9]|^decode" | sed -E "s/; next top5.*//; s/GPU0 expert swaps.*//" | tr "\n" " "; }
echo "== A"
for f in prompt60 prompt300 prompt3k_1024 prompt3k_2048; do for v in 0 1 1 0; do echo "$f cpu=$v: $(g $f "DSX_PREFILL_CPU=$v DSX_PREFILL_CPU_MAX_N=4096" 4)"; done; done
for v in 0 1 1 0; do echo "decode cpu=$v: $(g prompt3000 "DSX_PREFILL_CPU=$v" 300)"; done
echo "== B"
for d in off 3; do systemctl --user reset-failed ds41-w5.scope 2>/dev/null; sleep 3; E=""; [ $d = 3 ] && E="DSX_DEFER=3"
  echo "== EVPROF defer=$d"; env DSX_EVPROF=1 $E ./guard.sh w5 118 $B gen $M --ids @prompt3000.ids -n 300 $P 2>&1 | grep -v -E "engram L|^\s*$|GPU[01] |arena|resident|host buffer|Running|generated|^prefill"
done
systemctl --user reset-failed ds41-w5.scope 2>/dev/null; sleep 3
./guard.sh w5 118 nsys profile -o /tmp/dec64 --force-overwrite true --trace=cuda,osrt --sample=none --cpuctxsw=none $B gen $M --ids @prompt300.ids -n 64 $P > /tmp/dec64.out 2>&1
grep -E "^decode" /tmp/dec64.out; nsys export --type sqlite -o /tmp/dec64.sqlite --force-overwrite true /tmp/dec64.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec64.sqlite
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN5-DONE $(date)"
