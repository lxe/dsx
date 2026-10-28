#!/bin/bash
# Lever 7 (the ~2.4 ms per token outside the layers) and lever 4 (small-kernel gaps): one nsys trace (CUDA API +
# kernels + copies, no OS-runtime or CPU sampling) of production decode, 120 tokens after a 300-token prompt, defer 3,
# production flags; one untraced run right before for the tracing overhead. Serve stopped ~12 min, restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: nsys trace of decode (lever 7), serve stopped (~12 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"; P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json"
sleep 20
echo "untraced: $(DSX_DEFER=3 ./guard.sh n0$RANDOM 118 /tmp/dsx-cur gen $M --ids @prompt300.ids -n 120 $P 2>&1 | grep -E "^decode" | sed -E 's/GPU0 expert swaps.*//')"
sleep 30
rm -f /tmp/dec7.nsys-rep /tmp/dec7.sqlite
DSX_DEFER=3 ./guard.sh n1$RANDOM 118 /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none -o /tmp/dec7 \
  /tmp/dsx-cur gen $M --ids @prompt300.ids -n 120 $P > /tmp/dec7.out 2>&1
echo "traced: $(grep -E "^decode" /tmp/dec7.out | sed -E 's/GPU0 expert swaps.*//')"; tail -3 /tmp/dec7.out
/usr/local/bin/nsys export --type sqlite -o /tmp/dec7.sqlite /tmp/dec7.nsys-rep > /dev/null 2>&1; ls -la /tmp/dec7.*
grep STOP watchdog.log | tail -1
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN25-DONE $(date)"
