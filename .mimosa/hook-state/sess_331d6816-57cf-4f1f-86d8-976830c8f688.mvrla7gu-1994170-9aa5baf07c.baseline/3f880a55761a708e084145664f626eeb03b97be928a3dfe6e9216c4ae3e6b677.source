#!/bin/bash
# Server turns with the CPU share off vs on (current tree: CPU share, multi-token + small-call CPU kernels): one test
# server per setting (own scope and log), turns.py's conversation on each; then serve restarted on the production binary.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: server turn benchmark, serve replaced by test servers (~12 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
stop() {
  pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid
  while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
  tmux kill-session -t serve 2>/dev/null; tmux kill-session -t tsrv 2>/dev/null
  for u in $(systemctl --user list-units --type=scope --no-legend "ds41-tsrv*" | awk '{print $1}'); do systemctl --user stop "$u"; done
  systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 2
}
for v in 0 1 0 1; do
  stop
  tmux new-session -d -s tsrv "DSX_PREFILL_CPU=$v ./guard.sh tsrv$v$RANDOM 118 /tmp/dsx-server-cur --model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --host 127.0.0.1 --port 8090 --ctx 131072 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json --defer-rank 3 --vision hf-t16/vision.safetensors > /tmp/tsrv_$v.log 2>&1"
  ok=0; for i in $(seq 120); do curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8090/props | grep -q 200 && { ok=1; break; }; sleep 2; done
  [ $ok = 1 ] || { echo "test server $v did not come up"; tail -3 /tmp/tsrv_$v.log; continue; }
  echo "== CPU share $v"; python3 turns.py /tmp/tsrv_$v.log
done
stop
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN10-DONE $(date)"
