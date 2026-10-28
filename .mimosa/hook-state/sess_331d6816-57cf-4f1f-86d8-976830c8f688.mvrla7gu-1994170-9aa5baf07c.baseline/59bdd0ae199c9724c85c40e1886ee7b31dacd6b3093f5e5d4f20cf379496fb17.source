#!/bin/bash
# Dev4's small-call CPU expert benchmark (sc102.sh, serve stopped), then a fresh server's first requests: start serve,
# and as soon as it answers, three identical 200-token generations timed from this box (cold vs warm decode).
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: Dev4 small-call bench + fresh-server first-request test, serve stopped (~15 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; sleep 3
echo "== sc102"; bash ~/src/dsx-smallcall/sc102.sh > ~/ai/ds41/sc102.out 2>&1; tail -40 ~/ai/ds41/sc102.out
echo "== fresh server"
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
until curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8090/props | grep -q 200; do sleep 2; done
echo "up at $(date +%T)"
for i in 1 2 3; do
  curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H "content-type: application/json" -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 150-word story about a lighthouse keeper.\"}],\"max_tokens\":200,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" > /dev/null
  echo "request $i: $(grep -a "engine: done" ~/ai/ds41/serve.out | tail -1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "prompt_tokens=[0-9]+|generated=[0-9]+|prefill_ms=[0-9]+|decode_tps=\"[0-9.]+\"" | tr "\n" " ")"
done
echo "WIN9-DONE $(date)"
