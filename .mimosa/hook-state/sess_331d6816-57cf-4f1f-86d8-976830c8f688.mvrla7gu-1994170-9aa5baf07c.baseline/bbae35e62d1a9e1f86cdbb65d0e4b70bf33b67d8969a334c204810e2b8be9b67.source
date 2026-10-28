#!/bin/bash
# nsys of real server turns: server under `nsys launch` (CPU share on, production flags), base request uncaptured,
# then capture a ~60-token and a ~300-token turn. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: nsys of server turns, serve replaced (~8 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null; sleep 2
rm -f /tmp/conv.json /tmp/srvturn.nsys-rep /tmp/srvturn.sqlite
tmux new-session -d -s tsrv "./guard.sh nsrv$RANDOM 118 nsys launch --session-new=srvturn --trace=cuda,osrt --sample=none --cpuctxsw=none ~/ai/ds41/dsx/target/release/dsx-server --model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --host 127.0.0.1 --port 8090 --ctx 131072 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json --defer-rank 3 --vision hf-t16/vision.safetensors > /tmp/nsrv.log 2>&1"
ok=0; for i in $(seq 150); do curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8090/props | grep -q 200 && { ok=1; break; }; sleep 2; done
[ $ok = 1 ] || { echo "server under nsys did not come up"; tail -5 /tmp/nsrv.log; }
python3 turns2.py /tmp/nsrv.log base
python3 turns2.py /tmp/nsrv.log turn 240
nsys start --session=srvturn --output=/tmp/srvturn --force-overwrite=true
sleep 1
python3 turns2.py /tmp/nsrv.log turn 240
python3 turns2.py /tmp/nsrv.log turn 1200
sleep 1
nsys stop --session=srvturn
sleep 3
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; sleep 5; tmux kill-session -t tsrv 2>/dev/null
for u in $(systemctl --user list-units --type=scope --no-legend "ds41-nsrv*" | awk '{print $1}'); do systemctl --user stop "$u"; done
ls -la /tmp/srvturn.nsys-rep && nsys export --type sqlite -o /tmp/srvturn.sqlite --force-overwrite true /tmp/srvturn.nsys-rep > /dev/null 2>&1; ls -la /tmp/srvturn.sqlite
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN12-DONE $(date)"
