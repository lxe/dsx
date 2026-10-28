#!/bin/bash
# Re-baseline under lxe's new limits (GPUs 230 W via gpu-cool, CPU RAPL 125/180 W, pump 60%): production build,
# decode 400 tokens after 3000 with DSX_DEFER 3 / 1 / 2 interleaved (2 rounds), prefill 3000 and 300; GPU power,
# clocks and CPU package temperature logged per run. Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: re-baseline under the new power/thermal limits, serve stopped (~40 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
echo "limits: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"; P="--ctx 32768 --chunk 4096 --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json"
for r in 1 2; do for d in 3 1 2; do sleep 25
  ( nvidia-smi --query-gpu=index,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active --format=csv,noheader,nounits -lms 200 > /tmp/rb.csv ) & smi=$!
  out=$(DSX_DEFER=$d ./guard.sh rb$RANDOM 118 /tmp/dsx-cur gen $M --ids @prompt3000.ids -n 400 $P 2>&1 | grep -E "^prefill [0-9]|^decode" | sed -E "s/; next top5.*//; s/GPU0 expert swaps.*//" | tr "\n" " ")
  kill $smi
  echo "r$r defer=$d: $out | $(python3 -c "
rows=[l.strip().split(', ') for l in open('/tmp/rb.csv') if l.count(',')>=4]
for g in ('0','1'):
    R=[r for r in rows[-150:] if r[0]==g and r[3].strip()!='0']
    if R:
        sm=sorted(int(r[1]) for r in R); pw=sorted(float(r[2]) for r in R)
        print(f'GPU{g} busy sm {sm[len(sm)//2]} MHz pw {pw[len(pw)//2]:.0f} W', end='; ')
") cpu $(sensors | grep Package | awk '{print $4}')"
done; done
# teacher-forced decode timing (identical tokens: the fair A/B when numerics change), production threads and adaptation
for d in 3 1 1 3; do
  sleep 40
  DSX_DEFER=$d ./guard.sh rb$RANDOM 118 /tmp/dsx-cur eval $M --eval eval --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 104 --rank calib/rank.json --decode --seqs 2 > /tmp/tf_$d.out 2>&1
  t=$(grep "^seq 1" /tmp/tf_$d.out); j=$(grep "^{" /tmp/tf_$d.out)
  echo "eval --decode defer=$d: $t $j cpu $(sensors | grep Package | awk '{print $4}')"
done
grep STOP watchdog.log | tail -2
sleep 25; echo "prefill 300: $(./guard.sh rb$RANDOM 118 /tmp/dsx-cur gen $M --ids @prompt300.ids -n 2 $P 2>&1 | grep -E "^prefill [0-9]" | sed -E "s/; next top5.*//")"
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN23-DONE $(date)"
