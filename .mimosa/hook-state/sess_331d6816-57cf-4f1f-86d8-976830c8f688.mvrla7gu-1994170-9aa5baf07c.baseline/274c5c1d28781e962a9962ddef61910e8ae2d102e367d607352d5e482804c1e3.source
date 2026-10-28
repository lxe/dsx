#!/bin/bash
# t16_pod_watch.sh JOBDIR [CAP_USD]: every 2 min, appends the pod's progress and spend to JOBDIR/status.log and
# terminates the pod when every layer is uploaded (UPLOAD-DONE), on ABORT, or when the spend reaches CAP_USD (default
# 19.5; the run's cap is $20 including failures).
R=$(cd "$(dirname "$0")/../.." && pwd); JD=$1; CAP=${2:-19.5}; ST=$JD/pod.json
P=(python3 $R/quant/rent/pod.py ssh $ST)
while [ -f $ST ]; do
  spend=$(python3 -c "import json, time; s = json.load(open('$ST')); print(f'{(time.time() - s[\"started\"]) / 3600 * float(s[\"cost\"]):.2f}')")
  st=$(timeout 60 "${P[@]}" 'tail -3 /root/t16.flag | tr "\n" ";"; echo -n " packed "; ls /work/t16/packed/L*/ 2>/dev/null | grep -c safetensors; echo -n " uploaded "; grep -c "^UP-L" /root/t16.flag; echo -n " last "; tail -qn1 /root/worker_*.log 2>/dev/null | tr "\n" ";"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>&1 | tr "\n" " ")
  echo "$(date +%T) \$$spend $st" >> $JD/status.log
  if echo "$st" | grep -q -E "UPLOAD-DONE|ALL-DONE|ABORT"; then
    python3 $R/quant/rent/pod.py down $ST >> $JD/status.log 2>&1; mv $ST $JD/pod.done.json; echo "$(date +%T) terminated: $st" >> $JD/status.log; exit 0
  fi
  if python3 -c "import sys; sys.exit(0 if $spend >= $CAP else 1)"; then
    python3 $R/quant/rent/pod.py down $ST >> $JD/status.log 2>&1; mv $ST $JD/pod.done.json; echo "$(date +%T) terminated at the cap: \$$spend" >> $JD/status.log; exit 0
  fi
  sleep 120
done
