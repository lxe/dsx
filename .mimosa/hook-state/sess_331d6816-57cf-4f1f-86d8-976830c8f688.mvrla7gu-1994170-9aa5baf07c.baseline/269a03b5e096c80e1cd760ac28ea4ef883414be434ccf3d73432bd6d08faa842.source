#!/bin/bash
# t16_pod_rent.sh JOBDIR: rents 1x RTX PRO 6000 (Server or Workstation Edition) at community price that passes the network canary within 60 s of ssh
# (any failure: terminate at once and skip-list the host), copies JOBDIR to /root/t16job and starts t16_pod.sh (NW
# workers, default 2). Writes JOBDIR/pod.json (pod.py state). Never retries a skip-listed host.
R=$(cd "$(dirname "$0")/../.." && pwd); JD=$1; ST=$JD/pod.json
SKIP="74.2.96.196 87.120.211.207 $(cat $JD/skip.txt 2> /dev/null | tr '\n' ' ')"   # hosts that failed before, kept across runs
P=(python3 $R/quant/rent/pod.py ssh $ST)
GPUS=("NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")   # same GB202, 96 GB
for try in $(seq 1 ${T16_TRIES:-6}); do
  rm -f $ST
  gpu=${GPUS[$(( (try - 1) % ${#GPUS[@]} ))]}
  timeout 900 python3 $R/quant/rent/pod.py up $ST --name dsx-t16-full1 --gpu "$gpu" --cloud ${T16_CLOUD:-COMMUNITY} --disk 850 --tries 1 2>&1 | tail -2
  [ -f $ST ] || continue
  ip=$(python3 -c "import json; print(json.load(open('$ST'))['ip'])")
  if echo " $SKIP " | grep -q " $ip "; then echo "$ip skip-listed"; python3 $R/quant/rent/pod.py down $ST | tail -1; continue; fi
  "${P[@]}" -put $JD/canary.sh /root/ > /dev/null 2>&1
  r=$(timeout 60 "${P[@]}" 'bash /root/canary.sh' 2>&1 | tail -1)
  echo "$ip: ${r:-CANARY-BAD timeout}"
  case "$r" in CANARY-OK*) break ;; esac
  SKIP="$SKIP $ip"; echo $ip >> $JD/skip.txt; python3 $R/quant/rent/pod.py down $ST | tail -1; rm -f $ST
done
[ -f $ST ] || { echo "NO-POD"; exit 1; }
"${P[@]}" 'mkdir -p /root/t16job' && "${P[@]}" -put $(ls -d $JD/* | grep -v pod.json) /root/t16job/ || { echo "PUSH-FAILED"; exit 1; }
"${P[@]}" "cd /root && NW=${NW:-2} setsid nohup bash /root/t16job/t16_pod.sh > /root/t16_pod.log 2>&1 < /dev/null & echo started"
