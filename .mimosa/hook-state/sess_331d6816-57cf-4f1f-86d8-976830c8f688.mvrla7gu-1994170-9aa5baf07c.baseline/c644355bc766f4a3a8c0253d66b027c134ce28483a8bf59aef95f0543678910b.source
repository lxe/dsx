#!/bin/bash
# .102 watchdog: every 10 s log GPU temp/mem/util, free RAM, CPU temp; on a new NVIDIA Xid or MemAvailable < 6 GB or
# CPU >= 95 C, stop every ds41-* scope (GPU jobs first) before the box can wedge. Log: ~/ai/ds41/watchdog.log
log=~/ai/ds41/watchdog.log
seen=$(sudo -n dmesg | grep -c "NVRM: Xid")
stop_all() {
  echo "$(date '+%F %T') STOP: $1" >> $log
  for u in $(systemctl --user list-units --type=scope --no-legend 'ds41-*' | awk '{print $1}'); do
    systemctl --user kill -s SIGTERM "$u"; done
  sleep 15
  for u in $(systemctl --user list-units --type=scope --no-legend 'ds41-*' | awk '{print $1}'); do
    systemctl --user kill -s SIGKILL "$u"; done
}
while true; do
  g=$(nvidia-smi --query-gpu=index,temperature.gpu,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits 2>&1 | tr '\n' ';')
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  cpu=$(sensors 2>/dev/null | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')
  echo "$(date '+%F %T') gpu[$g] ram_avail=${avail}G cpu=${cpu}C" >> $log
  n=$(sudo -n dmesg | grep -c "NVRM: Xid")
  if [ "$n" -gt "$seen" ]; then stop_all "new Xid: $(sudo -n dmesg | grep 'NVRM: Xid' | tail -1)"; seen=$n; fi
  if echo "$g" | grep -qi "error\|unknown"; then stop_all "nvidia-smi error: $g"; fi
  [ "$avail" -lt 6 ] && stop_all "RAM available ${avail}G < 6G"
  [ -n "$cpu" ] && [ "$cpu" -ge 95 ] && stop_all "CPU ${cpu}C"
  for gt in $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null); do [ "$gt" -ge 90 ] && stop_all "GPU ${gt}C"; done
  sleep 10
done
