#!/bin/bash
# t16_pod_up.sh DUMPDIR PACKDIR PUTS [FIRST LAST]: uploads each finished layer of PACKDIR (DUMPDIR/LNN/DONE written by
# the chain, no dump left to claim, 384 experts packed) as one tar, PACKDIR's LNN/eNNN.safetensors, to its presigned
# PUT URL (PUTS lines: "LNN URL"). Flags UP-LNN / UPFAIL-LNN in /root/t16.flag (T16_FLAG); deletes what it uploaded.
D=$1; P=$2; U=$3; FIRST=${4:-0}; LAST=${5:-39}; F=${T16_FLAG:-/root/t16.flag}; TMP=${T16_UPTMP:-/work}
flag() { echo "$* $(date +%T)" >> "$F"; }
for L in $(seq $FIRST $LAST); do
  LL=$(printf %02d $L)
  url=$(awk -v l=L$LL '$1 == l { print $2 }' "$U")
  [ -n "$url" ] || { flag "UPFAIL-L$LL no URL"; continue; }
  until [ -f $D/L$LL/DONE ] && ! ls $D/L$LL/e*.pt* > /dev/null 2>&1 && [ $(ls $P/L$LL/e*.safetensors 2> /dev/null | wc -l) -ge 384 ]; do sleep 20; done
  tar -cf $TMP/up_L$LL.tar -C $P L$LL
  ok=0
  for i in 1 2 3 4 5; do curl -sf --retry 3 -T $TMP/up_L$LL.tar "$url" -o /dev/null && { ok=1; break; }; sleep 15; done
  if [ $ok = 1 ]; then
    flag "UP-L$LL $(du -m $TMP/up_L$LL.tar | cut -f1) MB"
    rm -rf $TMP/up_L$LL.tar $P/L$LL $D/L$LL
  else
    flag "UPFAIL-L$LL"
  fi
done
flag UPLOAD-DONE
