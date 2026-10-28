#!/bin/bash
# t16_pod_prep.sh JOBDIR [V41SRC]: the job directory for t16_pod.sh, built on this machine (R2 credentials stay here;
# the pod gets presigned URLs only):
#   manifest.tsv  presigned GETs (24 h) for hydrate.sh: quant2.tar.gz, ds41-engram.json, calib4 (as calib/), calib3
#   puts.tsv      presigned PUTs (36 h): R2 dsx/t16/full/L00.tar .. L39.tar
#   hydrate.sh canary.sh make_mul16_exllamav3.sh patch_qtf_opt.py qt_frac_opt.cuh t16_qworker.py t16_qselftest.py
#   t16_pod.sh t16_pod_up.sh, and v41.tgz (V41SRC, default this tree's quant/v41: v41_calib.py with V41_DUMP_ALL)
set -eo pipefail
JD=$1; R=$(cd "$(dirname "$0")/../.." && pwd); V41SRC=${2:-$R/quant/v41}
mkdir -p "$JD"; chmod 700 "$JD"
r2() { python3 $R/quant/rent/r2.py "$@" | tail -1; }
( umask 077
  { for k in quant2.tar.gz ds41-engram.json; do printf '%s\t%s\n' "$(r2 get dsx/stage/$k --hours 24)" $k; done
    for f in tokens.npy hashes.npy meta.json; do
      printf '%s\t%s\n' "$(r2 get dsx/t16/calib4/$f --hours 24)" calib/$f
      printf '%s\t%s\n' "$(r2 get dsx/stage/calib3/$f --hours 24)" calib3/$f
    done; } > "$JD/manifest.tsv"
  for L in $(seq 0 39); do LL=$(printf %02d $L); echo "L$LL $(r2 put dsx/t16/full/L$LL.tar --hours 36)"; done > "$JD/puts.tsv" )
[ $(grep -c "^https://" "$JD/manifest.tsv") = 8 ] && [ $(grep -c " https://" "$JD/puts.tsv") = 40 ] || { echo "presigning failed"; exit 1; }
cp $R/quant/rent/hydrate.sh $R/quant/rent/canary.sh "$JD/"
cp $R/quant/t16/{make_mul16_exllamav3.sh,patch_qtf_opt.py,qt_frac_opt.cuh,t16_qworker.py,t16_qselftest.py,t16_pod.sh,t16_pod_up.sh} "$JD/"
tar czf "$JD/v41.tgz" -C "$V41SRC" --exclude=__pycache__ .
grep -q V41_DUMP_ALL "$V41SRC/v41_calib.py" || { echo "v41_calib.py without V41_DUMP_ALL"; exit 1; }
echo "job dir ready: $JD ($(ls "$JD" | wc -l) files)"
