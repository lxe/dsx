#!/bin/bash
# w4_102.sh (on .102, inside a window with the dsx server stopped; takes no GPU_LOCK, sets no power limits): Dev 4's
# window on branch window-dev4. Raw logs go to $OUT (default ~/ai/ds41/w4-<time>), a summary to stdout.
#   1. kernels on GPU0, one process per config, interleaved, 25 s apart:
#      - engram wkv 25600 x 6144: gemv_f16 vs gemv_fp8b32 vs gemv_fp8b32_o (E1's kernel, bit-identical to gemv_f16)
#      - T16 decode layer call (6 experts of 64, t16_decode_speed), DSX_T16_GEMV=0 (old) vs 1 (cp.async), K 1.5/2 and 2/2.5
#      - t16_decode_bits with DSX_T16_GEMV=0 and 1: the two hashes must be equal
#   2. sudo ncu on sm_86 (budget section 7 and lever 5): gemv_q8r, t16d_gemv old / new, the three engram GEMVs:
#      warp instructions per DRAM byte, issue use, occupancy, stalls (bench/ncu_parse.py)
#   3. cpubench t16 at 2 threads (one P-core) and 14 (the pool), under the 125 W RAPL cap
#   4. teacher-forced decode A/B (`dsx eval --decode`), ROUNDS (default 2) interleaved rounds of 4 arms:
#        base  DSX_ENGRAM_FP8=0 DSX_T16_GEMV=0     e1  DSX_ENGRAM_FP8=1 DSX_T16_GEMV=0
#        t16   DSX_ENGRAM_FP8=0 DSX_T16_GEMV=1     both DSX_ENGRAM_FP8=1 DSX_T16_GEMV=1
#      Both changes are bit-identical, so every arm's KL / top-1 / nll must equal base's exactly; tok/s from the
#      "teacher-forced decode" line. A 60 s pause before each eval; DSX_CPU_THREADS=$CPU_THREADS (default 8, as in
#      window 24).
# EVAL_ARGS: `dsx eval` arguments as in window 24 (model, engram, --eval ~/ai/ds41/eval, --seqs 1, DEFER etc. via env),
# without --decode:
#   EVAL_ARGS="--model ... --engram ... --eval ~/ai/ds41/eval --seqs 1" bash bench/w4_102.sh > w4.out 2>&1
# ~40 min. Failures print the log's tail unfiltered.
cd "$(dirname "$0")/.." || exit 1
[ -n "$EVAL_ARGS" ] || { echo "set EVAL_ARGS (dsx eval arguments, no --decode)"; exit 1; }
case " $EVAL_ARGS " in *" --decode"*) echo "drop --decode from EVAL_ARGS"; exit 1;; esac
OUT=${OUT:-~/ai/ds41/w4-$(date +%m%d-%H%M)}; mkdir -p "$OUT"
NCU=${NCU:-/usr/local/cuda-13.4/bin/ncu}
TH=DSX_CPU_THREADS=${CPU_THREADS:-8}
echo "serve running: $(pgrep -fa '[d]sx.*serve' | wc -l); $(uptime); logs in $OUT; branch $(git rev-parse --short HEAD)"
temps() { echo "  $(date +%T) $(sensors 2> /dev/null | grep -m1 Package) GPU C,W,cap $(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | tr -d " " | tr "\n" " ")"; }
run() { # run LOG CMD...: full output to $OUT/LOG; on failure its last 30 lines
  local log=$OUT/$1; shift
  if ! "$@" > "$log" 2>&1; then echo "  FAILED ($*): tail of $log"; tail -30 "$log"; return 1; fi
}
s=$(date +%s)
(cd nvidia && ~/.cargo/bin/cargo build --release --features cuda --bin dsx --bin cpubench 2>&1 | tail -1 &&
  ~/.cargo/bin/cargo test --release --features cuda --test gpu_t16 --test gpu_engram_fp8 --test gpu_q8 --no-run 2>&1 | tail -1)
echo "build $(( $(date +%s) - s )) s"
B=target/release/dsx
bin() { ls -t target/release/deps/$1-* | grep -v '\.d$' | head -1; }
T16=$(bin gpu_t16); E8=$(bin gpu_engram_fp8); Q8=$(bin gpu_q8)

echo "== 1. kernels"; temps
run k_engram.log "$E8" fp8_gemv_speed --ignored --exact --nocapture && grep -E "gemv_" "$OUT/k_engram.log"
for r in 1 2; do
  for v in 0 1; do
    for K in 1.5,2 2,2.5; do
      sleep 25
      run k_t16_${r}_${v}_${K}.log env DSX_T16_GEMV=$v T16_SPEED_K=$K "$T16" t16_decode_speed --ignored --exact --nocapture &&
        echo "  round $r DSX_T16_GEMV=$v K $K: $(grep -E 'T16: mean' "$OUT/k_t16_${r}_${v}_${K}.log")"
    done
  done
done
for v in 0 1; do
  run bits_$v.log env DSX_T16_GEMV=$v "$T16" t16_decode_bits --exact --nocapture && echo "  DSX_T16_GEMV=$v: $(grep 'all cases' "$OUT/bits_$v.log")"
done
temps

echo "== 2. ncu (sm_86)"
M=$(cat bench/ncu_metrics.txt)
sleep 25
run ncu_q8.log sudo -n "$NCU" --csv --page raw -k "regex:^gemv_q8r$" -c 10 --metrics "$M" "$Q8" q8_variants --ignored --exact
for v in 0 1; do
  sleep 25
  run ncu_t16_$v.log sudo -n env DSX_T16_GEMV=$v T16_SPEED_K=1.5,2 "$NCU" --csv --page raw -k "regex:^t16d_gemv" -c 6 --metrics "$M" "$T16" t16_decode_speed --ignored --exact
done
sleep 25
run ncu_engram.log sudo -n "$NCU" --csv --page raw -k "regex:^gemv_(f16|fp8b32|fp8b32_o)$" -c 6 --metrics "$M" "$E8" fp8_gemv_speed --ignored --exact
for f in ncu_q8 ncu_t16_0 ncu_t16_1 ncu_engram; do [ -s "$OUT/$f.log" ] && python3 -I bench/ncu_parse.py "$OUT/$f.log" | cut -c1-260; done
temps

echo "== 3. cpubench t16 (CPU only)"
sleep 25
run cpu_t16.log env CPUB_THREADS=2,14 CPUB_NT= target/release/cpubench t16 && grep -E "threads:" "$OUT/cpu_t16.log"
temps

echo "== 4. teacher-forced decode A/B"
arm() { case $1 in base) echo "DSX_ENGRAM_FP8=0 DSX_T16_GEMV=0";; e1) echo "DSX_ENGRAM_FP8=1 DSX_T16_GEMV=0";; t16) echo "DSX_ENGRAM_FP8=0 DSX_T16_GEMV=1";; both) echo "DSX_ENGRAM_FP8=1 DSX_T16_GEMV=1";; esac; }
for r in $(seq 1 "${ROUNDS:-2}"); do
  for a in base e1 t16 both; do
    sleep 60; temps
    E=$(arm $a)
    if run eval_${r}_$a.log env $E $TH $B eval $EVAL_ARGS --decode; then
      echo "  round $r $a ($E): $(grep -E '^teacher-forced' "$OUT/eval_${r}_$a.log") $(grep -E 'kl_top' "$OUT/eval_${r}_$a.log")"
    fi
  done
done
temps; echo "done $(date)"
