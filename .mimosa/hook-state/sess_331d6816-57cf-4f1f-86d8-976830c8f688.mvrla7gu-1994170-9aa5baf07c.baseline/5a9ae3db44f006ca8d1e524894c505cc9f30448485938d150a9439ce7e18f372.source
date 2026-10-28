#!/bin/bash
# ef102.sh (on .102, inside a window with the dsx server stopped; takes no GPU_LOCK): DSX_ENGRAM_FP8 (engram wkv as
# FP8 + 32x32 E8M0 scales, converted exactly from the GGUF's F16) A/B on branch engram-fp8. Builds dsx, then:
#   0. kernel: gemv_f16 vs gemv_fp8b32 on the engram shape (random weights), interleaved medians
#   1. decode, ROUNDS interleaved rounds (default 2) of three arms, one cold `dsx gen` process each:
#        f16       DSX_ENGRAM_FP8=0, --main-gb $MAIN_GB
#        fp8-same  DSX_ENGRAM_FP8=1, --main-gb $MAIN_GB - 0.63: GPU0 keeps the f16 arm's expert count (kernel effect)
#        fp8       DSX_ENGRAM_FP8=1, --main-gb $MAIN_GB: the freed ~630 MB goes to GPU0 experts (kernel + placement)
#      then one DSX_PROF=1 run per engram setting for the "engram" section
#   2. quality: `dsx eval --decode --seqs 2`, DSX_ENGRAM_FP8 0 vs 1 (same weights, only GEMV summation order differs),
#      at DSX_CPU_THREADS=8 after a 60 s cool-down: a decode eval is a long all-core CPU-expert load that otherwise
#      reaches 96-97 C (the CPU kernels are bit-exact at any thread count, so KL is unchanged)
# 25 s pauses with the package and GPU temperatures between runs (.102's watchdog stops ds41 scopes at >= 95 C).
# GEN_ARGS: your usual `dsx gen` arguments for the 3K agent prompt and ~400 decode tokens, WITHOUT --main-gb/--gpu0-gb
# (MAIN_GB, default -1.5 = dsx's default headroom, sets it); EVAL_ARGS: `dsx eval` arguments for the Lucebox eval:
#   GEN_ARGS="--model ... --ids ~/ai/ds41/prompt3k.ids -n 400 ..." EVAL_ARGS="--model ... --eval ~/ai/ds41/eval" \
#     bash bench/ef102.sh > ef102.out 2>&1
# Lines to read: "gemv_" (kernel us, GB/s), "engram:" (FP8 exact per layer), "experts resident:", "decode N tokens",
# the "engram" DSX_PROF section, and eval's KL / top-1.
cd "$(dirname "$0")/.." || exit 1
[ -n "$GEN_ARGS" ] || { echo "set GEN_ARGS (dsx gen arguments, no --main-gb) and EVAL_ARGS (dsx eval arguments)"; exit 1; }
case " $GEN_ARGS " in *" --main-gb"*|*" --gpu0-gb"*) echo "drop --main-gb/--gpu0-gb from GEN_ARGS; set MAIN_GB"; exit 1;; esac
MAIN_GB=${MAIN_GB:--1.5}
SAME_GB=$(python3 -c "print(round($MAIN_GB - 0.63, 2))")
echo "serve running: $(pgrep -fa 'dsx.*serve' | grep -v pgrep | wc -l); $(uptime)"
s=$(date +%s); (cd nvidia && ~/.cargo/bin/cargo build --release --features cuda --bin dsx 2>&1 | tail -1); echo "build $(( $(date +%s) - s )) s"
B=target/release/dsx
cool() { sleep 25; echo "  $(date +%T) $(sensors 2> /dev/null | grep -m1 Package) GPU C,W,cap $(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | tr -d " " | tr "\n" " ")"; }
echo "== kernel (GPU0)"
(cd nvidia && ~/.cargo/bin/cargo test --release --features cuda --test gpu_engram_fp8 -- --include-ignored --nocapture --test-threads 1 2>&1 | grep -E "engram wkv|gemv_|test result|panicked")
cool
for r in $(seq 1 "${ROUNDS:-2}"); do
  for arm in f16 fp8-same fp8; do
    case $arm in
      f16) E="DSX_ENGRAM_FP8=0"; G=$MAIN_GB;;
      fp8-same) E="DSX_ENGRAM_FP8=1"; G=$SAME_GB;;
      fp8) E="DSX_ENGRAM_FP8=1"; G=$MAIN_GB;;
    esac
    echo "== round $r, $arm ($E --main-gb $G)"
    env $E $B gen $GEN_ARGS --main-gb $G 2>&1 | grep -E "^engram|^experts resident|^decode|rror|panic"
    cool
  done
done
for f in 0 1; do
  echo "== DSX_PROF=1, DSX_ENGRAM_FP8=$f"
  env DSX_PROF=1 DSX_ENGRAM_FP8=$f $B gen $GEN_ARGS --main-gb $MAIN_GB 2>&1 | grep -iE "engram|^decode|rror|panic"
  cool
done
if [ -n "$EVAL_ARGS" ]; then
  for f in 0 1; do
    sleep 35; cool
    echo "== eval --decode --seqs 2, DSX_ENGRAM_FP8=$f, DSX_CPU_THREADS=8"
    env DSX_CPU_THREADS=8 DSX_ENGRAM_FP8=$f $B eval $EVAL_ARGS --decode --seqs 2 2>&1 | grep -E "^engram|KL|kl|top|ppl|perplex|rror|panic" | tail -8
    cool
  done
fi
echo "done $(date)"
