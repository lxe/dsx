#!/bin/bash
# e2_102.sh (on .102, inside a window with the dsx server stopped; takes no GPU_LOCK): teacher-forced A/B of the
# engram changes on branch engram-fp8. Every arm decodes the same eval tokens (`dsx eval --decode`), so the speed
# comparison does not depend on generated text. Builds dsx, then:
#   0. kernel: gemv_f16 vs gemv_fp8b32 on the engram shape (random weights), interleaved medians
#   1. ROUNDS interleaved rounds (default 2) of the ARMS (default "f16 fp8 fp8e2"), one cold `dsx eval --decode`
#      process each; each prints "teacher-forced decode N steps in S s (tok/s)" and the JSON line with KL / top-1:
#        f16       DSX_ENGRAM_FP8=0
#        fp8       DSX_ENGRAM_FP8=1                       (E1; GPU0 gets the freed ~630 MB for experts)
#        fp8-same  DSX_ENGRAM_FP8=1 at --main-gb MAIN_GB - 0.63 (E1 at the f16 arm's GPU0 expert count)
#        fp8e2     DSX_ENGRAM_FP8=1 DSX_ENGRAM_PERSIST=1   (E1 + E2: persistent gather threads, layer 1's rows first)
#   2. one DSX_PROF=1 eval per arm (synchronizes per section: for the section times, not tok/s): "engram",
#      "host: engram join wait", and for fp8e2 "host: engram gather, 1st layer" and "engram rows not cached (n)"
# A 60 s pause before each eval (the package and GPU temperatures, power and power limit printed); evals run at the
# production CPU thread count unless CPU_THREADS is set (it changes speed, not KL: the CPU kernels are bit-exact).
#   EVAL_ARGS="--model ... --engram ... --eval ~/ai/ds41/eval --seqs 1" MAIN_GB=-1.5 bash bench/e2_102.sh > e2_102.out 2>&1
# EVAL_ARGS without --main-gb/--gpu0-gb and --decode (MAIN_GB, default -1.5 = dsx's default, sets it).
cd "$(dirname "$0")/.." || exit 1
[ -n "$EVAL_ARGS" ] || { echo "set EVAL_ARGS (dsx eval arguments, no --main-gb, no --decode)"; exit 1; }
case " $EVAL_ARGS " in *" --main-gb"*|*" --gpu0-gb"*|*" --decode"*) echo "drop --main-gb/--gpu0-gb/--decode from EVAL_ARGS"; exit 1;; esac
MAIN_GB=${MAIN_GB:--1.5}
SAME_GB=$(python3 -c "print(round($MAIN_GB - 0.63, 2))")
ARMS=${ARMS:-f16 fp8 fp8e2}
TH=${CPU_THREADS:+DSX_CPU_THREADS=$CPU_THREADS}
echo "serve running: $(pgrep -fa 'dsx.*serve' | grep -v pgrep | wc -l); $(uptime); arms: $ARMS; ${TH:-production CPU threads}"
s=$(date +%s); (cd nvidia && ~/.cargo/bin/cargo build --release --features cuda --bin dsx 2>&1 | tail -1); echo "build $(( $(date +%s) - s )) s"
B=target/release/dsx
temps() { echo "  $(date +%T) $(sensors 2> /dev/null | grep -m1 Package) GPU C,W,cap $(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | tr -d " " | tr "\n" " ")"; }
arm() {
  case $1 in
    f16) echo "DSX_ENGRAM_FP8=0 $MAIN_GB";;
    fp8) echo "DSX_ENGRAM_FP8=1 $MAIN_GB";;
    fp8-same) echo "DSX_ENGRAM_FP8=1 $SAME_GB";;
    fp8e2) echo "DSX_ENGRAM_FP8=1,DSX_ENGRAM_PERSIST=1 $MAIN_GB";;
  esac
}
echo "== kernel (GPU0)"; temps
(cd nvidia && ~/.cargo/bin/cargo test --release --features cuda --test gpu_engram_fp8 -- --include-ignored --nocapture --test-threads 1 2>&1 | grep -E "engram wkv|gemv_|test result|panicked")
for r in $(seq 1 "${ROUNDS:-2}"); do
  for a in $ARMS; do
    read -r E G <<< "$(arm $a)"
    sleep 60; temps
    echo "== round $r, $a (${E//,/ } $TH --main-gb $G)"
    env ${E//,/ } $TH $B eval $EVAL_ARGS --decode --main-gb $G 2>&1 | grep -E "^engram|^experts resident|^teacher-forced|kl_top|rror|panic"
  done
done
for a in $ARMS; do
  read -r E G <<< "$(arm $a)"
  sleep 60; temps
  echo "== DSX_PROF=1, $a"
  env DSX_PROF=1 ${E//,/ } $TH $B eval $EVAL_ARGS --decode --main-gb $G 2>&1 | grep -iE "engram|^teacher-forced|rror|panic"
done
temps; echo "done $(date)"
