#!/bin/bash
# sc102.sh (on .102, inside a window with the dsx server stopped; takes no GPU_LOCK): decode-sized T16 CPU expert calls
# on branch t16-smallcall. Builds cpubench, then: the RAM read ceiling and per-phase breakdown (DSX_T16_PROF) of
# 1-6-expert calls for the barrier schedule and the new default; 4 interleaved A/B rounds of those calls; 2 interleaved
# rounds of the 6-expert nt sweep at 14 threads. ~6 min. Usage: bash sc102.sh > sc102.out 2>&1
cd "$(dirname "$0")/nvidia" || exit 1
echo "serve running: $(pgrep -fa 'dsx.*serve' | grep -v pgrep | wc -l); $(uptime)"
s=$(date +%s); ~/.cargo/bin/cargo build --release --bin cpubench 2>&1 | tail -1; echo "build $(( $(date +%s) - s )) s"
B=../target/release/cpubench
lscpu | grep "Model name"
for mode in barrier default; do
  if [ $mode = barrier ]; then E="DSX_T16_TAIL=0 DSX_T16_PIPE=0"; else E=""; fi
  echo "== breakdown, $mode schedule"; env $E DSX_T16_PROF=1 CPUB_CALLS=400 $B small
done
for r in 1 2 3 4; do for mode in barrier default; do
  if [ $mode = barrier ]; then E="DSX_T16_TAIL=0 DSX_T16_PIPE=0"; else E=""; fi
  echo -n "AB $mode "; env $E CPUB_CALLS=400 $B small 2>&1 | grep -oE "ne=[0-9]+: [0-9.]+ ms" | awk '{printf "%s ", $2}'; echo
done; done
for r in 1 2; do for mode in barrier default; do
  if [ $mode = barrier ]; then E="DSX_T16_TAIL=0 DSX_T16_PIPE=0"; else E=""; fi
  echo "== nt sweep, $mode"; env $E CPUB_THREADS=14 CPUB_NT=1,2,4,8,16 $B t16 2>&1 | grep -E "^14 threads"
done; done
echo "done $(date)"
