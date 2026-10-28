#!/bin/bash
# Exactness of the integrate build (b0f4d07, /tmp/dsx-sk-h5) against base f1f53e4 (/tmp/dsx-sk-base), all at
# DSX_ADAPT=0 (static residency), teacher-forced eval --decode 1 seq x 1023 steps, defer 3. Window 34's adapt-0 pair
# base vs E (+ DSX_ENGRAM_FP8=1) differed (KL 0.160876 vs 0.168373): FP8 engram frees ~0.31 GB of GPU0, so more experts
# become resident. Arms: base; T = h5 defaults (all bit-identical changes, F16 engram): must equal base; E-same = h5
# DSX_ENGRAM_FP8=1 with --main-gb lowered by the freed bytes: GPU0 expert count must equal base's, then KL must too;
# E = h5 DSX_ENGRAM_FP8=1 at the production --main-gb (window 34's arm, for the resident count). Serve restarted after.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: integrate exactness at DSX_ADAPT=0 (4 evals), serve stopped (~15 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; systemctl --user reset-failed ds41-serve.scope 2>/dev/null
M=$HOME/ai/ds41/hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf
EV="--model $M --eval $HOME/ai/ds41/eval --aux-gb=-1.0 --arena-gb 104 --rank $HOME/ai/ds41/calib/rank.json --seqs 1 --decode"
run() { # label env binary main-gb
  sleep 30; echo "== $1"
  env $2 DSX_ADAPT=0 DSX_DEFER=3 /tmp/dsx-sk-$3 eval $EV --main-gb=$4 > /tmp/sk_eval.out 2>&1
  grep -E "^experts resident|^engram:|^teacher-forced|kl_top" /tmp/sk_eval.out | cut -c1-150 || tail -5 /tmp/sk_eval.out
}
run "base" "" base -1.5
run "T" "" h5 -1.5
run "E-same" "DSX_ENGRAM_FP8=1" h5 -1.814
run "E" "DSX_ENGRAM_FP8=1" h5 -1.5
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN35-DONE $(date)"
