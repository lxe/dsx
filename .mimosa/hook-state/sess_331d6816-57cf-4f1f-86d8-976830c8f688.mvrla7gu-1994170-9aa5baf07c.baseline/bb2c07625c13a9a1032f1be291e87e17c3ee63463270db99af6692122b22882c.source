#!/bin/bash
# t16_full.sh on an 8-GPU pod (files in /root/t16job): the full-model T16 quant and its two KL evals.
#   1. hydrate (FP8 checkpoint from HF, quant code + calib3 from R2), exllamav3 da68d57 + the T16 patch, dsx from GitHub,
#      dsx-ud from HF, Lucebox eval files
#   2. t16_fetch.py: Wikipedia / The Stack smol-xl / UltraChat (~550K tokens) -> calib4 = calib3 + mix (prep-mix);
#      eval2 = 8 x 1024 tokens of SWE-agent / GitHub code / OpenHermes (neither calibration set)
#   3. v41_gguf.py --rtn q8_0 --experts-in-place: dense Q8_0 (as dsx-ud), expert regions as holes
#   4. GPU 0: forward chain over calib4 (teacher propagation), every expert's W + H to /work/dump (backpressure: at most
#      ~3 layers of dumps); GPU 1: teacher chains (eval2, and Lucebox's tokens as a check of the teacher path), then
#      workers; GPUs 1-7: t16_worker.py x 3 each (T16, sigma_reg 2, UD-sized layout, Q8_0 in place)
#   5. dsx eval: T16 and dsx-ud on Lucebox and eval2 (all positions), T16 also from 512
# Flags in /root/t16.flag; logs in /root/*.log.
set -x
J=/root/t16job; W=/work/ds41; T=/work/t16
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
flag() { echo "$* $(date +%T)" >> /root/t16.flag; }
mkdir -p $T /root/eval
cat /sys/fs/cgroup/cpu.max > /root/cgroup.txt; nproc >> /root/cgroup.txt; free -g >> /root/cgroup.txt; nvidia-smi -L >> /root/cgroup.txt
while read f u; do curl -sf -o /root/eval/$f "$u"; done < $J/urls.txt
curl -sf -o $T/rank.json "$(cat $J/rank_url.txt)"
bash $J/hydrate.sh $J/manifest.tsv > /root/hydrate.log 2>&1 &
HPID=$!
( until python3 -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.11') else 1)" 2>/dev/null; do sleep 15; done
  pip install -q ninja rich marisa_trie pydantic pyyaml llguidance transformers pyarrow > /dev/null 2>&1
  mkdir -p /root/x3 && cd /root/x3 && git init -q && git remote add origin https://github.com/turboderp-org/exllamav3
  git fetch -q --depth 1 origin da68d57de61781e5b74ad68698ac4caf36408d7a && git checkout -q FETCH_HEAD
  bash $J/make_mul16_exllamav3.sh /root/x3 /root/x3t16 > /root/t16patch.log 2>&1 || { flag PATCH-FAILED; exit 1; }
  # H stays on the device after finalize (its CPU un-rotation under a cgroup CPU cap was 20x slow)
  python3 - /root/x3t16/exllamav3/modules/quant/exl3_lib/quantize.py <<'PY'
import sys; p = sys.argv[1]; s = open(p).read()
old = '        H = H.cpu()\n        H_data["H"] = H.cpu()\n'
assert old in s; open(p, "w").write(s.replace(old, '        H_data["H"] = H   # kept on the device (dsx)\n'))
PY
  cd /tmp && TORCH_CUDA_ARCH_LIST="12.0;9.0" MAX_JOBS=64 PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 python3 -c "import exllamav3.ext" > /root/build_t16.log 2>&1
  flag "BUILD-t16 $?" ) &
( command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q > /dev/null 2>&1; }
  source ~/.cargo/env
  mkdir -p /work/dsx && cd /work/dsx && git init -q && git remote add origin https://github.com/lxe/dsx
  git fetch -q --depth 1 origin main && git checkout -q FETCH_HEAD
  CUDA_ARCHS=120,90 NVCC=/usr/local/cuda/bin/nvcc cargo build -q --release -p dsx --features cuda > /root/dsx_build.log 2>&1
  flag "DSX-BUILD $?" ) &
( until command -v hf >/dev/null; do sleep 10; done; hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-ud --local-dir /work/dsx-ud --max-workers 16 > /root/dl_ud.log 2>&1; flag "DSXUD-DL $?" ) &
wait $HPID; flag "HYDRATE $?"; tail -3 /root/hydrate.log
until grep -q "BUILD-t16" /root/t16.flag; do sleep 10; done
export V41_MODEL=$W/fp8-1017 V41_REF=$W/fp8-1017/inference V41_ROWS=8192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  V41_SCREEN_TIERS="" V41_SCREEN_IQ2=0
# Q8_0 encoder of the workers == v41_formats.encode_q8_0 (bit-exact), and T16 imports on this build
mkdir -p /root/v41 && tar xzf $J/v41.tgz -C /root/v41
( cd /tmp && PYTHONPATH=/root/x3t16:/root/v41 TORCH_EXTENSIONS_DIR=/root/ext_t16 python3 $J/t16_selftest.py > /root/selftest.log 2>&1; flag "SELFTEST $?" )
grep -q "SELFTEST 0" /root/t16.flag || { flag ABORT-selftest; exit 1; }
# today's quant/v41 over the R2 snapshot (quant2.tar.gz: native libs, older python)
mkdir -p /root/v41 && tar xzf $J/v41.tgz -C /root/v41
for V in calib4 eval2 lucet; do mkdir -p /work/$V/quant && cp -r $W/quant2/* /work/$V/quant/ && cp -r /root/v41/* /work/$V/quant/; done
# 2. data
python3 $J/t16_fetch.py $W/fp8-1017 /root/eval/heldout.txt $T/data > /root/fetch.log 2>&1; flag "FETCH $?"
( cd /work/calib4/quant && V41_WORK=/work/calib4/w python v41_calib.py prep-mix --base $W/calib3 --mix $T/data/calib_mix.jsonl > /root/prepmix.log 2>&1; flag "PREPMIX $?" )
( cd /work/eval2/quant && V41_WORK=/work/eval2/w python v41_calib.py prep-ids $T/data/eval2.i32 --rows 8 --domain eval2 > /root/prep_eval2.log 2>&1
  cd /work/lucet/quant && V41_WORK=/work/lucet/w python v41_calib.py prep-ids /root/eval/heldout.i32 --rows 8 --domain lucebox >> /root/prep_eval2.log 2>&1; flag "PREPIDS $?" )
# 3. the in-place GGUF (dense Q8_0, expert holes)
( cd /work/calib4/quant && python v41_gguf.py --out $T/1017-t16q8.gguf --rtn q8_0 --experts-in-place --no-engram --threads 16 > /root/gguf.log 2>&1; flag "GGUF $?" )
grep -q "GGUF 0" /root/t16.flag || { flag ABORT-gguf; exit 1; }
# 4. chains and workers
mkdir -p /work/dump $T/stats
( cd /work/calib4/quant && export CUDA_VISIBLE_DEVICES=0 V41_WORK=/work/calib4/w V41_DUMP_ALL=/work/dump
  python v41_calib.py init 2>&1 | tail -1
  for L in $(seq 0 39); do
    while [ $(find /work/dump -name 'e*.pt*' | wc -l) -gt 1100 ]; do sleep 10; done   # backpressure: <= ~3 layers of dumps
    python v41_calib.py layer $L --device cuda --gc 2>&1 | grep --line-buffered -E "done in|Error|Traceback"
  done
  echo CHAIN-DONE ) > /root/chain.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  for V in eval2 lucet; do
    cd /work/$V/quant && export V41_WORK=/work/$V/w
    python v41_calib.py init 2>&1 | tail -1
    for L in $(seq 0 39); do python v41_calib.py layer $L --device cuda --gc 2>&1 | grep --line-buffered -E "done in|Error|Traceback"; done
    python v41_calib.py head --device cuda --teacher-out /work/$V/teacher 2>&1 | tail -3
  done
  flag TEACHERS-DONE
  python3 $J/teacher_check.py /root/eval /work/lucet/teacher > /root/teacher_check.log 2>&1
  for k in 0 1 2; do CUDA_VISIBLE_DEVICES=1 Q_THREADS=2 T16_PACKDIR=$T/packed PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 \
      python3 $J/t16_worker.py 1$k $T/1017-t16q8.gguf /work/dump $T/stats > /root/worker_1$k.log 2>&1 & done
  wait ) > /root/teachers.log 2>&1 &
for g in 2 3 4 5 6 7; do for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g Q_THREADS=2 T16_PACKDIR=$T/packed PYTHONPATH=/root/x3t16 TORCH_EXTENSIONS_DIR=/root/ext_t16 \
    python3 $J/t16_worker.py $g$k $T/1017-t16q8.gguf /work/dump $T/stats > /root/worker_$g$k.log 2>&1 &
done; done
wait
n=$(find $T/stats -name 'e*.json' | wc -l); flag "QUANT-DONE $n experts"
sync
# 5. evals (dsx, one GPU + RAM arena as on the earlier pods)
until grep -q "DSX-BUILD" /root/t16.flag && grep -q "DSXUD-DL" /root/t16.flag; do sleep 10; done
D=/work/dsx/target/release/dsx; P="--main-gb=-4 --aux-gb 0 --arena-gb ${ARENA_GB:-600} --rank $T/rank.json"
UD=$(ls /work/dsx-ud/DeepSeek-V4.1-Flash-1017-dsx-ud-00001-of-0000*.gguf | head -1)
ev() { echo "=== $1"; DSX_ADAPT=0 CUDA_VISIBLE_DEVICES=0 $D eval --model $2 --engram $W/fp8-1017 --eval $3 $P ${@:4} 2>&1 | grep -E "^\{|rror|panic"; }
{ ev "T16 Lucebox" $T/1017-t16q8.gguf /root/eval
  ev "dsx-ud eval2" $UD /work/eval2/teacher
  ev "T16 eval2" $T/1017-t16q8.gguf /work/eval2/teacher
  ev "dsx-ud Lucebox (same-pod control)" $UD /root/eval
  ev "T16 Lucebox from 512" $T/1017-t16q8.gguf /root/eval --from 512; } > /root/evals.log 2>&1
flag ALL-DONE
