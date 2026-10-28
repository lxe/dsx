#!/bin/bash
# eval_job.sh NAME - on the pod after quant_job.sh: split the GGUF into 4 shards (with dsx.expert_rot), then the Lucebox
# eval on the prefill path, all positions and positions 512+. One GPU, every expert in RAM.
set -eo pipefail
NAME=$1; W=/work/ds41; D=/work/dsx/target/release/dsx
cd $W
mkdir -p hf-$NAME
$D split out/1017-$NAME.gguf hf-$NAME/DeepSeek-V4.1-Flash-1017-dsx-2bit --shards 4 --name "DeepSeek-V4.1-Flash-1017 dsx 2-bit $NAME" --expert-rot-from qwork-$NAME/q
M=hf-$NAME/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf
P="--main-gb=-4 --aux-gb 0 --arena-gb 300 --rank $W/rank.json"
echo "=== $NAME eval, all positions";  DSX_ADAPT=0 $D eval --model $M --engram fp8-1017 --eval eval $P 2>&1 | grep -E "^\{|rror"
echo "=== $NAME eval, positions 512+"; DSX_ADAPT=0 $D eval --model $M --engram fp8-1017 --eval eval $P --from 512 2>&1 | grep -E "^\{|rror"
echo EVAL-DONE
