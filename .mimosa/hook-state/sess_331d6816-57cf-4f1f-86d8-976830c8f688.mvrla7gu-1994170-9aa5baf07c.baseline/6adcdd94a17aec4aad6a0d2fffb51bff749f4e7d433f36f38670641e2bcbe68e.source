#!/bin/bash
# when the joint chain and the t100 GGUF are done: build the joint GGUF, then eval both (finish_v2.sh)
cd ~/ai/ds41
until grep -q "quant chain done" qv2-joint.log && grep -q GGUF_T100_DONE gguf_t100.log; do
  grep -q FAILED qv2-joint.log && { echo "joint chain FAILED"; exit 1; }
  sleep 30
done
bash finish_v2.sh
