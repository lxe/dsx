#!/usr/bin/env bash
# Opt-in single-MI325X profile for low-temperature agentic coding, especially four sessions.
set -euo pipefail
DSX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$DSX_ROOT"
DSX_MODEL_DIR=${DSX_MODEL_DIR:-/data/models/v41-1017}
DSX_RANK=${DSX_RANK:-/data/models/dsx-mxfp4/expert-rank.json}
export DSXA_WEIGHTS=${DSXA_WEIGHTS:-/data/models/dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf}
export DSXA_DSPARK=${DSXA_DSPARK:-4}
export DSXA_RESERVE_GB=${DSXA_RESERVE_GB:-15.998758912}
export DSXA_BBLAS=${DSXA_BBLAS:-1}
export DSXA_BBLAS_MIN=${DSXA_BBLAS_MIN:-32}
export DSXA_BDENSE_ONLY=${DSXA_BDENSE_ONLY:-attn.wq_b.weight,attn.wo_a.weight,attn.wo_b.weight}
export DSXA_ATTN_FINE=${DSXA_ATTN_FINE:-1}
export DSXA_ATTN_MERGE_OUT=${DSXA_ATTN_MERGE_OUT:-1}
export DSXA_SWIGLU_PAIR=${DSXA_SWIGLU_PAIR:-1}
export DSXA_GPU_SAMPLE=${DSXA_GPU_SAMPLE:-1}
export DSXA_PROB_DRAFT=${DSXA_PROB_DRAFT:-1}
export DSXS_MIN=${DSXS_MIN:-0}
export DSXA_DRAFT_LIMIT=${DSXA_DRAFT_LIMIT:-5}
export DSXA_LOOKUP_LIMIT=${DSXA_LOOKUP_LIMIT:-6}
export DSXA_LOOKUP_CONTEXT=${DSXA_LOOKUP_CONTEXT:-5}
export DSXA_VERIFY_ROWS=${DSXA_VERIFY_ROWS:-31}
export DSXR_CHUNK=${DSXR_CHUNK:-512}
export DSXR_IDLE_CHUNK=${DSXR_IDLE_CHUNK:-2048}
export DSXR_DECODE_ROUNDS=${DSXR_DECODE_ROUNDS:-4}
export DSXA_TUNING=${DSXA_TUNING:-1}
export DSXA_BGRAPH=${DSXA_BGRAPH:-0}
export DSXR_SOLO=${DSXR_SOLO:-0}
test -f "$DSXA_WEIGHTS"
test -f "$DSX_MODEL_DIR/config.json"
test -f "$DSX_RANK"
exec "${DSXA_BIN:-$DSX_ROOT/target/release/dsx-amd}" serve \
  --model "$DSX_MODEL_DIR" --rank "$DSX_RANK" \
  --slots "${DSX_SLOTS:-8}" --batch-ctx 262144 --ctx 262200 \
  --host 127.0.0.1 --port "${DSX_PORT:-8090}" --admin-port "${DSX_ADMIN_PORT:-8091}" \
  --temperature "${DSX_TEMPERATURE:-0.1}" --top-p 1.0
