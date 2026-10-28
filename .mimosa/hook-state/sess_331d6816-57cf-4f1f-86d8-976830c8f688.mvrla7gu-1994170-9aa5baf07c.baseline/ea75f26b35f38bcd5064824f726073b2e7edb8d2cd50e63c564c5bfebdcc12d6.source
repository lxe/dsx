#!/bin/bash
# Start the dsx API server on .102 (OpenAI /v1/chat/completions, /v1/responses, Anthropic /v1/messages) under the
# memory cap, with experts placed by hotness: GPU1 and GPU0 hold the hottest, the RAM arena the rest.
# Usage: ./dsx-serve.sh [model.gguf] [port]
set -e
cd ~/ai/ds41
MODEL=${1:-hf-v2/DeepSeek-V4.1-Flash-1017-dsx-2bit-00001-of-00004.gguf}
PORT=${2:-8090}
sudo -n nvidia-smi -i 0 -pl 210 >/dev/null 2>&1 || true   # GPU0 is cooling-limited (see watchdog notes)
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog ~/ai/ds41/watchdog.sh
exec ~/ai/ds41/guard.sh serve 118 ./engine/target/release/dsx-server \
  --tokenizer fp8-1017/tokenizer.json --model "$MODEL" --engram fp8-1017 \
  --host 0.0.0.0 --port "$PORT" --ctx 131072 --chunk 4096 \
  --arena-gb ${ARENA_GB:-110} --rank calib/rank.json
