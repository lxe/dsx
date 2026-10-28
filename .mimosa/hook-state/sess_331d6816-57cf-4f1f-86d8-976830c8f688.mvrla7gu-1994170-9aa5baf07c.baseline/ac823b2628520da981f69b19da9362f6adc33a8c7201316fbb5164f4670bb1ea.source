#!/bin/bash
# network canary, run within 60 s of ssh: prints CANARY-OK <MB/s> or CANARY-BAD <reason>
timeout 25 pip download -q --no-deps -d /tmp/cn safetensors >/dev/null 2>&1 || { echo "CANARY-BAD pip"; exit; }
timeout 10 curl -sfI https://huggingface.co >/dev/null || { echo "CANARY-BAD hf-head"; exit; }
U=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model-00030-of-00048.safetensors
mkdir -p /tmp/cn; s=$(date +%s.%N)
for i in 0 1 2 3 4 5 6 7; do a=$((i * 134217728)); timeout 25 curl -sfL -r $a-$((a + 134217727)) -o /tmp/cn/p$i "$U" & done; wait
e=$(date +%s.%N); b=$(cat /tmp/cn/p* 2>/dev/null | wc -c); rm -rf /tmp/cn
r=$(python3 -c "print(int($b / 1e6 / ($e - $s)))")
[ "$b" -ge 1000000000 ] && [ "$r" -ge 600 ] && echo "CANARY-OK $r" || echo "CANARY-BAD rate $r MB/s ($b bytes)"
