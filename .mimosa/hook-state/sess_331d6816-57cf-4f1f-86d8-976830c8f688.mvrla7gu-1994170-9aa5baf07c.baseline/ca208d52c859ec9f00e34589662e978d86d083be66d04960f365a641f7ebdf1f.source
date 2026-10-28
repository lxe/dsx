#!/bin/bash
# Server-like turns: prompt3000 prefilled in chunks of 60 (or 300); chunks after 2400 tokens are 60- (300-) token turns
# on a warm process with 2.4-3K cached tokens. CPU share on vs off, alternating; mean of the late chunks.
cd ~/ai/ds41
[ -e GPU_LOCK ] && { echo "locked: $(cat GPU_LOCK)"; exit 1; }
echo "DSX Dev 1: chunked-turn prefill, serve stopped (~20 min) $(date)" > GPU_LOCK
trap "rm -f ~/ai/ds41/GPU_LOCK" EXIT
pid=$(pgrep -x dsx-server); [ -n "$pid" ] && kill $pid; while [ -n "$pid" ] && kill -0 $pid 2>/dev/null; do sleep 1; done
tmux kill-session -t serve 2>/dev/null; sleep 3
B=~/src/dsx-dev/target/release/dsx; M="--model hf-t16one/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf"
for ch in 60 300; do for v in 0 1 1 0; do systemctl --user reset-failed ds41-w8.scope 2>/dev/null; sleep 2
  DSX_PREFILL_CPU=$v ./guard.sh w8 118 $B gen $M --ids @prompt3000.ids -n 2 --ctx 32768 --chunk $ch --main-gb=-1.5 --aux-gb=-1.0 --arena-gb 110 --rank calib/rank.json 2>&1 | grep -E "^prefill chunk" > /tmp/ch.txt
  echo "chunk $ch cpu=$v: $(python3 -c "
import re
L=[(int(a),int(b),float(t)) for a,b,t in re.findall(r'chunk (\d+)\.\.(\d+): ([0-9.]+) s',open('/tmp/ch.txt').read())]
late=[t for a,b,t in L if a>=2400]
print(f'{len(L)} chunks, first {L[0][2]:.2f} s, chunks at 2400+: mean {sum(late)/len(late):.3f} s (n={len(late)}), min {min(late):.3f}')
")"
done; done
systemctl --user reset-failed ds41-serve.scope 2>/dev/null
tmux new-session -d -s serve -x 200 -y 50 "~/ai/ds41/dsx-serve.sh 2>&1 | tee ~/ai/ds41/serve.out"
echo "WIN8-DONE $(date)"
