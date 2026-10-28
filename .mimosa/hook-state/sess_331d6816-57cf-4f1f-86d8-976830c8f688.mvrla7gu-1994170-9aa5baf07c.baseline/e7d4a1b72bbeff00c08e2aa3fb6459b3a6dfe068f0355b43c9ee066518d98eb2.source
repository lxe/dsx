"""turns.py split in phases for profiling: `base` sends the ~3000-token first message, `turn N` sends a ~N-char new
user message on top of the saved conversation (/tmp/conv.json). Prints prefill_ms etc. from the server log."""
import json, os, re, sys, time, urllib.request
log, phase = sys.argv[1], sys.argv[2]
text = open('/home/lxe/ai/ds41/eval/heldout.txt').read()
st = json.load(open('/tmp/conv.json')) if phase != 'base' and os.path.exists('/tmp/conv.json') else {"msgs": [], "pos": 0}
content = text[:12000] if phase == 'base' else text[st["pos"]:st["pos"] + int(sys.argv[3])]
st["pos"] = 12000 if phase == 'base' else st["pos"] + int(sys.argv[3])
st["msgs"].append({"role": "user", "content": content})
body = json.dumps({"model": "x", "messages": st["msgs"], "max_tokens": 8, "temperature": 0, "thinking": {"type": "disabled"}}).encode()
r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body, {"content-type": "application/json"}), timeout=600).read())
st["msgs"].append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
json.dump(st, open('/tmp/conv.json', 'w'))
time.sleep(0.5)
L = [x for x in (re.sub(r"\x1b\[[0-9;]*m", "", l) for l in open(log, errors="ignore")) if "engine: done" in x]
kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', L[-1]))
print(f"{phase}: prompt {kv['prompt_tokens']}, cached {kv['cached_prompt_tokens']}, prefill {kv['prefill_ms']} ms, decode {kv.get('decode_ms')} ms")
