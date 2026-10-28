"""Server-side turn benchmark, as an agent sends them: a conversation that grows by appending messages. Request 0: a
user message of ~3000 tokens of held-out text (8 tokens out); each next request = the previous messages + the
assistant's actual reply + a new user message of ~60 or ~300 tokens. The server reuses the cached prefix. Prints
prefill_ms / prompt / cached tokens per request from the server log."""
import json, re, sys, time, urllib.request
log = sys.argv[1]
text = open('/home/lxe/ai/ds41/eval/heldout.txt').read()
msgs = []
def ask(content):
    msgs.append({"role": "user", "content": content})
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": 8, "temperature": 0, "thinking": {"type": "disabled"}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body, {"content-type": "application/json"}), timeout=600).read())
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
    time.sleep(0.5)
    L = [x for x in (re.sub(r"\x1b\[[0-9;]*m", "", l) for l in open(log, errors="ignore")) if "engine: done" in x]
    kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', L[-1]))
    return int(kv["prompt_tokens"]), int(kv["cached_prompt_tokens"]), int(kv["prefill_ms"])
pos = 12000
p, c, ms = ask(text[:pos]); print(f"base: prompt {p}, cached {c}, prefill {ms} ms")
res = {}
for i, add in enumerate([240, 1200, 240, 1200, 240, 1200, 240, 1200]):
    p, c, ms = ask(text[pos:pos + add]); pos += add
    print(f"turn {i}: +{p - c} new tokens on {c} cached: prefill {ms} ms")
    res.setdefault(add, []).append(ms)
print("means: " + ", ".join(f"~{a // 4} tokens {sum(v) / len(v):.0f} ms" for a, v in res.items()))
