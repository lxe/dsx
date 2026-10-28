"""collect.py - reasoning traces from the full-precision model for calibration.

Asks DeepSeek-V4.1-Flash-1017 (OpenRouter) Omni-MATH (difficulty 4-6.9) and SuperGPQA ("middle") questions with
thinking on, and writes one JSON line per answer: messages, reasoning, answer, usage. Rows the dsx evals draw from
(Omni-MATH difficulty >= 7, SuperGPQA "hard") are left out.

    OPENROUTER_API_KEY=... python3 collect.py OMNI.jsonl SUPERGPQA.jsonl OUT.jsonl [--math 96] [--mc 64] [--jobs 16]
"""
import argparse
import json
import os
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SYSTEM = "You are a careful expert assistant. Answer in exactly the format the question asks for."


def rows(omni, sgpqa, n_math, n_mc, seed):
    rng = random.Random(seed)
    m = [json.loads(l) for l in open(omni)]
    m = [r for r in m if 4 <= float(r.get("difficulty", 0)) < 7]
    rng.shuffle(m)
    out = [{"id": f"omni-{i}", "kind": "math", "user": r["problem"] + "\n\nSolve the problem. Give the final answer inside \\boxed{}."}
           for i, r in enumerate(m[:n_math])]
    q = [json.loads(l) for l in open(sgpqa)]
    q = [r for r in q if r.get("difficulty") == "middle"]
    rng.shuffle(q)
    for i, r in enumerate(q[:n_mc]):
        opts = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(r["options"]))
        out.append({"id": f"sgpqa-{i}", "kind": "mc", "user": f"{r['question']}\n\n{opts}\n\nThink it through, then end with a line 'Answer: X'."})
    for i, r in enumerate(out):
        r["effort"] = "low" if i % 2 else "high"   # dsx serves both
    return out


def ask(r, key):
    body = {"model": "deepseek/deepseek-v4.1-flash-1017", "max_tokens": 16384, "reasoning": {"effort": r["effort"]},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["user"]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json", "authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    m = d["choices"][0]["message"]
    return {**r, "system": SYSTEM, "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "answer": m.get("content") or "", "usage": d.get("usage"), "provider": d.get("provider"),
            "finish": d["choices"][0].get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("omni"); ap.add_argument("sgpqa"); ap.add_argument("out")
    ap.add_argument("--math", type=int, default=96); ap.add_argument("--mc", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--seed", type=int, default=1017)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [r for r in rows(a.omni, a.sgpqa, a.math, a.mc, a.seed) if r["id"] not in done]
    lock = threading.Lock()
    f = open(a.out, "a")

    def one(r):
        try:
            res = ask(r, key)
        except Exception as e:  # noqa: BLE001
            print(f"{r['id']}: {e}", file=sys.stderr)
            return
        with lock:
            f.write(json.dumps(res) + "\n"); f.flush()
            u = res.get("usage") or {}
            print(f"{r['id']} {r['effort']} {u.get('completion_tokens')} tok {res['finish']}", file=sys.stderr)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
