"""calib5_public.py OUT.jsonl: the public part of calibration set v5, rendered for v41_calib.py prep-mix ({"text", "domain"}).
Every text fits one 2048-token row (prep-mix keeps the first 2047 tokens). Sources (Hugging Face, downloaded to
./hfcache with HF_XET_HIGH_PERFORMANCE):
  agent_swe  SWE-bench/SWE-smith-trajectories (Claude 3.7 agent runs): text actions -> DSML bash tool calls, OBSERVATION
             turns -> tool results; windows as sessions_render.py (up to 2 per trajectory)
  think_r1   a-m-team/AM-DeepSeek-R1-0528-Distilled (R1-0528's reasoning; code / math / science / if / other), thinking
             mode, whole samples that fit
  wiki       wikimedia/wikipedia 20231101, random articles from a middle shard of 10 languages
  aya        CohereLabs/aya_dataset train, chat mode, many languages
  code, chat every other text of calib4's the-stack-smol-xl / UltraChat mix (t16data/calib_mix.jsonl)
Every document sharing a 50-character window with Lucebox's heldout.txt or eval2 is dropped."""
import glob, json, os, random, re, subprocess, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402
import dsv41_encoding as E  # noqa: E402
import sessions_render as SR  # noqa: E402

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/ai/ds41/fp8-1017"))
HELDOUT = os.path.expanduser("~/ai/ds41/eval/heldout.txt")
EVAL2 = os.path.expanduser("~/src/t16q/t16data/eval2.i32")
MIX4 = os.path.expanduser("~/src/t16q/t16data/calib_mix.jsonl")
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--scale", type=float, default=1.0, help="budget multiplier")
ap.add_argument("--prefix", default="", help="domain prefix, e.g. pool_ for the coverage candidates")
ap.add_argument("--exclude", action="append", default=[], help="jsonl files whose texts must not repeat")
ap.add_argument("--langs", default="en,de,fr,es,ru,zh,ja,ko,pt,ar")
ap.add_argument("--no-mix4", action="store_true")
A = ap.parse_args()
BUDGET = {k: int(v * A.scale) for k, v in {"agent_swe": 120_000, "think_r1": 150_000, "wiki": 80_000, "aya": 25_000}.items()}
MAXT = 2040
CACHE = "hfcache"
rng = random.Random(A.seed)
import hashlib
SEEN = set()
for f in A.exclude:
    for line in open(f):
        SEEN.add(hashlib.sha1(json.loads(line)["text"].encode()).hexdigest())
tok = AutoTokenizer.from_pretrained(MODEL)
ntok = lambda t: len(tok.encode(t, add_special_tokens=False))
norm = lambda t: re.sub(r"\s+", " ", t.lower())
H = norm(open(HELDOUT).read()) + " " + norm(tok.decode(np.fromfile(EVAL2, dtype=np.int32).tolist()))
SH = {H[i:i + 50] for i in range(len(H) - 49)}
dropped = Counter()


def clean(t, dom):
    h = hashlib.sha1(t.encode()).hexdigest()
    if h in SEEN:
        dropped[dom + " (repeat)"] += 1
        return False
    SEEN.add(h)
    n = norm(t)
    if any(n[i:i + 50] in SH for i in range(0, max(1, len(n) - 49), 25)):
        dropped[dom] += 1
        return False
    return True


def dl(repo, f):
    """plain HTTPS (a xet download of a Wikipedia shard stalled on .102), cached under CACHE"""
    p = os.path.join(CACHE, repo.replace("/", "__"), f)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", p + ".part", f"https://huggingface.co/datasets/{repo}/resolve/main/{f}"], check=True)
        os.replace(p + ".part", p)
    return p


def agent_swe(out):
    f = f"data/ticks-0000{A.seed % 8}-of-00008.parquet"
    p = glob.glob(f"{CACHE}/{f}") or [dl("SWE-bench/SWE-smith-trajectories", f)]
    rows = pq.read_table(p[0], columns=["messages"]).to_pylist()
    rng.shuffle(rows)
    total = 0
    for r in rows:
        ms = r["messages"] if isinstance(r["messages"], list) else json.loads(r["messages"])
        msgs, pending, n = [], None, 0
        for m in ms:
            c = m.get("content") or ""
            if m["role"] == "user":
                if c.startswith("OBSERVATION:") and pending:
                    msgs.append({"role": "tool", "tool_call_id": pending, "content": B.clip(c[len("OBSERVATION:"):].strip())})
                    pending = None
                else:
                    msgs.append({"role": "user", "content": c})
            elif m["role"] == "assistant":
                blocks = list(re.finditer(r"```[a-z]*\n(.*?)```", c, re.S))
                if blocks:
                    cid = f"call_{n}"; n += 1
                    msgs.append({"role": "assistant", "content": c[:blocks[-1].start()].strip(),
                                 "tool_calls": [B.tool_call(cid, "bash", {"command": blocks[-1].group(1).strip()})]})
                    pending = cid
                else:
                    msgs.append({"role": "assistant", "content": c}); pending = None
        starts = [i for i, m in enumerate(msgs) if m["role"] in ("user", "assistant")]
        if not starts or msgs[0]["role"] != "user":
            continue
        for i in sorted({0, starts[len(starts) // 2]}):
            w = SR.best_window(msgs, i, tok, MAXT)
            if w and w[1] >= MAXT // 3 and clean(w[0], "agent_swe"):
                out.append({"text": w[0], "domain": "agent_swe"}); total += w[1]
        if total >= BUDGET["agent_swe"]:
            break
    return total


def think_r1(out):
    base = "https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-0528-Distilled/resolve/main"
    per = BUDGET["think_r1"] // 5
    total = 0
    for cat in ("code", "math", "science", "if", "other"):
        f = f"{CACHE}/{cat}.jsonl"
        if not os.path.exists(f):   # the first ~60 MB is plenty (whole files are up to 20 GB)
            subprocess.run(["curl", "-sfL", "-r", "0-60000000", "-o", f, f"{base}/{cat}.jsonl"], check=True)
        lines = open(f, errors="ignore").read().split("\n")[:-1]
        rng.shuffle(lines)
        got = 0
        for line in lines:
            try:
                c = json.loads(line)["conversations"]
            except Exception:
                continue
            if len(c) != 2:
                continue
            v = c[1]["value"]
            th = re.search(r"<think>(.*?)</think>", v, re.S)
            an = re.search(r"<answer>(.*?)(?:</answer>|$)", v, re.S)
            if not th or not an:
                continue
            msgs = [{"role": "user", "content": c[0]["value"]},
                    {"role": "assistant", "content": an.group(1).strip(), "reasoning_content": th.group(1).strip()}]
            t = E.encode_messages(msgs, thinking_mode="thinking")
            n = ntok(t)
            if n > MAXT or n < 300 or not clean(t, "think_r1"):
                continue
            out.append({"text": t, "domain": "think_r1"}); got += n
            if got >= per:
                break
        total += got
    return total


def wiki(out):
    api = HfApi()
    files = api.list_repo_files("wikimedia/wikipedia", repo_type="dataset")
    langs = A.langs.split(",")
    per, total = BUDGET["wiki"] // len(langs), 0
    for lang in langs:
        fs = sorted(f for f in files if f.startswith(f"20231101.{lang}/") and f.endswith(".parquet"))
        t = pq.read_table(dl("wikimedia/wikipedia", fs[(len(fs) // 2 + A.seed) % len(fs)]), columns=["title", "text"]).to_pylist()
        rng.shuffle(t)
        got = 0
        for r in t:
            if len(r["text"]) < 1500:
                continue
            text = (r["title"] + "\n\n" + r["text"])[:6000]
            ids = tok.encode(text, add_special_tokens=False)[:MAXT - 2]
            if not clean(text, "wiki"):
                continue
            out.append({"text": tok.decode(ids), "domain": f"wiki_{lang}"}); got += len(ids)
            if got >= per:
                break
        total += got
    return total


def aya(out):
    t = pq.read_table(dl("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet"), columns=["inputs", "targets", "language"]).to_pylist()
    rng.shuffle(t)
    total, langs = 0, Counter()
    for r in t:
        if langs[r["language"]] >= 40:   # spread over languages
            continue
        msgs = [{"role": "user", "content": r["inputs"]}, {"role": "assistant", "content": r["targets"]}]
        text = E.encode_messages(msgs, thinking_mode="chat")
        n = ntok(text)
        if n > MAXT or not clean(text, "aya"):
            continue
        out.append({"text": text, "domain": "aya"}); total += n; langs[r["language"]] += 1
        if total >= BUDGET["aya"]:
            break
    return total


def mix4(out):
    by = defaultdict(list)
    for line in open(MIX4):
        r = json.loads(line)
        if r["domain"] in ("code", "chat"):
            by[r["domain"]].append(r["text"])
    total = 0
    for dom, texts in by.items():
        for t in texts[::2]:
            out.append({"text": t, "domain": dom}); total += min(MAXT, ntok(t))
    return total


def main():
    out, stats = [], {}
    for name, fn in (("agent_swe", agent_swe), ("think_r1", think_r1), ("wiki", wiki), ("aya", aya)) + ((() if A.no_mix4 else (("mix4", mix4),))):
        n0 = len(out)
        stats[name] = (fn(out), len(out) - n0)
        print(f"{name}: {stats[name][0]} tokens, {stats[name][1]} texts", file=sys.stderr, flush=True)
    with open(A.out, "w") as fh:
        for r in out:
            r["domain"] = A.prefix + r["domain"]
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {sum(v[0] for v in stats.values())} tokens; dropped by the shingle filter {dict(dropped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
