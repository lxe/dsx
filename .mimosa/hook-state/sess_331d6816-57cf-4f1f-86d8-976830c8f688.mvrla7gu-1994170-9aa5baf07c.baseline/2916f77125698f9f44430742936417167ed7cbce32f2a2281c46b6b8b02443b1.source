"""t16_fetch.py CKPT_DIR LUCEBOX_HELDOUT_TXT OUT_DIR: the full-model T16 run's extra calibration text and its second
eval set, from public Hugging Face datasets (parquet / JSON; HF_XET_HIGH_PERFORMANCE pulls).

OUT_DIR/calib_mix.jsonl ({"text", "domain"}, ~550K tokens, appended to calib3 by v41_calib.py prep-mix):
  wiki_en / wiki_de / wiki_fr / wiki_zh / wiki_ja  wikimedia/wikipedia 20231101 (~45K tokens each)
  code   bigcode/the-stack-smol-xl (python, javascript, typescript, java, c, c++, go, rust, shell, sql; ~150K)
  chat   HuggingFaceH4/ultrachat_200k train_sft, DeepSeek chat template (~175K)
OUT_DIR/eval2.i32 + eval2.json: 8 x 1024 tokens of text in neither calibration set (sequences start with BOS):
  3 agentic (nebius/SWE-agent-trajectories), 3 code (codeparrot/github-code-clean), 2 chat (teknium/OpenHermes-2.5).
Every document sharing a 50-character window with Lucebox's heldout.txt is dropped (whitespace-normalized, lowercase).
"""
import json, os, re, sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ckpt, heldout, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(ckpt)
BOS, EOS = tok.bos_token_id, tok.eos_token_id
norm = lambda t: re.sub(r"\s+", " ", t.lower())
H = norm(open(heldout).read())
SH = {H[i:i + 50] for i in range(len(H) - 49)}
dropped = 0


def clean(t):
    global dropped
    n = norm(t)
    if any(n[i:i + 50] in SH for i in range(0, max(len(n) - 49, 1), 25)):
        dropped += 1
        return False
    return True


def ntok(t):
    return len(tok.encode(t, add_special_tokens=False))


def dl(repo, path):
    return hf_hub_download(repo, path, repo_type="dataset")


def chat(turns, system=None):
    """[(role, text)] -> DeepSeek's template (non-thinking assistant turns)"""
    s = f"<｜System｜>{system}" if system else ""
    for role, text in turns:
        if role == "user":
            s += f"<｜User｜>{text}"
        else:
            s += f"<｜Assistant｜></think>{text}<｜end▁of▁sentence｜>"
    return s


mix = []
def take(domain, texts, budget):
    got = 0
    for t in texts:
        if not t or not clean(t):
            continue
        n = ntok(t)
        mix.append({"text": t, "domain": domain})
        got += n
        if got >= budget:
            break
    print(f"{domain}: {got} tokens", flush=True)


# --- calibration additions
for lang, budget in (("en", 45000), ("de", 45000), ("fr", 45000), ("zh", 45000), ("ja", 45000)):
    t = pq.read_table(dl("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-" + {"en": "00041", "de": "00020", "fr": "00017", "zh": "00006", "ja": "00015"}[lang] + ".parquet"), columns=["title", "text"])
    rows = t.to_pylist()
    take(f"wiki_{lang}", (f"{r['title']}\n\n{r['text']}"[:20000] for r in rows[::7]), budget)

code = []
for lang in ("python", "javascript", "typescript", "java", "c", "c++", "go", "rust", "shell", "sql"):
    try:
        p = dl("bigcode/the-stack-smol-xl", f"data/{lang}/data.json")
    except Exception as e:
        print("stack", lang, e); continue
    n = 0
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("content", "")
            if 300 < len(c) < 12000:
                code.append(c); n += 1
            if n >= 60:
                break
take("code", code, 150000)

uc = pq.read_table(dl("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"), columns=["messages"]).to_pylist()
take("chat", (chat([(m["role"], m["content"]) for m in r["messages"]]) for r in uc[::5]), 175000)

with open(os.path.join(out, "calib_mix.jsonl"), "w") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"calibration mix: {len(mix)} texts, {dropped} dropped by the heldout filter", flush=True)

# --- second eval set: 8 x 1024
def seqs_from(texts, n, L=1024):
    out_s = []
    cur = [BOS]
    for t in texts:
        if not t or not clean(t):
            continue
        special = t.startswith("<｜")
        ids = tok.encode(t, add_special_tokens=False) + ([] if special else [EOS])
        cur += ids[: L - len(cur)]
        if len(cur) == L:
            out_s.append(cur); cur = [BOS]
            if len(out_s) == n:
                return out_s
    raise SystemExit(f"eval: only {len(out_s)} of {n} sequences")


def swe_texts():
    t = pq.read_table(dl("nebius/SWE-agent-trajectories", "data/train-00007-of-00012.parquet"))
    cols = t.column_names
    print("SWE-agent columns:", cols, flush=True)
    col = next(c for c in ("trajectory", "messages", "conversation", "conversations") if c in cols)
    for r in t.column(col).to_pylist()[::13]:
        turns, system = [], None
        for m in r:
            role = (m.get("role") or m.get("from") or "").lower()
            text = m.get("content") or m.get("text") or m.get("value") or ""
            if isinstance(text, list):
                text = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            if role == "system":
                system = text
            elif role in ("assistant", "ai", "gpt"):
                turns.append(("assistant", text))
            else:
                turns.append(("user", text))
        yield chat(turns, system)


def code_texts():
    t = pq.read_table(dl("codeparrot/github-code-clean", "data/train-00500-of-00880.parquet"), columns=["code", "language"])
    seen = set()
    for r in t.to_pylist()[::97]:
        if 2000 < len(r["code"]) < 30000 and r["language"] not in seen:
            seen.add(r["language"])
            yield r["code"]


def hermes_texts():
    data = json.load(open(dl("teknium/OpenHermes-2.5", "openhermes2_5.json")))
    for r in data[::997]:
        turns = [("user" if m["from"] == "human" else "assistant", m["value"]) for m in r["conversations"] if m["from"] in ("human", "gpt")]
        sysm = next((m["value"] for m in r["conversations"] if m["from"] == "system"), None)
        if len(turns) >= 2:
            yield chat(turns, sysm)


ev = seqs_from(swe_texts(), 3) + seqs_from(code_texts(), 3) + seqs_from(hermes_texts(), 2)
np.array(ev, dtype=np.int32).astype("<i4").tofile(os.path.join(out, "eval2.i32"))
json.dump({"rows": ["agentic"] * 3 + ["code"] * 3 + ["chat"] * 2, "len": 1024,
           "sources": ["nebius/SWE-agent-trajectories train-00007", "codeparrot/github-code-clean train-00500",
                       "teknium/OpenHermes-2.5"], "heldout_filter_dropped": dropped}, open(os.path.join(out, "eval2.json"), "w"))
print("eval2: 8 x 1024 written", flush=True)
