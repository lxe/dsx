"""sessions_render.py IN.jsonl OUT.jsonl [--budget TOKENS] [--max-tokens 2040] [--per-session 3] [--domain agent_lxe]:
scrubbed agent sessions (sessions_extract.py) -> DeepSeek V4.1 renders for v41_calib.py prep-mix ({"text", "domain"}).

prep-mix keeps only the first 2047 tokens of each text, so every render is one window that fits a 2048-token row:
the agent system prompt with schemas for the tools the window calls, then messages from a start point to the latest
assistant turn that still fits. Start points are the session's user turns; long tool loops with few user turns also
start at assistant turns, behind a "Continue." user turn. Up to --per-session windows per session, spread over its
start points; sessions are taken round-robin over projects until --budget tokens. Thinking mode, as an agent client
sends it. Needs the tokenizer (V41_MODEL)."""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402
import dsv41_encoding as E  # noqa: E402

MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/ai/ds41/fp8-1017"))


def window(msgs, i, k, tok, max_tokens):
    """messages i..k-1 rendered, with a "Continue." user turn when the window does not start at a user turn"""
    ms = msgs[i:k]
    if ms[0]["role"] != "user":
        ms = [{"role": "user", "content": "Continue."}] + ms
    ids = {tc["id"] for m in ms if m["role"] == "assistant" for tc in m.get("tool_calls", [])}
    ms = [m for m in ms if m["role"] != "tool" or m["tool_call_id"] in ids]
    calls = [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for m in ms if m["role"] == "assistant"
             for tc in m.get("tool_calls", [])]
    sysm = {"role": "system", "content": B.SYSTEM}
    if calls:
        sysm["tools"] = B.synth_tools(calls)
    t = E.encode_messages([sysm] + ms, thinking_mode="thinking")
    return t, len(tok.encode(t, add_special_tokens=False))


def best_window(msgs, i, tok, max_tokens):
    ends = [k + 1 for k in range(i, len(msgs)) if msgs[k]["role"] == "assistant"]
    lo, hi, best = 0, len(ends) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            t, n = window(msgs, i, ends[mid], tok, max_tokens)
        except Exception:
            hi = mid - 1
            continue
        if n <= max_tokens:
            best, lo = (t, n), mid + 1
        else:
            hi = mid - 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--budget", type=int, default=250_000)
    ap.add_argument("--max-tokens", type=int, default=2040)
    ap.add_argument("--per-session", type=int, default=3)
    ap.add_argument("--domain", default="agent_lxe")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    by_project = defaultdict(list)
    for line in open(a.inp):
        s = json.loads(line)
        by_project[s["project"]].append(s)
    for v in by_project.values():
        v.sort(key=lambda s: -len(s["msgs"]))
    order = []
    projects = sorted(by_project, key=lambda p: -len(by_project[p]))
    while any(by_project[p] for p in projects):
        for p in projects:
            if by_project[p]:
                order.append(by_project[p].pop(0))
    total, n_out, kept = 0, 0, defaultdict(int)
    with open(a.out, "w") as fh:
        for s in order:
            msgs = s["msgs"]
            starts = [i for i, m in enumerate(msgs) if m["role"] == "user"]
            if len(starts) < a.per_session:   # long tool loops: also start at assistant turns
                starts = sorted(set(starts) | {i for i, m in enumerate(msgs) if m["role"] == "assistant"})
            if not starts:
                continue
            pick = sorted({starts[round(j * (len(starts) - 1) / max(1, a.per_session - 1))] for j in range(a.per_session)})
            for i in pick:
                w = best_window(msgs, i, tok, a.max_tokens)
                if not w or w[1] < a.max_tokens // 3:   # too short to be worth a row
                    continue
                fh.write(json.dumps({"text": w[0], "domain": a.domain, "project": s["project"], "file": s["file"],
                                     "tokens": w[1]}, ensure_ascii=False) + "\n")
                total += w[1]; n_out += 1; kept[s["project"]] += 1
            if total >= a.budget:
                break
    print(f"{n_out} windows, {total} tokens from {len(kept)} projects: {sorted(kept.items(), key=lambda kv: -kv[1])[:15]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
