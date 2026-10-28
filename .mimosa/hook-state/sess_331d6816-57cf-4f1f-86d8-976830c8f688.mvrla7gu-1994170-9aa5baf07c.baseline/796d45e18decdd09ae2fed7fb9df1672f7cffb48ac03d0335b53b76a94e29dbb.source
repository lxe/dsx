#!/usr/bin/env python3
"""Overthinking probe: S01 / S02 (Pocket Model v26 rows) against an OpenAI-compatible server, N runs each.

Per run: completion tokens, wall time, boxed answer, correct or not, and the count of "Wait" in the reasoning and
answer. Usage: overthink_probe.py URL STYLE OUT.jsonl [runs] [cases]
  STYLE: dsx (reasoning_effort "low") or llama (chat_template_kwargs reasoning_effort 50, the template's "low").
"""
import json, re, sys, time, urllib.request

CASES = {
    "S01": ("Determine the greatest possible value of \\(\\sum_{i=1}^{10} \\cos(3x_i)\\) for real numbers $x_1,x_2,\\dots,x_{10}$ "
            "satisfying \\(\\sum_{i=1}^{10} \\cos(x_i) = 0\\).\n\nSolve the problem. Give the final answer as a single number inside \\boxed{}.",
            "480/49"),
    "S02": ("We consider positive integers $n$ having at least six positive divisors. Let the positive divisors of $n$ be arranged in a "
            "sequence $(d_i)_{1\\le i\\le k}$ with $$1=d_1<d_2<\\dots <d_k=n\\quad (k\\ge 6).$$\nFind all positive integers $n$ such "
            "that $$n=d_5^2+d_6^2.$$\n\nSolve the problem. Give the final answer as a single number inside \\boxed{}.",
            "500"),
}
SYSTEM = ("You are a careful expert assistant. When tools are offered, call the ones the request needs (several at once if it asks "
          "for several things), with arguments taken from the request; if no offered tool fits, reply briefly in words and call "
          "none. Otherwise answer in exactly the format the question asks for.")


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7 : j - 1]


def norm(a):
    if a is None:
        return None
    a = a.replace("\\dfrac", "\\frac").replace(" ", "")
    m = re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", a)
    return f"{m.group(1)}/{m.group(2)}" if m else a


def main():
    url, style, out = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    cases = sys.argv[5].split(",") if len(sys.argv) > 5 else list(CASES)
    for case in cases:
        prompt, expected = CASES[case]
        for r in range(runs):
            body = {"model": "deepseek-v4.1-flash", "temperature": 1.0, "top_p": 1.0, "max_tokens": 128000,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            if style == "llama":
                body["chat_template_kwargs"] = {"reasoning_effort": 50}
                body["top_k"] = 0
                body["min_p"] = 0.0
            else:
                body["reasoning_effort"] = "low"
            req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
            t0 = time.time()
            d = json.load(urllib.request.urlopen(req, timeout=7200))
            wall = time.time() - t0
            msg = d["choices"][0]["message"]
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            content = msg.get("content") or ""
            ans = boxed(content) or boxed(reasoning)
            rec = {"case": case, "run": r, "completion_tokens": d.get("usage", {}).get("completion_tokens"), "wall_s": round(wall, 1),
                   "boxed": ans, "correct": norm(ans) == expected, "waits": len(re.findall(r"\bWait\b", reasoning + content)),
                   "finish": d["choices"][0].get("finish_reason"), "reasoning_chars": len(reasoning)}
            print(json.dumps(rec), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            with open(out.replace(".jsonl", f"-{case}-{r}.json"), "w") as f:
                json.dump(d, f)


if __name__ == "__main__":
    main()
