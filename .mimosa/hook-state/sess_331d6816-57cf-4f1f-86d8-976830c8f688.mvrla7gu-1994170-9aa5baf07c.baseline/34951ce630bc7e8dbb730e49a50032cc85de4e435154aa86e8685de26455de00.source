"""sessions_extract.py OUT.jsonl [ROOT ...]: Claude Code session transcripts (default ~/.claude/projects) -> scrubbed
structured agent sessions for calibration, run on the machine that holds the transcripts so only scrubbed text leaves it.

Per session (build_corpus.session_messages: user / assistant thinking + text + tool calls / tool results clipped to 6000
chars; sidechain and meta records skipped; >= 8 tool calls): {"file": sha1[:12], "project", "msgs", "calls"}.
Every string goes through build_corpus's scrub list plus the patterns below (presigned URLs, env-style credentials,
long mixed-case tokens, private hostnames of storage accounts). sessions_scan.py re-checks the output independently.
Rendering to DeepSeek V4.1 text and token budgeting happen elsewhere (sessions_render.py, with the tokenizer)."""
import glob, hashlib, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as B  # noqa: E402

B.SECRETS[:0] = [
    # presigned / signed URLs: drop the whole query string
    (re.compile(r"(https?://[^\s\"'<>?]+)\?[^\s\"'<>]*(?:X-Amz-|Signature=|sig=|token=|Expires=)[^\s\"'<>]*"), r"\1?REDACTED"),
    # object-storage account hosts (R2 / S3 endpoints carry the account id)
    (re.compile(r"\b[0-9a-f]{32}\.r2\.cloudflarestorage\.com"), "ACCOUNT.r2.cloudflarestorage.com"),
    # env / config style credentials: NAME_WITH_KEY_OR_SECRET=value
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH)[A-Z0-9_]*\s*[=:]\s*[\"']?)[^\s\"',}]{8,}"), r"\1REDACTED"),
    # RunPod / Cloudflare / OpenRouter / Anthropic style keys not covered below
    (re.compile(r"\brpa_[A-Za-z0-9]{20,}"), "rpa_REDACTED"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"), "sk_live_REDACTED"),
    # user:password@host in URLs
    (re.compile(r"(://)[^\s/:@\"']+:[^\s/@\"']+@"), r"\1REDACTED@"),   # any scheme, or none in front
]
# long tokens with upper case, lower case and digits (API keys, JWTs); plain hex (hashes, ids) is left alone
LONGTOK = re.compile(r"(?<![A-Za-z0-9_\-])(?=[A-Za-z0-9_\-]*[A-Z])(?=[A-Za-z0-9_\-]*[a-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{40,}(?![A-Za-z0-9_\-])")
B.SECRETS.append((LONGTOK, "TOKEN_REDACTED"))
JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")
B.SECRETS.insert(0, (JWT, "JWT_REDACTED"))


def main():
    out = sys.argv[1]
    roots = sys.argv[2:] or [os.path.expanduser("~/.claude/projects")]
    files = sorted(f for r in roots for f in glob.glob(os.path.join(r, "**", "*.jsonl"), recursive=True)
                   if os.path.getsize(f) > 50_000)
    n, stats = 0, Counter()
    with open(out + ".tmp", "w") as fh:
        for f in files:
            try:
                got = B.session_messages(f)
            except Exception as e:
                stats[f"error {type(e).__name__}"] += 1
                continue
            if not got:
                stats["not an agent session"] += 1
                continue
            msgs, calls = got
            cwd = ""
            for line in open(f, errors="ignore"):
                if '"cwd"' in line:
                    try:
                        cwd = json.loads(line).get("cwd", ""); break
                    except Exception:
                        pass
            project = B.scrub(os.path.basename(cwd.rstrip("/")) if cwd else "?")
            fh.write(json.dumps({"file": hashlib.sha1(f.encode()).hexdigest()[:12], "project": project, "msgs": msgs,
                                 "calls": [[c, B.scrub_obj(a)] for c, a in calls]}, ensure_ascii=False) + "\n")
            n += 1
            stats["kept"] += 1
    os.replace(out + ".tmp", out)
    print(f"{len(files)} files -> {n} sessions; {dict(stats)}; {os.path.getsize(out) / 1e6:.0f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
