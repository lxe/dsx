"""sessions_scan.py FILE.jsonl: independent leak scan of scrubbed calibration text before it leaves the machine.
Counts matches of credential-shaped patterns (none should remain) and prints a few redacted-context samples of any
hit; exits 1 if any hard pattern matches."""
import re, sys
from collections import Counter

HARD = {
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "hf_token": r"hf_[A-Za-z0-9]{30,}",
    "openai_anthropic": r"sk-(?:ant-|or-|proj-)?[A-Za-z0-9_\-]{20,}",
    "github": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}",
    "slack": r"xox[abpr]-[A-Za-z0-9\-]{10,}",
    "google": r"AIza[0-9A-Za-z_\-]{30,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY",
    "amz_signed_url": r"X-Amz-(?:Signature|Credential)=[A-Za-z0-9%]{16,}",
    "r2_account_host": r"[0-9a-f]{32}\.r2\.cloudflarestorage",
    "jwt": r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.",
    "url_password": r"://(?!REDACTED@)[^\s/:@\"'\[\]()]+:[^\s/@\"'\[\]()]{3,}@",
    "runpod": r"\brpa_[A-Za-z0-9]{20,}",
}
SOFT = {
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    "assignment_secret": r"(?i)(?:secret|password|passwd|api_?key|token)\s*[=:]\s*[\"']?(?!REDACTED)[^\s\"']{8,}",
}
text = open(sys.argv[1], errors="ignore").read()
hits = Counter()
for name, rx in {**HARD, **SOFT}.items():
    ms = list(re.finditer(rx, text))
    hits[name] = len(ms)
    for m in ms[:3]:
        s = text[max(0, m.start() - 40): m.end() + 10].replace("\n", " ")
        print(f"  {name}: ...{s[:40]}[{len(m.group())} chars]...", file=sys.stderr)
emails = Counter(m.group() for m in re.finditer(SOFT["email"], text))
print("hits:", dict(hits), "| emails:", emails.most_common(8))
sys.exit(1 if any(hits[k] for k in HARD) else 0)
