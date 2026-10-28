#!/usr/bin/env python3
"""mirror.py STATE POD_DIR R2_PREFIX - copy a directory from a rented pod to R2: this machine starts a multipart upload
per file and presigns its parts, the pod uploads them (r2up.py, files one after another, 16 parts at a time), this
machine completes each upload from the ETags the pod reports. Files already in R2 with the same size are skipped."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2

state, src, prefix = sys.argv[1:4]
pod = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pod.py"), "ssh", state]
listing = subprocess.run(pod + [f"cd {src} && find . -type f ! -name '*.ok' -printf '%s %P\\n'"], capture_output=True, text=True, check=True).stdout
have = dict((k, n) for k, n in r2.ls(prefix))
todo = []
for line in listing.splitlines():
    n, rel = line.split(" ", 1)
    key = f"{prefix}/{rel}"
    if have.get(key) == int(n):
        continue
    todo.append((rel, key, int(n)))
print(f"{len(todo)} files, {sum(t[2] for t in todo) / 1e9:.1f} GB to upload", flush=True)
tmp = tempfile.mkdtemp()
jobs = []
for i, (rel, key, n) in enumerate(todo):
    m = r2.mpu_start(key, n, part_mb=512, hours=24)
    json.dump(m, open(f"{tmp}/m{i}.json", "w"))
    jobs.append((i, rel, key, m["upload_id"]))
subprocess.run(pod + ["mkdir -p /work/mirror && rm -f /work/mirror/*"], check=True)
subprocess.run(pod + ["-put"] + [f"{tmp}/m{i}.json" for i, *_ in jobs] + ["/work/mirror/"], check=True)
script = "; ".join(f"python /work/rent/r2up.py {src}/{rel!r} /work/mirror/m{i}.json /work/mirror/e{i}.json" for i, rel, _, _ in jobs)
subprocess.run(pod + [script], check=True)
for i, rel, key, uid in jobs:
    subprocess.run(pod + ["-get", f"/work/mirror/e{i}.json", f"{tmp}/e{i}.json"], check=True)
    r2.mpu_done(key, uid, json.load(open(f"{tmp}/e{i}.json")))
print(f"mirrored {len(jobs)} files to {prefix}", flush=True)
