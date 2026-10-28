#!/usr/bin/env python3
"""pod.py - rent and release one RunPod GPU pod for a quantization run.

  pod.py up STATE [--gpu ID] [--image IMAGE] [--disk GB] [--cloud C]  rent, wait for ssh, write STATE (json: id, ip, port, cost)
  pod.py down STATE                                           terminate the pod in STATE
  pod.py ssh STATE CMD...                                     run CMD on it (also: -put SRC... DEST, -get SRC DEST)

The API key comes from RUNPOD_API_KEY or the env file named by RUNPOD_ENV (environment or ~/.config/dsx/rent.env) and stays
on this machine; the pod gets this
machine's ~/.ssh/id_ed25519.pub.
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request


def _defaults():
    """KEY=VALUE lines from ~/.config/dsx/rent.env (R2_ACCOUNT, R2_BUCKET, R2_ENV, RUNPOD_ENV) as environment defaults."""
    p = os.path.expanduser("~/.config/dsx/rent.env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip('"'))


_defaults()

API = "https://rest.runpod.io/v1/pods"
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]


def key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    for line in open(os.path.expanduser(os.environ["RUNPOD_ENV"])):
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("RUNPOD_API_KEY not found")


def call(method, url, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": "dsx-quant/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def create(a, cloud):
    body = {"name": a.name, "imageName": a.image, "gpuTypeIds": [a.gpu], "gpuCount": a.count, "cloudType": cloud,
            "containerDiskInGb": a.disk, "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()}}  # the image starts sshd with it
    return call("POST", API, body)


def up(a):
    """Rent; a pod without an ssh address within --wait seconds is terminated and another one rented (up to --tries)."""
    for attempt in range(a.tries):
        cloud = a.cloud if a.cloud != "ALT" else ("SECURE" if attempt % 2 == 0 else "COMMUNITY")
        try:
            d = create(a, cloud)
        except urllib.error.HTTPError as e:
            print(f"{cloud}: {e.code} {e.read()[:200]!r}", flush=True)
            continue
        pid, t0 = d["id"], time.time()
        print(f"{pid} {a.count}x {a.gpu} ({cloud}) at ${d.get('costPerHr')}/h", flush=True)
        while time.time() - t0 < a.wait:
            d = call("GET", f"{API}/{pid}")
            ip, port = d.get("publicIp"), (d.get("portMappings") or {}).get("22")
            if ip and port and subprocess.run(["ssh", *SSH, "-p", str(port), f"root@{ip}", "true"], capture_output=True).returncode == 0:
                json.dump({"id": pid, "ip": ip, "port": port, "started": t0, "cost": d.get("costPerHr")}, open(a.state, "w"))
                print(f"ssh root@{ip} -p {port} ({time.time() - t0:.0f} s)")
                return
            time.sleep(5)
        call("DELETE", f"{API}/{pid}")
        print(f"{pid}: no ssh in {a.wait} s, terminated", flush=True)
    sys.exit(f"no pod with ssh after {a.tries} tries")


def down(a):
    st = json.load(open(a.state))
    call("DELETE", f"{API}/{st['id']}")
    h = (time.time() - st["started"]) / 3600
    print(f"terminated {st['id']} after {h:.2f} h (~${h * float(st.get('cost') or 0):.2f})")


def ssh(a):
    st = json.load(open(a.state))
    ip, port = st["ip"], str(st["port"])
    c = a.cmd
    if c and c[0] == "-put":
        os.execvp("scp", ["scp", "-q", *SSH, "-P", port, *c[1:-1], f"root@{ip}:{c[-1]}"])
    if c and c[0] == "-get":
        os.execvp("scp", ["scp", "-q", "-r", *SSH, "-P", port, f"root@{ip}:{c[1]}", c[2]])
    os.execvp("ssh", ["ssh", "-n", *SSH, "-p", port, f"root@{ip}", *c])


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd_", required=True)
    q = s.add_parser("up")
    q.add_argument("state")
    q.add_argument("--gpu", default="NVIDIA RTX PRO 6000 Blackwell Server Edition")
    q.add_argument("--count", type=int, default=1)
    q.add_argument("--image", default="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")
    q.add_argument("--disk", type=int, default=1000)
    q.add_argument("--name", default="dsx-quant")
    q.add_argument("--wait", type=int, default=120, help="seconds to ssh before a pod is replaced")
    q.add_argument("--tries", type=int, default=6)
    q.add_argument("--cloud", default="ALT", choices=["ALT", "SECURE", "COMMUNITY"], help="ALT: secure, then community")
    q = s.add_parser("down")
    q.add_argument("state")
    q = s.add_parser("ssh")
    q.add_argument("state")
    q.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    {"up": up, "down": down, "ssh": ssh}[a.cmd_](a)


if __name__ == "__main__":
    main()
