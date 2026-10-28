#!/usr/bin/env python3
"""r2.py - Cloudflare R2 objects for rented quantization pods, through presigned URLs only.

Configuration: R2_ACCOUNT, R2_BUCKET and credentials (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY, or an env file named by
R2_ENV with BULLET_STORAGE_* keys), from the environment or ~/.config/dsx/rent.env. Credentials stay on this machine; pods get URLs that expire. Large objects go up as multipart uploads: this machine starts the upload
and presigns every part, the pod PUTs the parts and reports their ETags, this machine completes it.

  r2.py get KEY [--hours H]                       presigned GET URL
  r2.py put KEY [--hours H]                       presigned PUT URL (single objects up to 5 GB)
  r2.py mpu-start KEY SIZE [--part-mb 512]        start a multipart upload: JSON {key, upload_id, part_size, urls}
  r2.py mpu-done KEY UPLOAD_ID ETAGS_JSON         complete it (ETAGS_JSON: list of the parts' ETags in order)
  r2.py ls PREFIX                                 keys and sizes under PREFIX
"""
import argparse, datetime, hashlib, hmac, json, os, sys, urllib.parse, urllib.request
import xml.etree.ElementTree as ET


def _defaults():
    """KEY=VALUE lines from ~/.config/dsx/rent.env (R2_ACCOUNT, R2_BUCKET, R2_ENV, RUNPOD_ENV) as environment defaults."""
    p = os.path.expanduser("~/.config/dsx/rent.env")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip('"'))


_defaults()

ACCOUNT = os.environ.get("R2_ACCOUNT", "")
HOST = f"{ACCOUNT}.r2.cloudflarestorage.com"
BUCKET = os.environ.get("R2_BUCKET", "")


def creds():
    if os.environ.get("R2_ACCESS_KEY_ID"):
        return os.environ["R2_ACCESS_KEY_ID"], os.environ["R2_SECRET_ACCESS_KEY"]
    env = {}
    for line in open(os.path.expanduser(os.environ["R2_ENV"])):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v.strip('"')
    return env["BULLET_STORAGE_ACCESS_KEY_ID"], env["BULLET_STORAGE_SECRET_ACCESS_KEY"]


def _key(sk, day):
    k = ("AWS4" + sk).encode()
    for part in (day, "auto", "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return k


def _qs(q):
    return "&".join(f"{urllib.parse.quote(k, safe='~')}={urllib.parse.quote(str(v), safe='~')}" for k, v in sorted(q.items()))


def _path(key):
    return "/" + BUCKET + "/" + urllib.parse.quote(key, safe="/~")


def presign(method, key, seconds=12 * 3600, query=None):
    """SigV4 query-string presigned URL (region auto)."""
    ak, sk = creds()
    now = datetime.datetime.now(datetime.timezone.utc)
    amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/auto/s3/aws4_request"
    q = dict(query or {})
    q.update({"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{ak}/{scope}", "X-Amz-Date": amz,
              "X-Amz-Expires": str(int(seconds)), "X-Amz-SignedHeaders": "host"})
    canonical = f"{method}\n{_path(key)}\n{_qs(q)}\nhost:{HOST}\n\nhost\nUNSIGNED-PAYLOAD"
    to_sign = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    sig = hmac.new(_key(sk, day), to_sign.encode(), hashlib.sha256).hexdigest()
    return f"https://{HOST}{_path(key)}?{_qs(q)}&X-Amz-Signature={sig}"


def call(method, key, query=None, body=b""):
    """A header-signed request from this machine (multipart start / complete, listing)."""
    ak, sk = creds()
    now = datetime.datetime.now(datetime.timezone.utc)
    amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/auto/s3/aws4_request"
    ph = hashlib.sha256(body).hexdigest()
    q = _qs(query or {})
    path = _path(key) if key is not None else "/" + BUCKET
    canonical = f"{method}\n{path}\n{q}\nhost:{HOST}\nx-amz-content-sha256:{ph}\nx-amz-date:{amz}\n\nhost;x-amz-content-sha256;x-amz-date\n{ph}"
    to_sign = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    sig = hmac.new(_key(sk, day), to_sign.encode(), hashlib.sha256).hexdigest()
    auth = f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={sig}"
    req = urllib.request.Request(f"https://{HOST}{path}" + (f"?{q}" if q else ""), data=body if method != "GET" else None,
                                 method=method, headers={"Authorization": auth, "x-amz-date": amz, "x-amz-content-sha256": ph})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _strip(tag):
    return tag.split("}", 1)[-1]


def mpu_start(key, size, part_mb=512, hours=24):
    x = ET.fromstring(call("POST", key, {"uploads": ""}))
    uid = next(e.text for e in x.iter() if _strip(e.tag) == "UploadId")
    ps = part_mb << 20
    n = max(1, -(-size // ps))
    urls = [presign("PUT", key, hours * 3600, {"partNumber": i + 1, "uploadId": uid}) for i in range(n)]
    return {"key": key, "upload_id": uid, "part_size": ps, "size": size, "urls": urls}


def mpu_done(key, uid, etags):
    parts = "".join(f"<Part><PartNumber>{i + 1}</PartNumber><ETag>{e}</ETag></Part>" for i, e in enumerate(etags))
    call("POST", key, {"uploadId": uid}, f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>".encode())


def ls(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        x = ET.fromstring(call("GET", None, q))
        for c in x:
            if _strip(c.tag) == "Contents":
                d = {_strip(e.tag): e.text for e in c}
                out.append((d["Key"], int(d["Size"])))
        nxt = [e.text for e in x if _strip(e.tag) == "NextContinuationToken"]
        if not nxt:
            return out
        token = nxt[0]


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for c in ("get", "put"):
        q = s.add_parser(c)
        q.add_argument("key")
        q.add_argument("--hours", type=float, default=12)
    q = s.add_parser("mpu-start")
    q.add_argument("key")
    q.add_argument("size", type=int)
    q.add_argument("--part-mb", type=int, default=512)
    q = s.add_parser("mpu-done")
    q.add_argument("key")
    q.add_argument("upload_id")
    q.add_argument("etags")
    q = s.add_parser("ls")
    q.add_argument("prefix")
    a = p.parse_args()
    if a.cmd in ("get", "put"):
        print(presign(a.cmd.upper(), a.key, a.hours * 3600))
    elif a.cmd == "mpu-start":
        json.dump(mpu_start(a.key, a.size, a.part_mb), sys.stdout)
    elif a.cmd == "mpu-done":
        mpu_done(a.key, a.upload_id, json.load(open(a.etags)))
    else:
        for k, n in ls(a.prefix):
            print(n, k)


if __name__ == "__main__":
    main()
