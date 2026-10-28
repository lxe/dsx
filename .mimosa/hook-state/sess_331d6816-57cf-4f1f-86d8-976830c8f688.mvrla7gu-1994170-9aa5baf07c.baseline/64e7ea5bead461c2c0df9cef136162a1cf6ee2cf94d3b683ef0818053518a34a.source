#!/usr/bin/env python3
"""r2up.py FILE MPU_JSON ETAGS_OUT - upload FILE's parts to the presigned multipart URLs in MPU_JSON (r2.py mpu-start),
16 at a time with retries; write the parts' ETags in order to ETAGS_OUT."""
import concurrent.futures as cf, http.client, json, os, sys, time, urllib.parse

f, m, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
ps, size = m["part_size"], os.path.getsize(sys.argv[1])
assert size == m["size"], (size, m["size"])


def put(i):
    u = urllib.parse.urlsplit(m["urls"][i])
    with open(f, "rb") as fh:
        fh.seek(i * ps)
        data = fh.read(ps)
    for attempt in range(6):
        try:
            c = http.client.HTTPSConnection(u.netloc, timeout=600)
            c.request("PUT", u.path + "?" + u.query, body=data, headers={"Content-Length": str(len(data))})
            r = c.getresponse()
            r.read()
            if r.status == 200:
                return r.getheader("ETag")
        except OSError:
            pass
        time.sleep(2 ** attempt)
    raise SystemExit(f"part {i + 1} failed")


t = time.time()
with cf.ThreadPoolExecutor(16) as ex:
    tags = list(ex.map(put, range(len(m["urls"]))))
json.dump(tags, open(out, "w"))
print(f"{f}: {size / 1e9:.1f} GB in {time.time() - t:.0f} s ({size / 1e6 / (time.time() - t):.0f} MB/s)")
