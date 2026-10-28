#!/usr/bin/env python3
"""ckpt_list.py PREFIX DEST > ckpt.aria2 - an aria2 input list of presigned GETs for every object under PREFIX in R2,
saved under DEST with the same relative paths (for hydrate.sh)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2
prefix, dest = sys.argv[1].rstrip("/"), sys.argv[2]
for key, _ in r2.ls(prefix + "/"):
    rel = key[len(prefix) + 1:]
    print(r2.presign("GET", key, 24 * 3600))
    print(f"  dir={os.path.join(dest, os.path.dirname(rel))}".rstrip("/"))
    print(f"  out={os.path.basename(rel)}")
