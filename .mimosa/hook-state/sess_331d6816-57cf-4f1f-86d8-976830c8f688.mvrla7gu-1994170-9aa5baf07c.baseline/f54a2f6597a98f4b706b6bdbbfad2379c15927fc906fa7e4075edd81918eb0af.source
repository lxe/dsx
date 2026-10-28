#!/usr/bin/env python3
"""Lucebox eval (.npy/.npz) -> raw little-endian files for `dsx-amd eval`. usage: eval_bins.py EVAL_DIR"""
import sys, numpy as np
d = sys.argv[1]
np.load(f"{d}/heldout_8x1024.npy").astype("<i4").tofile(f"{d}/heldout.i32")
z = np.load(f"{d}/teacher_8x1024.npz")
z["idx"].astype("<i4").tofile(f"{d}/teacher_idx.i32")
z["lp"].astype("<f4").tofile(f"{d}/teacher_lp.f32")
print("ok", z["idx"].shape)
