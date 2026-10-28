"""DRAM GB/s from `perf stat -I 500 -x,` of uncore_imc_free_running/data_read|data_write (MiB per interval): the
median over the steady intervals of a phase, given as fractions of the run [lo, hi]."""
import sys, statistics
mode = sys.argv[2]
rows = {}
for l in open(sys.argv[1]):
    p = l.strip().split(',')
    if len(p) < 4 or not p[0].replace('.', '').isdigit():
        continue
    t, v, ev = float(p[0]), p[1], p[3]
    try:
        rows.setdefault(t, {})['r' if 'read' in ev else 'w'] = float(v)
    except ValueError:
        pass
ts = sorted(rows)
if mode == 'last':  # the last N seconds before the counters stopped, minus 1.5 s of exit
    b = ts[-1] - 1.5; a = b - float(sys.argv[3])
else:
    lo, hi = float(sys.argv[3]), float(sys.argv[4]); a, b = ts[int(len(ts) * lo)], ts[min(len(ts) - 1, int(len(ts) * hi))]
sel = [rows[t] for t in ts if a <= t <= b]
dt = ts[1] - ts[0]
rd = statistics.median(r.get('r', 0) for r in sel) / dt / 1024
wr = statistics.median(r.get('w', 0) for r in sel) / dt / 1024
print(f"intervals {len(sel)} ({a:.1f}-{b:.1f} s): DRAM read {rd:.1f} GB/s, write {wr:.1f} GB/s")
