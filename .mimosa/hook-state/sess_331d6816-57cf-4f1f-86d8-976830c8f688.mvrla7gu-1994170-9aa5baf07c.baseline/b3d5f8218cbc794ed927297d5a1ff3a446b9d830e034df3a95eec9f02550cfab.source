#!/usr/bin/env python3
"""ncu_parse.py CSV...: one line per kernel launch from `ncu --csv --page raw --metrics $(cat bench/ncu_metrics.txt)`
(time, DRAM bytes and GB/s, warp instructions per DRAM byte, issue use, occupancy, pipes, top stalls)."""
import csv, sys
for f in sys.argv[1:]:
    lines = open(f).read().splitlines()
    rows = list(csv.reader(lines[next(i for i, l in enumerate(lines) if l.startswith("\"ID\"")):]))
    hdr, units, data = rows[0], rows[1], rows[2:]
    ix = {h: i for i, h in enumerate(hdr)}
    DRAMKEY = next(k for k in ("dram__bytes_op_read.sum", "dram__bytes_read.sum", "dram__bytes.sum") if k in ix)
    def v(r, k):
        x = r[ix[k]].replace(",", "")
        try: return float(x)
        except ValueError: return x
    print(f"== {f.split('/')[-1]}  (time unit {units[ix['gpu__time_duration.sum']]}, bytes unit {units[ix[DRAMKEY]]})")
    for r in data:
        t = v(r, "gpu__time_duration.sum"); tu = units[ix["gpu__time_duration.sum"]]
        t_s = t * {"ns": 1e-9, "us": 1e-6, "usecond": 1e-6, "nsecond": 1e-9, "msecond": 1e-3}.get(tu, 1e-9)
        b = v(r, DRAMKEY); bu = units[ix[DRAMKEY]]
        b *= {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}.get(bu, 1)
        inst = v(r, "smsp__inst_executed.sum")
        stalls = {k.split("stalled_")[1].split("_per_issue")[0]: v(r, k) for k in hdr if "issue_stalled" in k}
        top = sorted(stalls.items(), key=lambda x: -x[1])[:4]
        print(f"  {r[ix['Kernel Name']][:14]:14s} grid {int(v(r, 'launch__grid_size')):6d} {t_s * 1e6:7.1f} us {b / 1e6:7.2f} MB {b / t_s / 1e9:6.0f} GB/s (dram {v(r, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'):4.0f}%) "
              f"warp-inst/B {inst / b:.3f} issue {v(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):4.1f}% occ {v(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):4.1f}% "
              f"alu {v(r, 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active'):4.1f}% fma {v(r, 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active'):4.1f}% regs {int(v(r, 'launch__registers_per_thread'))} "
              f"clk {v(r, 'sm__cycles_elapsed.avg.per_second') / 1e9 if isinstance(v(r, 'sm__cycles_elapsed.avg.per_second'), float) else '?'} | " + " ".join(f"{k} {x:.1f}" for k, x in top))
