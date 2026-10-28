import collections
rows = [l.strip().split(", ") for l in open("/tmp/clk.csv") if l.count(",") >= 6]
for g in ("0", "1"):
    R = [r for r in rows if r[0] == g and r[5].strip() not in ("0", "[N/A]")]
    if not R:
        continue
    sm = sorted(int(r[1]) for r in R); pw = sorted(float(r[4]) for r in R)
    ps = collections.Counter(r[3] for r in R); gen = collections.Counter(r[6] for r in R)
    print(f"GPU{g} busy samples {len(R)} of {sum(1 for r in rows if r[0] == g)}: sm MHz p10 {sm[len(sm)//10]} median {sm[len(sm)//2]} p90 {sm[9*len(sm)//10]}; power median {pw[len(pw)//2]:.0f} W; pstates {dict(ps)}; pcie gen {dict(gen)}")
