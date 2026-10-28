# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Draws the model card's charts: speed.png, quality.png, size.png, size-vs-quality.png.

Run:  uv run cards/dsx-2bit/assets/make_charts.py      (uv installs playwright; any Chromium works)
Each chart is an SVG drawn here, rendered by headless Chromium at 2x (1200 px wide) on a white card.
Chromium: PW_CHROMIUM=/path/to/chrome, else the first one found in ~/.cache/ms-playwright, else
`uv run --with playwright playwright install chromium` once. Fonts: Pathway Extreme from Google Fonts,
Iosevka for numbers (CHART_NUM_FONT=/path/to/font.woff2 to use a specific file); both fall back to system fonts.
All numbers are the ones below; edit them here and rerun.
"""
import glob, os, pathlib
from playwright.sync_api import sync_playwright

OUT = pathlib.Path(__file__).resolve().parent
W = 600  # CSS px; PNGs are 2x

# ---------------------------------------------------------------- data
SPEED = {  # tok/s, exact routing, greedy
    "2x RTX 3090 + 128 GB DDR4": {"prose": 28.9, "agent": 28.7, "p3k": 301, "p8k": 470},  # v4
    "RTX 5090 + 128 GB DDR5": {"prose": 12.0, "agent": 15.2, "p3k": 93, "p8k": 172},
}
QUALITY = [  # Lucebox 1017 eval set: 8 x 1024 tokens vs FP8 top-256 log-probs
    # name, KL, top-1 %, perplexity, highlight
    ("FP8 original", 0.0, 100.0, 8.56, False),
    ("Lucebox ROCMFP, 2.5 bpw", 0.253, 78.7, 9.78, False),
    ("dsx 2-bit, 2.03 bpw", 0.373, 74.4, 10.68, True),
]
SIZE = [  # GB to download, without the 203 GB Engram tables unless marked
    ("FP8 original", 510, "includes Engram", False),
    ("Q2_K GGUF (mrfakename)", 265, "", False),
    ("EXL3 3.5 bpw", 258, "", False),
    ("EXL3 2.9 bpw", 211, "", False),
    ("dsx 2-bit", 148, "", True),
    ("VQ (Baekpica)", 115, "approx.", False),
]
SIZE_QUALITY = [  # sizes all include the Engram tables; perplexity on the Lucebox eval
    # name, GB, GB range or None, perplexity, highlight
    ("FP8 original", 510, None, 8.56, False),
    ("Lucebox ROCMFP 2.5 bpw", 395, (383, 405), 9.78, False),
    ("dsx 2-bit", 351, None, 10.68, True),
]

# ---------------------------------------------------------------- style
BLUE, BLUE2, GREY, GREY2, INK, MUTED, GRID = "#4d6bfe", "#a9b7ff", "#9aa3ad", "#5e6872", "#14181f", "#5d6670", "#e4e8ec"
FONT = os.environ.get("CHART_NUM_FONT") if os.path.exists(os.environ.get("CHART_NUM_FONT", "")) else None
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Pathway+Extreme:opsz,wdth,wght@8..12,75..100,400..600&display=swap');
{f"@font-face{{font-family:Num;src:url(file://{FONT})}}" if FONT else ""}
html,body{{margin:0;background:transparent}}
.card{{width:{W}px;box-sizing:border-box;background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:20px 22px 16px}}
h1{{margin:0;font:600 21px/1.15 'Pathway Extreme',system-ui,sans-serif;font-variation-settings:'wdth' 85;color:{INK}}}
.sub{{margin-top:4px;font:400 12.5px/1.35 system-ui,-apple-system,'Segoe UI',sans-serif;color:{MUTED}}}
svg{{display:block;margin-top:10px;overflow:visible}}
svg text{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:{INK};font-size:12px}}
svg .n{{font-family:Num,Iosevka,'Iosevka Term',ui-monospace,Menlo,monospace}}
svg .m{{fill:{MUTED}}}
svg .v{{paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}}
.key{{display:flex;gap:18px;margin-top:10px;font:400 12.5px system-ui,sans-serif;color:{INK}}}
.key i{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.note{{margin-top:8px;font:400 11.5px/1.35 system-ui,sans-serif;color:{MUTED}}}
"""

def page(title, sub, svg, key="", note="", width=W):
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style><div class=card style='width:{width}px'><h1>{title}</h1><div class=sub>{sub}</div>{key}{svg}{f'<div class=note>{note}</div>' if note else ''}</div>"

def t(x, y, s, cls="", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{s}</text>'

def key(items):
    return "<div class=key>" + "".join(f"<span><i style='background:{c}'></i>{n}</span>" for n, c in items) + "</div>"

# ---------------------------------------------------------------- charts
def speed():
    machines = list(SPEED)
    cols = [BLUE, GREY]
    panels = [("Decode", [("prose", "prose"), ("agent", "agent, 3K prompt")], 45, 10),
              ("Prefill", [("p3k", "3K prompt"), ("p8k", "8K prompt")], 650, 100)]
    pw, h, top, bot = 262, 230, 26, 196
    s = []
    for pi, (name, groups, mx, step) in enumerate(panels):
        ox = pi * (pw + 32) + 34
        s.append(t(ox - 30, 12, name + " (tok/s)", extra='font-weight="600"'))
        y = lambda v: bot - v / mx * (bot - top)
        for v in range(0, mx + 1 - step // 2, step):
            s.append(f'<line x1="{ox}" x2="{ox+pw-34}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID if v else "#b8c0c8"}"/>')
            s.append(t(ox - 6, y(v) + 4, v, "n m", "end", 'font-size="10.5"'))
        gw = (pw - 34) / len(groups)
        for gi, (k, label) in enumerate(groups):
            gx = ox + gi * gw + (gw - 2 * 40 - 6) / 2
            for mi, mname in enumerate(machines):
                v = SPEED[mname][k]; x = gx + mi * 46
                s.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="40" height="{bot-y(v):.1f}" rx="2" fill="{cols[mi]}"/>')
                s.append(t(x + 20, y(v) - 5, f"{v:.1f}" if mx < 100 else f"{v:g}", "n v", "middle", 'font-size="11.5"'))
            s.append(t(ox + gi * gw + gw / 2, bot + 17, label, "", "middle"))
    svg = f'<svg width="{W-44}" height="{h}">{"".join(s)}</svg>'
    return page("dsx speed: DeepSeek-V4.1-Flash-1017 2-bit", "Tokens per second, exact routing, greedy decoding.", svg, key(list(zip(machines, cols))))

def quality():
    lw, pw, gap, rh, top = 168, 112, 16, 30, 24
    panels = [("Perplexity ↓", 3, 12, "{:.2f}"), ("Top-1 match ↑", 2, 100, "{:.1f}%"), ("KL divergence ↓", 1, 0.5, "{:.3f}")]
    s = []
    for pi, (name, idx, mx, fmt) in enumerate(panels):
        ox = lw + pi * (pw + gap)
        s.append(t(ox, 12, name, extra='font-weight="600"'))
        for ri, row in enumerate(QUALITY):
            v, hl = row[idx], row[4]
            y = top + ri * rh
            c = BLUE if hl else (GREY2 if ri == 0 else GREY)
            bw = v / mx * (pw - 44)
            if bw > 0: s.append(f'<rect x="{ox}" y="{y+4}" width="{bw:.1f}" height="18" rx="2" fill="{c}"/>')
            s.append(t(ox + bw + 5, y + 17, fmt.format(v), "n v", extra=f'font-size="11.5" {"font-weight=\"700\"" if hl else ""}'))
    for ri, row in enumerate(QUALITY):
        s.append(t(0, top + ri * rh + 17, row[0], "", extra=f'{"font-weight=\"700\"" if row[4] else ""}'))
    svg = f'<svg width="{W-44}" height="{top + rh*len(QUALITY)}">{"".join(s)}</svg>'
    return page("Quality against the FP8 original", "Lucebox 1017 eval set: 8 × 1,024 tokens, compared with FP8's top-256 log-probs.", svg)

def size():
    lw, rh, top, mx = 168, 28, 4, 550
    bx = lambda v: lw + v / mx * (W - 44 - lw - 165)
    s = []
    for g in range(0, mx + 1, 100):
        s.append(f'<line x1="{bx(g):.1f}" x2="{bx(g):.1f}" y1="{top}" y2="{top + rh*len(SIZE)}" stroke="{GRID}"/>')
        s.append(t(bx(g), top + rh * len(SIZE) + 14, g, "n m", "middle", 'font-size="10.5"'))
    for ri, (name, gb, tag, hl) in enumerate(SIZE):
        y = top + ri * rh
        c = BLUE if hl else GREY
        pat = ' fill="url(#hatch)"' if tag == "includes Engram" else f' fill="{c}"'
        s.append(t(0, y + 16, name, "", extra='font-weight="700"' if hl else ""))
        s.append(f'<rect x="{lw}" y="{y+4}" width="{bx(gb)-lw:.1f}" height="17" rx="2"{pat}/>')
        s.append(t(bx(gb) + 6, y + 16, ("~" if tag == "approx." else "") + f"{gb} GB" + (" · includes Engram" if tag == "includes Engram" else ""), "n", extra=f'font-size="11.5" {"font-weight=\"700\"" if hl else ""}'))
    defs = f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="{GREY}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" stroke-width="2.5"/></pattern></defs>'
    svg = f'<svg width="{W-44}" height="{top + rh*len(SIZE) + 20}">{defs}{"".join(s)}</svg>'
    return page("Download size of DeepSeek-V4.1-Flash-1017 builds", "Gigabytes, without the 203 GB Engram tables unless marked.", svg)

def size_quality():
    X0, X1, Y0, Y1 = 46, W - 44 - 20, 226, 14
    xl, xh, yl, yh = 300, 550, 8, 12
    x = lambda v: X0 + (v - xl) / (xh - xl) * (X1 - X0)
    y = lambda v: Y0 - (v - yl) / (yh - yl) * (Y0 - Y1)
    s = []
    for g in range(xl, xh + 1, 50):
        s.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{Y1}" y2="{Y0}" stroke="{GRID}"/>')
        s.append(t(x(g), Y0 + 15, g, "n m", "middle", 'font-size="10.5"'))
    for g in range(yl, yh + 1):
        s.append(f'<line x1="{X0}" x2="{X1}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{GRID}"/>')
        s.append(t(X0 - 7, y(g) + 4, g, "n m", "end", 'font-size="10.5"'))
    s.append(t((X0 + X1) / 2, Y0 + 34, "Size with Engram tables (GB)", "m", "middle"))
    s.append(t(X0 - 34, (Y0 + Y1) / 2, "Perplexity ↓", "m", "middle", f'transform="rotate(-90 {X0-34} {(Y0+Y1)/2})"'))
    place = {"FP8 original": (-10, 4, "end"), "Lucebox ROCMFP 2.5 bpw": (0, -14, "middle"), "dsx 2-bit": (12, 4, "start")}
    for name, gb, rng, ppl, hl in SIZE_QUALITY:
        c = BLUE if hl else (BLUE2 if name.startswith("dsx") else GREY2)
        if rng: s.append(f'<line x1="{x(rng[0]):.1f}" x2="{x(rng[1]):.1f}" y1="{y(ppl):.1f}" y2="{y(ppl):.1f}" stroke="{GREY2}" stroke-width="2"/>')
        s.append(f'<circle cx="{x(gb):.1f}" cy="{y(ppl):.1f}" r="{7 if hl else 6}" fill="{c}"/>')
        dx, dy, a = place[name]
        size_txt = f"~{gb} ({rng[0]}–{rng[1]})" if rng else f"{gb}"
        s.append(t(x(gb) + dx, y(ppl) + dy, f'{name} <tspan class="n m">{size_txt} GB, {ppl:.2f}</tspan>', "", a, 'font-weight="700"' if hl else ""))
    svg = f'<svg width="{W-44}" height="{Y0 + 42}">{"".join(s)}</svg>'
    return page("Size against perplexity", "Lucebox 1017 eval set. Sizes include the 203 GB Engram tables.", svg)

BRAND_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Pathway+Extreme:ital,opsz,wdth,wght@0,8..12,75..100,300..600;1,8..12,75..100,300..600&display=swap');
{f"@font-face{{font-family:Num;src:url(file://{FONT})}}" if FONT else ""}
html,body{{margin:0}}
.card{{width:1600px;height:900px;box-sizing:border-box;position:relative;overflow:hidden;padding:64px 76px;color:#f0f6f7;
  font:400 22px/1.4 Num,Iosevka,ui-monospace,Menlo,monospace;
  background:radial-gradient(60% 70% at 88% 0%,rgba(125,240,212,.16),transparent 65%),radial-gradient(50% 60% at 0% 100%,rgba(154,168,255,.10),transparent 70%),#0d1318}}
.eyebrow{{font-weight:700;font-size:20px;letter-spacing:.16em;color:#7df0d4;text-transform:uppercase}}
h1{{margin:18px 0 0;font:300 66px/1.02 'Pathway Extreme',sans-serif;font-variation-settings:'wdth' 75,'opsz' 12;letter-spacing:-.01em}}
h1 em{{color:#7df0d4;font-style:italic}}
.hero{{margin-top:54px;display:grid;grid-template-columns:repeat(3,1fr);gap:28px}}
.tile{{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:26px 30px}}
.big{{font:500 88px/1 'Pathway Extreme',sans-serif;font-variation-settings:'wdth' 75;color:#7df0d4;white-space:nowrap}}
.big small{{font-size:40px;margin-left:10px;color:#bff7e9}}
.lab{{margin-top:12px;color:#a9b8bd;font-size:21px}}
.cmp{{margin-top:44px}}
.cmp h2{{margin:0 0 18px;font:600 26px/1 Num,Iosevka,ui-monospace,monospace}}
.cmp h2 span{{color:#a9b8bd;font-weight:400}}
.row{{display:grid;grid-template-columns:260px 1fr 110px;align-items:center;gap:22px;height:46px}}
.nm{{font-weight:700;font-size:24px}} .pc{{font-weight:700;font-size:30px;text-align:right}}
.track{{height:26px;border-radius:6px;background:rgba(255,255,255,.06)}} .track b{{display:block;height:100%;border-radius:6px}}
.foot{{position:absolute;left:76px;right:76px;bottom:40px;display:flex;justify-content:space-between;color:#8a9a9f;font-size:20px}}
"""

POST = {  # the X post: a few big numbers, one comparison
    "speed": "30+", "size": "148", "prefill": "500",  # prefill: 508 tok/s on an 8K prompt
    # top-1 match with FP8 on the Lucebox 1017 eval set (QUALITY above), rounded
    "match": [("FP8 original", 100, "#c9d6da"), ("Lucebox 2.5-bit", 79, "#9aa8ff"), ("dsx 2-bit", 74, "#7df0d4")],
}

def post():
    """16:9 social card in Pocket Model's site style: dark ground, mint accent, Pathway Extreme headings."""
    bars = "".join(
        f"<div class=row><span class=nm style='color:{c}'>{n}</span><div class=track><b style='width:{v}%;background:{c}'></b></div>"
        f"<span class=pc style='color:{c}'>{v}%</span></div>" for n, v, c in POST["match"])
    css = BRAND_CSS
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · 2-bit quant + engine</div>
<h1>DeepSeek-V4.1-Flash-1017 (552B MoE)<br><em>on 2x RTX 3090 + 128 GB RAM</em></h1>
<div class=hero>
  <div class=tile><div class=big>{POST['speed']}<small>tok/s</small></div><div class=lab>generation speed</div></div>
  <div class=tile><div class=big>{POST['size']}<small>GB</small></div><div class=lab>model download</div></div>
  <div class=tile><div class=big>{POST['prefill']}<small>tok/s</small></div><div class=lab>prompt processing</div></div>
</div>
<div class=cmp><h2>Same next token as FP8 <span>top-1 match</span></h2>{bars}</div>
<div class=foot><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div>
</div>"""

EVAL_POST = {  # Pocket Model dogfood v26b: dsx 2-bit vs the hosted model, temperature 1.0, think low, graded in code
    "rows": [("Tool calls", 20, 16, 14), ("Olympiad math", 3, 3, 3), ("Hard multiple choice", 3, 1, 1), ("All", 26, 20, 18)],
    "dsx": ("dsx 2-bit, at home", "#7df0d4"), "hosted": ("Full model, hosted", "#5b7cff"),
}

def post_eval():
    """16:9 social card: dsx 2-bit vs the hosted model on a 26-row Pocket Model eval."""
    dn, dc = EVAL_POST["dsx"]; hn, hc = EVAL_POST["hosted"]
    groups = ""
    for name, n, d, h in EVAL_POST["rows"]:
        bar = lambda k, c: (f"<div class=vb><span style='color:{c}'>{k}/{n}</span>"
                            f"<b style='height:{k / n * 240:.0f}px;background:{c}'></b></div>")
        groups += (f"<div class='grp{' all' if name == 'All' else ''}'><div class=pair>{bar(d, dc)}{bar(h, hc)}</div>"
                   f"<div class=gn>{name}</div><div class=gr>{n} rows</div></div>")
    css = BRAND_CSS + """
h1{font-size:54px;margin-top:14px}
.card{padding-top:52px}
.body{margin-top:30px;display:grid;grid-template-columns:1fr 430px;gap:40px}
.chart{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:22px 30px 16px}
.chart h2{margin:0;font:600 25px/1 Num,Iosevka,ui-monospace,monospace}
.chart .k{display:flex;gap:26px;margin-top:14px;font-size:20px}
.chart .k i{display:inline-block;width:16px;height:16px;border-radius:4px;margin-right:9px;vertical-align:-2px}
.groups{margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;align-items:end;height:310px;border-bottom:2px solid rgba(228,241,244,.3)}
.grp{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.grp.all{background:rgba(255,255,255,.04);border-radius:12px 12px 0 0}
.pair{display:flex;gap:10px;align-items:flex-end}
.vb{width:62px;display:flex;flex-direction:column;align-items:center}
.vb span{font-weight:700;font-size:23px;margin-bottom:8px;white-space:nowrap}
.vb b{display:block;width:100%;border-radius:5px 5px 0 0}
.gn,.gr{position:relative;top:0}
.grp .gn{order:3;margin-top:0}
.labels{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;text-align:center;margin-top:12px}
.labels b{display:block;font-size:20px} .labels span{color:#a9b8bd;font-size:17px}
.tiles{display:flex;flex-direction:column;gap:16px}
.tiles .tile{padding:18px 26px}
.tiles .big{font-size:66px}
.tiles .lab{margin-top:8px;font-size:20px}
.foot{flex-direction:column;gap:6px;bottom:28px;font-size:18px}
.foot .r{display:flex;justify-content:space-between}
.foot .note{color:#a9b8bd}
"""
    names = [r[0] for r in EVAL_POST["rows"]]; ns = [r[1] for r in EVAL_POST["rows"]]
    groups = groups.replace("<div class=gn>", "<div style='display:none'>").replace("<div class=gr>", "<div style='display:none'>")
    labels = "".join(f"<div><b>{a}</b><span>{n} rows</span></div>" for a, n in zip(names, ns))
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · Pocket Model eval · 26 hard rows</div>
<h1>2-bit DeepSeek-V4.1-Flash-1017 on two RTX 3090s<br><em>matches the hosted full model</em></h1>
<div class=body>
 <div class=chart><h2>Rows passed</h2>
  <div class=k><span><i style='background:{dc}'></i>{dn}</span><span><i style='background:{hc}'></i>{hn}</span></div>
  <div class=groups>{groups}</div><div class=labels>{labels}</div></div>
 <div class=tiles>
  <div class=tile><div class=big>2×<small>tokens</small></div><div class=lab>thinking on math and multiple choice</div></div>
  <div class=tile><div class=big>29<small>tok/s</small></div><div class=lab>2x RTX 3090 + 128 GB RAM</div></div>
  <div class=tile><div class=big>$0<small>per token</small></div><div class=lab>self-hosted, 148 GB</div></div>
 </div>
</div>
<div class=foot><div class=note>Both at temperature 1.0, think low, graded in code. Within noise (p = 0.69): the hosted model scored 20/26 here at temperature 0.</div>
<div class=r><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div></div>
</div>"""

OVERTHINK = {  # reasoning tokens per run at temperature 1.0 (the model's recommended sampling); every run correct
    "builds": [  # key, label, colour
        ("hosted", "Full model", "#5b7cff"), ("b26", "2.6-bit", "#ffc95c"),
        ("v3a", "v3a 2-bit", "#7df0d4"), ("v2", "v2 2-bit", "#ff6b7a")],
    "problems": [
        ("Problem A", {"hosted": [17426, 11069, 10465, 6371, 6799, 5665, 8426, 13485], "b26": [9735, 10302, 11585],
                       "v3a": [10469, 10263, 15311], "v2": [10126, 39479, 12159, 31487]}),
        ("Problem B", {"hosted": [12523, 7507, 11651, 9523, 8879, 13870, 9741, 12147], "b26": [29133, 32858],
                       "v3a": [21489, 20180], "v2": [26462, 22645]}),
    ],
    "xmax": 42000,
    # perplexity vs download size, Lucebox 1017 eval; sizes include the 203 GB Engram tables
    "ppl": [  # label, GB, GB range or None, perplexity, colour, label offset (dx, dy, anchor)
        ("FP8 original", 510, None, 8.56, "#e6eef0", (0, 30, "middle")),
        ("MXFP4 (MI325X)", 473, None, 8.71, "#c79bff", (0, -18, "middle")),
        ("Lucebox 2.5 bpw", 395, (383, 405), 9.78, "#8a9a9f", (0, -18, "middle")),
        ("2.6-bit", 391, None, 9.37, "#ffc95c", (18, 6, "start")),
        ("v3a 2-bit", 351, None, 10.81, "#7df0d4", (18, 18, "start")),
        ("v2 2-bit", 351, None, 10.93, "#ff6b7a", (18, -8, "start")),
    ],
}

def post_overthink():
    """16:9 social card: perplexity vs size beside reasoning tokens per run on two hard math problems."""
    # left: perplexity against size
    PW, PH = 470, 520
    X0, X1, Y0, Y1 = 58, PW - 16, PH - 64, 64
    xl, xh, yl, yh = 330, 530, 8.4, 11.2
    px = lambda v: X0 + (v - xl) / (xh - xl) * (X1 - X0)
    py = lambda v: Y0 - (v - yl) / (yh - yl) * (Y0 - Y1)
    s = ['<text x="0" y="24" class="pt">Perplexity vs size</text>', '<text x="0" y="50" class="ps">lower is better · with Engram tables</text>']
    for g in range(350, 531, 50):
        s.append(f'<line x1="{px(g):.1f}" x2="{px(g):.1f}" y1="{Y1}" y2="{Y0}" stroke="rgba(228,241,244,.10)"/><text x="{px(g):.1f}" y="{Y0 + 26}" class="ax" text-anchor="middle">{g}</text>')
    for g in (9, 10, 11):
        s.append(f'<line x1="{X0}" x2="{X1}" y1="{py(g):.1f}" y2="{py(g):.1f}" stroke="rgba(228,241,244,.10)"/><text x="{X0 - 10}" y="{py(g) + 6}" class="ax" text-anchor="end">{g}</text>')
    s.append(f'<text x="{X1}" y="{Y0 + 54}" class="ax" text-anchor="end">download, GB</text>')
    for name, gb, rng, ppl, col, (dx, dy, an) in OVERTHINK["ppl"]:
        if rng: s.append(f'<line x1="{px(rng[0]):.1f}" x2="{px(rng[1]):.1f}" y1="{py(ppl):.1f}" y2="{py(ppl):.1f}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<circle cx="{px(gb):.1f}" cy="{py(ppl):.1f}" r="10" fill="{col}" stroke="#0d1318" stroke-width="2.5"/>')
        s.append(f'<text x="{px(gb) + dx:.1f}" y="{py(ppl) + dy:.1f}" class="dl" text-anchor="{an}" style="fill:{col}">{name}</text>')
    left = f'<div class="panel pl"><svg width="{PW}" height="{PH}">{"".join(s)}</svg></div>'
    # right: reasoning tokens per run
    W, L, R, rowh = 860, 150, 24, 38
    xmax = OVERTHINK["xmax"]; builds = OVERTHINK["builds"]
    x = lambda v: L + min(v, xmax) / xmax * (W - L - R)
    s = ['<text x="0" y="24" class="pt">Reasoning tokens per run</text>', '<text x="0" y="50" class="ps">two olympiad math problems · every run correct</text>']
    top = 96
    for pi, (name, runs) in enumerate(OVERTHINK["problems"]):
        blk = top + pi * (rowh * len(builds) + 62)
        s.append(f'<text x="0" y="{blk + 4}" class="pn">{name}</text>')
        y0 = blk + 16
        lo, hi = min(runs["hosted"]), max(runs["hosted"])
        s.append(f'<rect x="{x(lo):.1f}" y="{y0}" width="{x(hi) - x(lo):.1f}" height="{rowh * len(builds)}" rx="6" fill="rgba(91,124,255,.17)"/>')
        for t in range(0, xmax + 1, 10000):
            s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{y0}" y2="{y0 + rowh * len(builds)}" stroke="rgba(228,241,244,.10)"/>')
            s.append(f'<text x="{x(t):.1f}" y="{y0 + rowh * len(builds) + 22}" class="ax" text-anchor="middle">{"0" if t == 0 else f"{t // 1000}K"}</text>')
        for i, (k, label, col) in enumerate(builds):
            cy = y0 + rowh * i + rowh / 2
            s.append(f'<text x="0" y="{cy + 7}" class="rl" style="fill:{col}">{label}</text>')
            for v in sorted(runs[k]):
                s.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="11" fill="{col}" stroke="#0d1318" stroke-width="2.5"/>')
    H = top + 2 * (rowh * len(builds) + 62) - 22
    right = f'<div class="panel pr"><svg width="{W}" height="{H}">{"".join(s)}</svg></div>'
    css = BRAND_CSS + """
.card{padding-top:50px}
h1{font-size:54px;margin-top:12px}
.v3note{margin-top:14px;font-size:21px;line-height:1.4;color:#c9d6da;white-space:nowrap}
.v3note b{color:#7df0d4}
.panels{margin-top:30px;display:grid;grid-template-columns:auto 1fr;gap:24px;align-items:stretch}
.panel{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:22px 26px 12px}
svg{display:block;overflow:visible}
svg text{font-family:Num,Iosevka,ui-monospace,Menlo,monospace;fill:#f0f6f7}
.pt{font-size:25px;font-weight:700} .ps{font-size:18px;fill:#a9b8bd} .pn{font-size:20px;font-weight:700;fill:#c9d6da}
.rl{font-size:20px;font-weight:700} .dl{font-size:18px;font-weight:700} .ax{font-size:16px;fill:#8a9a9f}
.foot{flex-direction:column;gap:6px;bottom:26px;font-size:18px}
.foot .r{display:flex;justify-content:space-between} .foot .note{color:#a9b8bd}
"""
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · 2-bit DeepSeek-V4.1-Flash for 2x RTX 3090 · now with image input</div>
<h1>Better perplexity doesn't stop overthinking</h1>
<div class=v3note><b>v3a:</b> v2's bits and size (148 GB, 2.03 bpw), GPTQ recalibrated with the full model's own reasoning traces in the calibration text.</div>
<div class=panels>{left}{right}</div>
<div class=foot><div class=note>Temperature 1.0, each dot one run. Blue band: the full model's range. On problem B every quantized build shown, the 2.6-bit one included, thinks 2–3× longer than the full model.</div>
<div class=r><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div></div>
</div>"""

UNSLOTH = {  # dsx 2-bit v3a vs Unsloth's 1-bit GGUFs; reasoning tokens per run at temperature 1.0, effort low, one prompt template; every run correct
    # baseline: dsx MXFP4 (near-lossless, KL 0.049), run on an MI325X; the hosted model is left out (its provider template is unverified)
    "builds": [  # key, label, colour
        ("mxfp4", "dsx MXFP4", "#c79bff"), ("v3a", "dsx v3a 2-bit", "#7df0d4"),
        ("iq1s", "Unsloth IQ1_S", "#ffc95c"), ("iq1m", "Unsloth IQ1_M", "#ff9a5c")],
    "problems": [
        ("Problem A", {"mxfp4": [5239, 7956, 10955], "v3a": [10469, 10263, 15311],
                       "iq1s": [10565, 19855, 13080], "iq1m": [10955, 13289]}),
        ("Problem B", {"mxfp4": [17866, 16374, 22216], "v3a": [21489, 20180],
                       "iq1s": [33922, 29765, 21215], "iq1m": [20319, 20270]}),
    ],
    "xmax": 36000,
    # Lucebox 1017 eval: 8 x 1024 tokens vs FP8's top-256 log-probs; weights without the Engram tables
    # same machine, 2x RTX 3090 + 128 GB DDR4, each build in its own runtime; IQ1_S wasn't run there
    "speed": {"v3a": ("dsx", "~30", 30, 334), "iq1m": ("llama.cpp PR #30817", "12.3", 12.3, 109), "iq1s": None},
    "quality": [  # label, colour, weights GB, KL, top-1 %, perplexity
        ("dsx MXFP4", "#c79bff", "270", 0.049, 90.6, 8.71),
        ("dsx v3a 2-bit", "#7df0d4", "148", 0.397, 74.1, 10.81),
        ("Unsloth IQ1_S", "#ffc95c", "~122", 0.361, 75.1, 10.76),
    ],
}

def post_unsloth():
    """16:9 social card, a plain measurement card: reasoning tokens, speed on one machine, per-token quality."""
    import statistics
    builds = UNSLOTH["builds"]; col = {k: c for k, _, c in builds}; lab = {k: l for k, l, _ in builds}
    total = {k: sum(statistics.median(p[1][k]) for p in UNSLOTH["problems"]) for k, _, _ in builds}
    order = ["mxfp4", "v3a", "iq1m", "iq1s"]
    mx = max(total.values())
    reason = "".join(
        f"<div class=hb><div class=hn style='color:{col[k]}'>{lab[k]}</div><div class=ht><b style='width:{total[k] / mx * 100:.1f}%;background:{col[k]}'></b></div>"
        f"<div class=hv><b style='color:{col[k]}'>{total[k] / 1000:.1f}K</b><span>{'baseline' if k == 'mxfp4' else f'{total[k] / total['mxfp4']:.2f}×'}</span></div></div>"
        for k in order)
    def speed_rows(i, mxv):
        out = ""
        for k in ["v3a", "iq1m", "iq1s"]:
            sp = UNSLOTH["speed"][k]
            if sp is None:
                continue
            v = sp[2] if i == 0 else sp[3]; txt = sp[1] if i == 0 else f"{sp[3]}"
            out += (f"<div class=hb><div class=hn style='color:{col[k]}'>{lab[k]}</div><div class=ht><b style='width:{v / mxv * 100:.1f}%;background:{col[k]}'></b></div>"
                    f"<div class=hv><b style='color:{col[k]}'>{txt}</b><span>tok/s</span></div></div>")
        return out
    qrows = "".join(
        f"<tr><td style='color:{c}'><b>{n}</b><span>{gb} GB weights</span></td><td>{kl:.3f}</td><td>{t1:.1f}%</td><td>{ppl:.2f}</td></tr>"
        for n, c, gb, kl, t1, ppl in UNSLOTH["quality"])
    css = BRAND_CSS + """
.card{padding-top:50px}
h1{font-size:54px;margin-top:12px}
.panels{margin-top:30px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px;align-items:stretch}
.panel{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:22px 24px 16px}
.ptitle{font-size:24px;font-weight:700} .psub{font-size:17px;color:#a9b8bd;margin-top:6px;line-height:1.35}
.sec{margin-top:20px;font-size:19px;font-weight:700;color:#c9d6da}
.hb{display:grid;grid-template-columns:1fr auto;grid-template-areas:'n v' 't t';column-gap:10px;margin-top:14px}
.hn{grid-area:n;font-size:19px;font-weight:700} .hv{grid-area:v;text-align:right;white-space:nowrap}
.hv b{font-size:24px} .hv b.na{color:#8a9a9f} .hv span{font-size:16px;color:#a9b8bd;margin-left:8px}
.ht{grid-area:t;height:16px;border-radius:5px;background:rgba(255,255,255,.06);margin-top:6px} .ht b{display:block;height:100%;border-radius:5px}
table.q{margin-top:22px;border-collapse:collapse;width:100%}
table.q th{font-size:16px;color:#a9b8bd;font-weight:400;text-align:right;padding:0 0 10px 10px}
table.q td{font-size:22px;font-weight:700;text-align:right;padding:12px 0 12px 10px;border-top:1px solid rgba(228,241,244,.12)}
table.q td:first-child{text-align:left;font-size:19px;padding-left:0} table.q td span{display:block;font-size:15px;font-weight:400;color:#a9b8bd;margin-top:4px}
.qnote{margin-top:12px;font-size:16px;color:#8a9a9f}
.rs .hb{margin-top:30px}
.rs .hv b{font-size:30px}
.foot{flex-direction:column;gap:6px;bottom:26px;font-size:17px}
.foot .r{display:flex;justify-content:space-between} .foot .note{color:#a9b8bd}
"""
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>DeepSeek-V4.1-Flash-1017 · dsx 2-bit v3a and Unsloth 1-bit GGUFs</div>
<h1>Reasoning, speed and quality at 1–2 bits</h1>
<div class=panels>
 <div class='panel rs'><div class=ptitle>Reasoning tokens</div><div class=psub>median on problem A + median on problem B, two olympiad math problems, every answer correct. Baseline: dsx MXFP4, near-lossless (KL 0.049), same template</div>{reason}</div>
 <div class=panel><div class=ptitle>Speed</div><div class=psub>2x RTX 3090 + 128 GB DDR4; v3a in dsx, IQ1_M in llama.cpp PR #30817</div>
  <div class=sec>Decode</div>{speed_rows(0, 30)}<div class=sec>Prefill, 3K-token prompt</div>{speed_rows(1, 334)}<div class=qnote style='margin-top:22px'>IQ1_S wasn't run on this machine.</div></div>
 <div class=panel><div class=ptitle>Per-token quality vs FP8</div><div class=psub>same eval, 8 × 1,024 tokens</div>
  <table class=q><tr><th></th><th>KL ↓</th><th>top-1 ↑</th><th>ppl ↓</th></tr>{qrows}</table><div class=qnote>IQ1_M wasn't measured on this eval.</div></div>
</div>
<div class=foot><div class=note>Reasoning at temperature 1.0, effort low, 2–3 runs per build. MXFP4 ran on an AMD MI325X; token counts don't depend on the hardware. Engram tables not in the sizes: dsx 203 GB, Unsloth 110 GB.</div>
<div class=r><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div></div>
</div>"""

# ---------------------------------------------------------------- render
def chromium():
    if os.environ.get("PW_CHROMIUM"): return os.environ["PW_CHROMIUM"]
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    return found[-1] if found else None

if __name__ == "__main__":
    charts = {"speed.png": speed(), "quality.png": quality(), "size.png": size(), "size-vs-quality.png": size_quality(), "post.png": post(), "post-eval.png": post_eval(), "post-overthink.png": post_overthink(), "post-unsloth.png": post_unsloth()}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium(), args=["--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": 840, "height": 900}, device_scale_factor=2)
        for name, html in charts.items():
            tmp = OUT / f".{name}.html"
            tmp.write_text(html)
            pg.set_viewport_size({"width": 1600, "height": 900} if name.startswith("post") else {"width": 840, "height": 900})
            pg.goto(tmp.as_uri(), wait_until="networkidle")
            pg.evaluate("document.fonts.ready")
            pg.locator(".card").screenshot(path=str(OUT / name), omit_background=True)
            tmp.unlink()
            print("wrote", OUT / name)
        b.close()
