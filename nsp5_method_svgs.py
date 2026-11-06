"""Compact SVG figures for the NS-P5 method slides, from logs/nsp5/method_figures.json.

Integer coordinates, implicit line-to, no <text> (labels are <p>s pinned over the SVG).
Prints the pixel positions the slide labels need. The figures are uploaded to the deck as
artifact assets (F3 placement, F5 M2 minus top-k are the ones on the slides since 2026-11-04);
the data comes from nsp5_method_figures.py.

    env -u VIRTUAL_ENV .venv/bin/python nsp5_method_svgs.py [figures.json] [out_dir]
"""
import json
import math
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "logs/nsp5/method_figures.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "logs/nsp5/method_svgs"
os.makedirs(OUT, exist_ok=True)
R = json.load(open(SRC))
ex = R["example"]
T = ex["T"]
GREEN, CRIMSON, PERI, GRAY, DARK, MID, LIGHT, RULE = ("#019881", "#BF1E42", "#839DFB", "#9AA0A5",
                                                      "#1B1C1D", "#6F7478", "#E3E6E9", "#CFD3D7")


def pts(xs, ys):
    return "M" + " ".join(f"{int(round(x))} {int(round(y))}" for x, y in zip(xs, ys))


def pool(a, b, how="max"):
    a = np.asarray(a, float)
    n = int(math.ceil(len(a) / b))
    pad = np.full(n * b, np.nan)
    pad[:len(a)] = a
    pad = pad.reshape(n, b)
    return np.nanmax(pad, 1) if how == "max" else np.nansum(pad, 1)


def save(name, w, h, body, label):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" aria-label="{label}" width="{w}" '
           f'height="{h}" viewBox="0 0 {w} {h}">' + "".join(body) + "</svg>")
    open(os.path.join(OUT, name + ".svg"), "w").write(svg)
    print(f"{name}: {len(svg) / 1024:.1f} KB")


def tx(t, w, x0=0):
    return x0 + t / T * w


lo, hi = 1470, 1490                    # evidence 24:30-24:50
# ---------------------------------------------------------------- F1 option tracks
W, PITCH, TOP, H = 1120, 112, 36, 60
oc = np.array(ex["option_curves"])     # [4, T]
stem = np.array(ex["stem_curve"])
mx = np.array(ex["maxopt_curve"])
B = 4
xs = (np.arange(int(math.ceil(T / B))) * B + B / 2) / T * W
def ymap(z, i):
    z = np.clip(z, 5, 50)
    return i * PITCH + TOP + H * (1 - (z - 5) / 45)
body = []
for i in range(5):
    y0 = i * PITCH + TOP
    body.append(f'<rect x="{int(tx(lo, W))}" y="{y0}" width="{max(6, int(tx(hi, W) - tx(lo, W)))}" '
                f'height="{H}" fill="{PERI}" opacity="0.4"/>')
    body.append(f'<line x1="0" y1="{y0 + H}" x2="{W}" y2="{y0 + H}" stroke="{RULE}" stroke-width="1"/>')
tracks = [(0, stem, GRAY, 2), (1, oc[1], GREEN, 2.5), (2, oc[3], CRIMSON, 2),
          (3, oc[0], GRAY, 1.5), (3, oc[2], RULE, 1.5), (4, mx, DARK, 2.5)]
for i, c, col, sw in tracks:
    body.append(f'<path d="{pts(xs, ymap(pool(c, B), i))}" fill="none" stroke="{col}" '
                f'stroke-width="{sw}" stroke-linejoin="round"/>')
dots = [(0, int(np.argmax(stem)), stem, GRAY), (1, int(np.argmax(oc[1])), oc[1], GREEN),
        (2, int(np.argmax(oc[3])), oc[3], CRIMSON), (4, int(np.argmax(mx)), mx, DARK),
        (4, int(np.argmax(oc[1])), mx, GREEN)]
for i, s, c, col in dots:
    body.append(f'<circle cx="{int(tx(s + 0.5, W))}" cy="{int(ymap(c[s], i))}" r="7" fill="{col}"/>')
    print(f"  F1 dot track {i} second {s} ({s // 60}:{s % 60:02d}) x={int(tx(s + 0.5, W))} y={int(ymap(c[s], i))}")
save("f1_options", W, 5 * PITCH, body,
     "Score curves over a 33-minute video for one LVBench question. The question alone peaks at 4:24, far "
     "from the 20-second evidence at 24:30. Adding option B, a box of needles, peaks at 24:58 beside the "
     "evidence; option D, a box of cards, peaks at 27:45 where a box of cards is on the table. The maximum "
     "over options keeps both peaks; the evidence ranks third.")
print(f"  F1 evidence x {tx(lo, W):.0f}-{tx(hi, W):.0f}; minutes x:",
      {m: int(tx(m * 60, W)) for m in (0, 10, 20, 30)})

# ---------------------------------------------------------------- F2a NLL vs tau
t = R["tau"]["maxopt"]
taus, nl = np.array(t["taus"]), np.array(t["nll_mean"])
keep = taus >= 0.85
W2, H2, XL, XR, YT, YB = 760, 400, 70, 740, 20, 360
lx = lambda v: XL + (math.log(v) - math.log(0.85)) / (math.log(30) - math.log(0.85)) * (XR - XL)
ly = lambda v: YB - (v - 3.0) / (8.0 - 3.0) * (YB - YT)
body = [f'<line x1="{XL}" y1="{YB}" x2="{XR}" y2="{YB}" stroke="{GRAY}" stroke-width="1.5"/>',
        f'<line x1="{XL}" y1="{YT}" x2="{XL}" y2="{YB}" stroke="{GRAY}" stroke-width="1.5"/>']
for v in (4, 5, 6, 7):
    body.append(f'<line x1="{XL}" y1="{int(ly(v))}" x2="{XR}" y2="{int(ly(v))}" stroke="{LIGHT}" stroke-width="1"/>')
uy = int(ly(t["uniform_nll"]))
body.append(f'<line x1="{XL}" y1="{uy}" x2="{XR}" y2="{uy}" stroke="{MID}" stroke-width="2" stroke-dasharray="10 8"/>')
body.append(f'<path d="{pts([lx(v) for v in taus[keep]], [ly(v) for v in nl[keep]])}" fill="none" '
            f'stroke="{DARK}" stroke-width="3.5" stroke-linejoin="round"/>')
j = int(np.argmin(nl))
body.append(f'<circle cx="{int(lx(taus[j]))}" cy="{int(ly(nl[j]))}" r="10" fill="{GREEN}"/>')
for v in (1, 30):
    k = int(np.argmin(abs(taus - v)))
    body.append(f'<circle cx="{int(lx(taus[k]))}" cy="{int(ly(nl[k]))}" r="7" fill="{DARK}"/>')
    print(f"  F2a point tau {taus[k]:.2f} nll {nl[k]:.2f} at x={int(lx(taus[k]))} y={int(ly(nl[k]))}")
save("f2a_nll", W2, H2, body,
     "Average negative log probability on the evidence against the temperature, over 1,534 LVBench "
     "questions. It falls from 6.96 at temperature 1 to a minimum of 3.66 at 3.58, then rises to 4.82 at 30; "
     "a uniform guess scores 5.13.")
print(f"  F2a min x={int(lx(taus[j]))} y={int(ly(nl[j]))}; uniform y={uy}; xticks",
      {v: int(lx(v)) for v in (1, 3, 10, 30)}, "yticks", {v: int(ly(v)) for v in (4, 5, 6, 7)})

# ---------------------------------------------------------------- F2b probability at three taus
W3, PITCH3, TOP3, H3 = 860, 130, 40, 76
body = []
z = np.array(ex["maxopt_curve"])
xs3 = (np.arange(int(math.ceil(T / B))) * B + B / 2) / T * W3
for i, tau in enumerate((1.0, 3.58, 10.0)):
    p = np.exp((z - z.max()) / tau)
    p /= p.sum()
    m = pool(p, B, "sum")
    y0 = i * PITCH3 + TOP3
    ys = y0 + H3 * (1 - m / m.max())
    body.append(f'<rect x="{int(tx(lo, W3))}" y="{y0}" width="{max(6, int(tx(hi, W3) - tx(lo, W3)))}" '
                f'height="{H3}" fill="{PERI}" opacity="0.4"/>')
    area = pts(xs3, ys) + f" {int(xs3[-1])} {y0 + H3} {int(xs3[0])} {y0 + H3}Z"
    col = GREEN if tau == 3.58 else MID
    body.append(f'<path d="{area}" fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
    body.append(f'<line x1="0" y1="{y0 + H3}" x2="{W3}" y2="{y0 + H3}" stroke="{RULE}" stroke-width="1"/>')
save("f2b_prob", W3, 3 * PITCH3, body,
     "The same question's probability over the 33 minutes at three temperatures. At 1 it sits on a few "
     "seconds; at 3.58 it keeps a main peak by the evidence and a smaller one at the cards; at 10 it is "
     "nearly flat.")
print("  F2b minutes x:", {m: int(tx(m * 60, W3)) for m in (0, 10, 20, 30)})

# ---------------------------------------------------------------- F3 Kelly CDF and placements
W4, X0 = 1680, 60
WT = W4 - X0
CT, CB = 16, 236
mix = np.array(ex["p"]["3.58"])                      # M2 places frames at the quantiles of p
F = np.concatenate([[0.0], np.cumsum(mix)])          # F at integer seconds 0..T
secs = np.arange(0, T + 1, 4)
body = [f'<rect x="{int(tx(lo, WT, X0))}" y="{CT}" width="{max(6, int(tx(hi, WT) - tx(lo, WT)))}" '
        f'height="{CB - CT}" fill="{PERI}" opacity="0.4"/>',
        f'<line x1="{X0}" y1="{CB}" x2="{W4}" y2="{CB}" stroke="{GRAY}" stroke-width="1.5"/>',
        f'<line x1="{X0}" y1="{CT}" x2="{X0}" y2="{CB}" stroke="{GRAY}" stroke-width="1.5"/>',
        f'<line x1="{X0}" y1="{CT}" x2="{W4}" y2="{CT}" stroke="{LIGHT}" stroke-width="1"/>',
        f'<line x1="{X0}" y1="{(CT + CB) // 2}" x2="{W4}" y2="{(CT + CB) // 2}" stroke="{LIGHT}" stroke-width="1"/>']
kt = np.array(ex["kelly_times"])
rung, drop = [], []
for i, tt in enumerate(sorted(kt)):
    q = (i + 0.5) / 64
    y = CB - q * (CB - CT)
    x = tx(tt, WT, X0)
    rung.append(f"M{X0} {y:.0f}h14")
    drop.append(f"M{x:.0f} {y:.0f}V{CB}")
body.append(f'<path d="{"".join(rung)}" stroke="{GREEN}" stroke-width="2" fill="none"/>')
body.append(f'<path d="{"".join(drop)}" stroke="{GREEN}" stroke-width="1" stroke-opacity="0.45" fill="none"/>')
body.append(f'<line x1="{X0}" y1="{CB}" x2="{W4}" y2="{CT}" stroke="{MID}" stroke-width="2" stroke-dasharray="10 8"/>')
body.append(f'<path d="{pts([tx(s, WT, X0) for s in secs], [CB - F[min(s, T)] * (CB - CT) for s in secs])}" '
            f'fill="none" stroke="{GREEN}" stroke-width="4" stroke-linejoin="round"/>')
rows = [("uniform_times", GRAY), ("topk_times", MID), ("kelly_times", GREEN)]
RT, RP, RH = 296, 80, 32
for r, (key, col) in enumerate(rows):
    y0 = RT + r * RP
    d = "".join(f"M{tx(tt, WT, X0):.0f} {y0}v{RH}" for tt in ex[key])
    body.append(f'<rect x="{int(tx(lo, WT, X0))}" y="{y0 - 4}" width="{max(6, int(tx(hi, WT) - tx(lo, WT)))}" '
                f'height="{RH + 8}" fill="{PERI}" opacity="0.4"/>')
    body.append(f'<line x1="{X0}" y1="{y0 + RH}" x2="{W4}" y2="{y0 + RH}" stroke="{RULE}" stroke-width="1"/>')
    body.append(f'<path d="{d}" stroke="{col}" stroke-width="3" fill="none"/>')
save("f3_kelly", W4, RT + 2 * RP + RH + 6, body,
     "M2's Kelly placement for the example. The running total of the probability climbs in steps: near "
     "4:00 (the stage), at 24:58 (the needles) and at 27:45 (the cards). Frames sit where it crosses 64 "
     "evenly spaced levels. Three rows compare the 64 frames: uniform puts none on the evidence, top-k puts "
     "5 there and bunches the rest on high-scoring seconds, M2 puts 6 there, 18 within 30 seconds and 5 at "
     "the cards, and leaves gaps of up to 262 seconds.")
print("  F3 rows y:", [RT + r * RP for r in range(3)], "cdf", CT, CB, "minutes x:",
      {m: int(tx(m * 60, WT, X0)) for m in (0, 10, 20, 30)},
      "needles x", int(tx(1498, WT, X0)), "cards x", int(tx(1665, WT, X0)),
      "F(24:30)", round(F[1470], 3), "F(25:30)", round(F[1530], 3), "F(27:30)", round(F[1650], 3), "F(28:00)", round(F[1680], 3))

# ---------------------------------------------------------------- F4 E3 ladder: accuracy vs frames on evidence
e3 = R["e3"]
W5, H5, XL5, XR5, YT5, YB5 = 800, 420, 60, 780, 16, 380
fx = lambda v: XL5 + (math.log(v) - math.log(4)) / (math.log(50) - math.log(4)) * (XR5 - XL5)
fy = lambda v: YB5 - (v - 30) / (66 - 30) * (YB5 - YT5)
body = [f'<line x1="{XL5}" y1="{YB5}" x2="{XR5}" y2="{YB5}" stroke="{GRAY}" stroke-width="1.5"/>',
        f'<line x1="{XL5}" y1="{YT5}" x2="{XL5}" y2="{YB5}" stroke="{GRAY}" stroke-width="1.5"/>']
for v in (35, 45, 55, 65):
    body.append(f'<line x1="{XL5}" y1="{int(fy(v))}" x2="{XR5}" y2="{int(fy(v))}" stroke="{LIGHT}" stroke-width="1"/>')
lad = ["e3_uniform", "e3_gauss_s240", "e3_gauss_s60", "e3_gauss_s15"]
P = [(e3[a]["mean_frames_on_evidence"], e3[a]["acc"]) for a in lad]
body.append(f'<path d="{pts([fx(a) for a, _ in P], [fy(b) for _, b in P])}" fill="none" stroke="{PERI}" stroke-width="4" stroke-linejoin="round"/>')
for (a, b), name in zip(P, lad):
    col = GRAY if name == "e3_uniform" else PERI
    body.append(f'<circle cx="{int(fx(a))}" cy="{int(fy(b))}" r="11" fill="{col}"/>')
    print(f"  F4 {name}: frames {a} acc {b} at x={int(fx(a))} y={int(fy(b))}")
a, b = e3["e3_offpeak_s60"]["mean_frames_on_evidence"], e3["e3_offpeak_s60"]["acc"]
body.append(f'<circle cx="{int(fx(a))}" cy="{int(fy(b))}" r="11" fill="{CRIMSON}"/>')
print(f"  F4 offpeak: frames {a} acc {b} at x={int(fx(a))} y={int(fy(b))}; xticks",
      {v: int(fx(v)) for v in (5, 10, 20, 40)}, "yticks", {v: int(fy(v)) for v in (35, 45, 55, 65)})
save("f4_ladder", W5, H5, body,
     "LVBench accuracy against the average number of frames on the evidence, for the E3 placements: "
     "uniform 6.2 frames and 40.3%, oracle sigma 240 s 11.4 and 48.8%, sigma 60 s 19.6 and 57.6%, sigma 15 s "
     "33.2 and 62.1%. Gains shrink as frames double. A placebo clustered 10 minutes away scores 35.1%.")

# ---------------------------------------------------------------- F5 Kelly minus top-k with CIs
# M2 minus top-k (q8opt_topk_nr), registered measures; METHOD.md section 11
rows5 = [("LVBench", 1.43, -0.32, 3.21), ("Video-MME long", 1.22, -0.78, 3.33),
         ("LongVideoBench", 4.04, 2.29, 5.85), ("MLVU mean over tasks", -0.54, -2.05, 0.99),
         ("CG-Bench", 1.63, 0.39, 2.80), ("Pooled, 10,745 questions", 1.30, 0.69, 1.95)]
W6, H6, XL6, XR6, PIT6, T6 = 560, 470, 20, 540, 72, 40
gx = lambda v: XL6 + (v + 5) / 11 * (XR6 - XL6)
body = [f'<line x1="{int(gx(0))}" y1="10" x2="{int(gx(0))}" y2="{T6 + 5 * PIT6 + 30}" stroke="{GRAY}" stroke-width="2" stroke-dasharray="8 8"/>',
        f'<line x1="{XL6}" y1="{T6 + 5 * PIT6 - 36}" x2="{XR6}" y2="{T6 + 5 * PIT6 - 36}" stroke="{RULE}" stroke-width="1"/>']
for i, (name, m, l, h) in enumerate(rows5):
    y = T6 + i * PIT6
    col = GREEN if l > 0 else (CRIMSON if h < 0 else MID)
    body.append(f'<line x1="{int(gx(l))}" y1="{y}" x2="{int(gx(h))}" y2="{y}" stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
    body.append(f'<circle cx="{int(gx(m))}" cy="{y}" r="10" fill="{col}"/>')
print("  F5 rows y:", [T6 + i * PIT6 for i in range(6)], "xticks", {v: int(gx(v)) for v in (-4, -2, 0, 2, 4)})
save("f5_alloc", W6, H6, body,
     "M2 minus top-k on the same 8B option scores, with 95% intervals clustered by video: LVBench +1.4, "
     "Video-MME long +1.2, LongVideoBench +4.0 (significant), MLVU -0.5, CG-Bench +1.6 (significant); pooled "
     "over all 10,745 questions +1.3, significant.")
