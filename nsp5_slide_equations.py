"""Typeset the per-question equations of the NS-P5 method slide (3.1) as SVG.

The deck's slide format takes no MathML and no sub- or superscripts, so the equations are drawn
with matplotlib's mathtext in STIX (the deck's serif), glyphs as paths, and uploaded to the deck
as image assets. One SVG unit is one pixel on the slide; each file prints its size.

    uv run --no-sync python research/TimeLens/nsp5_slide_equations.py [out_dir]
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "research/TimeLens/logs/nsp5/slide_eqs"
INK, GRAY = "#1B1C1D", "#6F7478"
EQ, NOTE, GAP = 30, 24, 12          # px on the slide
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral",
                     "svg.fonttype": "path"})

FIGS = {
    "eq_score": [(r"$z(t) = \max_{k}\ 100\,\cos(f_t,\, q_k)$", EQ, INK),
                 (r"$f_t$: the frame at second $t$", NOTE, GRAY),
                 (r"$q_k$: the question with option $k$", NOTE, GRAY)],
    "eq_prob": [(r"$p(t) = \dfrac{\exp(z(t)/\tau)}{\sum_{s}\ \ \exp(z(s)/\tau)}$", EQ, INK),
                (r"$\tau = 3.58$, fit by likelihood", NOTE, GRAY),
                (r"$s$ runs over every second", NOTE, GRAY)],
}


def render(name: str, lines: list) -> None:
    """Stack the lines top to bottom, left-aligned, and save a cropped SVG and a PNG preview."""
    fig = plt.figure(figsize=(8, 5), dpi=72)
    renderer = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    y = h - 2
    for i, (text, size, color) in enumerate(lines):
        t = fig.text(2 / w, y / h, text, fontsize=size, color=color, va="top", ha="left")
        y -= t.get_window_extent(renderer).height + (GAP if i == 0 else 4)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".svg")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=2 / 72, transparent=True,
                metadata={"Date": None, "Creator": None, "Format": None, "Type": None})
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=144, bbox_inches="tight",
                pad_inches=2 / 72, facecolor="#FCFCFC")
    plt.close(fig)
    # the deck takes no URLs, comments or <style> in an SVG: keep shapes and local refs only
    svg = open(path, encoding="utf-8").read()
    svg = re.sub(r"<\?xml[^>]*\?>\s*|<!DOCTYPE[^>]*>\s*|<!--.*?-->\s*", "", svg, flags=re.S)
    svg = re.sub(r"<metadata>.*?</metadata>\s*|<style[^>]*>.*?</style>\s*", "", svg, flags=re.S)
    svg = svg.replace("xlink:href", "href").replace(' xmlns:xlink="http://www.w3.org/1999/xlink"', "")
    open(path, "w", encoding="utf-8").write(svg)
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    print(f"{name}: {float(vb[2]):.0f} x {float(vb[3]):.0f} px, {len(svg) / 1024:.1f} KB")


for fig_name, fig_lines in FIGS.items():
    render(fig_name, fig_lines)
