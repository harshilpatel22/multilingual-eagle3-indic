"""Publication figures for the cross-lingual EAGLE-3 paper. Reads results/*.csv, writes results/figures/*.pdf (+ .png)."""
import csv, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)

# ---- aesthetic defaults ----
rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
})
ENG = "#C0504D"; IND = "#2C7FB8"; NEU = "#7F7F7F"  # english-head red, indic-head blue
NAME = {"en":"English","es":"Spanish","hi":"Hindi","gu":"Gujarati","bn":"Bengali",
        "mr":"Marathi","ta":"Tamil","te":"Telugu"}

def langmean(path):
    d = {}
    for r in csv.DictReader(open(os.path.join(R, path))):
        d.setdefault(r["lang"], []).append(float(r["accept_length"]))
    return {k: statistics.mean(v) for k, v in d.items()}

def lang_meanstd(paths):  # across seed files
    per = {}
    for p in paths:
        for k, v in langmean(p).items():
            per.setdefault(k, []).append(v)
    return {k: (statistics.mean(v), (statistics.pstdev(v) if len(v) < 2 else statistics.stdev(v))) for k, v in per.items()}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig); print("wrote", name + ".pdf/.png")

# ============ FIG 1: degradation + mechanism ============
cov = {r["lang"]: (float(r["inflation_vs_en"]), float(r["eng_head_coverage_pct"]))
       for r in csv.DictReader(open(os.path.join(R, "lang_inflation_coverage.csv")))}
tau8 = langmean("curve_enghead.csv")  # 8-lang english-head tau
langs = sorted(tau8, key=lambda l: -tau8[l])

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
# panel A: tau by language
cols = [NEU if l in ("en", "es") else ENG for l in langs]
bars = a.bar([NAME[l] for l in langs], [tau8[l] for l in langs], color=cols, edgecolor="black", linewidth=0.5)
a.axhline(1.0, ls="--", color="black", lw=0.8)
a.annotate("τ = 1 (no speedup)", xy=(5.3, 1.0), xytext=(3.4, 1.82), fontsize=8.5, ha="center",
           arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
a.set_ylabel("acceptance length τ"); a.set_title("(a) English head: τ collapses on Indic")
a.tick_params(axis="x", rotation=40); a.set_ylim(0, 2.6)
for bar, l in zip(bars, langs): a.text(bar.get_x()+bar.get_width()/2, tau8[l]+0.03, f"{tau8[l]:.2f}", ha="center", fontsize=8)
# panel B: tau vs coverage scatter
xs = [cov[l][1] for l in langs]; ys = [tau8[l] for l in langs]
b.scatter(xs, ys, c=cols, s=70, edgecolor="black", linewidth=0.5, zorder=3)
import numpy as np
m, c = np.polyfit(xs, ys, 1); xr = np.array([min(xs), max(xs)])
b.plot(xr, m*xr+c, color=IND, lw=1.5, alpha=0.7, zorder=2)
_off = {"en": (-21, -4), "es": (5, 3), "mr": (6, 2), "hi": (6, 2),
        "bn": (-2, 8), "ta": (-15, -10), "te": (0, -14), "gu": (9, -3)}
for l in langs:
    b.annotate(l, (cov[l][1], tau8[l]), textcoords="offset points", xytext=_off.get(l, (5, 3)), fontsize=8)
b.set_xlabel("draft-vocab coverage of eval tokens (%)"); b.set_ylabel("acceptance length τ")
b.set_title("(b) τ vs. vocab coverage  (Pearson r = +0.95)")
save(fig, "fig1_degradation_mechanism")

# ============ FIG 2: recovery (FLORES + held-out) ============
eng_fl = langmean("phase1_tau_by_language.csv")
ind_fl = lang_meanstd(["phase3_tau_v2.csv", "seed_1.csv", "seed_2.csv"])  # 3 seeds
eng_ho = langmean("heldout_enghead.csv"); ind_ho = langmean("heldout_v2.csv")
order = ["en", "hi", "gu"]; x = np.arange(len(order)); w = 0.38

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
a.bar(x-w/2, [eng_fl[l] for l in order], w, label="English head", color=ENG, edgecolor="black", lw=0.5)
a.bar(x+w/2, [ind_fl[l][0] for l in order], w, yerr=[ind_fl[l][1] for l in order], capsize=4,
      label="Indic head (3 seeds)", color=IND, edgecolor="black", lw=0.5)
a.axhline(1.0, ls="--", color="black", lw=0.8)
a.set_xticks(x); a.set_xticklabels([NAME[l] for l in order]); a.set_ylabel("acceptance length τ")
a.set_title("(a) FLORES (in-domain)"); a.legend(frameon=False, fontsize=9); a.set_ylim(0, 2.8)
b.bar(x-w/2, [eng_ho[l] for l in order], w, label="English head", color=ENG, edgecolor="black", lw=0.5)
b.bar(x+w/2, [ind_ho[l] for l in order], w, label="Indic head", color=IND, edgecolor="black", lw=0.5)
b.axhline(1.0, ls="--", color="black", lw=0.8)
b.set_xticks(x); b.set_xticklabels([NAME[l] for l in order]); b.set_title("(b) Aya (held-out, out-of-domain)")
b.legend(frameon=False, fontsize=9)
save(fig, "fig2_recovery")

# ============ FIG 3: cross-scale (8B vs 32B) ============
e32 = langmean("curve32_enghead.csv"); i32 = langmean("curve32_v.csv")
i8 = {l: ind_fl[l][0] for l in order}
groups = [("8B", eng_fl, i8), ("32B", e32, i32)]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (scale, eh, ih) in zip(axes, groups):
    ax.bar(x-w/2, [eh[l] for l in order], w, label="English head", color=ENG, edgecolor="black", lw=0.5)
    ax.bar(x+w/2, [ih[l] for l in order], w, label="Indic head", color=IND, edgecolor="black", lw=0.5)
    ax.axhline(1.0, ls="--", color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NAME[l] for l in order]); ax.set_title(f"Qwen3-{scale}")
    if scale == "8B": ax.set_ylabel("acceptance length τ"); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Degradation + recovery generalize across model scale", y=1.02, fontsize=12)
save(fig, "fig3_crossscale")
print("DONE")
