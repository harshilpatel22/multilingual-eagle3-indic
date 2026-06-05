"""Robustness stats for the τ results: bootstrap 95% CIs + paired held-out recovery test.
Reads results/*.csv (per-request accept_length). Local, CPU-only. Addresses reviewer point #1.
"""
import csv
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(0)


def load(fname):
    d = {}
    for r in csv.DictReader(open(R / fname)):
        d.setdefault(r["lang"], []).append(float(r["accept_length"]))
    return d


def boot_ci(vals, B=10000):
    a = np.asarray(vals, float)
    means = rng.choice(a, (B, a.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return a.mean(), a.std(ddof=1), lo, hi, a.size


FILES = {
    "EngHead  · FLORES":      "phase1_tau_by_language.csv",
    "v1 indic-heavy · FLORES":"phase3_tau_newhead.csv",
    "v2 balanced · FLORES":   "phase3_tau_v2.csv",
    "EngHead  · Aya(held-out)":"heldout_enghead.csv",
    "v2 balanced · Aya(held-out)":"heldout_v2.csv",
}

print(f"{'condition':30s} {'lang':4s} {'mean':>6s} {'95% CI':>17s} {'sd':>5s} {'n':>3s}")
print("-" * 72)
for name, f in FILES.items():
    d = load(f)
    for lang in ["en", "hi", "gu"]:
        if lang in d:
            m, sd, lo, hi, n = boot_ci(d[lang])
            print(f"{name:30s} {lang:4s} {m:6.3f} [{lo:5.3f}, {hi:5.3f}] {sd:5.3f} {n:3d}")

# Paired held-out recovery (same Aya prompts -> pair by (lang,id))
eng = {(r["lang"], r["id"]): float(r["accept_length"]) for r in csv.DictReader(open(R / "heldout_enghead.csv"))}
v2 = {(r["lang"], r["id"]): float(r["accept_length"]) for r in csv.DictReader(open(R / "heldout_v2.csv"))}
print("\n=== Held-out (Aya) PAIRED recovery  v2 − EngHead  (same prompts) ===")
for lang in ["en", "hi", "gu"]:
    diffs = np.array([v2[k] - eng[k] for k in eng if k[0] == lang and k in v2])
    bs = rng.choice(diffs, (10000, diffs.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    win = (diffs > 0).mean() * 100
    sig = "significant (CI excludes 0)" if lo * hi > 0 else "n.s. (CI spans 0)"
    print(f"  {lang}: Δτ = {diffs.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   v2 wins {win:4.0f}% of prompts   {sig}")
