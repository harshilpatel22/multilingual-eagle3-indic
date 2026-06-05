from eagle3.analysis.tokenization import (
    inflation_ratios, per_sample_ratio, build_tokenization_table,
)


def test_inflation_ratios_basic():
    out = inflation_ratios({"en": [10, 10], "hi": [20, 20]}, base="en")
    assert out["en"]["inflation_vs_base"] == 1.0
    assert out["hi"]["inflation_vs_base"] == 2.0
    assert out["hi"]["n"] == 2


def test_per_sample_ratio_uses_aligned_pairs():
    out = per_sample_ratio({"en": [10, 20], "hi": [20, 20]}, base="en")
    assert out["hi"]["mean_ratio"] == 1.5      # (20/10 + 20/20)/2


def test_build_table_marks_tau_drop_unmeasured():
    texts = {"en": ["a", "b"], "hi": ["aaa", "bbb"]}
    fake_count = lambda xs: [len(x) for x in xs]   # 'a'->1, 'aaa'->3
    rows = build_tokenization_table(texts, fake_count, base="en")
    hi = next(r for r in rows if r["lang"] == "hi")
    assert hi["inflation_vs_en"] == 3.0
    assert hi["tau_drop"] is None                  # never fabricated; filled in Phase 1
