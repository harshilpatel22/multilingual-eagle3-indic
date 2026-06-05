"""Tests for offline draft-vocab construction + coverage (no tokenizer, fake token ids)."""
from eagle3.data.vocab_coverage import (
    token_frequencies, build_combined_frequency, build_draft_vocab,
    coverage, language_token_share, sweep, meets_threshold,
)


def test_token_frequencies_counts_across_seqs():
    assert token_frequencies([[1, 1, 2], [2, 3]]) == {1: 2, 2: 2, 3: 1}


def test_build_draft_vocab_takes_top_k_by_frequency():
    assert build_draft_vocab({10: 5, 20: 3, 30: 1}, 2) == {10, 20}


def test_build_draft_vocab_tie_break_smaller_id_first():
    # 7 and 3 tie at count 2 -> smaller id (3) ranks first; top-1 -> {3}
    assert build_draft_vocab({7: 2, 3: 2, 9: 1}, 1) == {3}


def test_build_draft_vocab_handles_k_larger_than_unique():
    assert build_draft_vocab({1: 3, 2: 1}, 10) == {1, 2}


def test_coverage_fraction_inside_vocab():
    assert coverage([[1, 2, 3], [2, 9]], {1, 2, 3}) == 0.8  # 9 is out -> 4/5


def test_coverage_empty_is_zero():
    assert coverage([], {1}) == 0.0


def test_language_token_share_normalizes():
    assert language_token_share({"en": 3, "hi": 1}) == {"en": 0.75, "hi": 0.25}


def test_combined_frequency_scales_by_budget():
    # each language's sample frequency is scaled so it contributes `budget` tokens
    freq = build_combined_frequency({"en": [[1, 1]], "hi": [[2, 2]]},
                                    {"en": 100, "hi": 100})
    assert freq[1] == 100.0 and freq[2] == 100.0


def test_combined_frequency_more_english_budget_outranks_indic():
    freq = build_combined_frequency({"en": [[1]], "hi": [[2]]},
                                    {"en": 1000, "hi": 10})
    assert build_draft_vocab(freq, 1) == {1}  # english token wins the single slot


def test_sweep_enlarging_k_recovers_indic_coverage():
    # english tokens 1..3 share the budget heavily; hi tokens 100..102 get a smaller budget
    seqs = {"en": [[1, 2, 3, 1, 2, 3]], "hi": [[100, 101, 102]]}
    eval_by_lang = {"en": [[1, 2, 3]], "hi": [[100, 101, 102]]}
    rows = sweep(seqs, eval_by_lang, {"en": 600, "hi": 300}, [3, 6])
    cov = {(r["K"], r["lang"]): r["coverage"] for r in rows}
    assert cov[(3, "en")] == 1.0 and cov[(3, "hi")] == 0.0   # K=3: only english fits
    assert cov[(6, "hi")] == 1.0                             # K=6: indic recovered


def test_meets_threshold():
    rows = [{"lang": "en", "coverage": 0.97}, {"lang": "hi", "coverage": 0.99}]
    assert meets_threshold(rows, {"en": 0.96, "hi": 0.98}) is True
    assert meets_threshold(rows, {"en": 0.98, "hi": 0.98}) is False
