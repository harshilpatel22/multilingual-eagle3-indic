"""Offline replication of EAGLE-3 draft-vocabulary construction + coverage measurement.

SpecForge builds the EAGLE-3 draft vocab (the d2t/t2d tensors) by selecting the top
`draft_vocab_size` token IDs by FREQUENCY over the training corpus. That is pure
tokenization + counting, so we reproduce it offline (no GPU) to choose draft_vocab_size
and the English:Indic mix BEFORE training. Pure functions only; the Qwen3 tokenizer and
real data live in scripts/vocab_sweep.py so these stay unit-testable offline.
"""
from collections import Counter


def token_frequencies(token_id_seqs):
    """token_id_seqs: iterable of iterables of int -> dict {token_id: count}."""
    c = Counter()
    for seq in token_id_seqs:
        c.update(seq)
    return dict(c)


def build_combined_frequency(seqs_by_lang, token_budget_by_lang):
    """Simulate a training mix without regenerating data.

    seqs_by_lang: {lang: list[list[int]]} representative tokenized text per language.
    token_budget_by_lang: {lang: int} target token volume per language in the simulated
        corpus. Each language's sample frequency is scaled by budget/sample_total, so a
        small representative sample can stand in for a larger corpus of the same
        distribution. Returns Counter {token_id: float_weight} over the combined corpus.
    """
    combined = Counter()
    for lang, budget in token_budget_by_lang.items():
        if budget <= 0:
            continue
        sample = token_frequencies(seqs_by_lang.get(lang, []))
        sample_total = sum(sample.values())
        if sample_total == 0:
            continue
        scale = budget / sample_total
        for tid, cnt in sample.items():
            combined[tid] += cnt * scale
    return combined


def build_draft_vocab(freq, draft_vocab_size):
    """Top-`draft_vocab_size` token ids by frequency. Deterministic tie-break: higher
    count first, then smaller token id. Returns a set (<= draft_vocab_size ids)."""
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return {tid for tid, _ in ordered[:draft_vocab_size]}


def coverage(eval_token_seqs, draft_vocab_set):
    """Fraction (0..1) of eval tokens that fall inside the draft vocab."""
    total = inside = 0
    for seq in eval_token_seqs:
        for tid in seq:
            total += 1
            if tid in draft_vocab_set:
                inside += 1
    return (inside / total) if total else 0.0


def language_token_share(token_budget_by_lang):
    """Each language's share (0..1) of total corpus tokens."""
    total = sum(max(0, b) for b in token_budget_by_lang.values())
    if total == 0:
        return {lang: 0.0 for lang in token_budget_by_lang}
    return {lang: max(0, b) / total for lang, b in token_budget_by_lang.items()}


def sweep(seqs_by_lang, eval_by_lang, token_budget_by_lang, draft_vocab_sizes):
    """For each K in draft_vocab_sizes, build the combined-corpus draft vocab and report
    per-language coverage of the eval tokens + per-language token share.

    Returns list of rows: {"K", "lang", "coverage", "token_share"}.
    """
    freq = build_combined_frequency(seqs_by_lang, token_budget_by_lang)
    shares = language_token_share(token_budget_by_lang)
    rows = []
    for K in draft_vocab_sizes:
        vocab = build_draft_vocab(freq, K)
        for lang in eval_by_lang:
            rows.append({
                "K": K, "lang": lang,
                "coverage": coverage(eval_by_lang[lang], vocab),
                "token_share": shares.get(lang, 0.0),
            })
    return rows


def meets_threshold(rows, thresholds):
    """rows for a single K; thresholds: {lang: min_coverage}. True iff all langs pass."""
    cov = {r["lang"]: r["coverage"] for r in rows}
    return all(cov.get(lang, 0.0) >= thr for lang, thr in thresholds.items())
