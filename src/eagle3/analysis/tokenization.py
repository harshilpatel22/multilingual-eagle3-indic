"""Token-inflation analysis (Innovation 2, independent-variable half).

Pure functions (testable offline) + a tokenizer loader. No torch / no model weights.
"""
import numpy as np


def inflation_ratios(tokens_by_lang, base="en"):
    base_mean = float(np.mean(tokens_by_lang[base]))
    out = {}
    for lang, counts in tokens_by_lang.items():
        arr = np.asarray(counts, dtype=float)
        out[lang] = {
            "n": int(arr.size),
            "mean_tokens": float(arr.mean()),
            "median_tokens": float(np.median(arr)),
            "inflation_vs_base": float(arr.mean() / base_mean),
        }
    return out


def per_sample_ratio(tokens_by_lang, base="en"):
    base_arr = np.asarray(tokens_by_lang[base], dtype=float)
    out = {}
    for lang, counts in tokens_by_lang.items():
        ratios = np.asarray(counts, dtype=float) / base_arr
        out[lang] = {"mean_ratio": float(ratios.mean()),
                     "median_ratio": float(np.median(ratios))}
    return out


def build_tokenization_table(texts_by_lang, count_tokens_fn, base="en"):
    """count_tokens_fn: list[str] -> list[int]. Returns list of per-language dict rows."""
    tokens = {lang: count_tokens_fn(texts) for lang, texts in texts_by_lang.items()}
    agg = inflation_ratios(tokens, base=base)
    psr = per_sample_ratio(tokens, base=base)
    rows = []
    for lang in texts_by_lang:
        rows.append({
            "lang": lang,
            "n": agg[lang]["n"],
            "mean_tokens": agg[lang]["mean_tokens"],
            "median_tokens": agg[lang]["median_tokens"],
            "inflation_vs_en": agg[lang]["inflation_vs_base"],
            "mean_per_sample_ratio": psr[lang]["mean_ratio"],
            "tau_drop": None,   # filled in Phase 1 (GPU) — never fabricate
        })
    return rows


def load_tokenizer(model_id="Qwen/Qwen3-8B"):
    from transformers import AutoTokenizer  # lazy import keeps unit tests offline
    return AutoTokenizer.from_pretrained(model_id)


def count_tokens(texts, tokenizer):
    enc = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in enc]
