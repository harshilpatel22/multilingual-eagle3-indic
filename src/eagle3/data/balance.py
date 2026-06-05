"""Corpus-sizing helpers: convert a chosen English:Indic token ratio (from the vocab
sweep) into a concrete number of ShareGPT conversations to include in the v3 corpus."""
import math


def english_convs_for_token_ratio(indic_tokens, ratio, mean_en_tokens_per_conv):
    """Number of English conversations so english_tokens ~= ratio * indic_tokens.

    Rounds UP so we never under-fill the target English volume. Returns 0 for ratio<=0.
    """
    if ratio <= 0 or mean_en_tokens_per_conv <= 0:
        return 0
    target_en_tokens = ratio * indic_tokens
    return math.ceil(target_en_tokens / mean_en_tokens_per_conv)
