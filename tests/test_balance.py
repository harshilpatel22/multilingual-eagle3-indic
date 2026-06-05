from eagle3.data.balance import english_convs_for_token_ratio


def test_english_convs_for_token_ratio_basic():
    # indic corpus = 1000 tokens, want en:indic = 2.0 -> 2000 english tokens;
    # english mean 100 tokens/conv -> 20 conversations
    assert english_convs_for_token_ratio(indic_tokens=1000, ratio=2.0,
                                         mean_en_tokens_per_conv=100) == 20


def test_english_convs_rounds_up():
    # 1550 english tokens / 100 -> 16 (round up, never under-fill)
    assert english_convs_for_token_ratio(1000, 1.55, 100) == 16


def test_english_convs_zero_ratio():
    assert english_convs_for_token_ratio(1000, 0.0, 100) == 0
