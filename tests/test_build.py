from eagle3.data.build import to_per_language_rows


def test_to_per_language_rows_splits_and_tags():
    aligned = [
        {"id": 0, "en": "hello", "hi": "namaste", "gu": "kem cho"},
        {"id": 1, "en": "world", "hi": "duniya", "gu": "duniya"},
    ]
    out = to_per_language_rows(aligned, ["en", "hi", "gu"])
    assert out["hi"] == [
        {"id": 0, "lang": "hi", "text": "namaste"},
        {"id": 1, "lang": "hi", "text": "duniya"},
    ]
    assert [r["text"] for r in out["en"]] == ["hello", "world"]
    assert set(out.keys()) == {"en", "hi", "gu"}
