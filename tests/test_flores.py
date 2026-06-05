from eagle3.data.flores import align_parallel, LANG_CODES


def test_lang_codes_are_flores_codes():
    assert LANG_CODES == {"en": "eng_Latn", "hi": "hin_Deva", "gu": "guj_Gujr"}


def test_align_parallel_joins_on_id():
    per_lang = {
        "en": {0: "hello", 1: "world"},
        "hi": {0: "namaste", 1: "duniya"},
    }
    assert align_parallel(per_lang) == [
        {"id": 0, "en": "hello", "hi": "namaste"},
        {"id": 1, "en": "world", "hi": "duniya"},
    ]


def test_align_parallel_keeps_only_common_ids():
    per_lang = {"en": {0: "a", 1: "b"}, "hi": {0: "x"}}
    assert [r["id"] for r in align_parallel(per_lang)] == [0]
