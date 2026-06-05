"""Load FLORES+ (FLORES-200) and align parallel sentences across languages.

Verified 2026-05-31: load_dataset("openlanguagedata/flores_plus", "<code>", split="devtest"),
configs are language codes, sentence field is "text", aligned "id" field. Dataset is GATED.
"""

LANG_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "gu": "guj_Gujr"}
DATASET_ID = "openlanguagedata/flores_plus"


def align_parallel(per_lang):
    """per_lang: {lang: {id: text}} -> [{"id": id, lang: text, ...}] joined on common ids."""
    common = set.intersection(*(set(d.keys()) for d in per_lang.values()))
    rows = []
    for sid in sorted(common):
        row = {"id": sid}
        for lang in per_lang:
            row[lang] = per_lang[lang][sid]
        rows.append(row)
    return rows


def load_parallel(lang_codes=None, split="devtest", dataset_id=DATASET_ID):
    """Download FLORES+ for each language and return aligned parallel rows.

    Requires `huggingface-cli login` + accepting terms (dataset is gated).
    """
    from datasets import load_dataset  # imported lazily so unit tests stay offline

    lang_codes = lang_codes or LANG_CODES
    per_lang = {}
    for short, code in lang_codes.items():
        ds = load_dataset(dataset_id, code, split=split)
        per_lang[short] = {row["id"]: row["text"] for row in ds}
    return align_parallel(per_lang)
