"""Transform aligned parallel rows into per-language eval rows: {id, lang, text}."""


def to_per_language_rows(aligned_rows, langs):
    out = {lang: [] for lang in langs}
    for row in aligned_rows:
        for lang in langs:
            out[lang].append({"id": row["id"], "lang": lang, "text": row[lang]})
    return out
