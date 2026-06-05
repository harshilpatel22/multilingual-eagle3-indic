from eagle3.common.io import read_jsonl, write_jsonl, ensure_dir


def test_write_then_read_roundtrip(tmp_path):
    rows = [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"}]
    p = tmp_path / "sub" / "out.jsonl"
    write_jsonl(p, rows)            # also creates parent dir
    assert read_jsonl(p) == rows


def test_write_jsonl_preserves_unicode_unescaped(tmp_path):
    p = tmp_path / "u.jsonl"
    write_jsonl(p, [{"text": "नमस्ते ગુજરાતી"}])
    content = p.read_text(encoding="utf-8")
    assert "नमस्ते" in content and "ગુજરાતી" in content   # not \uXXXX escapes


def test_ensure_dir_creates(tmp_path):
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()
