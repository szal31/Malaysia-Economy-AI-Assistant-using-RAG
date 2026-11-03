import pytest

import reports


def words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_chunk_never_crosses_pages():
    chunks = reports.chunk([(1, words(600, "a")), (2, words(30, "b"))], words=250, overlap=50)
    assert {c["page"] for c in chunks} == {1, 2}
    assert sum(c["page"] == 2 for c in chunks) == 1
    assert all(len(c["text"].split()) <= 250 for c in chunks)
    assert all(c["text"].startswith("a") for c in chunks if c["page"] == 1)


def test_chunk_overlap():
    p1 = [c for c in reports.chunk([(1, words(600, "a"))], words=250, overlap=50)]
    assert p1[0]["text"].split()[-50:] == p1[1]["text"].split()[:50]
    assert [c["chunk_no"] for c in p1] == [0, 1, 2]


def test_fetch_rejects_changed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "RAW", tmp_path)
    (tmp_path / "x").write_bytes(b"not what was hashed")
    row = {"doc_id": "x", "kind": "pdf", "url": "http://invalid", "sha256": "0" * 64}
    with pytest.raises(reports.SourceChanged):
        reports.fetch(row)
