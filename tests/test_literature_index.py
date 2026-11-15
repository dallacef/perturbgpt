"""Tests for the PubMed literature RAG pipeline.

No network access and no sentence-transformers model download are required:
the embedding model is replaced with a deterministic stand-in.  FAISS-backed
tests are skipped when faiss is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from perturbgpt.rag import literature_index as li
from perturbgpt.rag.pubmed_ingest import (
    PubMedRecord,
    _parse_efetch_xml,
    load_stored_abstracts,
)

DIM = 8


class FakeSentenceTransformer:
    """Deterministic character-histogram embedder (no model download)."""

    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for i, ch in enumerate(text.lower()):
                v[i % DIM] += (ord(ch) % 17) / 17.0
            rows.append(v)
        arr = np.stack(rows)
        if normalize_embeddings:
            norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            arr = arr / norms
        return arr.astype(np.float32)


RECORDS = [
    PubMedRecord(pmid="1", title="KLF1 erythroid", abstract="KLF1 controls erythroid genes",
                 journal="Blood", year="2020"),
    PubMedRecord(pmid="2", title="HIF1A hypoxia", abstract="HIF1A drives hypoxia response",
                 journal="Cell", year="2021"),
    PubMedRecord(pmid="3", title="MYC proliferation", abstract="MYC drives proliferation",
                 journal="Nature", year="2022"),
]


@pytest.fixture()
def fake_models(monkeypatch):
    pytest.importorskip("faiss")
    monkeypatch.setattr(li, "_require_sentence_transformers", lambda: FakeSentenceTransformer)
    return FakeSentenceTransformer


def test_retrieve_returns_ranked_hits(fake_models):
    index = li.LiteratureIndex(RECORDS, relevance_threshold=0.0)
    hits = index.retrieve_literature("KLF1 erythroid", top_k=3)
    assert hits
    assert hits[0]["pmid"] == "1"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_relevance_threshold_suppresses_weak_matches(fake_models):
    index = li.LiteratureIndex(RECORDS, relevance_threshold=1.01)
    assert index.retrieve_literature("KLF1 erythroid", top_k=3) == []


def test_save_and_load_roundtrip(fake_models, tmp_path):
    index = li.LiteratureIndex(RECORDS, relevance_threshold=0.0)
    index.save(tmp_path / "index")

    assert (tmp_path / "index" / "literature.faiss").exists()
    payload = json.loads((tmp_path / "index" / "literature.json").read_text())
    assert payload["model_name"] == li.DEFAULT_MODEL
    assert len(payload["records"]) == len(RECORDS)

    loaded = li.LiteratureIndex.load(tmp_path / "index")
    assert [r.pmid for r in loaded.records] == [r.pmid for r in RECORDS]
    assert loaded.relevance_threshold == index.relevance_threshold
    hits_before = index.retrieve_literature("HIF1A hypoxia", top_k=2)
    hits_after = loaded.retrieve_literature("HIF1A hypoxia", top_k=2)
    assert [h["pmid"] for h in hits_before] == [h["pmid"] for h in hits_after]


def test_build_literature_index_rejects_empty_file(fake_models, tmp_path):
    path = tmp_path / "abstracts.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="No abstracts"):
        li.build_literature_index(path)


def test_load_stored_abstracts_roundtrip(tmp_path):
    path = tmp_path / "abstracts.json"
    payload = [
        {"pmid": "42", "title": "t", "abstract": "a", "journal": "j", "year": "2024"}
    ]
    path.write_text(json.dumps(payload))
    records = load_stored_abstracts(path)
    assert records == [PubMedRecord("42", "t", "a", "j", "2024")]
    assert load_stored_abstracts(tmp_path / "missing.json") == []


def test_parse_efetch_xml_extracts_fields():
    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>KLF1 is a master regulator</ArticleTitle>
            <Abstract><AbstractText>KLF1 activates globin genes.</AbstractText></Abstract>
            <Journal><Title>Blood</Title></Journal>
          </Article>
        </MedlineCitation>
        <PubmedData><History><PubDate><Year>2019</Year></PubDate></History></PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>"""
    records = _parse_efetch_xml(xml)
    assert len(records) == 1
    rec = records[0]
    assert rec.pmid == "12345"
    assert rec.title == "KLF1 is a master regulator"
    assert "globin" in rec.abstract
    assert rec.journal == "Blood"
    assert rec.year == "2019"
