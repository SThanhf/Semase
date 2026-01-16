# src/utils.py
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Dict, List, Optional, Literal

from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()



class DocumentMetaData(BaseModel):
    title: str
    url: str
    content_type: str
    last_modified: datetime
    storage_size: int


class Passage(BaseModel):
    chunk_id: str
    text: str
    score: float
    retriever: Literal["bm25", "vector"]


class SearchHit(BaseModel):
    id: str  # document_id

    # raw scores (optional; depends on which retrieval produced the hit)
    bm25: Optional[float] = None
    cosine: Optional[float] = None

    # fused score used for sorting
    score: float = 0.0

    # populated later
    meta: Optional[DocumentMetaData] = None

    # NEW: passages/snippets to show in UI / ground the answer
    passages: List[Passage] = []


def aggregate_search_hits(
    hits: Iterable[SearchHit],
    *,
    w_text: float = 1.0,
    w_vector: float = 1.0,
    rrf_k: int = 60,
    max_passages_per_doc: int = 3,
) -> List[SearchHit]:
    """
    Hybrid fusion using RRF + keep top passages per doc.
    - Uses ranking positions rather than raw scores (BM25 vs vector score are not comparable).
    """
    hits_list = list(hits)

    text_hits = [h for h in hits_list if h.bm25 is not None]
    vector_hits = [h for h in hits_list if h.cosine is not None]

    # Rank maps: doc_id -> rank (1-based)
    text_rank: Dict[str, int] = {}
    for i, h in enumerate(text_hits, start=1):
        text_rank.setdefault(h.id, i)

    vector_rank: Dict[str, int] = {}
    for i, h in enumerate(vector_hits, start=1):
        vector_rank.setdefault(h.id, i)

    merged: Dict[str, SearchHit] = {}

    def _ensure(doc_id: str) -> SearchHit:
        if doc_id not in merged:
            merged[doc_id] = SearchHit(id=doc_id, passages=[])
        return merged[doc_id]

    for h in hits_list:
        m = _ensure(h.id)

        # Keep best raw scores
        if h.bm25 is not None:
            m.bm25 = max(m.bm25 or float("-inf"), h.bm25)
        if h.cosine is not None:
            m.cosine = max(m.cosine or float("-inf"), h.cosine)

        # Collect passages
        if h.passages:
            m.passages.extend(h.passages)

    # Compute RRF fused score
    for doc_id, m in merged.items():
        s = 0.0
        if doc_id in text_rank:
            s += w_text * (1.0 / (rrf_k + text_rank[doc_id]))
        if doc_id in vector_rank:
            s += w_vector * (1.0 / (rrf_k + vector_rank[doc_id]))
        m.score = s

        # Deduplicate passages by chunk_id, keep best score per chunk
        best_by_chunk: Dict[str, Passage] = {}
        for p in m.passages:
            old = best_by_chunk.get(p.chunk_id)
            if old is None or p.score > old.score:
                best_by_chunk[p.chunk_id] = p

        # Sort passages by score desc, keep top N
        m.passages = sorted(best_by_chunk.values(), key=lambda x: x.score, reverse=True)[:max_passages_per_doc]

    return sorted(merged.values(), key=lambda x: x.score, reverse=True)
