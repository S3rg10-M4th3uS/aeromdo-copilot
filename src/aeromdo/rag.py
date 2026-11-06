"""Hybrid Retrieval-Augmented Generation (RAG) over the 2026 AeroDesign Preliminary Report.

Features:
- Baseline lexical overlap search vs. Hybrid Vector/Lexical search (TF-IDF word 1-2 grams + char 3-5 grams + Requirement ID boost)
- Mandatory source citations (chunk_id, page, section, requirement_ids)
- Calibrated abstention threshold when evidence is insufficient or query is out-of-domain
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aeromdo.data_pipeline import DEFAULT_DB_PATH
from aeromdo.schemas import Citation, RAGResponse


STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois",
    "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram",
    "essa", "essas", "esse", "esses", "esta", "estas", "este", "estes", "eu", "foi",
    "for", "foram", "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me",
    "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nem", "no",
    "nos", "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "não", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "ais",
    "quais", "quando", "que", "quem", "se", "sem", "seu", "seus", "só", "sua",
    "suas", "são", "também", "te", "tem", "tendo", "tenho", "ter", "teu", "teus",
    "tive", "todos", "tu", "tua", "tuas", "tudo", "um", "uma", "umas", "uns", "você",
    "vocês", "vos", "à", "às", "é", "qual", "quais", "sobre", "relatório", "equipe",
}


class AeroRAGIndex:
    """In-memory hybrid retrieval engine backed by SQLite `doc_chunks`."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, abstention_threshold: float = 0.12) -> None:
        self.db_path = db_path
        self.abstention_threshold = abstention_threshold
        self.chunks_df: pd.DataFrame = pd.DataFrame()
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.word_matrix: Any = None
        self.char_matrix: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load chunks from SQLite and build dual TF-IDF (word + character n-gram) indexes."""
        with sqlite3.connect(self.db_path) as conn:
            self.chunks_df = pd.read_sql_query("SELECT * FROM doc_chunks", conn)

        corpus = (
            self.chunks_df["section"].fillna("")
            + " "
            + self.chunks_df["requirement_ids"].fillna("")
            + " "
            + self.chunks_df["content"].fillna("")
        ).tolist()

        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-\./]+\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(corpus)
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def search_naive_lexical(self, query: str, top_k: int = 5) -> list[Citation]:
        """Baseline naive keyword matching (counting exact content token overlaps without section/req/ngram weighting)."""
        self.ensure_loaded()
        q_tokens = {
            tok
            for tok in re.findall(r"[a-zA-Zà-úÀ-Ú]{4,}", query.lower())
            if tok not in STOPWORDS_PT
        }
        scores: list[float] = []
        for row in self.chunks_df.itertuples():
            # Naive baseline only checks raw content words > 3 chars (misses numbers, codes like REQ-SYS-001, and morphology)
            doc_tokens = set(re.findall(r"[a-zA-Zà-úÀ-Ú]{4,}", str(row.content).lower()))
            if not q_tokens or not doc_tokens:
                scores.append(0.0)
            else:
                overlap = len(q_tokens & doc_tokens) / math.sqrt(len(doc_tokens) + 10.0)
                scores.append(float(overlap))

        order = np.argsort(scores)[::-1][:top_k]
        return self._format_citations(order, scores)

    def search_hybrid(self, query: str, top_k: int = 5) -> list[Citation]:
        """Hybrid retrieval combining word n-grams, character n-grams, requirement codes, and domain boosts."""
        self.ensure_loaded()
        assert self.word_vectorizer is not None and self.char_vectorizer is not None

        q_word = self.word_vectorizer.transform([query])
        q_char = self.char_vectorizer.transform([query])

        sim_word = cosine_similarity(q_word, self.word_matrix).ravel()
        sim_char = cosine_similarity(q_char, self.char_matrix).ravel()
        combined = 0.65 * sim_word + 0.35 * sim_char

        # Exact boost for requirement IDs (e.g. REQ-SYS-001, REQ-AEROD-010, DG-01) and table numbers
        req_codes = re.findall(r"REQ-[A-Z]+-\d+|DG-\d+", query.upper())
        table_refs = re.findall(r"tabela\s+\d+\.\d+|eq(?:uação|\.)?\s+\d+\.\d+|seção\s+\d+(?:\.\d+)+", query.lower())

        for idx, row in enumerate(self.chunks_df.itertuples()):
            text_upper = f"{row.section} {row.requirement_ids} {row.content}".upper()
            text_lower = text_upper.lower()
            for req in req_codes:
                if req in text_upper:
                    combined[idx] += 0.45
            for tref in table_refs:
                if tref in text_lower:
                    combined[idx] += 0.30

        order = np.argsort(combined)[::-1][:top_k]
        return self._format_citations(order, combined.tolist())

    def _format_citations(self, indices: Any, scores: list[float]) -> list[Citation]:
        citations: list[Citation] = []
        for idx in indices:
            score = float(scores[int(idx)])
            if score <= 0.0:
                continue
            row = self.chunks_df.iloc[int(idx)]
            reqs = json.loads(row["requirement_ids"]) if row["requirement_ids"] else []
            excerpt = str(row["content"]).strip()
            if len(excerpt) > 360:
                excerpt = excerpt[:357] + "..."
            citations.append(
                Citation(
                    chunk_id=str(row["chunk_id"]),
                    page=int(row["page"]),
                    section=str(row["section"]),
                    requirement_ids=reqs,
                    score=round(score, 4),
                    excerpt=excerpt,
                )
            )
        return citations

    def answer_question(self, query: str, top_k: int = 5) -> RAGResponse:
        """Retrieve relevant chunks, check abstention threshold, and synthesize a cited response."""
        citations = self.search_hybrid(query, top_k=top_k)
        top_score = citations[0].score if citations else 0.0
        confidence = round(float(min(top_score / 0.55, 1.0)), 4)

        if not citations or top_score < self.abstention_threshold:
            return RAGResponse(
                query=query,
                abstained=True,
                confidence=confidence,
                answer=(
                    "Abstenção: Não foram encontradas evidências suficientes no Relatório de Projeto 2026 "
                    "ou nos requisitos da competição SAE AeroDesign para responder a esta pergunta com segurança."
                ),
                citations=[],
            )

        best = citations[0]
        second = citations[1] if len(citations) > 1 and citations[1].score >= self.abstention_threshold else None

        req_str = f" ({', '.join(best.requirement_ids)})" if best.requirement_ids else ""
        parts = [
            f"Com base no Relatório de Projeto 2026 [Pág. {best.page}, Seção {best.section}{req_str}]: "
            f"{best.excerpt}"
        ]
        if second and second.chunk_id != best.chunk_id:
            req2_str = f" ({', '.join(second.requirement_ids)})" if second.requirement_ids else ""
            parts.append(
                f"Complemento [Pág. {second.page}, Seção {second.section}{req2_str}]: {second.excerpt}"
            )

        return RAGResponse(
            query=query,
            abstained=False,
            confidence=confidence,
            answer="\n\n".join(parts),
            citations=citations,
        )
