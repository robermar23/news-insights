from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from transformers import pipeline, AutoTokenizer

from ..config import get_settings, resolve_zero_shot_labels
from ..db.models import Article, Prediction

logger = logging.getLogger("news_insights")


@dataclass
class Analyzer:
    sentiment_model: str
    emotion_model: str
    zeroshot_model: str
    candidate_labels: List[str]

    # Chunking configuration
    enable_chunking: bool
    max_tokens_per_chunk: Optional[int]
    chunk_overlap: int
    long_doc_agg: str  # "mean" or "max"
    max_chunks_per_doc: Optional[int]

    # Runtime pipeline instances (typed as Any for compatibility with HF stubs)
    sentiment: Any = field(init=False)
    emotion: Any = field(init=False)
    zeroshot: Any = field(init=False)

    # Tokenizers (for chunking)
    tokenizer_sent: Any = field(init=False, default=None)
    tokenizer_emo: Any = field(init=False, default=None)
    tokenizer_zs: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        logger.info("Loading transformers pipelines (this may take a while on first run)...")
        # mypy: transformers pipeline typing is not precise; ignore overload resolution
        self.sentiment = pipeline("sentiment-analysis", model=self.sentiment_model)  # type: ignore[call-overload]
        self.emotion = pipeline("text-classification", model=self.emotion_model, top_k=None)  # type: ignore[call-overload]
        self.zeroshot = pipeline("zero-shot-classification", model=self.zeroshot_model)  # type: ignore[call-overload]

        if self.enable_chunking:
            # Initialize fast tokenizers for token-based chunking
            try:
                self.tokenizer_sent = AutoTokenizer.from_pretrained(self.sentiment_model, use_fast=True)
            except Exception:
                self.tokenizer_sent = None
            try:
                self.tokenizer_emo = AutoTokenizer.from_pretrained(self.emotion_model, use_fast=True)
            except Exception:
                self.tokenizer_emo = None
            try:
                self.tokenizer_zs = AutoTokenizer.from_pretrained(self.zeroshot_model, use_fast=True)
            except Exception:
                self.tokenizer_zs = None

    @classmethod
    def from_settings(
        cls,
        candidate_labels: Optional[List[str]] = None,
        enable_chunking: Optional[bool] = None,
        max_tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        long_doc_agg: Optional[str] = None,
        max_chunks_per_doc: Optional[int] = None,
    ) -> "Analyzer":
        s = get_settings()
        labels = candidate_labels or resolve_zero_shot_labels(s.zero_shot_preset)
        return cls(
            s.hf_model_sentiment,
            s.hf_model_emotion,
            s.hf_model_zeroshot,
            labels,
            enable_chunking if enable_chunking is not None else s.enable_chunking,
            max_tokens_per_chunk if max_tokens_per_chunk is not None else s.max_tokens_per_chunk,
            chunk_overlap if chunk_overlap is not None else s.chunk_overlap,
            (long_doc_agg or s.long_doc_agg).lower(),
            max_chunks_per_doc if max_chunks_per_doc is not None else s.max_chunks_per_doc,
        )

    def analyze_text(self, text: str) -> Dict[str, Dict[str, float]]:
        if self.enable_chunking:
            return self._analyze_long_text(text)

        # Non-chunked path: single-shot
        sent = self.sentiment(text, truncation=True)
        sent_label = sent[0]["label"].lower()
        sent_score = float(sent[0]["score"]) if isinstance(sent[0], dict) else 0.0

        emo_raw = self.emotion(text, truncation=True)
        emo_scores: Dict[str, float] = {}
        if isinstance(emo_raw, list) and len(emo_raw) > 0 and isinstance(emo_raw[0], dict) and "label" in emo_raw[0]:
            for item in emo_raw:
                emo_scores[item["label"].lower()] = float(item["score"])  # type: ignore[index]
        elif isinstance(emo_raw, list) and len(emo_raw) > 0 and isinstance(emo_raw[0], list):
            for item in emo_raw[0]:
                emo_scores[item["label"].lower()] = float(item["score"])  # type: ignore[index]

        zs = self.zeroshot(text, candidate_labels=self.candidate_labels, multi_label=True)
        intent_scores: Dict[str, float] = {}
        for label, score in zip(zs["labels"], zs["scores"]):
            intent_scores[label.lower()] = float(score)

        return {
            "sentiment": {sent_label: sent_score},
            "emotion": emo_scores,
            "intent": intent_scores,
        }

    def predict_and_store(self, session: Session, article: Article) -> None:
        results = self.analyze_text(article.content or article.title or "")

        # Persist predictions
        for task, scores in results.items():
            for label, score in scores.items():
                session.add(
                    Prediction(
                        article_id=article.id,
                        task=task,
                        label=label,
                        score=score,
                        model_name=self._model_name_for(task),
                    )
                )

    def _model_name_for(self, task: str) -> str:
        def get_name(p: Any) -> str:
            try:
                m = getattr(p, "model", None)
                if m is not None:
                    name = getattr(m, "name_or_path", None)
                    if isinstance(name, str):
                        return name
                # Some pipelines expose model name differently
                name2 = getattr(p, "model_name", None)
                if isinstance(name2, str):
                    return name2
            except Exception:
                pass
            return "unknown"

        if task == "sentiment":
            return get_name(self.sentiment)
        if task == "emotion":
            return get_name(self.emotion)
        if task == "intent":
            return get_name(self.zeroshot)
        return "unknown"

    # ---- Long document handling ----
    def _safe_max_len(self, tok: Any) -> int:
        if tok is None:
            return 512
        # Many tokenizers set a very large sentinel when unknown; treat >1e6 as unknown
        m = getattr(tok, "model_max_length", None)
        if isinstance(m, int) and m > 0 and m < 1_000_000:
            # Safety margin to avoid special token overflow
            return max(32, m - 16)
        return 512

    def _chunk_text_by_tokens(self, text: str, tok: Any, max_tokens: int, overlap: int) -> List[str]:
        if tok is None:
            # Fallback: naive char-based splitting approx by 4 chars per token
            approx = max_tokens * 4
            stride_chars = max(1, approx - overlap * 4)
            out: List[str] = []
            start = 0
            while start < len(text):
                end = min(len(text), start + approx)
                out.append(text[start:end])
                if end == len(text):
                    break
                start += stride_chars
            return out
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        offsets = enc.get("offset_mapping") or []
        if not offsets:
            return [text] if text else []
        out2: List[str] = []
        start_idx = 0
        stride = max_tokens - overlap if max_tokens > overlap else max_tokens
        n_tokens = len(offsets)
        while start_idx < n_tokens:
            end_idx = min(start_idx + max_tokens, n_tokens)
            start_char = offsets[start_idx][0]
            end_char = offsets[end_idx - 1][1]
            out2.append(text[start_char:end_char])
            if end_idx >= n_tokens:
                break
            start_idx += stride
        return out2

    def _agg_scores(self, dicts: List[Dict[str, float]], mode: str) -> Dict[str, float]:
        if not dicts:
            return {}
        keys = set().union(*(d.keys() for d in dicts))
        out: Dict[str, float] = {}
        if mode == "max":
            for k in keys:
                out[k] = max(float(d.get(k, 0.0)) for d in dicts)
        else:  # mean
            n = len(dicts)
            for k in keys:
                out[k] = sum(float(d.get(k, 0.0)) for d in dicts) / n
        return out

    def _analyze_long_text(self, text: str) -> Dict[str, Dict[str, float]]:
        # Resolve per-task chunk sizes
        max_len_common = self.max_tokens_per_chunk
        max_sent = max_len_common or self._safe_max_len(self.tokenizer_sent)
        max_emo = max_len_common or self._safe_max_len(self.tokenizer_emo)
        max_zs = max_len_common or self._safe_max_len(self.tokenizer_zs)
        ov = max(0, int(self.chunk_overlap))

        chunks_sent = self._chunk_text_by_tokens(text, self.tokenizer_sent, max_sent, ov)
        chunks_emo = self._chunk_text_by_tokens(text, self.tokenizer_emo, max_emo, ov)
        chunks_zs = self._chunk_text_by_tokens(text, self.tokenizer_zs, max_zs, ov)

        # Optionally cap number of chunks per doc
        if isinstance(self.max_chunks_per_doc, int) and self.max_chunks_per_doc > 0:
            chunks_sent = chunks_sent[: self.max_chunks_per_doc]
            chunks_emo = chunks_emo[: self.max_chunks_per_doc]
            chunks_zs = chunks_zs[: self.max_chunks_per_doc]

        # Sentiment: need per-class scores; use return_all_scores=True
        sent_norm: List[Dict[str, float]] = []
        if chunks_sent:
            raw = self.sentiment(
                chunks_sent,
                truncation=True,
                max_length=max_sent,
                return_all_scores=True,
            )  # type: ignore[call-overload]
            for item in raw:
                # item is list[{'label','score'}]
                d = {e["label"].lower(): float(e["score"]) for e in item}
                sent_norm.append(d)
        sent_scores = self._agg_scores(sent_norm, self.long_doc_agg)
        # keep only winning class for 'sentiment' to match previous shape
        if sent_scores:
            best = max(sent_scores, key=lambda k: sent_scores[k])
            sent_scores = {best: sent_scores[best]}

        # Emotion: top_k=None already returns all classes per chunk
        emo_norm: List[Dict[str, float]] = []
        if chunks_emo:
            raw_emo = self.emotion(
                chunks_emo,
                truncation=True,
                max_length=max_emo,
                top_k=None,
            )  # type: ignore[call-overload]
            for item in raw_emo:
                d = {e["label"].lower(): float(e["score"]) for e in item}
                emo_norm.append(d)
        emo_scores = self._agg_scores(emo_norm, self.long_doc_agg)

        # Zero-shot intent
        intent_norm: List[Dict[str, float]] = []
        if chunks_zs:
            raw_zs = self.zeroshot(
                chunks_zs,
                candidate_labels=self.candidate_labels,
                multi_label=True,
            )  # type: ignore[call-overload]
            if isinstance(raw_zs, dict):
                raw_zs = [raw_zs]
            for item in raw_zs:
                d = {lbl.lower(): float(score) for lbl, score in zip(item["labels"], item["scores"])}
                intent_norm.append(d)
        intent_scores = self._agg_scores(intent_norm, self.long_doc_agg)

        return {
            "sentiment": sent_scores,
            "emotion": emo_scores,
            "intent": intent_scores,
        }


def predict_pending(
    session: Session,
    limit: Optional[int] = None,
    candidate_labels: Optional[List[str]] = None,
    preset: Optional[str] = None,
    # Chunking overrides
    chunking: Optional[bool] = None,
    max_tokens_per_chunk: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    agg: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> int:
    labels = resolve_zero_shot_labels(preset=preset, labels=candidate_labels)
    analyzer = Analyzer.from_settings(
        candidate_labels=labels,
        enable_chunking=chunking,
        max_tokens_per_chunk=max_tokens_per_chunk,
        chunk_overlap=chunk_overlap,
        long_doc_agg=(agg.lower() if isinstance(agg, str) else None),
        max_chunks_per_doc=max_chunks,
    )

    # Find articles without any predictions
    subq = select(Prediction.article_id)
    q = select(Article).where(~Article.id.in_(subq)).order_by(Article.published_at.desc().nullslast())
    if limit:
        q = q.limit(limit)

    articles = list(session.scalars(q))
    logger.info("Predicting for %d articles...", len(articles))

    cnt = 0
    for art in articles:
        analyzer.predict_and_store(session, art)
        cnt += 1
        if limit and cnt >= limit:
            break
    return cnt

