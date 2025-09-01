from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(50), default="rss")
    url: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    articles: Mapped[List[Article]] = relationship("Article", back_populates="source")


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    url: Mapped[str] = mapped_column(String(2048))
    canonical_url: Mapped[str | None] = mapped_column(String(2048))
    url_hash: Mapped[str] = mapped_column(String(64), index=True)

    title: Mapped[str | None] = mapped_column(String(1024))
    content: Mapped[str | None] = mapped_column(Text())
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    language: Mapped[str | None] = mapped_column(String(16))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    source: Mapped[Source | None] = relationship("Source", back_populates="articles")
    predictions: Mapped[List[Prediction]] = relationship("Prediction", back_populates="article", cascade="all, delete-orphan")
    labels: Mapped[List[Label]] = relationship("Label", back_populates="article", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("url_hash", name="uq_articles_url_hash"),
        Index("ix_articles_source_published", "source_id", "published_at"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    task: Mapped[str] = mapped_column(String(64))  # e.g., sentiment, emotion, intent
    label: Mapped[str] = mapped_column(String(128))
    score: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    article: Mapped[Article] = relationship("Article", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_article_task", "article_id", "task"),
    )


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    task: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(128))
    annotator: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    article: Mapped[Article] = relationship("Article", back_populates="labels")

    __table_args__ = (
        Index("ix_labels_article_task", "article_id", "task"),
    )

