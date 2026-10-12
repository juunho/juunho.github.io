---
title: Customer-support AI agent for T world
kind: Retrieval & Agent
period: 2024 — 2026
order: 40
summary: >-
  Owned retrieval and the support agent from beta to general availability
  (about 600k monthly users), and led the 2025 improvement track. Trained a
  domain embedding model with hard-negative mining (MRR@5 0.84 → 0.96,
  search failures −89%), added hybrid sparse–dense search, moved inference
  to quantized ONNX on CPU (2.7× faster, under 0.1% quality loss, roughly 68%
  lower cost), and swapped the cross-encoder reranker for an LLM reranker that
  takes new domain cues without retraining.
stack:
  - PyTorch
  - ONNX Runtime
  - LangGraph
  - vLLM
link:
  href: /posts/embedding-model-contrastive-learning/
  label: How the embedding model was trained
---
