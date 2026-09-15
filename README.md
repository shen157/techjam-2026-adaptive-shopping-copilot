# Adaptive Shopping Copilot

### Stateful Hybrid Retrieval for Multi-Turn Product Search

Adaptive Shopping Copilot is a conversational product-search system that maintains evolving user preferences across multiple turns, combines lexical and semantic retrieval, and adapts when earlier recommendations are insufficient.

Built as a **solo project for TikTok TechJam 2026 — Track 4: Shopping Copilot**.

### Highlights

- **92.5% Hit@10** on the official 200-session public development set
- **TechnicalScore: 0.795744**, improved from a 0.106710 BM25 starter
- Stateful multi-turn preference and constraint tracking
- BM25 + MiniLM hybrid retrieval
- Adaptive clarification and failure-aware candidate recovery
- Fully local inference with **0 external API tokens**

---

## Demo & Links

- **Video Demo:** [https://youtu.be/0H4lH9e4XUU](https://www.youtube.com/watch?v=0H4lH9e4XUU)
- **Offline Runtime Assets:** https://github.com/shen157/techjam-2026-adaptive-shopping-copilot/releases/tag/v1

---

## System Architecture

The system separates conversation state, retrieval, ranking, clarification, and recommendation selection into modular stages.

<img width="1200" alt="figure_1_system_architecture" src="https://github.com/user-attachments/assets/0d01559a-671f-44d3-94a8-4ec5ee26e6e3" />

---

## Core Design

- **Stateful conversation:** tracks product category, hard constraints, soft preferences, asked/ignored attributes, and preference changes across turns.
- **Buying vs. browsing routing:** sessions with hard constraints use a precision-oriented buying route; otherwise the system remains exploratory.
- **Hybrid retrieval:** combines SQLite FTS5 BM25 with local MiniLM semantic retrieval.
- **Precision-preserving fusion:** protects BM25 Top 4 for buying and Top 8 for browsing before weighted Reciprocal Rank Fusion.
- **Constraint-aware reranking:** prioritizes hard-constraint matches, then soft-preference matches, then the original hybrid rank.
- **Adaptive clarification:** uses a stable early-turn order and candidate coverage × diversity for later questions.
- **Failure-aware recovery:** expands retrieval from 50 to 100 candidates from Turn 4 onward, with semantic diversification for late browsing sessions.

---

## Development Results

The system was developed through controlled public-development experiments.

<img width="1200" alt="figure_3_development_evolution" src="https://github.com/user-attachments/assets/b3474627-d9ff-4ae6-8fa8-5db5bd889203" />

The largest improvements came from adding conversational state, structured clarification, and constraint-aware ranking.

---

## What Did Not Work

Not every experiment was retained.

| Experiment | Result |
|---|---|
| Dense-only retrieval | Reduced ranking quality |
| Explicit product-price reranking | Reduced public-development performance |
| Direct profile product reranking | Reduced public-development performance |

These results shaped the final architecture rather than simply adding more features.

---

## Public Evaluation Results

Final results on the official 200-session public development set:

<img width="1200" alt="figure_2_public_development_results" src="https://github.com/user-attachments/assets/9d4f1e04-5436-445e-a068-f3d70f4bfa47" />

| Metric | Result |
|---|---:|
| Hit@10 | **0.925** |
| MRR | **0.606480** |
| TechnicalScore | **0.795744** |
| External API Tokens | **0** |

These are **public-development results only** and are not claims about the organizer's private evaluation set.

---

## Setup

The project was developed and tested with:

```text
Python 3.12.3
```

### 1. Clone the repository

```bash
git clone https://github.com/shen157/techjam-2026-adaptive-shopping-copilot.git
cd techjam-2026-adaptive-shopping-copilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the competition catalog

The frozen 50,000-product competition catalog is **not redistributed** in this repository.

Obtain it through the official TechJam participant-kit instructions and place it at:

```text
data/catalog.jsonl
```

### 4. Download offline runtime assets

Download `runtime_assets.zip` from:

https://github.com/shen157/techjam-2026-adaptive-shopping-copilot/releases/tag/v1

Extract it into the repository root.

The following paths should then exist:

```text
data/dense_embeddings.npy
models/all-MiniLM-L6-v2/
```

### 5. Run the public evaluator

```bash
python -m evaluator.local_evaluator
```

Expected result:

```text
Hit@10        = 0.925
MRR           = 0.606480
MTTC          = 3.435
Efficiency    = 0.7565
TechnicalScore = 0.795744
```

---

## Limitations

- Intent and slot parsing is lightweight and rule-based, so unseen phrasing can still be misinterpreted.
- Constraint reranking relies mainly on lexical evidence rather than full semantic entailment.
- Offline embeddings and semantic clusters require refresh when the product catalog changes.

For production, I would explore stronger semantic parsing, ANN retrieval, multilingual support, learned clarification policies, monitoring, and online A/B testing.

---

## Contribution

This is a **solo project**. I designed and implemented the conversational state system, hybrid retrieval and ranking pipeline, adaptive clarification, failure-aware recovery, semantic diversification, ablation experiments, offline reproducibility workflow, and project documentation.

---

## Data & Model Attribution

The competition data is derived from **Amazon Reviews 2023** and distributed through the official TechJam participant kit. Semantic retrieval uses `sentence-transformers/all-MiniLM-L6-v2`.

See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for detailed attribution.
