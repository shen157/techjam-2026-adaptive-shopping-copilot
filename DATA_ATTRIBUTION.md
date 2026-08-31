# Data and Model Attribution

This project was developed for TikTok TechJam 2026 Track 4:
**Shopping Copilot — AI Conversational Search and Recommendations**.

The project uses the organizer-provided frozen competition catalog and public
evaluation sessions together with locally generated semantic retrieval assets.

---

## 1. Competition Dataset

The competition dataset is derived from **Amazon Reviews 2023**, published by
McAuley Lab at UCSD.

- Source dataset: Amazon Reviews 2023
- Selected category: `Clothing_Shoes_and_Jewelry`
- Product identifier: `parent_asin`
- Competition modality: text and structured product metadata
- Frozen competition catalog size: 50,000 products
- Public development sessions: 200
- Private organizer evaluation sessions: not included in this repository

Original dataset documentation:

https://amazon-reviews-2023.github.io/

The competition catalog and public evaluation sessions were prepared and frozen
by the TechJam organizers.

This project does not modify the organizer-provided product catalog.

---

## 2. Competition Assets

The following assets originate from the official participant kit:

```text
data/public_set.jsonl
docs/competition_specification.md
docs/agent_api_contract.json
docs/evaluation_config.json
docs/baseline_results.json
evaluator/
```

The frozen product catalog is distributed separately according to the official
participant-kit instructions and is expected at:

`data/catalog.jsonl`
The organizer's private evaluation sessions and organizer-only materials are not
included in this project repository.

---

## 3. Sentence Encoder

Semantic retrieval uses:

`sentence-transformers/all-MiniLM-L6-v2`

The model is used only as a local sentence encoder.

It converts user search queries and product text into 384-dimensional semantic
embeddings.

No external model API is required during inference.

For reproducible offline execution, the model is stored locally under:

`models/all-MiniLM-L6-v2/`

Original model source:

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Users redistributing the model should review and follow the licensing and usage
terms published with the original model.

---

## 4. Derived Dense Retrieval Assets

The following files are generated from the frozen competition catalog using the
MiniLM sentence encoder:

```text
data/dense_embeddings.npy
data/dense_asins.json
```

`dense_embeddings.npy` contains the precomputed semantic embeddings used for
runtime dense retrieval.

`dense_asins.json` preserves the `parent_asin` ordering corresponding to the
embedding matrix.

These files are derived retrieval assets and do not introduce additional
products or modify catalog records.

---

## 5. Semantic Cluster Asset

The project additionally generates:

`data/product_clusters.json`

This file assigns each frozen catalog product to an offline semantic cluster.

Clusters are produced using MiniBatchKMeans over the precomputed MiniLM product
embeddings.

The clustering step is implemented in:

`build_clusters.py`

The final system uses these clusters only for diversity-aware recommendation
selection during late-stage exploratory sessions.

No mock ASINs, synthetic catalog entries, or catalog mutations are introduced.

---

## 6. Public Evaluation Results

All reported development metrics are computed using the official deterministic
local evaluator and the released 200-session public development set.
Reported results do not represent or claim performance on the organizer's
private 800-session evaluation set.

---

## 7. Data Use

The underlying Amazon-derived product information remains subject to the
applicable terms and conditions of the original dataset and the competition
participant kit.
This project claims no ownership over the underlying Amazon product content.
Derived embeddings and cluster assignments are used solely to support the
competition retrieval pipeline and related technical analysis.