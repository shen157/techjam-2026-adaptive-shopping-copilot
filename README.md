# Adaptive Shopping Copilot

### Failure-Aware Hybrid Conversational Retrieval for E-Commerce

Adaptive Shopping Copilot is a conversational product-search agent developed for
TikTok TechJam 2026 Track 4: Shopping Copilot — AI Conversational Search and
Recommendations.

The system is designed to distinguish high-intent buying from exploratory
browsing, maintain user preferences across multiple turns, handle changing
requirements, ask useful clarification questions, and adapt its retrieval
strategy when earlier recommendations fail.

The final system achieves:

| Metric | Public Development Set |
| --- | ---: |
| Hit Rate@10 | 0.925 |
| MRR | 0.606480 |
| MTTC | 3.435 |
| Efficiency | 0.7565 |
| TechnicalScore | 0.795744 |
| API Token Usage | 0 |

The official weak BM25 starter achieves a TechnicalScore of approximately
0.106710 on the same 200-session public development set.

---

## 1. Problem

Traditional keyword search works well when users know exactly what they want,
but conversational shopping often involves incomplete, evolving, or ambiguous
preferences.

A shopper may begin with an exploratory request, gradually reveal constraints,
reject an attribute, or completely change an earlier preference.

Adaptive Shopping Copilot addresses this problem with a stateful and
runtime-adaptive retrieval pipeline rather than a single static search query.

The system focuses on four challenges:

1. Distinguishing precision-oriented buying from exploration-oriented browsing.
2. Accumulating and updating preferences across multiple conversation turns.
3. Recovering when early recommendations do not find the intended product.
4. Balancing ranking precision with semantic exploration.

---

## 2. System Architecture

The agent uses the following high-level pipeline:

```text
User Message
     |
     v
Conversation State Tracker
     |
     v
Intent / Constraint Routing
     |
     +-------------------+
     |                   |
     v                   v
Buying Mode          Browsing Mode
Precision            Exploration
     |                   |
     +---------+---------+
               |
               v
        Hybrid Retrieval
       BM25 + MiniLM Dense
               |
               v
        Weighted RRF Fusion
               |
               v
     Constraint-Aware Reranking
       Hard > Soft > Base Rank
               |
               v
      Adaptive Clarification
               |
               v
        Repeated Failure?
          Turn >= 4
               |
               v
     Retrieval Pool Expansion
          Top 50 -> Top 100
               |
               v
      Browsing / Low-Constraint?
               |
               v
   Semantic Cluster Diversification
      Top-1 Precision Anchor
      Diverse Exploration Tail
               |
               v
        Top-10 Recommendations
```

A lightweight long-term profile prior is also used when two clarification
questions have nearly equal information value. Current-session evidence always
takes priority over historical profile signals.

---

## 3. Conversational State Tracking

Each session maintains an isolated state containing:
- detected product category;
- structured preference slots;
- hard constraints;
- soft preference history;
- previously asked attributes;
- attributes explicitly rejected or marked as unimportant;
- the most recent clarification attribute;
- weak profile-derived clarification preferences.
Preferences accumulate over turns rather than rebuilding the session from only
the latest message.
The parser also handles intent changes such as preference overrides and several
equivalent no-preference expressions.
For example, an earlier soft preference can be removed when the user changes
their mind, while a boundary response such as having no preference for an
attribute prevents the system from repeatedly asking about that attribute.

---

## 4. Buying vs Browsing Routing

The agent uses the current conversational state to choose a retrieval strategy.

### Buying mode

When hard constraints are available, the session is treated as
precision-oriented.
The system protects more of the strongest lexical candidates and prioritizes
products satisfying accumulated constraints.

### Browsing mode

When no hard constraint has yet been established, the system treats the session
as exploratory.
Browsing allows more semantic exploration and can activate diversity-aware
recovery during difficult late-turn sessions.
This routing is based on runtime state rather than on access to the evaluator's
hidden scenario label.

---

## 5. Hybrid Retrieval

The retrieval system combines two complementary signals.
### BM25 lexical retrieval
SQLite FTS5 provides an in-memory lexical search index over product metadata,
including:
- title;
- categories;
- features;
- details;
- store;
- description.
Field-specific BM25 weights emphasize the most informative product metadata.
### Dense semantic retrieval
The system uses the local
sentence-transformers/all-MiniLM-L6-v2 sentence encoder to generate
384-dimensional query embeddings.
Catalog embeddings are precomputed once and stored locally, allowing runtime
retrieval to use efficient NumPy similarity computation rather than repeatedly
encoding all 50,000 products.
### Rank fusion
BM25 and dense results are combined using weighted Reciprocal Rank Fusion
(RRF).
The lexical signal remains dominant while the dense signal provides semantic
recall for less exact wording.

---

## 6. Constraint-Aware Reranking

Retrieval candidates are reranked using accumulated conversational evidence.
The ordering follows:

```text
Hard-constraint matches
        >
Soft-preference matches
        >
Original hybrid retrieval rank
```

This design keeps explicit current-session requirements stronger than weaker
historical preferences.
Several alternative ranking strategies were tested during development,
including direct price-aware and profile-aware product reranking. They were not
retained because they reduced public-set ranking quality.

---

## 7. Adaptive Clarification

The agent can recommend products and ask one structured clarification question
in the same turn.
During early turns, the system uses a stable clarification order to obtain useful
constraints quickly.
During later turns, it estimates the value of candidate attributes using:

```text
Question Value = Candidate Coverage x Attribute Diversity
```

The system therefore prefers questions that are both well represented in the
candidate pool and capable of separating candidate products.
Long-term profile information is used only as a weak tie-break when multiple
clarification attributes have nearly equal information value.

---

## 8. Failure-Aware Retrieval Recovery

A key design insight is that continued interaction itself provides useful
information.
If the conversation reaches Turn 4, previous recommendation rounds have not yet
produced a conversion in the evaluation setting.
The agent therefore treats continued interaction as an implicit failure signal
and changes its workflow:

```text
Turns 1-3:
    Retrieve up to 50 candidates

Turn 4+:
    Retrieve up to 100 candidates
```
This increases recall only when the additional search depth is needed instead
of paying the cost on every interaction.

---

## 9. Semantic Browsing Diversity

Late-stage exploratory sessions can suffer from recommendation redundancy:
many highly ranked products may be semantically very similar.
To reduce this problem, the 50,000 catalog-product embeddings are grouped into
128 semantic clusters offline using MiniBatchKMeans.
For browsing-mode recovery sessions:

```text
Rank 1:
    Preserved as a precision anchor

Ranks 2-10:
    Prefer candidates from previously unseen semantic clusters
```
If insufficient unique clusters are available, remaining positions are filled
using the original ranking.
This creates a deliberate precision-exploration trade-off only after earlier
recommendations have failed.
A small sensitivity study on the public development set compared different
protected-prefix sizes. Protecting only the first result produced the strongest
observed TechnicalScore while preserving Hit Rate@10.

---

## 10. Offline and Failure-Safe Execution

The final system does not require a paid LLM API.
For the official offline configuration, the MiniLM encoder is loaded from a
local model directory and requires no network access.
The public repository may omit large binary runtime assets depending on
distribution constraints. See the setup instructions for the required asset
layout.
Runtime behavior follows:

```text
Local MiniLM available
    -> BM25 + Dense hybrid retrieval

Dense model or dense assets unavailable
    -> BM25-only fallback
```
The fallback path was tested independently and completes all 200 public
development sessions without failure.
The full hybrid pipeline was also tested with Hugging Face and Transformers
forced into offline mode and reproduced the same final public-set metrics.
No external API calls are required during evaluation.

---

## 11.Latency and Cost

The final offline configuration was benchmarked by running the complete
official 200-session public evaluator.

```text
Total evaluation runtime   = 209.70 seconds
Average runtime per session = 1.05 seconds
Amortized runtime per agent turn = 0.31 seconds
Reported prompt tokens      = 0
Reported completion tokens  = 0
Estimated external API cost = $0
Network access required     = No
```

The benchmark includes model initialization, local retrieval-index setup,
agent execution, and evaluator overhead, so the amortized per-turn value
should not be interpreted as isolated model-inference latency.

Benchmark hardware: Intel(R) Core(TM) Ultra 5 125H.

Runtime measurements are hardware-dependent.

---

## 12. Public Evaluation Results

Final V15 results on the official 200-session public development set:
| Scenario | Samples | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 0.9000 | 0.654167 | 6.6000 |
| Browsing | 80 | 0.9625 | 0.602684 | 2.9500 |
| Buying | 80 | 0.9375 | 0.578968 | 2.7250 |
| Intent Override | 30 | 0.8000 | 0.674074 | 5.566667 |
| **Overall** | **200** | **0.9250** | **0.606480** | **3.4350** |


Overall:

```text
Efficiency        = 0.7565
TechnicalScore    = 0.795744
Prompt tokens     = 0
Completion tokens = 0
External API cost = $0
```

These results are development-set measurements only and are not claims about the
organizer's private 800-session evaluation set.

---

## 13. Selected Ablation Results

The system was developed through controlled experiments rather than retaining
every attempted feature.
| Experiment | Hit@10 | MRR | MTTC | TechnicalScore | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Official weak BM25 starter | 0.125 | 0.068034 | 9.810 | 0.106710 | Baseline |
| Conversation memory | 0.270 | 0.151381 | 8.600 | 0.228414 | Keep |
| Structured state + clarification | 0.830 | 0.535266 | 4.500 | 0.705580 | Keep |
| Dense-only retrieval | 0.445 | 0.202817 | 7.320 | 0.356945 | Reject |
| Constraint-aware reranking | 0.915 | 0.594722 | 3.510 | 0.785717 | Keep |
| Direct profile product reranking | 0.890 | 0.522829 | 3.895 | 0.743949 | Reject |
| Late candidate-pool recovery | 0.925 | 0.605694 | 3.465 | 0.794908 | Keep |
| **Final semantic diversity + profile tie-break** | **0.925** | **0.606480** | **3.435** | **0.795744** | **Final** |


Rejected experiments were retained as design evidence rather than being hidden
from the development history.

---

## 14. Project Structure

```text
starter/
    agent.py
        Final conversational shopping agent.

evaluator/
    local_evaluator.py
        Official deterministic public evaluator.

data/
    public_set.jsonl
        Official 200-session development set.

    dense_embeddings.npy
        Precomputed MiniLM embeddings for the frozen catalog.

    dense_asins.json
        ASIN ordering corresponding to dense embeddings.

    product_clusters.json
        Offline semantic cluster assignments.

models/
    all-MiniLM-L6-v2/
        Local SentenceTransformer model used by the full offline submission.
        Large model files may be omitted from the public Git repository.

experiments/
    ...
        Saved experiment metrics and selected code snapshots.

build_clusters.py
    Offline MiniBatchKMeans cluster-generation script.

DATA_ATTRIBUTION.md
    Data and third-party asset attribution.
```

The organizer-provided frozen catalog is kept read-only.

---

## 15. Setup

Python 3.10 or later is recommended.

The final submission was developed and tested with Python 3.12.3.

Install project dependencies:

```bash
pip install -r requirements.txt
```

Download the official frozen competition catalog using the participant-kit
release instructions and place it at:

```text
data/catalog.jsonl
```

The product catalog is intentionally not modified by this project.

The fully offline competition submission includes the local MiniLM model and
precomputed dense embeddings. These large binary assets may be omitted from the
public Git repository to keep the repository lightweight.

---

## 16. Reproduce the Final Result

From the repository root, run:

```bash
python -m evaluator.local_evaluator
```

The evaluator writes aggregate and per-session results to:

```text
results.json
```

Expected public-development result for the final configuration:

```text
Hit Rate@10     = 0.925
MRR             = 0.606480
MTTC            = 3.435
Efficiency      = 0.7565
TechnicalScore  = 0.795744
```

Because the evaluator and public development sessions are deterministic, the
same environment and assets should reproduce the reported result.

---

## 17. Optional Runtime Controls

Dense retrieval can be disabled for fallback testing:

```bash
DISABLE_DENSE=1
```
If the bundled local model is unavailable, Hugging Face Hub access is disabled
by default. Development environments may explicitly permit Hub loading through
the project's optional runtime flag.
The official final configuration uses the bundled local model and requires no
network access.

---

## 18. Tools and Libraries

Core runtime components:
- Python
- SQLite FTS5
- NumPy
- Sentence Transformers
- Hugging Face Transformers runtime dependencies
Offline asset generation additionally uses scikit-learn for MiniBatchKMeans.
No external paid model API is used during inference.

---

## 19. Limitations

The current system has several limitations.
First, the intent and slot parser is intentionally lightweight and rule based.
Although common preference-override and no-preference paraphrases are supported,
unseen linguistic structures may still be interpreted incorrectly.
Second, hard and soft constraint reranking relies largely on lexical evidence in
catalog metadata. It does not perform full natural-language entailment between
every user constraint and every product.
Third, semantic clusters are generated offline from a frozen catalog. A
production catalog with frequent additions or deletions would require periodic
embedding and cluster refreshes.
Fourth, long-term user-profile information is deliberately used only as a weak
clarification prior. Direct product-level personalization was tested but
rejected because it could conflict with stronger current-session intent.
Finally, all reported results are based on the released 200-session public
development set. Performance on unseen users and products may differ.
Given more development time, future work would include stronger semantic
constraint parsing, uncertainty-aware intent routing, learned clarification
policies, dynamic cluster refresh, and evaluation on broader conversational
language variations.

---

## 20. Data and Model Attribution

The competition catalog and public evaluation sessions are derived from Amazon
Reviews 2023 and distributed through the official TechJam participant kit.
The local semantic encoder is
`sentence-transformers/all-MiniLM-L6-v2`, used for semantic search and offline
product clustering.
See `DATA_ATTRIBUTION.md` for source and usage details.

---

## 21. Contribution

This is a solo submission.
The participant is responsible for system architecture, implementation
integration, experiment design, local evaluation, ablation analysis,
reproducibility testing, and submission documentation.