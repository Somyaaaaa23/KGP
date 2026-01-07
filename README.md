Perfect. Below is a **tightened, sharper, submission-ready README**.
Same substance, **less fluff**, clearer structure, stronger wording — exactly what evaluators like.

You can paste this **as-is** into your repo.

---

# 📘 Track B — BDH-Driven Continuous Narrative Reasoning

**Kharagpur Data Science Hackathon 2026**

## 🌊 **Now with Pathway Integration!**

This project now uses **[Pathway](https://github.com/pathwaycom/pathway)** - a Python framework for stream processing and incremental computation - perfectly aligned with Track B requirements for continuous, incremental reasoning.

## Overview

This project tackles the core challenge of the Kharagpur Data Science Hackathon 2026:
**determining whether a hypothetical character backstory is causally and logically consistent with a full-length narrative (novel).**

The solution follows **Track B**, emphasizing **continuous reasoning over long contexts**, inspired by **Baby Dragon Hatchling (BDH)** principles such as persistent internal state, incremental belief updates, and long-term constraint tracking.

Rather than relying on one-shot retrieval or surface plausibility, the system reads narratives progressively, builds a structured narrative memory, and evaluates whether the proposed backstory could realistically lead to the observed future events.

### ✨ Key Technologies

- **PyTorch** - Neural model training and inference
- **Sentence Transformers** - Semantic text encoding
- **Pathway** - Incremental stream processing *(Track B requirement!)*
- **Python 3.10+** - Development framework

---

## Problem Definition

### Task

Given:

* A complete novel (100k+ words, untruncated)
* A newly written, plausible hypothetical backstory for one central character

Predict whether the backstory is:

* **Consistent** with the narrative
* **Inconsistent** due to logical, causal, or constraint violations

### Output Labels

| Label | Meaning                   |
| ----- | ------------------------- |
| 1     | Backstory is consistent   |
| 0     | Backstory is inconsistent |

This is a **binary classification task** requiring **global narrative reasoning**, not local text matching.

---

## Why Track B

Track B focuses on *how* reasoning is formed across time.

This project explicitly models:

* Persistent narrative memory
* Incremental belief and constraint updates
* Long-term causal consistency
* **Pathway-powered differential computation**

The approach is **BDH-inspired**, but lightweight and practical, prioritizing reasoning quality over architectural scale.

### 🌊 Pathway Integration

**Pathway** is a Python framework for stream processing that enables:

- **Incremental Computation**: Only processes data changes (not full recomputation)
- **Stateful Operations**: Maintains persistent memory across stream updates
- **Differential Dataflow**: Automatically optimizes computation graph
- **Unified Batch & Streaming**: Same code for both modes
- **Real-time Processing**: Native support for continuous data streams

**Why it matters for Track B:**
- Persistent state ✓ (Pathway reducers)
- Incremental updates ✓ (Differential dataflow)
- Long-term tracking ✓ (Stateful groupby/reduce)
- Continuous reasoning ✓ (Stream processing)
- Efficient computation ✓ (Only recomputes changes)

See `PATHWAY_GUIDE.md` for detailed documentation.

---

## Dataset Description

### Files Provided

* `train.csv`
* `test.csv`
* `novels/` directory containing full `.txt` novels

### `train.csv`

| Column       | Description                      |
| ------------ | -------------------------------- |
| `story_id`   | Unique narrative identifier      |
| `novel_path` | Path to full novel text          |
| `backstory`  | Hypothetical character backstory |
| `label`      | Ground truth (1 or 0)            |

### `test.csv`

| Column       | Description            |
| ------------ | ---------------------- |
| `story_id`   | Unique identifier      |
| `novel_path` | Path to novel text     |
| `backstory`  | Hypothetical backstory |

**Notes**

* Test labels are hidden
* Novels are complete and untruncated
* No summaries are provided or allowed

---

## Input & Output

### Input

For each example:

1. Full novel text (`.txt`)
2. Hypothetical backstory (text)

### Output

* `results.csv` with one row per test example

Example:

```csv
story_id,prediction
001,1
002,0
```

---

## System Architecture (Track B)

### High-Level Flow

```
Novel → Chunking → Encoder → Persistent Narrative Memory
                                      ↓
                              Backstory Comparison
                                      ↓
                               Binary Prediction
```

---

## Detailed Methodology

### 1. Narrative Chunking

* Novel is split into ordered chunks (chapter-wise or fixed length)
* Temporal order is strictly preserved

Purpose:

* Enable incremental reading
* Maintain long-range dependencies

---

### 2. Chunk Encoding

Each chunk is encoded into a semantic representation using lightweight encoders or BDH-style representations.

Encodings are **not used independently**, but to update a shared narrative state.

---

### 3. Persistent Narrative Memory (Core Component)

A structured **Narrative State** is maintained across the entire novel.

It stores:

* Observed character traits
* Key events
* Motivations and belief shifts
* Constraints (facts that must remain true)
* Ruled-out possibilities

This memory:

* Persists across chunks
* Updates selectively
* Mimics BDH’s continuous internal state

---

### 4. Incremental Belief Updates

For each chunk:

* Identify new or conflicting information
* Update only affected beliefs or constraints
* Avoid full reprocessing

This ensures:

* Sparse updates
* Temporal consistency
* Continuous reasoning

---

### 5. Backstory Consistency Check

After processing the full novel:

* Final narrative state is compared with the backstory

Checks include:

* Logical contradictions
* Violated constraints
* Unsupported motivations
* Causal incompatibilities

---

### 6. Classification

A lightweight classifier outputs:

* `1` if backstory fits all narrative constraints
* `0` if major inconsistencies exist

Evidence rationale is optional (as per Track B rules).

---

## Training Strategy

* Train on `train.csv`
* Learn associations between narrative memory patterns and backstory validity
* Focus on robustness and generalization over raw accuracy

---

## Evaluation Focus

Aligned with Track B evaluation criteria:

* Correctness and robustness
* Meaningful use of BDH-style mechanisms
* Effective long-context reasoning

---

## Tech Stack

* **Python 3.10+**
* **PyTorch** - Deep learning framework
* **Sentence Transformers** - Semantic embeddings
* **Pathway** - Stream processing & incremental computation ⭐
* **Pandas, NumPy** - Data manipulation
* **Scikit-learn** - Metrics and utilities

### Key Libraries

- `pathway>=0.8.0` - Incremental stream processing
- `torch>=2.0.0` - Neural networks
- `transformers>=4.30.0` - Language models
- `sentence-transformers>=2.2.0` - Text embeddings

---

## Limitations

* Long narratives increase compute cost
* Memory abstraction may miss subtle stylistic cues
* No external world knowledge is assumed

---

## How to Run

### Quick Start

```bash
# Install dependencies (includes Pathway!)
pip install -r requirements.txt

# Train model
python train.py

# Predict with Pathway (recommended)
python predict_pathway.py --use_pathway

# OR predict traditionally
python predict.py
```

### With Pathway (Incremental Processing)

```bash
# Pathway-based prediction (Track B optimized!)
python predict_pathway.py \
    --use_pathway \
    --model_path checkpoints/best_model.pt \
    --test_csv test.csv
```

**Benefits:**
- ✅ Incremental computation
- ✅ Stateful memory operations
- ✅ Differential updates
- ✅ Streaming support

### Traditional Batch Processing

```bash
# Standard prediction
python predict.py --model_path checkpoints/best_model.pt
```

Output:

```bash
results.csv
```

For detailed usage, see `PATHWAY_GUIDE.md` and `USAGE.md`.

---

## Key Contribution

This project demonstrates that **persistent narrative memory and incremental belief updates** provide a strong foundation for long-form causal reasoning, aligning closely with the goals of **Track B**.

# KGP
