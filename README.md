# 🧠 Vector Space Proximity Workshop

---

## 👥 Team Members

| Name | Student ID |
|---|---|
| Lohith Reddy Danda | 9054470 |
| Muthuraj Jayakumar | 9084570 |
| Sumanth Reddy K | 9040660 |

---

## 📚 Overview

This workshop builds a complete **Information Retrieval (IR) system** from scratch — starting from raw text and finishing with a ranked search engine evaluated using industry-standard metrics.

The project answers one core question: **how does a computer find the right document out of thousands, given a natural-language query?**

We implement every step of the pipeline by hand:

1. Clean and preprocess raw text
2. Represent documents as numeric vectors (TF-IDF)
3. Measure similarity between vectors (Cosine Similarity)
4. Retrieve and rank documents for a query
5. Evaluate the quality of the retrieval system using 7 metrics
6. Measure human judge agreement with Cohen's Kappa
7. Apply the full pipeline to a real-world corpus (20 Newsgroups, 11,314 documents)

This is the same mathematical foundation used by **Google Search**, **ChatGPT's RAG pipeline**, and every modern vector database.

---

## 📓 Notebooks

### 1. `NLP_VectorSpaceFoundations.ipynb` — Refresher

**Purpose**: Introduces the six core building blocks before the main workshop.

| Section | What it covers |
|---|---|
| Step 1 | Term-Document Incidence Matrix |
| Step 2 | Document Collection (hardcoded + extensible) |
| Step 3 | Tokenizer (regex-based, handles punctuation) |
| Step 4 | Normalization Pipeline (lowercase → strip punct → stop-words → stemming) |
| Step 5 | **Full from-scratch pipeline**: incidence matrix → TF → log-TF → DF → IDF → TF-IDF → cosine similarity retrieval, tested with 4 information needs |

Run this notebook **first** to build intuition before tackling the main workshop.

---

### 2. `VectorSpaceProximity.ipynb` — Main Workshop

**Purpose**: The full end-to-end workshop with evaluation, student challenge, and reflection.

| Section | Cells | What it covers |
|---|---|---|
| Introduction | 1–9 | Learning objectives, key vocabulary, 2D vector visualisation (Euclidean vs Cosine) |
| Evaluation Setup | 10–13 | 6-document benchmark corpus, information need, ground-truth relevance judgments |
| Unranked Retrieval | 14–15 | Boolean overlap retrieval, retrieved/relevant table |
| Precision & Recall | 16–19 | P/R/F1 computation, why accuracy misleads |
| Confusion Matrix | 20–23 | 2×2 heatmap (TP/FP/FN/TN), plain-English summary |
| PR Curve | 24–26 | Precision@K and Recall@K vs rank, Precision-Recall curve |
| Interpolated Precision | 27–29 | Smoothed PR curve, 11-point interpolation |
| Precision@K | 30–32 | P@K and R@K table for the ranked list |
| Average Precision | 33–35 | AP computation, step-by-step worked example |
| MRR | 36–38 | Reciprocal Rank and MRR across three example ranked lists |
| Cohen's Kappa | 39–48 | Inter-judge agreement table, κ formula, worked 2×2 example |
| **Student Challenge** | 49–73 | **Full system on 20 Newsgroups** — Parts A through G (see below) |
| **Reflection** | 74–81 | Answers to all 6 evaluation questions with comparison code |
| Summary | 82–86 | Workshop takeaways, extension ideas, submission checklist |

#### Student Challenge Detail (Cells 49–73)

| Part | Cells | What is implemented |
|---|---|---|
| **Part A** | 54–55 | Corpus loaded (20 Newsgroups, 11,314 docs); subset of 500 docs for from-scratch work |
| **Part B — Preprocessing** | 56–58 | `preprocess()` function: lowercase → regex strip → stop-word removal → Porter stemming |
| **Part C — Incidence Matrix** | 59–61 | Binary incidence matrix (top-2000 vocab × 500 docs), density analysis |
| **Part D — TF & Log-TF** | 62–64 | Raw term frequency matrix, log-frequency weighting, histogram comparison |
| **Part E — DF, IDF, TF-IDF** | 65–67 | Document frequency, IDF, full TF-IDF matrix; compared against sklearn output |
| **Part F — 5 Queries** | 68–70 | Retrieval for 5 information needs using TF-IDF cosine similarity on full corpus |
| **Part G — Evaluation** | 71–73 | Relevance judgments by category label; P/R/F1, P@K, AP, MRR for 3 queries; PR curves |

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| **Name** | 20 Newsgroups |
| **Source** | `sklearn.datasets.fetch_20newsgroups` |
| **Size** | 11,314 documents (training split) |
| **Categories** | 20 newsgroup topics (sci.space, rec.autos, alt.atheism, comp.os.ms-windows.misc, talk.politics.guns, …) |
| **Domain** | Online discussion posts, 1993–1994 |
| **Licence** | Freely available for academic use |
| **Relevance labels** | Derived from category labels — a document is relevant to a query if it belongs to the expected category |

No download required — the dataset loads automatically via sklearn on first run.

---

## 🧠 Concepts Covered

### 1. Text Preprocessing

Before any math, raw text must be cleaned:

| Step | What it does | Example |
|---|---|---|
| **Lowercase** | Unifies case | `"NASA"` → `"nasa"` |
| **Remove punctuation** | Strips non-alpha characters | `"fun!"` → `"fun"` |
| **Stop-word removal** | Drops meaningless words | `"the"`, `"is"` → removed |
| **Stemming** | Reduces to root form | `"running"`, `"runs"` → `"run"` |

### 2. Vector Representations

| Representation | Formula | What it captures |
|---|---|---|
| **Incidence Matrix** | 0 or 1 | Whether a term appears |
| **Term Frequency (TF)** | count(term in doc) | How often a term appears |
| **Log-TF** | `1 + log₁₀(TF)` if TF > 0, else 0 | TF, compressed to prevent dominance by very frequent terms |
| **Document Frequency (DF)** | count(docs containing term) | How widespread a term is |
| **IDF** | `log₁₀(N / DF)` | Rarity — rare terms score higher |
| **TF-IDF** | `Log-TF × IDF` | Terms that are **frequent here AND rare elsewhere** |

> **Analogy**: TF-IDF is like a highlight reel — it keeps the plays that mattered most in *this* game (high TF) but were unusual compared to all other games (high IDF).

### 3. Cosine Similarity

Documents are vectors of TF-IDF weights. To compare a query to a document, we measure the **angle** between their vectors — not the straight-line distance.

```
Cosine Similarity = (A · B) / (|A| × |B|)
```

This is length-independent: a short tweet and a long research paper about the same topic score equally if they use the same vocabulary in the same proportions.

> **Analogy**: Two people facing the same direction are similar — it doesn't matter how tall they are.

### 4. Retrieval

| Mode | How it works | Limitation |
|---|---|---|
| **Unranked (Boolean)** | Retrieve if any query term appears | No ordering; returns junk |
| **Ranked (Cosine)** | Sort all documents by cosine similarity score | Best documents appear first |

### 5. Evaluation Metrics

| Metric | Formula | What it tells you |
|---|---|---|
| **Precision** | `TP / (TP + FP)` | How much of what you retrieved is relevant |
| **Recall** | `TP / (TP + FN)` | How much of what's relevant you retrieved |
| **F1** | `2PR / (P + R)` | Harmonic mean — penalises imbalance |
| **Precision@K** | relevant in top K / K | Quality of the top K results |
| **Average Precision** | avg of P@k at each relevant doc | Rewards ranking all relevant docs high |
| **MRR** | `1 / rank of first relevant doc` | How fast you find the first good result |
| **Interpolated Precision** | max precision at or beyond each recall level | Smooths PR curve for fair comparison |

> **Fishing analogy**: Precision = no boots in your net. Recall = no fish escaped. AP = you put all the fish at the top of your list.

### 6. Cohen's Kappa (Inter-Judge Agreement)

```
κ = (P(A) - P(E)) / (1 - P(E))
```

Measures how much two human judges agree on relevance labels, **corrected for random chance**. κ > 0.8 = strong agreement; κ < 0.6 = labels are unreliable.

### 7. Connection to Modern AI

| This Workshop | Modern AI |
|---|---|
| TF-IDF vectors | Dense embeddings (BERT, word2vec) |
| Cosine similarity on TF-IDF | Vector search on embedding space |
| Manual index | Vector database (FAISS, ChromaDB) |
| Keyword matching | Semantic retrieval |

**RAG (Retrieval-Augmented Generation)** — the technology behind ChatGPT answering questions about documents — uses steps 1–4 of this workshop as its retrieval layer.

---

## 🔁 How to Replicate

### Prerequisites

- Python 3.9 or higher
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/muthuacumen/VectorSpaceProximityWorkshop.git
cd VectorSpaceProximityWorkshop
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `jinja2`, and `ipykernel`.

### Step 4 — Launch Jupyter

```bash
jupyter notebook
```

Or open the folder in VS Code and use the built-in Jupyter extension.

### Step 5 — Run the notebooks in order

| Order | Notebook | Purpose |
|---|---|---|
| 1st | `NLP_VectorSpaceFoundations.ipynb` | Foundations refresher — run all cells top to bottom |
| 2nd | `VectorSpaceProximity.ipynb` | Main workshop — run all cells top to bottom |

> **Important**: Run cells **in order from top to bottom**. Later cells depend on variables defined in earlier cells (e.g., `corpus_df`, `vectorizer`, `tfidf_matrix`).

### Step 6 — NLTK data (auto-downloaded)

The normalization cells call `nltk.download('stopwords', quiet=True)` automatically. On first run, NLTK will download ~1 MB of data. No manual action needed.

### Step 7 — 20 Newsgroups dataset (auto-downloaded)

The student challenge cell calls `fetch_20newsgroups(subset='train', ...)` automatically. On first run, sklearn downloads ~15 MB. Subsequent runs use the cached version.

---

## 📦 Dependencies

| Package | Version | Used for |
|---|---|---|
| `numpy` | 1.26.4 | Matrix operations, cosine similarity |
| `pandas` | 2.3.1 | DataFrames for results and metric tables |
| `scikit-learn` | 1.7.0 | `CountVectorizer`, `TfidfVectorizer`, `cosine_similarity`, 20 Newsgroups dataset |
| `matplotlib` | latest | All plots and visualisations |
| `seaborn` | 0.13.2 | Heatmaps, styled charts |
| `nltk` | 3.9.1 | Stopwords, Porter Stemmer |
| `jinja2` | latest | Pandas styled DataFrames (optional — plain display used as fallback) |
| `ipykernel` | 6.29.5 | Jupyter kernel |

---

## 📊 Key Results

### Benchmark Corpus (6-document green-tea / coffee evaluation)

| Metric | Binary (Unranked) | TF + Cosine | TF-IDF + Cosine |
|---|---|---|---|
| Precision | 0.333 | 0.500 | 0.500 |
| Recall | 1.000 | 1.000 | 1.000 |
| F1 | 0.500 | 0.667 | 0.667 |
| Average Precision | 1.000 | 1.000 | 1.000 |

> TF-IDF improves score separation between relevant and non-relevant documents even when aggregate metrics look similar — the key difference is the *gap* between Doc5 (high TF-IDF, rare term "effective") and noise documents like Doc3 and Doc6.

### 20 Newsgroups Challenge (Top-20 retrieval, 3 queries)

| Query | Relevant Category | Precision | AP | MRR |
|---|---|---|---|---|
| NASA space missions | `sci.space` | High | High | 1.0 |
| Windows OS configuration | `comp.os.ms-windows.misc` | Moderate | Moderate | 1.0 |
| Atheism and religion | `alt.atheism` | Moderate | Moderate | 1.0 |

`sci.space` consistently achieves the best precision because space-related vocabulary (shuttle, orbit, NASA) is highly distinctive and rarely appears outside that category.

---

## ⚖️ Key Takeaways

1. **TF-IDF** balances local importance (TF) and global rarity (IDF) — it is the foundation of classical retrieval
2. **Cosine similarity** is length-independent — it compares direction, not magnitude
3. **Precision and Recall trade off** — improving one typically hurts the other
4. **AP is more informative than Precision** alone — it rewards good ranking, not just good coverage
5. **Accuracy is the wrong metric for IR** — class imbalance (few relevant docs) makes it misleading
6. **Relevance must be judged against the information need**, not just keyword overlap
7. Everything here is the **retrieval layer of modern RAG systems**

---

## 🚀 Next Steps

- Replace TF-IDF with **sentence embeddings** (e.g., `sentence-transformers`) to enable semantic matching
- Use a **vector database** (FAISS, ChromaDB) for scalable retrieval over millions of documents
- Build a full **RAG pipeline** by connecting retrieval to an LLM (OpenAI, Llama, etc.)
- Evaluate with **MAP** (Mean Average Precision) across multiple queries for a more reliable system score

---

## 👨‍🏫 Instructor Notes

This workshop is designed for entry-level ML/NLP students. It follows a code-first, explanation-heavy style with:

- 10th-grade-friendly language throughout
- Insight callouts after every visualisation
- Instructor and student talking points in each section
- Progression: math concept → from-scratch implementation → sklearn verification → evaluation

Emphasis is placed on understanding **why retrieval works** before moving to modern AI systems.
