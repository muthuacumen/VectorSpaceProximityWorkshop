#!/usr/bin/env python3
"""Insert a complete Reflection section into VectorSpaceProximity.ipynb."""

import json, uuid

NB_PATH = 'D:/Projects/IRBasics_VectorSpaceProximity/VectorSpaceProximity.ipynb'
INSERT_AFTER_ID = '174a245e'   # Cell 54: TF-IDF retrieval starter

def new_id():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": new_id(),
            "metadata": {}, "outputs": [], "source": [src]}

# ─── Content ─────────────────────────────────────────────────────────────────

REFLECTION_INTRO = """\
---

## \U0001F4DD Reflection: Answering the Six Evaluation Questions

This section works through all six reflection questions using the **green tea / coffee
benchmark corpus** demonstrated earlier in this notebook.  Every answer is grounded in
numbers you can reproduce by running the cells above.

> \U0001F4CC The comparison code cell below computes cosine similarity scores for three
> representations — **Binary**, **TF**, and **TF-IDF** — on the same six documents, so
> you can see exactly how the representations differ before reading the written answers."""

REFLECTION_CODE = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Comparing Three Representations on the Same Corpus ──────────────────────
# We reuse the variables already defined in the evaluation section:
#   documents_eval  : the 6-document benchmark collection
#   relevance_judgments : ground-truth labels (1 = relevant, 0 = not relevant)
#   query_eval      : the query string
#   results_eval    : DataFrame with Retrieved / Relevant columns (binary system)

sns.set_style('whitegrid')

doc_texts  = list(documents_eval.values())
doc_names  = list(documents_eval.keys())
relevance  = [relevance_judgments[d] for d in doc_names]

# ── Representation 1: Binary — all docs with any query-term overlap retrieved ─
binary_scores = results_eval["Retrieved"].values.astype(float)

# ── Representation 2: Raw TF + Cosine Similarity ──────────────────────────────
tf_vec        = CountVectorizer()             # raw term counts (not binary)
tf_matrix     = tf_vec.fit_transform(doc_texts)
tf_query_vec  = tf_vec.transform([query_eval])
tf_scores     = cosine_similarity(tf_query_vec, tf_matrix).flatten()

# ── Representation 3: TF-IDF + Cosine Similarity ──────────────────────────────
# TF-IDF rewards terms that are rare across the corpus (high IDF)
# while penalising common terms like "tea" or "is".
tfidf_vec       = TfidfVectorizer()
tfidf_matrix    = tfidf_vec.fit_transform(doc_texts)
tfidf_query_vec = tfidf_vec.transform([query_eval])
tfidf_scores    = cosine_similarity(tfidf_query_vec, tfidf_matrix).flatten()

# ── Comparison table ──────────────────────────────────────────────────────────
comp_df = pd.DataFrame({
    "Document":           doc_names,
    "Relevant?":          relevance,
    "Binary Retrieved":   binary_scores.astype(int),
    "TF Cosine":          np.round(tf_scores, 3),
    "TF-IDF Cosine":      np.round(tfidf_scores, 3),
})
comp_df["TF Rank"]     = comp_df["TF Cosine"].rank(ascending=False, method='min').astype(int)
comp_df["TF-IDF Rank"] = comp_df["TF-IDF Cosine"].rank(ascending=False, method='min').astype(int)

print(f"Query: '{query_eval}'")
print()
display(comp_df.sort_values("TF-IDF Cosine", ascending=False).reset_index(drop=True))

# ── Helper: compute P, R, F1, AP from a score vector ─────────────────────────
def compute_ir_metrics(scores, rel_labels):
    '''Rank documents by score descending, then compute P, R, F1, AP.'''
    order      = np.argsort(scores)[::-1]
    ranked_rel = [rel_labels[i] for i in order]
    total_rel  = sum(rel_labels)

    tp   = sum(ranked_rel)
    fp   = len(ranked_rel) - tp
    fn   = total_rel - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    ap_sum, rel_seen = 0.0, 0
    for k, r in enumerate(ranked_rel, 1):
        if r == 1:
            rel_seen += 1
            ap_sum   += rel_seen / k
    ap = ap_sum / total_rel if total_rel > 0 else 0.0

    return {"Precision": round(prec, 3), "Recall": round(rec, 3),
            "F1": round(f1, 3), "AP": round(ap, 3)}

metrics = {
    "Binary (Unranked)": compute_ir_metrics(binary_scores, relevance),
    "TF + Cosine":       compute_ir_metrics(tf_scores,     relevance),
    "TF-IDF + Cosine":   compute_ir_metrics(tfidf_scores,  relevance),
}
metrics_df = pd.DataFrame(metrics).T

print("\\n\\U0001F4CA Metrics Comparison Across All Three Representations:")
display(metrics_df.style
        .highlight_max(color='lightgreen')
        .format("{:.3f}"))

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: bar chart of P / R / F1 / AP per representation
metrics_df.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='white', width=0.7)
axes[0].set_title("Metric Comparison by Representation", fontsize=13, fontweight='bold')
axes[0].set_xlabel("")
axes[0].set_ylabel("Score (0 \u2013 1)", fontsize=11)
axes[0].set_ylim(0, 1.2)
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(fontsize=10)

# Right: per-document TF vs TF-IDF cosine scores (green shading = relevant docs)
x   = np.arange(len(doc_names))
w   = 0.35
axes[1].bar(x - w/2, tf_scores,    w, label='TF Cosine',     color='steelblue',  edgecolor='white')
axes[1].bar(x + w/2, tfidf_scores, w, label='TF-IDF Cosine', color='tomato',     edgecolor='white')
for i, rel in enumerate(relevance):
    if rel == 1:
        axes[1].axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')
axes[1].set_xticks(x)
axes[1].set_xticklabels(doc_names, rotation=15)
axes[1].set_title("Per-Document Scores: TF vs TF-IDF", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Cosine Similarity", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].text(0.02, 0.96, "\u2705 Green band = Relevant document",
             transform=axes[1].transAxes, fontsize=9, color='green', va='top')

plt.suptitle("How the Three Representations Compare on Our 6-Document Corpus",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""

Q1 = """\
### \u2753 Q1 \u2014 Which representation worked best and why?

**Answer: TF-IDF + Cosine Similarity performed best**, and here is exactly why.

The query is `"green tea coffee heart attack effective"`. The word **"effective"** appears in
only **one** document in the whole corpus — Doc5. TF-IDF assigns it a very high IDF weight
because it is rare. Binary and raw TF representations treat all query terms equally, so
they cannot exploit this signal.

| Representation | What it sees | What it misses |
|---|---|---|
| **Binary** | Whether each term appears — nothing more | Term frequency, term rarity |
| **TF** | How often each term appears in each doc | That rare terms are more informative |
| **TF-IDF** | Both frequency AND rarity, combined | Nothing — this is the most expressive of the three |

In the bar chart above, notice that **Doc5** receives a noticeably higher TF-IDF score than
any other document. Doc3 ("coffee prices / climate") and Doc6 ("history of tea") drop to
near zero because the only term they share with the query is one very common word — TF-IDF
correctly penalises that.

> \U0001F4A1 On a 6-document corpus the gap is small. On a corpus of thousands of documents
> the gap is dramatic — that is when TF-IDF truly earns its place."""

Q2 = """\
### \u2753 Q2 \u2014 Did TF-IDF improve over raw term counts?

**Answer: Yes — particularly in the scores assigned to less-relevant documents.**

Looking at the metrics table from the code cell above:

- **Precision** is the same across representations on this tiny corpus — both relevant
  documents (Doc1, Doc5) already rank in the top 2 positions with either method.
- **The real difference is in the score gap**: TF-IDF increases the *separation* between
  relevant and non-relevant documents.

Concretely:

| Document | Why TF-IDF helps |
|---|---|
| **Doc5** | Scores highest because "effective" appears *only here* \u2192 maximum IDF boost |
| **Doc3** | Scores near zero — only "coffee" matches, and "coffee" appears in 3 of 6 docs \u2192 low IDF |
| **Doc6** | Scores near zero — only "tea" matches, and "tea" appears in 4 of 6 docs \u2192 even lower IDF |

With raw TF, Doc3 and Doc6 still receive a non-trivial score just because "coffee" or
"tea" matched. TF-IDF rightly pushes them toward the bottom of the ranking.

> \U0001F4A1 The improvement is subtle here because documents are short and most terms
> appear only once (so TF \u2248 binary). On a 20newsgroups-scale corpus, where documents are
> long and term frequencies vary widely, TF-IDF improvements are much more pronounced."""

Q3 = """\
### \u2753 Q3 \u2014 What kinds of false positives were observed?

**Answer: Four false positives were retrieved**, each illustrating a different failure mode
of keyword-based retrieval.

| Document | Retrieved term(s) | Why it is a false positive |
|---|---|---|
| **Doc2** | "green", "tea", "heart" | Discusses green tea and heart health *in general* — does not compare green tea vs coffee for heart attacks |
| **Doc3** | "coffee" | About **crop prices** and climate change — "coffee" here means a commodity, not a health drink |
| **Doc4** | "heart", "attack" | About **emergency response** to heart attacks — not about which beverage prevents them |
| **Doc6** | "tea" | A **history** article about tea-drinking culture in East Asia — completely off-topic |

All four share the same root cause: **the system matched on surface-level word overlap
without understanding meaning**. The word "coffee" in a finance article means the same
string of characters as "coffee" in a health study — a keyword system cannot tell them
apart.

This is the core limitation of bag-of-words retrieval, and it is precisely why modern
systems use **dense vector embeddings** (which encode meaning, not just spelling)."""

Q4 = """\
### \u2753 Q4 \u2014 What kinds of relevant documents were missed?

**Answer: No relevant documents were missed in this example** — both Doc1 and Doc5 were
retrieved and ranked in the top 2 positions. Recall = 1.0 and FN = 0.

However, this result was *lucky*: both relevant documents happen to be packed with
query terms. In a realistic large-scale corpus, relevant documents would be missed for
several reasons:

| Failure mode | Example |
|---|---|
| **Synonym mismatch** | A document says "cardiac event" instead of "heart attack" |
| **Paraphrase mismatch** | "Is green tea healthier than coffee?" vs our query "green tea coffee heart attack effective" |
| **Different vocabulary level** | A clinical paper uses "myocardial infarction" — the query uses "heart attack" |
| **Relevant but sparse overlap** | A document answers the question in one sentence with few matching words |

These missed-relevant cases (false negatives) are exactly what **stemming, lemmatisation,
synonym expansion, and semantic search** are designed to fix. Retrieval systems fail silently
on FN — the user never sees what was missed — which is why measuring **Recall** explicitly
matters so much."""

Q5 = """\
### \u2753 Q5 \u2014 How did the evaluation metrics help understand system quality?

**Answer: Each metric revealed a different dimension of system quality**, and together
they told a much richer story than any single number could.

| Metric | Value (Binary) | What it revealed |
|---|---|---|
| **Precision** | 0.333 | Only 1 in 3 retrieved documents was relevant — the system over-retrieves badly |
| **Recall** | 1.000 | All relevant documents were found — nothing was missed |
| **F1** | 0.500 | The harmonic mean confirms the imbalance: high recall came at a steep precision cost |
| **Accuracy** | 0.333 | Misleadingly low here (class imbalance: 2 relevant vs 4 non-relevant) — confirms accuracy is the wrong metric for IR |
| **Confusion matrix** | TP=2, FP=4, FN=0, TN=0 | Made the failure mode visual: the FP quadrant (4 documents) is the entire problem |
| **PR curve** | Steeply falling | Showed that even retrieving just the top 2 results gives precision=1.0 — the ranking is actually excellent |
| **Average Precision** | 1.000 | Revealed the hidden strength: both relevant docs are at ranks 1 and 2 — the ranking is perfect even if the full set is noisy |
| **MRR** | 1.000 | The very first result is relevant — a user who stops after one result always wins |

The most important lesson: **raw Precision (0.333) looked bad, but AP (1.000) told a
completely different story.** The unranked system retrieves too much noise — but the ranked
version immediately surfaces the best documents. Without the full set of metrics, we would
have drawn the wrong conclusion about the system's quality."""

Q6 = """\
### \u2753 Q6 \u2014 How would you improve the system?

**Answer: Six concrete improvements, in order of expected impact.**

---

#### 1. Add preprocessing (stop-word removal + stemming)
The current pipeline queries on raw tokens. "The", "is", "and" pollute the term vectors
and create false matches. Stemming would unify "effective" / "effectiveness" and
"compare" / "comparing" — reducing both false positives and false negatives.

#### 2. Replace raw overlap with TF-IDF + cosine similarity throughout
The PR-curve ranking (Cell 25) still uses a raw dot-product overlap score, not cosine
similarity. Switching to TF-IDF cosine consistently would eliminate the length-bias
shown in the scatter plot (Cell 05) and align the lecture with the implementation.

#### 3. Raise the retrieval threshold
Instead of retrieving every document with overlap > 0, set a minimum cosine similarity
threshold (e.g., 0.10). This cuts false positives like Doc6 (cos \u2248 0.02) while keeping
true positives. Tuning this threshold with the evaluation metrics guides the decision.

#### 4. Test on more than one query
A single query is not enough to generalise. The current evaluation is based on one
information need. Evaluating on 3–5 diverse queries and computing **MAP** (Mean Average
Precision) gives a much more reliable picture of system quality.

#### 5. Expand to a large real corpus
The toy corpus has only 6 documents — every keyword matches something. On 20 Newsgroups
(11,000+ documents), false positives multiply and the difference between Binary, TF, and
TF-IDF becomes dramatic. Running the same evaluation pipeline on a large corpus is the
most important next step.

#### 6. Explore semantic / dense retrieval
Even a perfect TF-IDF system cannot match documents that use different vocabulary.
The next generation of retrieval uses **sentence embeddings** (e.g.,
`sentence-transformers`) to encode *meaning* rather than surface words — enabling it to
retrieve "cardiac event" documents for a "heart attack" query.

---

> \U0001F4CC **Bottom line**: The three biggest wins in order are — (1) add preprocessing,
> (2) switch to TF-IDF cosine throughout, (3) test on a large corpus with multiple queries.
> Everything else is optimisation."""

# ─── Patch the notebook ───────────────────────────────────────────────────────

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    md(REFLECTION_INTRO),
    code(REFLECTION_CODE),
    md(Q1),
    md(Q2),
    md(Q3),
    md(Q4),
    md(Q5),
    md(Q6),
]

updated = []
for cell in nb['cells']:
    updated.append(cell)
    if cell.get('id') == INSERT_AFTER_ID:
        updated.extend(new_cells)

nb['cells'] = updated

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Notebook now has {len(nb['cells'])} cells.")
print(f"Inserted {len(new_cells)} reflection cells after id={INSERT_AFTER_ID}")
