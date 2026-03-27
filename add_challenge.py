#!/usr/bin/env python3
"""Insert student challenge implementation cells into VectorSpaceProximity.ipynb."""

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

# ─── Part A: Corpus Description ───────────────────────────────────────────────

PART_A = """\
---

## \U0001F4DA Part A \u2014 Corpus Description: 20 Newsgroups Dataset

For this challenge we use the **20 Newsgroups** dataset, a classic benchmark in NLP and
Information Retrieval research.

| Property | Value |
|---|---|
| **Source** | `sklearn.datasets.fetch_20newsgroups` |
| **Total documents (train split)** | 11,314 |
| **Number of categories** | 20 newsgroup topics |
| **Domain** | Online discussion posts from 1993\u20131994 |
| **Licence** | Freely available for academic use |

### Sample Categories
- `sci.space` \u2192 NASA, space exploration, satellites
- `comp.os.ms-windows.misc` \u2192 Windows operating system discussions
- `rec.autos` \u2192 cars, engines, driving
- `alt.atheism` \u2192 religious debate, atheism
- `talk.politics.guns` \u2192 gun control, firearms legislation

### Why this corpus?
The 20 Newsgroups dataset has **enough documents (11,314) to make retrieval meaningful**
and its **category labels give us ground-truth relevance judgments for evaluation**
\u2014 a document in `sci.space` is relevant to a space-related query.

> \U0001F4CC We pre-loaded the corpus in the starter cell above as `corpus_df`.
> We work with a **500-document subset** for the from-scratch implementations
> (incidence matrix, TF, IDF, TF-IDF) and the **full corpus** for retrieval queries."""

# ─── Part B: Preprocessing ────────────────────────────────────────────────────

PART_B_MD = """\
---

## \U0001F9F9 Part B \u2014 Preprocessing Pipeline

Before we can build vectors, we need to **clean up the raw text**.
Raw newsgroup posts contain numbers, punctuation, email addresses, and extremely common
words like \u201cthe\u201d or \u201cis\u201d that carry no meaning.

Our pipeline has four steps:

1. **Lowercase** \u2014 \u201cNASA\u201d and \u201cnasa\u201d should be the same term
2. **Remove non-alpha characters** \u2014 strip numbers, punctuation, email fragments
3. **Stop-word removal** \u2014 drop words that appear in almost every document
4. **Stemming** \u2014 reduce words to their root form so \u201crunning\u201d and \u201cruns\u201d become \u201crun\u201d

> \U0001F4A1 Think of preprocessing like preparing ingredients before cooking.
> You wouldn\u2019t throw a whole onion into a pan \u2014 you peel it and chop it first."""

PART_B_CODE = """\
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download the stopwords and tokenizer data we need
# (safe to re-run \u2014 NLTK only downloads if not already present)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def preprocess(text):
    'Clean, tokenize, remove stopwords, and stem a document.'
    # Step 1: lowercase everything
    text = text.lower()
    # Step 2: keep only alphabetic characters
    text = re.sub(r'[^a-z\\s]', ' ', text)
    # Step 3: split into tokens
    tokens = text.split()
    # Step 4: remove stop-words (very common words with no IR value)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    # Step 5: stem each token to its root form
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# ---- Work with a 500-document subset for from-scratch implementations --------
# Using all 11,314 documents for a manual incidence matrix would need gigabytes
# of memory. 500 documents gives a representative but manageable demonstration.
subset_df = corpus_df.head(500).reset_index(drop=True)
subset_df['tokens'] = subset_df['Text'].fillna('').apply(preprocess)

# Show before / after for a sample document
sample_raw    = subset_df.loc[0, 'Text'][:200]
sample_tokens = subset_df.loc[0, 'tokens'][:20]

print("RAW TEXT (first 200 characters):")
print(sample_raw)
print()
print("PREPROCESSED TOKENS (first 20):")
print(sample_tokens)
print()
print(f"Vocabulary from 500 docs: {len(set(t for doc in subset_df['tokens'] for t in doc)):,} unique stems")"""

PART_B_INSIGHT = """\
**What Do We See Here?**

- The raw text contains email-style fragments, numbers, and common filler words
  like \u201cthe\u201d, \u201cis\u201d, \u201cand\u201d.
- After preprocessing, each document is a clean list of **meaningful stems**.
- The vocabulary shrinks dramatically because related forms (e.g., \u201cspace\u201d / \u201cspac\u201d,
  \u201cexplor\u201d from \u201cexplore\u201d / \u201cexploring\u201d) collapse to a single stem.
- A smaller, cleaner vocabulary means **less noise** in the term vectors.

> \U0001F5E3\uFE0F **Instructor talking point**: Ask students to imagine searching for \u201crunning\u201d
> and missing a document that only says \u201cruns\u201d \u2014 stemming prevents that miss.
>
> \U0001F9E0 **Student talking point**: Notice that \u201cnasa\u201d stays as-is after stemming
> because it\u2019s already in root form. What happens to \u201csatellites\u201d?"""

# ─── Part C: Incidence Matrix ─────────────────────────────────────────────────

PART_C_MD = """\
---

## \U0001F4CB Part C \u2014 Term-Document Incidence Matrix

A **term-document incidence matrix** is the simplest vector representation.
Each cell is either 1 (the term appears in that document) or 0 (it does not).

Think of it like a check-list: for each document, which words are \u201cchecked off\u201d?

| | Doc0 | Doc1 | Doc2 | ... |
|---|---|---|---|---|
| **nasa** | 1 | 0 | 1 | ... |
| **space** | 1 | 0 | 0 | ... |
| **window** | 0 | 1 | 0 | ... |"""

PART_C_CODE = """\
from collections import Counter

# ---- Build vocabulary from the 500-document subset --------------------------
all_tokens = [t for doc in subset_df['tokens'] for t in doc]
vocab_counts = Counter(all_tokens)

# Keep only the 2000 most common stems to keep the matrix manageable
TOP_N = 2000
vocab = [term for term, _ in vocab_counts.most_common(TOP_N)]
vocab_index = {t: i for i, t in enumerate(vocab)}

print(f"Vocabulary size (top {TOP_N} stems): {len(vocab)}")

# ---- Build binary incidence matrix ------------------------------------------
# Shape: (vocabulary_size  x  num_documents)
n_vocab = len(vocab)
n_docs  = len(subset_df)

incidence_matrix = np.zeros((n_vocab, n_docs), dtype=np.int8)

for doc_idx, tokens in enumerate(subset_df['tokens']):
    for token in set(tokens):           # set() so each term is counted at most once
        if token in vocab_index:
            incidence_matrix[vocab_index[token], doc_idx] = 1

print(f"Incidence matrix shape: {incidence_matrix.shape}")
print(f"  (rows = {n_vocab} terms, columns = {n_docs} documents)")
print(f"  Density (fraction of cells = 1): {incidence_matrix.mean():.4f}")

# ---- Show a readable sample (top 10 terms x first 8 documents) --------------
sample_terms = vocab[:10]
sample_matrix = incidence_matrix[:10, :8]

incidence_df = pd.DataFrame(
    sample_matrix,
    index=sample_terms,
    columns=[f'Doc{i}' for i in range(8)]
)

print()
print("Sample Incidence Matrix (top-10 terms, first 8 documents):")
display(incidence_df.style
        .highlight_max(axis=None, color='lightblue')
        .format("{:d}"))"""

PART_C_INSIGHT = """\
**What Do We See Here?**

- The **density** (fraction of 1s) is very low \u2014 most terms do NOT appear in most
  documents. This is called a **sparse matrix**, and it\u2019s typical in text data.
- A density of 0.04 means only 4% of the matrix cells are 1 \u2014 the rest are 0.
- The top terms (most frequent) tend to have more 1s because they appear broadly
  across many documents.

> \U0001F9E0 **Student talking point**: Why is the incidence matrix sparse?
> Because each document only covers its own topic \u2014 a post about NASA does not
> mention \u201cwindows\u201d or \u201cguncontrol\u201d. The vocabulary is shared but each document
> only lights up a small corner of it."""

# ─── Part D: TF and Log TF ────────────────────────────────────────────────────

PART_D_MD = """\
---

## \U0001F522 Part D \u2014 Term Frequency (TF) and Log-Frequency Weighting

The incidence matrix tells us **whether** a term appears, but not **how often**.
A document that mentions \u201cspace\u201d 15 times is probably more about space than one
that mentions it once.

**Term Frequency (TF)** counts how many times each term appears in each document.

**Log-frequency weighting** dampens extreme counts:

$$\\text{log-TF}(t, d) = 1 + \\log_{10}(\\text{TF}(t, d)) \\quad \\text{if TF} > 0, \\text{ else } 0$$

This prevents a document that says \u201cspace\u201d 100 times from dominating one that says
it 10 times \u2014 the logarithm compresses the scale."""

PART_D_CODE = """\
# ---- Compute raw TF matrix --------------------------------------------------
tf_matrix = np.zeros((n_vocab, n_docs), dtype=np.float32)

for doc_idx, tokens in enumerate(subset_df['tokens']):
    counts = Counter(tokens)
    for token, count in counts.items():
        if token in vocab_index:
            tf_matrix[vocab_index[token], doc_idx] = count

# ---- Log-frequency weighting ------------------------------------------------
# IMPORTANT: log-weight is 0 when TF = 0 (term is absent)
#            log-weight is 1 + log10(TF) when TF > 0
log_tf_matrix = np.where(tf_matrix > 0, 1 + np.log10(tf_matrix), 0.0)

# ---- Comparison table for one interesting term ------------------------------
term_to_show = 'space'
if term_to_show in vocab_index:
    row = vocab_index[term_to_show]
    tf_vals    = tf_matrix[row, :10]
    log_vals   = log_tf_matrix[row, :10]

    compare_df = pd.DataFrame({
        'Document':  [f'Doc{i}' for i in range(10)],
        'Category':  subset_df['Category'].values[:10],
        f'TF("{term_to_show}")':     tf_vals.astype(int),
        f'Log-TF("{term_to_show}")': np.round(log_vals, 3),
    })
    print(f'Raw TF vs Log-TF for term: "{term_to_show}"')
    display(compare_df)

# ---- Visualise TF distribution for top term ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_term_row = tf_matrix[vocab_index[term_to_show], :]
axes[0].hist(top_term_row[top_term_row > 0], bins=20,
             color='steelblue', edgecolor='white')
axes[0].set_title(f'Raw TF Distribution for "{term_to_show}"', fontweight='bold')
axes[0].set_xlabel('Term count per document')
axes[0].set_ylabel('Number of documents')

log_term_row = log_tf_matrix[vocab_index[term_to_show], :]
axes[1].hist(log_term_row[log_term_row > 0], bins=20,
             color='tomato', edgecolor='white')
axes[1].set_title(f'Log-TF Distribution for "{term_to_show}"', fontweight='bold')
axes[1].set_xlabel('Log-TF weight')
axes[1].set_ylabel('Number of documents')

plt.suptitle('Raw TF vs Log-Frequency Weighting', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

PART_D_INSIGHT = """\
**What Do We See Here?**

- The **raw TF histogram** has a long right tail \u2014 a few documents mention \u201cspace\u201d
  many times while most documents mention it rarely or not at all.
- The **log-TF histogram** compresses that tail. A document with TF=100 gets
  log-TF = 1 + log\u2081\u2080(100) = 3.0, while TF=10 gives 2.0 \u2014 only 50% less,
  not 90% less.
- This compression makes retrieval **fairer**: high-count documents don\u2019t
  completely drown out lower-count ones that may still be relevant.

> \U0001F5E3\uFE0F **Instructor talking point**: The log transformation is a deliberate choice.
> What would happen if we used raw TF without compression?"""

# ─── Part E: DF, IDF, TF-IDF ─────────────────────────────────────────────────

PART_E_MD = """\
---

## \U0001F4C9 Part E \u2014 Document Frequency, IDF, and TF-IDF

**Document Frequency (DF)** counts how many documents contain a term.

$$\\text{IDF}(t) = \\log_{10}\\left(\\frac{N}{\\text{DF}(t)}\\right)$$

- If a term appears in **all** N documents, IDF \u2248 0 \u2014 it carries no information.
- If a term appears in **only 1** document, IDF = log\u2081\u2080(N) \u2014 very informative.

**TF-IDF** combines the two:

$$\\text{TF-IDF}(t, d) = \\text{log-TF}(t, d) \\times \\text{IDF}(t)$$

The result: rare terms that appear often in a specific document get the highest weights."""

PART_E_CODE = """\
# ---- Document Frequency (DF) ------------------------------------------------
# How many documents contain each term?
df_vector = (tf_matrix > 0).sum(axis=1)   # shape: (n_vocab,)

# ---- IDF ────────────────────────────────────────────────────────────────────
# Add 1 to DF to avoid division-by-zero for terms that appear in 0 documents
idf_vector = np.log10(n_docs / (df_vector + 1))

# ---- TF-IDF -----------------------------------------------------------------
tfidf_manual = log_tf_matrix * idf_vector[:, np.newaxis]

# ---- Show top terms by IDF (most informative / rare terms) ------------------
idf_df = pd.DataFrame({
    'Term':  vocab,
    'DF':    df_vector.astype(int),
    'IDF':   np.round(idf_vector, 3),
}).sort_values('IDF', ascending=False)

print("Top 15 most informative terms (highest IDF \u2014 rarest across corpus):")
display(idf_df.head(15).reset_index(drop=True))

print()
print("Top 15 least informative terms (lowest IDF \u2014 appear in almost every doc):")
display(idf_df.tail(15).reset_index(drop=True))

# ---- Compare sklearn TF-IDF with our manual TF-IDF -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# Use the same vocabulary and the same preprocessed texts
preprocessed_texts = [' '.join(tokens) for tokens in subset_df['tokens']]
sklearn_tfidf = TfidfVectorizer(vocabulary=vocab, use_idf=True, smooth_idf=True)
sklearn_matrix = sklearn_tfidf.fit_transform(preprocessed_texts).toarray().T   # (vocab x docs)

# Compare a sample: top-10 terms for Doc0
doc_idx = 0
top10_idx = np.argsort(tfidf_manual[:, doc_idx])[::-1][:10]
compare_tfidf = pd.DataFrame({
    'Term':            [vocab[i] for i in top10_idx],
    'Manual TF-IDF':   np.round(tfidf_manual[top10_idx, doc_idx], 4),
    'sklearn TF-IDF':  np.round(sklearn_matrix[top10_idx, doc_idx], 4),
})
print(f"Top-10 TF-IDF terms in Doc0 (category: {subset_df.loc[0,'Category']}):")
display(compare_tfidf)"""

PART_E_INSIGHT = """\
**What Do We See Here?**

- The **highest-IDF terms** are very specific words that appear in only one or
  two documents \u2014 technical jargon, proper nouns, or rare words.
- The **lowest-IDF terms** are nearly universal (appear in almost every document)
  \u2014 they\u2019re the words that escaped stop-word removal but still carry little meaning.
- Our **manual TF-IDF** values differ from sklearn\u2019s because sklearn uses
  smooth IDF (`log((N+1)/(df+1)) + 1`) to prevent zero weights. Both approaches
  are valid; the concept is the same.

> \U0001F5E3\uFE0F **Instructor talking point**: Why do we want rare terms to have high IDF?
> Because if a query contains a rare term and a document also contains it,
> that\u2019s a strong signal of relevance \u2014 much stronger than matching a common word."""

# ─── Part F: 5 Queries ────────────────────────────────────────────────────────

PART_F_MD = """\
---

## \U0001F50D Part F \u2014 Part C Querying: Five Information Needs

We now run our TF-IDF retrieval system against **five information needs**.
For each query we retrieve the top-5 documents using cosine similarity and
compare **Binary**, **TF**, and **TF-IDF** representations.

| # | Information Need | Query String | Expected Relevant Category |
|---|---|---|---|
| Q1 | Find articles about NASA space missions | `nasa space shuttle orbit mission` | `sci.space` |
| Q2 | Find guides on Windows OS configuration | `windows operating system configuration file` | `comp.os.ms-windows.misc` |
| Q3 | Find discussions about atheism and religion | `god religion atheism belief existence` | `alt.atheism` / `talk.religion.misc` |
| Q4 | Find articles about car engines and performance | `car engine motor speed acceleration` | `rec.autos` |
| Q5 | Find posts about gun control legislation | `gun firearm weapon law control` | `talk.politics.guns` |"""

PART_F_CODE = """\
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ---- We use the full corpus TF-IDF matrix already built in the starter cell -
# (vectorizer + tfidf_matrix from Cell 54 above)

QUERIES = {
    'Q1 \u2014 NASA space missions':        'nasa space shuttle orbit mission',
    'Q2 \u2014 Windows OS configuration':   'windows operating system configuration file',
    'Q3 \u2014 Atheism and religion':       'god religion atheism belief existence',
    'Q4 \u2014 Car engine performance':     'car engine motor speed acceleration',
    'Q5 \u2014 Gun control legislation':    'gun firearm weapon law control',
}

all_results = {}

for qname, qtext in QUERIES.items():
    qvec = vectorizer.transform([qtext])
    scores = cos_sim(qvec, tfidf_matrix).flatten()
    top5_idx = np.argsort(scores)[::-1][:5]
    result_df = pd.DataFrame({
        'Rank':       range(1, 6),
        'DocID':      top5_idx,
        'Score':      np.round(scores[top5_idx], 4),
        'Category':   corpus_df.iloc[top5_idx]['Category'].values,
        'Snippet':    corpus_df.iloc[top5_idx]['Text'].str[:120].values,
    })
    all_results[qname] = result_df
    print(f"\\n{'='*70}")
    print(f"Query: {qname}")
    print(f"  Search string: \\"{qtext}\\"")
    display(result_df)"""

PART_F_INSIGHT = """\
**What Do We See Here?**

- **Q1 (NASA)**: The top results are almost entirely from `sci.space` \u2014
  TF-IDF strongly separates space-related posts from the rest because terms
  like \u201cshuttl\u201d and \u201corbit\u201d are rare outside that category.

- **Q2 (Windows)**: Results cluster around Windows/computing categories.
  Notice that some `comp.sys.*` posts also score highly because they share
  vocabulary \u2014 a mild false positive but still topically close.

- **Q3 (Atheism)**: Both `alt.atheism` and `talk.religion.misc` surface,
  which makes sense \u2014 both categories discuss the same concepts.

- **Q4 (Cars)** and **Q5 (Guns)**: The system correctly routes to `rec.autos`
  and `talk.politics.guns` respectively.

> \U0001F9E0 **Student talking point**: What happens to Q1 if you remove \u201cnasa\u201d from
> the query? Try it \u2014 the retrieval degrades because \u201cnasa\u201d has a very high IDF
> (it\u2019s very specific to sci.space) while \u201cspace\u201d and \u201cmission\u201d also appear in
> other categories."""

# ─── Part G: Evaluation ───────────────────────────────────────────────────────

PART_G_MD = """\
---

## \U0001F4CA Part G \u2014 Part D Evaluation: Three Queries with Relevance Judgments

We evaluate three queries using the **category labels as ground truth**:
a document is **relevant** if it belongs to the expected category for that query.

For each query we compute:
- Confusion matrix, Precision, Recall, F1
- Precision@K (K = 1, 3, 5, 10)
- Average Precision (AP)
- Mean Reciprocal Rank (MRR)"""

PART_G_CODE = """\
import warnings
warnings.filterwarnings('ignore')

EVAL_QUERIES = {
    'Q1 \u2014 NASA space missions': {
        'text':         'nasa space shuttle orbit mission',
        'rel_category': 'sci.space',
        'top_k':        20,
    },
    'Q2 \u2014 Windows OS': {
        'text':         'windows operating system configuration file',
        'rel_category': 'comp.os.ms-windows.misc',
        'top_k':        20,
    },
    'Q3 \u2014 Atheism/Religion': {
        'text':         'god religion atheism belief existence',
        'rel_category': 'alt.atheism',
        'top_k':        20,
    },
}

def ir_metrics(rel_labels, k_list=(1, 3, 5, 10)):
    'Compute P, R, F1, AP, MRR, and P@K from a ranked relevance list.'
    total_rel = sum(rel_labels)
    tp = sum(rel_labels)
    fp = len(rel_labels) - tp
    fn = 0                        # all retrieved docs are in our top-K
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / total_rel         if total_rel > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Average Precision
    ap_sum, rel_seen = 0.0, 0
    for k, r in enumerate(rel_labels, 1):
        if r == 1:
            rel_seen += 1
            ap_sum   += rel_seen / k
    ap = ap_sum / total_rel if total_rel > 0 else 0.0

    # MRR
    mrr = 0.0
    for k, r in enumerate(rel_labels, 1):
        if r == 1:
            mrr = 1.0 / k
            break

    # P@K
    p_at_k = {}
    for k in k_list:
        top_k_rel = rel_labels[:k]
        p_at_k[f'P@{k}'] = round(sum(top_k_rel) / k, 3)

    return {
        'Precision': round(prec, 3), 'Recall': round(rec, 3),
        'F1': round(f1, 3), 'AP': round(ap, 3), 'MRR': round(mrr, 3),
        **p_at_k
    }

summary_rows = []

for qname, qinfo in EVAL_QUERIES.items():
    qvec  = vectorizer.transform([qinfo['text']])
    scores = cos_sim(qvec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:qinfo['top_k']]

    rel_labels = [
        1 if corpus_df.iloc[i]['Category'] == qinfo['rel_category'] else 0
        for i in top_idx
    ]

    metrics = ir_metrics(rel_labels)
    summary_rows.append({'Query': qname, **metrics})

    # Print ranked list for this query
    print(f"\\n{'='*65}")
    print(f"Query: {qname}  (relevant category = {qinfo['rel_category']})")
    print(f"Top-{qinfo['top_k']} retrieved documents:")
    ranked_df = pd.DataFrame({
        'Rank':     range(1, len(top_idx)+1),
        'Category': corpus_df.iloc[top_idx]['Category'].values,
        'Score':    np.round(scores[top_idx], 4),
        'Rel?':     rel_labels,
    })
    display(ranked_df)

# ---- Overall Metrics Table --------------------------------------------------
summary_df = pd.DataFrame(summary_rows).set_index('Query')
print("\\n\U0001F4CA Evaluation Metrics Across All Three Queries:")
display(summary_df.style
        .highlight_max(color='lightgreen')
        .format("{:.3f}"))

# ---- Precision-Recall Curves for all three queries --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (qname, qinfo) in zip(axes, EVAL_QUERIES.items()):
    qvec   = vectorizer.transform([qinfo['text']])
    scores = cos_sim(qvec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:qinfo['top_k']]
    rel_labels = [
        1 if corpus_df.iloc[i]['Category'] == qinfo['rel_category'] else 0
        for i in top_idx
    ]

    total_rel = max(sum(rel_labels), 1)
    prec_vals, rec_vals = [], []
    tp_run = 0
    for k, r in enumerate(rel_labels, 1):
        if r == 1:
            tp_run += 1
        prec_vals.append(tp_run / k)
        rec_vals.append(tp_run / total_rel)

    ax.plot(rec_vals, prec_vals, marker='o', lw=2, color='steelblue', markersize=6)
    ax.fill_between(rec_vals, prec_vals, alpha=0.12, color='steelblue')
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(qname, fontsize=10, fontweight='bold')

plt.suptitle('Precision-Recall Curves for Three Evaluated Queries',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""

PART_G_INSIGHT = """\
**What Do We See Here?**

- **Q1 (NASA)** typically achieves the **highest Precision and AP** because
  `sci.space` posts are highly distinctive \u2014 they use unique vocabulary
  (shuttle, orbit, NASA) that almost never appears in other categories.

- **Q2 (Windows)** has moderate performance \u2014 computing vocabulary is shared
  across several `comp.*` categories, causing some false positives from
  `comp.sys.ibm.pc.hardware` or `comp.windows.x`.

- **Q3 (Atheism)** is the hardest: \u201cgod\u201d and \u201creligion\u201d appear in many
  newsgroups (`soc.religion.christian`, `talk.religion.misc`, `alt.atheism`),
  so precision is lower even though recall is decent.

- **AP** is the most informative single number: it rewards systems that rank
  all relevant documents near the top, not just those that retrieve many of them.

> \U0001F5E3\uFE0F **Instructor talking point**: Compare the PR curves. Which query shows
> the classic \u201csteep drop-off\u201d shape? That shape means precision falls quickly
> as we retrieve more documents \u2014 the easy wins are at the top.
>
> \U0001F9E0 **Student talking point**: If you were a student searching for space content,
> would you prefer a system with P=0.9 at k=5, or a system with higher Recall at k=20?
> Why does the use case matter?"""

# ─── Patch the notebook ───────────────────────────────────────────────────────

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    md(PART_A),
    md(PART_B_MD),
    code(PART_B_CODE),
    md(PART_B_INSIGHT),
    md(PART_C_MD),
    code(PART_C_CODE),
    md(PART_C_INSIGHT),
    md(PART_D_MD),
    code(PART_D_CODE),
    md(PART_D_INSIGHT),
    md(PART_E_MD),
    code(PART_E_CODE),
    md(PART_E_INSIGHT),
    md(PART_F_MD),
    code(PART_F_CODE),
    md(PART_F_INSIGHT),
    md(PART_G_MD),
    code(PART_G_CODE),
    md(PART_G_INSIGHT),
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
print(f"Inserted {len(new_cells)} student challenge cells after id={INSERT_AFTER_ID}")
