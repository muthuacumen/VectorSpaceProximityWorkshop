# 🧠 NLP to Vector Space Proximity to RAG Workshop

## 📚 Overview

This workshop introduces foundational Natural Language Processing (NLP)
concepts and progressively builds toward **Vector Space Proximity**,
**Word Embeddings**, and **LLM-powered applications**.

Students implement: - Term-Document Incidence Matrix - Term Frequency
(TF) - Log Frequency Weighting - Document Frequency (DF) - Inverse
Document Frequency (IDF) - TF-IDF - Word2Vec Embeddings - Cosine
Similarity

These are then applied to real-world systems: - Chatbots -
Recommendation Engines - Semantic Search Systems

------------------------------------------------------------------------

## 🎯 Learning Objectives

By the end of this workshop, students will be able to:

-   Represent text as vectors using multiple techniques
-   Compute similarity between documents using cosine similarity
-   Build a simple semantic chatbot
-   Understand how embeddings power modern AI systems
-   Connect classical IR techniques to modern LLM workflows

------------------------------------------------------------------------

## 🧪 Hands-On Components

Students will: - Build preprocessing pipelines (tokenization,
normalization, stopword removal) - Implement TF, IDF, and TF-IDF from
scratch - Train a Word2Vec model - Compute vector similarity - Build a
travel assistant chatbot - Extend the chatbot using OpenAI APIs

------------------------------------------------------------------------

## 🤖 From Vector Space to LLMs

The workshop bridges traditional IR with modern AI:

  Classical IR        Modern AI
  ------------------- --------------------------------------
  TF-IDF              Embeddings
  Cosine Similarity   Semantic Search
  Document Matching   Retrieval-Augmented Generation (RAG)

------------------------------------------------------------------------

## 🔍 Relevance to RAG (Retrieval-Augmented Generation)

RAG systems combine: 1. **Retrieval** (vector search) 2. **Generation**
(LLMs)

### How This Workshop Connects:

-   TF-IDF and embeddings → represent documents as vectors
-   Cosine similarity → retrieve relevant documents
-   Word2Vec → semantic understanding
-   Chatbot → generation layer

### RAG Pipeline:

1.  User query → embedding
2.  Retrieve top-k similar documents (vector proximity)
3.  Pass context + query to LLM
4.  Generate grounded response

------------------------------------------------------------------------

## 🏗️ Example Use Case

Travel Assistant Chatbot: - User: "Is there an evening bus to
Toronto?" - System: - Convert query to vector - Find closest matching
schedule - Use LLM to generate natural response

------------------------------------------------------------------------

## ⚖️ Key Takeaways

-   Vector space proximity is the **foundation of modern AI retrieval**
-   Embeddings enable **semantic understanding**
-   LLMs add **reasoning and language generation**
-   Together, they form the backbone of **RAG systems**

------------------------------------------------------------------------

## 🚀 Next Steps

-   Integrate FAISS or Chroma for vector storage
-   Use Sentence Transformers for better embeddings
-   Build full RAG pipelines
-   Deploy chatbot with Streamlit or FastAPI

------------------------------------------------------------------------

## 👨‍🏫 Instructor Notes

This workshop is designed for: - Entry-level ML/NLP students - Hands-on,
code-first learning - Progressive understanding from math →
implementation → applications
