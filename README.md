# 🎵 Music Recommendation Agentic Chatbot

A conversational AI music recommendation system powered by LLM tool-calling, semantic search (FAISS), and graph queries (Neo4j).  
Ask for music by mood, genre, artist, or song name—get smart, context-aware recommendations!

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/HelithaNimnaka/Music-Recommendation-Agentic-Chatbot)

---

## 🚀 Features

- **Agentic LLM chatbot** using [LangChain](https://github.com/langchain-ai/langchain)
- **Semantic music search** with FAISS and Sentence Transformers
- **Graph-based queries** for artist, genre, or song-specific searches with Neo4j
- **Friendly conversational interface**
- **Prompt engineering** for sensitive, context-aware responses

---

## 🛠️ Tech Stack

- **LangChain** (LLM agent & tools)
- **Neo4j** (Graph database, Cypher queries)
- **FAISS** (Vector search for similar music)
- **SentenceTransformers** (`all-MiniLM-L6-v2` for embeddings)
- **Pandas & NumPy** (Data handling)
- **Llama 3 (Groq)** (Language model)
- **Python**

---

**Why both vector and graph databases?**  
This project combines **vector search** (for understanding the meaning, mood, or vibe behind user requests) with a **graph database** (for precise, structured queries about artist, genre, and song relationships). This dual approach enables both flexible, semantic recommendations and accurate, data-driven searches.

---
