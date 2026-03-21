# 🚀 StatGuide – Self-Improving AI Tutor with Adaptive Retrieval

StatGuide is an AI-powered tutor for Mathematical Statistics that goes beyond traditional RAG systems by **evaluating, correcting, and improving its own retrieval process**.

Instead of just answering questions, the system focuses on **retrieval quality, reasoning, and answer reliability**.

---

## 🧠 What Makes It Different

Most AI systems:

* Retrieve context → Generate answer

StatGuide:

* Retrieves context
* Evaluates relevance
* Rewrites queries if needed
* Dynamically switches between sources
* Generates grounded, reliable answers

---

## 💥 Key Features

* 🔁 **Self-Correcting RAG Pipeline**
  Retrieval → grading → query rewrite → re-retrieval → answer generation

* 🔀 **Adaptive Routing System**
  Automatically chooses between:

  * Vector search (FAISS)
  * Web search
  * Hybrid retrieval

* 📊 **RAG vs Non-RAG Comparison Mode**
  Compares answers and evaluates using RAGAS metrics

* 🧮 **Deterministic Solver**
  Solves numerical/statistical problems without relying only on LLM

* 🧠 **Session Memory**
  Supports follow-up queries with context awareness

* 🖼️ **OCR Input Support**
  Accepts image-based questions

---

## 🏗️ Architecture

```id="d3f9p3"
User Query
   ↓
Router (RAG / Web / Hybrid)
   ↓
Retriever (FAISS + BM25 + Web)
   ↓
Document Grader (LLM)
   ↓
Query Rewriter (if needed)
   ↓
Generator (LLM)
   ↓
Final Answer
```

---

## ⚙️ Tech Stack

* Flask
* LangChain + LangGraph
* Groq LLM
* FAISS (Vector DB)
* HuggingFace Embeddings
* RAGAS (evaluation)
* Tesseract OCR

---

## 🔬 Engineering Highlights

* Modular service-based architecture
* Hybrid retrieval (semantic + keyword)
* LLM-based document relevance grading
* Query rewriting for retrieval improvement
* Dynamic routing between knowledge sources
* Environment-based secure configuration

---

## ⚠️ Challenges Solved

* Poor retrieval in RAG → solved using grading + rewriting
* LLM hallucination → reduced via grounded context
* Numerical inaccuracies → solved with deterministic solver
* Evaluation issues → improved using RAGAS

---

## 🛡️ Security & Reliability

* Safe FAISS loading (env-controlled)
* No hardcoded debug mode
* Environment validation at startup
* Structured logging

---

## ⚠️ Limitations

* Limited automated tests
* Evaluation can be further improved
* Web routing can be optimized

---

## 🚀 Setup

### 1. Create Virtual Environment

```powershell id="g5eqpj"
py -m venv statenv
.\statenv\Scripts\activate
```

---

### 2. Install Dependencies

```powershell id="n8qgqg"
pip install flask python-dotenv langchain langgraph langchain-groq langchain-community langchain-huggingface langchain-text-splitters faiss-cpu pypdf requests ragas datasets pillow pytesseract rank-bm25
```

---

### 3. Configure `.env`

```env id="s3lf42"
GROQ_API_KEY=your_key
LLM_MODEL=your_model
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
TESSERACT_CMD=C:\path\to\tesseract.exe
FLASK_DEBUG=false
ALLOW_DANGEROUS_FAISS_DESERIALIZATION=false
```

---

### 4. Build Index

```python id="g9e3lo"
from src.vectorstore.index_pipeline import IndexBuilder
IndexBuilder("Stat_Book.pdf").build_index()
```

---

### 5. Run App

```powershell id="fd7s2a"
python app.py
```

Open:

```id="v87q2k"
http://localhost:5000
```

---

## 💡 Future Work

* Personalized learning paths
* Weak topic detection
* Multi-subject expansion
* Better evaluation metrics
* Full test coverage

---

## 🧠 Key Insight

> Most RAG systems fail due to poor retrieval.
> StatGuide actively detects and corrects retrieval failures.

---

## 👨‍💻 Author

Priyanshu Kumar
