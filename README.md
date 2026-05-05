# YourStatGuide

YourStatGuide is an AI-powered statistics learning platform built with Flask. It is designed as a single study workspace where a user can ask statistics questions, explore concepts, practice topic-wise questions, generate notes, and create visualizations.

The main RAG knowledge source used in this project is:

- `Fundamentals of Mathematical Statistics (A Modern Approach)` by S.C. Gupta and V.K. Kapoor

## What the project does

This project brings together multiple learning tasks in one web app:

- Ask AI: RAG-based statistics question answering
- Concepts: concept-wise learning support
- Practice: topic-wise practice support
- Notes: note generation and PDF export
- Visualizations: statistical plots and interpretation
- Math Rendering: formulas rendered using MathJax
- Authentication: sign up / sign in before using the app

## Tech stack

- Frontend: HTML, CSS, JavaScript
- Backend: Flask
- LLM workflow: LangChain, LangGraph
- LLM provider: Groq
- Embeddings: HuggingFace Embeddings
- Vector store: FAISS
- Visualization: Matplotlib, Seaborn
- Math rendering: MathJax

## Repository structure

```text
ISI_Project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- report_template.tex
|-- src/
|   |-- llm_model.py
|   |-- graph_builder.py
|   |-- graph_node.py
|   |-- rag_generator.py
|   |-- retrieval_grader.py
|   |-- question_rewriter.py
|   |-- video_search.py
|   `-- vectorstore/
|       |-- vector.py
|       `-- index_pipeline.py
|-- templates/
|-- static/
|-- faiss_index/        # generated locally
|-- data/               # generated locally
`-- Stat_Book.pdf       # kept locally
```

## Before you run it

This project depends on a few local files and API keys.

You will need:

- Python 3.10 or newer
- a `.env` file with valid API keys
- `Stat_Book.pdf` in the project root
- a FAISS index inside `faiss_index/`

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ISI_Project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv statenv
.\statenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the environment file

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Then fill the values in `.env`.

Required variables:

- `GROQ_API_KEY`
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `FLASK_SECRET_KEY`
- `YOUTUBE`

Recommended values:

```env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=llama-3.1-8b-instant
```

## Required local files

### 1. Source book PDF

Place the source PDF in the project root with this exact name:

```text
Stat_Book.pdf
```

This file is used as the main knowledge source for the RAG system.

### 2. FAISS index

The app loads the vector database from:

```text
faiss_index/
```

If this folder is missing, you must build the index first.

## How to build the FAISS index

Run this in Python:

```python
from src.vectorstore.index_pipeline import IndexBuilder

IndexBuilder("Stat_Book.pdf").build_index()
```

What this does:

- loads the PDF
- splits it into chunks
- generates embeddings
- builds the FAISS index
- saves the index locally in `faiss_index/`

## How to run the project

After setup is complete, start the Flask app:

```powershell
python app.py
```

Then open this in your browser:

```text
http://127.0.0.1:5000
```

## How the app works

### Landing and authentication

- open the landing page
- create an account or sign in
- enter the protected workspace

### Inside the app

You can use:

- Ask AI
- Concepts
- Practice
- Visualizations
- Notes

## Notes for anyone cloning the project

- `.env` is intentionally not committed
- `faiss_index/` is ignored in `.gitignore`
- `data/` is generated locally
- `Stat_Book.pdf` is not committed

So a fresh clone will not run unless:

1. you create a valid `.env`
2. you add `Stat_Book.pdf`
3. you generate the FAISS index

## Current limitations

- authentication is local JSON-based
- the app expects the source PDF locally
- first run may take time if the embedding model is not cached
- HuggingFace model download may require stable internet on first setup

## Main files

- `app.py` - main Flask app and routes
- `src/vectorstore/vector.py` - PDF loading, chunking, embeddings, FAISS
- `src/vectorstore/index_pipeline.py` - FAISS index building
- `src/graph_builder.py` - LangGraph workflow
- `src/graph_node.py` - retrieval, grading, rewrite, generation nodes
- `src/rag_generator.py` - final answer generation prompt
- `src/retrieval_grader.py` - relevance grading
- `src/question_rewriter.py` - query rewriting
- `src/video_search.py` - YouTube search integration

## Authors

- Priyanshu Kumar 
- Sahil 

Mentor:

- Bidisha Dobe
