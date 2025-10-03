# LegalMate AI: Core RAG Prototype (MiniLM + FAISS + Gemini)

This repository contains the foundational Machine Learning components for the **LegalMate AI** project.  
The architecture uses **Retrieval-Augmented Generation (RAG)** to provide factually grounded legal advice by combining:

- **MiniLM + FAISS** → for fast vector retrieval  
- **Gemini LLM** → for text generation and simplification  

---

## 1. Architecture Overview

The system operates in two phases:

**Indexing (Offline):**  
- Large legal documents (Supreme Court Judgments, Bare Acts) are converted into embeddings (numerical vectors).  
- Stored in a searchable index using **FAISS**.  

**Runtime (Online):**  
- A user query is converted into a vector.  
- FAISS retrieves the most relevant legal text.  
- The retrieved text is passed to **Gemini LLM**, which generates a simplified and accurate answer **only** based on the provided context.  

---

## 2. Project Structure

```

LAWLLM
└── legalmate-ai
├── ai-models/
│   ├── data/                   # RAW INPUT: Downloaded ZIPs/PDFs (IGNORED by Git)
│   │   └── supreme_court/      # SC Judgment ZIPs
│   └── rag/                    # SCRIPTS: Core ML logic
│       ├── index_sc_cases_incremental.py  # Builds FAISS index from SC data
│       └── run_query.py                   # End-to-end RAG query execution
├── embeddings/                 # OUTPUT: Searchable indexes (.bin, .pkl) (IGNORED by Git)
├── legalmate_env/              # Python Virtual Environment (IGNORED by Git)
└── .gitignore

````

---

## 3. Setup and Dependencies

### A. Environment Setup

```bash
# 1. Clone this repository

# 2. Create Python Virtual Environment (venv)
python -m venv legalmate_env

# 3. Activate venv
# On Windows
.\legalmate_env\Scripts\activate
# On macOS/Linux
source legalmate_env/bin/activate
````

### B. Install Dependencies

```bash
pip install transformers sentence-transformers faiss-cpu pypdf numpy
pip install google-genai
```

### C. Configure Gemini API Key

```bash
# On macOS/Linux
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# On Windows (Command Prompt)
set GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

---

## 4. Data Acquisition (Indexing Phase)

### Step 1: Install AWS CLI (Prerequisite)

Ensure AWS CLI is installed so the `aws` command is recognized.

### Step 2: Download Supreme Court Judgments (ZIPs)

```bash
cd ai-models/data/supreme_court
aws s3 cp s3://indian-supreme-court-judgments/data/zip/ . \
    --recursive --exclude "*" --include "*.zip" --no-sign-request
```

### Step 3: Run the Indexing Script

```bash
# From project root
python ai-models/rag/index_sc_cases_incremental.py
```

> **Note:**
>
> * If you need to index the Constitution PDF:
>
>   1. Run `create_vector_store.py`
>   2. Modify `index_sc_cases_incremental.py` to load and merge that index at startup

---

## 5. Running the End-to-End RAG Query

Once indexing is complete, the final `.bin` and `.pkl` files in the `embeddings/` folder can be queried.

```bash
# From project root
python ai-models/rag/run_query.py
```

This script will:

* Take a query
* Retrieve the most relevant case law
* Generate simplified legal advice using **Gemini LLM**

---

✅ With this setup, **LegalMate AI** provides factually grounded, simplified, and context-driven legal answers.
