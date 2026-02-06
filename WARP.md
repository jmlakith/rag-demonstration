# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) demonstration called "Company Copilot" that uses Streamlit, LangChain, FAISS, and Ollama to create an intelligent company policy assistant. The application loads company documents, creates vector embeddings, and uses a local LLM to answer questions with source attribution.

## Essential Commands

### Setup and Installation
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and setup Ollama (if not already installed)
brew install ollama
ollama pull mistral
```

### Running the Application
```bash
# Start Ollama service (required)
ollama serve

# Run the Streamlit app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Development
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Test with different LLM models
ollama pull llama3  # Pull alternative model
# Then modify app.py line 48 to use the new model
```

## Architecture

### Core Components

**app.py** - Main application with three key sections:
1. **Document Loading & Vectorization** (`load_vector_db()` function):
   - Loads all `.txt` files from `docs/` directory
   - Chunks documents using `RecursiveCharacterTextSplitter` (300 chars, 50 overlap)
   - Creates embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
   - Stores vectors in FAISS for efficient similarity search
   - Uses `@st.cache_resource` to avoid reloading on each query

2. **LLM Integration**:
   - Uses Ollama for local inference (default: Mistral model)
   - Temperature set to 0 for deterministic responses
   - Connects to Ollama service running on localhost

3. **RAG Chain**:
   - `RetrievalQA` chain combines retriever + LLM
   - Retrieves k=3 most relevant document chunks per query
   - Returns both answer and source documents for transparency

**docs/** - Document storage:
- All `.txt` files are automatically loaded
- Currently contains `employee_handbook.txt` and `it_policy.txt`
- Add new documents here and restart the app to include them

### Key Configuration Parameters

**Chunking** (lines 28-31):
- `chunk_size=300` - Size of text chunks for embedding
- `chunk_overlap=50` - Overlap between chunks for context continuity

**Retrieval** (line 42):
- `k=3` - Number of document chunks retrieved per query

**LLM** (lines 47-50):
- `model="mistral"` - Ollama model to use
- `temperature=0` - Deterministic output

## Adding Documents

Simply add `.txt` files to the `docs/` directory. The app automatically loads all text files on startup. Supported formats can be extended by modifying the loader logic in `load_vector_db()`.

## Troubleshooting

**Ollama connection errors**: Ensure `ollama serve` is running in a separate terminal.

**Model not found**: Run `ollama pull mistral` (or your chosen model).

**Empty responses**: Check that documents exist in `docs/` and contain text.

**Slow responses**: First query loads the embedding model; subsequent queries are faster due to caching.

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (local inference)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Python**: 3.10+
