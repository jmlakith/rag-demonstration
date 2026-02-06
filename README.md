# NotebookLM Lite 🤖

A lightweight, privacy-focused RAG (Retrieval-Augmented Generation) application that runs 100% locally. Think of it as your own personal NotebookLM that works offline with your private documents.

Built with Streamlit, LangChain, FAISS, and Ollama - no cloud APIs, no data sharing, completely free.

## ✨ Features

- 🔒 **100% Local & Private**: All processing happens on your machine
- 📚 **Easy Document Loading**: Drop `.txt` files in the `docs/` folder
- 🔍 **Semantic Search**: Uses FAISS vector database for intelligent retrieval
- 🤖 **Multiple LLM Support**: Works with any Ollama model (Mistral, Llama, etc.)
- 📝 **Source Attribution**: Always shows which documents were used
- 💬 **Clean Chat Interface**: Simple, intuitive Streamlit UI
- 🆓 **Completely Free**: No API costs, no subscriptions

## 🎯 Use Cases

- Personal knowledge base from notes and documents
- Company policy assistant
- Research paper Q&A
- Study material helper
- Technical documentation explorer

## 📋 Prerequisites

- **Python 3.10 or higher**
- **4GB+ RAM** (8GB recommended for larger documents)
- **macOS, Linux, or Windows**

## 🚀 Quick Start

### 1. Install Ollama

**macOS/Linux:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Download from [ollama.ai](https://ollama.ai/download/windows)
- Run the installer

### 2. Download an LLM Model

```bash
# Start Ollama service (runs in background)
ollama serve

# In a new terminal, pull Mistral (recommended, ~4GB)
ollama pull mistral

# OR use Llama 3 (larger but more capable, ~8GB)
ollama pull llama3
```

### 3. Clone This Repository

```bash
git clone https://github.com/jmlakith/rag-demonstration.git
cd rag-demonstration
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📁 Adding Your Own Documents

1. Add any `.txt` files to the `docs/` folder
2. Restart the app
3. Start asking questions!

**Tips:**
- Break large documents into smaller files for better context
- Use descriptive filenames (they appear in source citations)
- Plain text works best, but you can extend the loader for PDFs

## 💡 Example Questions

With the default demo documents:
- "Can I work remotely?"
- "Do I need manager approval for remote work?"
- "What devices can I use for production access?"
- "Do I need to use a VPN?"

## How It Works

1. **Document Loading**: All `.txt` files in the `docs/` directory are loaded
2. **Chunking**: Documents are split into manageable chunks (300 characters with 50 character overlap)
3. **Embedding**: Text is converted to vectors using Sentence Transformers
4. **Vector Storage**: Embeddings are stored in FAISS for fast retrieval
5. **Query Processing**: User questions are embedded and matched against document chunks
6. **LLM Generation**: Ollama generates answers based on retrieved context
7. **Source Display**: Shows which document sections were used

## Adding More Documents

Simply add more `.txt` files to the `docs/` directory. The app will automatically load them on the next restart.

## Customization

### Change the LLM Model

Edit `app.py` line 48:

```python
llm = Ollama(
    model="llama3",  # Change from "mistral" to "llama3" or other models
    temperature=0
)
```

### Adjust Retrieval Settings

Modify the number of document chunks retrieved (line 42):

```python
retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # Retrieve 5 chunks instead of 3
```

### Customize Chunk Size

Adjust chunking parameters (lines 28-31):

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for larger chunks
    chunk_overlap=100    # Increase overlap for better context
)
```

## Troubleshooting

### Ollama Connection Error

Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found

Pull the model first:
```bash
ollama pull mistral
```

### Import Errors

Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

## License

MIT

## Contributing

Feel free to add more company documents or enhance the RAG pipeline!
