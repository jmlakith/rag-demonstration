# Company Copilot 🤖

A RAG (Retrieval-Augmented Generation) demonstration using Streamlit, LangChain, FAISS, and Ollama to create an intelligent company policy assistant.

## Features

- **Document Loading**: Automatically loads company documents from the `docs/` directory
- **Vector Storage**: Uses FAISS for efficient semantic search
- **Local LLM**: Powered by Ollama (Mistral) for privacy-focused inference
- **Source Attribution**: Shows which documents were used to answer each question
- **Interactive UI**: Clean Streamlit chat interface

## Prerequisites

### Software Requirements

- Python 3.10+
- pip
- Ollama installed and running

### Install Ollama

```bash
brew install ollama
ollama pull mistral
```

You can replace `mistral` with `llama3` if you prefer.

## Project Structure

```
rag-demonstration/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore patterns
└── docs/                      # Company documents
    ├── employee_handbook.txt  # Remote work policy
    └── it_policy.txt          # IT and security policies
```

## Installation

1. **Clone or navigate to the project directory**

```bash
cd rag-demonstration
```

2. **Create a virtual environment (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

1. **Make sure Ollama is running**

```bash
ollama serve
```

2. **Run the Streamlit app**

```bash
streamlit run app.py
```

3. **Open your browser** to the URL shown (typically `http://localhost:8501`)

4. **Ask questions** like:
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
