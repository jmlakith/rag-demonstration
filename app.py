import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Company Copilot", page_icon="🤖")
st.title("Company Copilot 🤖")

# -----------------------------
# Load Company Documents
# -----------------------------
@st.cache_resource
def load_vector_db():
    docs = []

    for file in os.listdir("docs"):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join("docs", file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Local LLM (Ollama)
# -----------------------------
llm = Ollama(
    model="mistral",
    temperature=0
)

# -----------------------------
# RAG Chain
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# Copilot Chat UI
# -----------------------------
question = st.chat_input("Ask a company question")

if question:
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        result = qa(question)

    st.chat_message("assistant").write(result["result"])

    with st.expander("Sources used"):
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content)
