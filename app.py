import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
prompt_template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Copilot Chat UI
# -----------------------------
question = st.chat_input("Ask a company question")

if question:
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        # Get answer
        answer = rag_chain.invoke(question)
        # Get source documents
        source_docs = retriever.invoke(question)

    st.chat_message("assistant").write(answer)

    with st.expander("Sources used"):
        for i, doc in enumerate(source_docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content)
