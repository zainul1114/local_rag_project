import os
import time
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd

from qdrant_client import QdrantClient
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Docker internal endpoints
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DATA_DIR = "./data"
COLLECTION_NAME = "docker_rag"

# --- Page Configuration ---
st.set_page_config(page_title="RAG & General AI Workspace", page_icon="⚡", layout="wide")

# --- Authentication ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"], config["cookie"]["name"], config["cookie"]["key"], config["cookie"]["expiry_days"]
)

try:
    authenticator.login("main")
except TypeError:
    authenticator.login(location="main")

if st.session_state.get("authentication_status") == False:
    st.error("Username/password is incorrect")
    st.stop()
elif st.session_state.get("authentication_status") == None:
    st.warning("Please enter your username and password")
    st.stop()

# --- Helper Function: Load Various File Types ---
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return PyPDFLoader(file_path).load()
        elif ext == '.docx':
            return Docx2txtLoader(file_path).load()
        elif ext == '.txt':
            return TextLoader(file_path).load()
        elif ext == '.csv':
            return CSVLoader(file_path).load()
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
            temp_csv_path = file_path + ".csv"
            df.to_csv(temp_csv_path, index=False)
            return CSVLoader(temp_csv_path).load()
        else:
            return []
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

# --- Persistent Vector Store Setup ---
@st.cache_resource
def get_vector_store():
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    return vector_store, client

# --- Main App Execution ---
if st.session_state.get("authentication_status"):
    authenticator.logout("Logout", "sidebar")
    st.sidebar.markdown(f"Welcome, **{st.session_state.get('name')}**!")
    st.sidebar.divider()

    vector_store, qdrant_client = get_vector_store()

    # Get Qdrant chunk status
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
    except Exception:
        points_count = 0

    st.sidebar.markdown(f"**Vector Database:** `{points_count}` chunks indexed")

    # --- Mode Selection ---
    st.sidebar.header("⚙️ Mode Settings")
    mode = st.sidebar.radio(
        "Select Operation Mode:",
        ["Document RAG Mode", "Direct LLM Mode (General/Code Question)"]
    )

    # --- Document Upload & Index Timing ---
    if mode == "Document RAG Mode":
        st.sidebar.header("📁 Document Management")
        uploaded_files = st.sidebar.file_uploader(
            "Upload Files", 
            type=["pdf", "txt", "docx", "xlsx", "xls", "csv"], 
            accept_multiple_files=True
        )
        
        process_btn = st.sidebar.button("Process & Index")

        if process_btn and uploaded_files:
            with st.spinner("Processing & indexing documents..."):
                start_index_time = time.perf_counter()
                
                os.makedirs(DATA_DIR, exist_ok=True)
                all_docs = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    docs = load_document(file_path)
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)

                if all_docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(all_docs)
                    vector_store.add_documents(chunks)
                    
                    elapsed_index_time = time.perf_counter() - start_index_time
                    st.sidebar.success(f"✅ Indexed in **{elapsed_index_time:.2f}s**!")
                    st.rerun()

    # --- Main Content Area ---
    st.title("⚡ AI Workspace")
    st.caption(f"Current Mode: **{mode}**")
    st.divider()

    user_query = st.text_input("Ask your question:", placeholder="Type a document question or code request...")

    if user_query:
        llm = Ollama(model="llama3.2", temperature=0.0, base_url=OLLAMA_BASE_URL)

        # MODE 1: Document RAG Mode
        if mode == "Document RAG Mode":
            if points_count == 0:
                st.warning("No documents indexed in Qdrant yet. Upload files in the sidebar first.")
            else:
                with st.spinner("Executing RAG Pipeline..."):
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                    # 1. Measure Retrieval Time
                    start_retrieval = time.perf_counter()
                    retrieved_docs = retriever.invoke(user_query)
                    retrieval_time = time.perf_counter() - start_retrieval

                    # 2. Measure Generation Time
                    template = """You are a reliable assistant. Answer using only the context below. 
If context is insufficient, say: "I don't have enough information."
Include citations.

Context:
{context}

Question: {question}

Answer with Citations:"""
                    prompt = ChatPromptTemplate.from_template(template)

                    def format_docs(docs):
                        return "\n\n".join([f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs])

                    rag_chain = (
                        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    start_gen = time.perf_counter()
                    response = rag_chain.invoke(user_query)
                    gen_time = time.perf_counter() - start_gen

                    # Display Response
                    st.markdown("### Answer")
                    st.write(response)

                    # Display Latency Metrics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieval Time (Qdrant)", f"{retrieval_time:.3f} s")
                    col2.metric("LLM Generation Time", f"{gen_time:.2f} s")
                    col3.metric("Total Processing Time", f"{(retrieval_time + gen_time):.2f} s")

                    with st.expander("View Retrieved Context Snippets"):
                        for idx, d in enumerate(retrieved_docs):
                            st.markdown(f"**Snippet {idx + 1} | Source:** `{d.metadata.get('source')}`")
                            st.caption(d.page_content)

        # MODE 2: Direct LLM Mode (General & Code Questions)
        else:
            with st.spinner("Generating answer with Llama 3.2..."):
                start_gen = time.perf_counter()
                response = llm.invoke(user_query)
                gen_time = time.perf_counter() - start_gen

                st.markdown("### Answer")
                st.write(response)

                st.divider()
                st.metric("LLM Generation Time", f"{gen_time:.2f} s")
