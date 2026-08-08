import os
import time
import json
import urllib.request
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
from bs4 import BeautifulSoup

from qdrant_client import QdrantClient
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader,
    RecursiveUrlLoader
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
st.set_page_config(
    page_title="Enterprise AI Workspace", 
    page_icon="🎡", 
    layout="wide"
)

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

# --- Helper Function: Load Documents ---
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

# --- Helper Function: Fetch Ollama Models ---
def fetch_ollama_models():
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", []) if "embed" not in m["name"].lower()]
            return models if models else ["llama3.2:latest", "llama3.2:1b"]
    except Exception:
        return ["llama3.2:latest", "llama3.2:1b"]

# --- Initialize Sequential Messages History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_formatted_history():
    history = ""
    for msg in st.session_state.messages[-6:]:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return history

# --- Main App Execution ---
if st.session_state.get("authentication_status"):
    vector_store, qdrant_client = get_vector_store()

    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
    except Exception:
        points_count = 0

    # --- Sidebar Setup ---
    with st.sidebar:
        st.markdown(f"Welcome, **{st.session_state.get('name')}**!")
        st.caption("Tech Stack: Docker Compose - Ollama, Qdrant, Streamlit and Python")
        authenticator.logout("Logout", "sidebar")
        st.divider()

        st.markdown(f"**Vector Database:** `{points_count}` chunks indexed")

        st.header("⚙️ Mode Settings")
        mode = st.radio(
            "Select Operation Mode:",
            ["Document RAG Mode", "Direct LLM Mode (General/Code Question)"]
        )

        selected_model = "llama3.2" 
        if mode == "Direct LLM Mode (General/Code Question)":
            st.subheader("🧠 Model Selection")
            available_models = fetch_ollama_models()
            selected_model = st.selectbox("Choose an LLM:", available_models)

        st.divider()
        
        # --- Data Ingestion ---
        with st.expander("📚 Add Data to Knowledge Base", expanded=True):
            st.markdown("**📁 Upload Documents**")
            uploaded_files = st.file_uploader(
                "Upload Files", 
                type=["pdf", "txt", "docx", "xlsx", "xls", "csv"], 
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            process_btn = st.button("Process Files")

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
                        st.success(f"✅ Indexed in **{elapsed_index_time:.2f}s**!")
                        st.rerun()

            st.divider()
            
            st.markdown("**🌐 Ingest Web URL**")
            url_input = st.text_input("Enter documentation URL:", label_visibility="collapsed", placeholder="https://...")
            url_btn = st.button("Process URL")

            if url_btn and url_input:
                with st.spinner(f"Crawling and scraping {url_input} (this may take a minute)..."):
                    start_index_time = time.perf_counter()
                    try:
                        def bs4_extractor(html: str) -> str:
                            soup = BeautifulSoup(html, "html.parser")
                            return soup.get_text(separator=" ", strip=True)

                        loader = RecursiveUrlLoader(
                            url=url_input,
                            max_depth=2, 
                            extractor=bs4_extractor
                        )
                        web_docs = loader.load()
                        
                        if web_docs:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            chunks = text_splitter.split_documents(web_docs)
                            vector_store.add_documents(chunks)
                            
                            elapsed_index_time = time.perf_counter() - start_index_time
                            st.success(f"✅ Crawled and Indexed {len(web_docs)} pages in **{elapsed_index_time:.2f}s**!")
                            st.rerun()
                        else:
                            st.warning("No content found at the provided URL.")
                    except Exception as e:
                        st.error(f"Failed to crawl URL: {e}")

    # --- Main Content Header with Colorful Icon styling ---
    st.markdown(
        """
        <h1 style='display: flex; align-items: center; gap: 15px;'>
            <span style='font-size: 2.5rem;'> 🎡 </span>
            <span style='background: linear-gradient(45deg, #FE6B8B, #FF8E53, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Enterprise AI Workspace
            </span>
        </h1>
        """, 
        unsafe_allow_html=True
    )
    
    st.caption(f"Current Mode: **{mode}**")
    st.divider()

    # --- Render All Previous Q&A Turns ---
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**Question:** {msg['content']}")
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                
                if "telemetry" in msg and msg["telemetry"]:
                    t = msg["telemetry"]
                    st.divider()
                    if t.get("retrieval", 0) > 0:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Retrieval Time (Qdrant)", f"{t['retrieval']:.3f} s")
                        col2.metric("LLM Generation Time", f"{t['gen']:.2f} s")
                        col3.metric("Total Processing Time", f"{t['total']:.2f} s")
                    else:
                        st.metric("LLM Generation Time", f"{t['gen']:.2f} s")
                
                if "sources" in msg and msg["sources"]:
                    with st.expander("View Retrieved Context Snippets"):
                        for idx, d in enumerate(msg["sources"]):
                            st.markdown(f"**Snippet {idx + 1} | Source:** `{d['source']}`")
                            st.caption(d['content'])

    # --- Input Box for Asking Questions ---
    user_query = st.chat_input("Ask a question about your documents, code, or general topics...")

    if user_query:
        # Display & record user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(f"**Question:** {user_query}")

        # Process Answer
        with st.chat_message("assistant"):
            if mode == "Document RAG Mode":
                llm = Ollama(model="llama3.2", temperature=0.0, base_url=OLLAMA_BASE_URL)
                
                if points_count == 0:
                    warning_msg = "⚠️ No documents indexed in Qdrant yet. Upload files or URLs in the sidebar first."
                    st.warning(warning_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                else:
                    with st.spinner("Executing RAG Pipeline..."):
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                        start_retrieval = time.perf_counter()
                        retrieved_docs = retriever.invoke(user_query)
                        retrieval_time = time.perf_counter() - start_retrieval

                        template = """You are a reliable assistant. Answer using only the context below. 
If context is insufficient, say: "I don't have enough information."
Include citations pointing to the source URL or file.

Recent Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer with Citations:"""
                        prompt = ChatPromptTemplate.from_template(template)

                        def format_docs(docs):
                            return "\n\n".join([f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs])

                        # Evaluate the history in the main thread first to avoid LangChain threading errors
                        current_chat_history = get_formatted_history()

                        rag_chain = (
                            {
                                "context": lambda x: format_docs(retrieved_docs), 
                                "chat_history": lambda x: current_chat_history,
                                "question": RunnablePassthrough()
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )

                        start_gen = time.perf_counter()
                        response = rag_chain.invoke(user_query)
                        gen_time = time.perf_counter() - start_gen

                        st.markdown(response)

                        sources_list = [
                            {"source": d.metadata.get("source", "Unknown"), "content": d.page_content}
                            for d in retrieved_docs
                        ]
                        telemetry = {
                            "retrieval": retrieval_time, 
                            "gen": gen_time, 
                            "total": retrieval_time + gen_time
                        }

                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Retrieval Time (Qdrant)", f"{retrieval_time:.3f} s")
                        col2.metric("LLM Generation Time", f"{gen_time:.2f} s")
                        col3.metric("Total Processing Time", f"{(retrieval_time + gen_time):.2f} s")

                        with st.expander("View Retrieved Context Snippets"):
                            for idx, d in enumerate(sources_list):
                                st.markdown(f"**Snippet {idx + 1} | Source:** `{d['source']}`")
                                st.caption(d['content'])

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "telemetry": telemetry,
                            "sources": sources_list
                        })

            else:
                llm = Ollama(model=selected_model, temperature=0.0, base_url=OLLAMA_BASE_URL)
                
                with st.spinner(f"Connecting to {selected_model}..."):
                    start_gen = time.perf_counter()
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    contextual_prompt = f"Previous conversation:\n{get_formatted_history()}\n\nUser Question: {user_query}"
                    
                    for chunk in llm.stream(contextual_prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    gen_time = time.perf_counter() - start_gen

                    telemetry = {"retrieval": 0.0, "gen": gen_time, "total": gen_time}
                    st.divider()
                    st.metric("LLM Generation Time", f"{gen_time:.2f} s")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "telemetry": telemetry
                    })
