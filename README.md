## The Technical Stack 

The application uses a modern, completely local, privacy-first AI architecture. Here is what each component is responsible for:

*   **Docker Compose (Infrastructure):** Acts as the backbone. It runs three isolated containers (Streamlit, Qdrant, Ollama) and networks them together so they can communicate internally on your machine without exposing data to the internet. It also mounts persistent volumes so your vector data and downloaded models survive reboots.
*   **Streamlit & Python (Frontend):** The user interface. It handles user authentication (`streamlit-authenticator`), captures your text inputs, manages session states (like chat history), and streams the final output to your screen.
*   **LangChain (The Orchestrator):** The "glue" written in Python. It provides the tools (`TextSplitter`, `WebBaseLoader`, `ChatPromptTemplate`) to parse documents, format prompts, and chain the different services together.
*   **Qdrant (Vector Database):** A highly efficient storage engine. Instead of storing raw text, it stores "embeddings" (arrays of numbers representing the semantic meaning of text) and performs lightning-fast mathematical similarity searches to find relevant information.
*   **Ollama (LLM Engine):** The AI brain. It runs entirely on your local hardware and serves two distinct models simultaneously via REST APIs:
    *   `nomic-embed-text`: Converts your PDFs, web pages, and queries into mathematical vectors.
    *   `llama3.2` (or `1b`): The generative model that reads the retrieved context and writes the final human-readable answer.

---

## Step-by-Step Execution Flow

### Phase 1: Data Ingestion (Adding Knowledge)

1.  **Parse:** You upload a PDF or paste a URL in Streamlit. LangChain extracts the raw text using tools like `PyPDFLoader` or `RecursiveUrlLoader`.
2.  **Chunk:** LangChain's `RecursiveCharacterTextSplitter` chops the massive text block into smaller, overlapping chunks (500 characters each) so the LLM can easily digest them later.
3.  **Embed:** Streamlit sends these chunks to Ollama's `nomic-embed-text` model, which translates the human text into high-dimensional numerical vectors.
4.  **Store:** These vectors, along with their source metadata (file name or URL), are saved permanently into the Qdrant database container.

### Phase 2: Retrieval-Augmented Generation (Answering Questions)

1.  **Query Processing:** You type a question. Streamlit takes that exact question and asks Ollama (`nomic-embed-text`) to turn it into a vector.
2.  **Similarity Search:** The vector query is sent to Qdrant. Qdrant compares the math of your question against the math of all your stored documents and returns the top 3 closest matching chunks.
3.  **Prompt Assembly:** LangChain takes the retrieved document chunks, your chat history, and your current question, and injects them all into a strict instruction template.
4.  **Generation:** This massive prompt is sent to Ollama (`llama3.2`). Because the model now has the exact document snippets in its "short-term memory," it can read them and generate an accurate, cited answer, streaming the tokens back to your Streamlit UI one by one.

---

## Deployment & Execution

To deploy the application, execute the following commands in your terminal:

1.  **Pull the code repository** to your local machine.
2.  **Pull the Docker image:**
    ```bash
    docker pull zainul1114/local_rag_project:latest
    ```
3.  **Run the containers:**
    ```bash
    docker compose up -d
    ```
4.  **Access the User Interface:**
    Open your web browser and navigate to: [http://192.168.1.2:8501/](http://192.168.1.2:8501/)
