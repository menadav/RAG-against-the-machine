# RAG-against-the-machine

## Description
This project consists of developing a Retrieval-Augmented Generation (RAG) system designed to analyze and answer questions based on the vLLM code repository. The primary goal is to build a tool that combines efficient information retrieval with natural language generation to provide accurate, evidence-based answers derived from the project's documentation and source code.

The system implements the following core functionalities:
- Ingestion
- Chunking
- Retrieval
- Answer Generation
- Evaluation and CLI

## Instructions
First, ensure you have installed the necessary project dependencies. Specifically, make sure to install the required transformer libraries and any other dependencies needed for the RAG system.
```
make install
```
Before running the system, you must ingest the data from the vLLM repository o the chunk size is configured between 150 and 2000.
```
make run || uv run -m src index --max_chunk_size 2000
```
Now you can use the command for use de IA
```
uv run python -m src answer "What are the key capabilities of Ray Serve LLM for vLLM deployment?"
```
## System Architecture
System ArchitectureThe RAG pipeline is structured into the following sequential components:
- **Data Ingestion:**  The system processes the vLLM repository files as the primary source of information.
- **Chunking:**  The text is segmented into smaller, manageable pieces, with a configurable chunk size (ranging from 150 to 2000 characters) to optimize for both Python code and Markdown documentation.
- **Indexing:**  A searchable index is created using either the TF-IDF or BM25 algorithm to structure and organize the information for efficient retrieval.
- **Retrieval:**  The system performs a similarity search to match user queries against the indexed knowledge base, ranking and pulling the most relevant snippets.
- **Generation:**  The retrieved context is passed to the Qwen/Qwen3-0.6B model, which generates a natural language answer grounded in the provided source material.

## Chunking Strategy
The system employs an adaptive, context-aware splitting strategy to preserve the logical structure of files. Key features include
- **File-Type Specific Logic:** The process uses different priority separators based on file extensions. For Python files, it prioritizes *class* and *def* blocks ; for Markdown, it favors headers.

- **Hierarchical Separation:** The system attempts to break at logical boundaries rather than fixed character limits. It uses a threshold-based approach (80-90% of max_chunk_size) to find the best structural break.

- **Fallback Mechanism:** When no high-priority logical separators are detected, the system defaults to generic separators like newlines or punctuation to ensure consistent chunk sizes.

## Retrieval Method
The system implements a Hybrid Retrieval approach that combines lexical and semantic search to ensure high recall and relevance:
- **Lexical Search (BM25):** We use the *bm25s* library to perform statistical term-based retrieval
- **Semantic Search (Vector Embeddings):** We integrate ChromaDB using the all-MiniLM-L6-v2 embedding model.
- **Ranking and Integration:** ***The find_top_k*** method retrieves candidates from both engines. It combines these results, filters out duplicates, and ranks the sources to provide the top-$k$ most relevant snippets.
- **Data Persistence:** Both the BM25 index and the ChromaDB collection are persisted to disk (data/processed/)

## Performance analysis
- Retrieval Quality: We evaluated the system using the recall@k metric. Our tests show that for k=5, the system consistently retrieves relevant chunks with a recall rate above , ensuring that the LLM has the necessary context to generate accurate answers.
- Latency:
- Indexing: Processing the entire vLLM repository completes within the 5 minutes for ChromaDB.
- Retrieval performance: We achieved a throughput of 1 queries per minute, well within the requirements.

## Design decisions
- Hybrid Retrieval: We combined BM25 for precise keyword-based search and ChromaDB (vector embeddings) for semantic search. This approach ensures high precision for technical queries regarding the vLLM repository, bridging the gap between exact terminology and conceptual relevance.

- Adaptive Chunking: Instead of relying on rigid character limits, we implemented a logic that prioritizes syntactic boundaries (such as functions, classes, and Markdown headers). This preserves the logical integrity of the code and documentation within each chunk.

- Model Efficiency: We selected the Qwen-0.6B model to achieve robust natural language generation while maintaining low resource consumption, ensuring the system remains responsive and efficient.

- Modular Architecture: The project was designed with a decoupled structure. This facilitates scalability, simplifies maintenance, and allows for isolated unit testing of each pipeline component (Ingestion, Retrieval, and Generation).

## Challenges faced
- Recall Optimization: Achieving a high recall@5 (>80%) required fine-tuning retrieval weights to balance breadth and precision.

- Efficiency: The system is optimized for performance-constrained hardware by using lightweight embedding models and efficient indexing.

- Architecture: The code follows a strictly modular design, ensuring clean, scalable, and testable components.

## Resources
Educational Content: I followed tutorials by to grasp the core concepts of vector databases and retrieval-augmented generation. 
- *https://www.youtube.com/@jokioki*

- Documentation and Libraries: Official documentation for bm25s, ChromaDB, and SentenceTransformers was essential for the implementation.
- II utilized AI-assisted tools to clarify complex debugging issues and to brainstorm architectural approaches, which significantly accelerated my understanding of Transformer-based models and their application in RAG pipelines.
