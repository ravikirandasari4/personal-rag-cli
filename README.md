# personal_RAG_CLI
A minimal personal RAG CLI system for building a document vector index, querying with natural language, and summarizing source documents. Includes parser, embedder, vector store, and LLM retriever components for self-hosted knowledge retrieval response generation pipelines CLI ready.

# personal_rag

A lightweight Python personal Retrieval-Augmented Generation (RAG) system for building a searchable semantic index over documents and asking natural language questions.

## Overview

`personal_rag` assembles a simple RAG pipeline using:
- document parsing for `.txt`, `.pdf`, and `.docx`
- sentence embedding generation with `sentence-transformers`
- vector similarity search with `faiss`
- response generation using an Ollama-backed LLM

The project is designed around a command-line interface in `main.py` and modular components in `src/`.

## Features

- Parse supported documents from `documents/`
- Chunk long text and generate embeddings
- Store embeddings in a persistent FAISS index
- Load an existing index for repeated use
- Ask questions over the document collection
- Generate a summary of the document contents

## Repository Structure

- `main.py` — CLI entrypoint for building/loading the index, querying, and summarizing
- `documents/` — document input folder (create or populate with files before indexing)
- `data/vector_store` — saved vector store path used by the app
- `src/document_parser.py` — document ingestion and file format parsing
- `src/embedder.py` — chunking and embedding generation
- `src/vector_store.py` — FAISS vector store, save/load, and search
- `src/retriever.py` — LLM prompt assembly and answer generation
- `src/rag_system.py` — orchestration of the full RAG workflow

## Requirements

Recommended Python dependencies:

```bash
pip install -r requirements.txt
```

> If you use a GPU or a different FAISS build, install the appropriate `faiss` package for your environment.

## Setup

1. Create or activate a Python environment.
2. Install the required packages.
3. Place your documents into the `documents/` directory. Supported formats:
   - `.txt`
   - `.pdf`
   - `.docx`

## Usage

Run the CLI from the repository root:

```bash
python3 main.py
```

Then use one of the supported commands:

- `build` — parse documents and build a new vector index
- `load` — load an existing index from `data/vector_store`
- `query <question>` — ask the system a question about the indexed documents
- `summarize` — generate a summary of the indexed content
- `quit` — exit the CLI

### Example session

```text
Personal RAG System
==================
Commands:
1. 'build' - Build index from documents/
2. 'load' - Load existing index
3. 'query <question>' - Ask a question
4. 'summarize' - Get document summary
5. 'quit' - Exit

Enter command: build
Enter command: query What are the main findings?
Enter command: summarize
Enter command: quit
```

## How it works

1. `main.py` initializes `PersonalRAG` and manages CLI commands.
2. `DocumentParser` reads supported files and returns text along with metadata.
3. `TextEmbedder` splits long documents into overlapping chunks and computes embeddings using a sentence transformer.
4. `VectorStore` stores normalized embeddings in FAISS and enables similarity search.
5. `LLMRetriever` sends retrieved document chunks to the Ollama model and returns an answer or summary.

## Notes

- The index is saved to `data/vector_store.index` and `data/vector_store.pkl`.
- If the `documents/` folder is empty, the `build` command will prompt for documents.
- For best results, keep your documents concise and relevant.
- If the Ollama client fails, check your Ollama installation and model availability.

## Extending the project

Suggestions for future improvements:

- support additional file formats such as `.md` or `.html`
- add a web UI or Gradio front-end
- add batching for larger document collections
- integrate a configurable retrieval prompt or more advanced reranking
- add a requirements file and `.gitignore`

## License

This project does not include a license file by default.

---

`personal_rag` is ideal for experimenting with local document search and RAG workflows using Python, FAISS, and Ollama.