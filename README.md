# RAG AI System — Ask Your Documents, Privately

> A fully local Retrieval-Augmented Generation (RAG) pipeline with a Streamlit chat UI — upload any PDF, ask questions in plain English, and get grounded, streamed answers. Zero cloud. Zero API keys. Zero data leaving your machine.

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Qwen](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct-7C3AED?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

</p>

---

## What Is This?

RAG AI System is a fully local Retrieval-Augmented Generation pipeline that lets you upload a PDF and ask natural-language questions against it — with the answer streamed token-by-token directly to your browser.

Most document Q&A tools send your files to OpenAI or some other cloud service. This project keeps the entire pipeline on your machine: a MiniLM embedding model converts your document into semantic vectors, a pure-Python cosine-search index retrieves the most relevant paragraphs, and Qwen2.5-3B-Instruct generates a grounded answer from those paragraphs alone.

There are no external API calls, no database services, and no internet connection required after the models are downloaded.

| File | Responsibility |
|---|---|
| `main.py` | Streamlit UI — chat interface, file upload, streaming output |
| `local_llm.py` | `AiModel` — loads Qwen, builds RAG prompt, streams token output |
| `local_embedding.py` | `LocalEmbedding` — MiniLM wrapper and vector index interface |
| `vector_index.py` | `VectorIndex` — pure-stdlib in-memory cosine/Euclidean vector store |
| `pdf_reader.py` | `PdfReader` — PDF text extraction and paragraph splitting |

---

## Feature List

### Retrieval-Augmented Generation
- PDF ingestion with automatic paragraph extraction and whitespace normalisation
- Batch embedding of all paragraphs in a single GPU pass — no per-paragraph round trips
- Cosine similarity search over 384-dimensional MiniLM vectors
- Configurable top-k retrieval to pass the most relevant context chunks to the LLM
- Strict grounding prompt — the model is instructed to answer only from the provided document text

### Streaming Output
- `TextIteratorStreamer` runs `model.generate()` in a background daemon thread
- Tokens are yielded through the streamer queue without blocking the Streamlit main thread
- `st.write_stream()` renders tokens progressively as they arrive in the browser

### Caching and Session Management
- `@st.cache_resource` loads the LLM once per server process; Streamlit reruns do not reload model weights
- `st.session_state` persists the embedding index across follow-up questions without re-indexing
- Uploading a new PDF resets the session and builds a fresh index automatically

### Pure-Python Vector Store
- `VectorIndex` uses no NumPy, FAISS, or external vector database
- Vectors are L2-normalised at index time; similarity search reduces to a dot-product scan over stored vectors
- Both Euclidean and cosine metrics are available in the same class

---

## How It Works

1. **PDF → Paragraphs** — `PdfReader` reads every page, normalises whitespace, and splits on paragraph breaks, yielding a clean list of text chunks.

2. **Paragraphs → Vectors** — `LocalEmbedding.build_index()` batch-embeds all paragraphs in one GPU pass using `all-MiniLM-L6-v2` and stores the 384-dimensional L2-normalised vectors in `VectorIndex`.

3. **Question → Context** — at query time the question is embedded and compared against every stored vector by cosine distance; the top-k highest-scoring chunks are concatenated into a single context string.

4. **Context + Question → Answer** — `AiModel` wraps the context and question in a strict RAG prompt, launches `model.generate()` in a background daemon thread, and yields tokens through `TextIteratorStreamer` so `st.write_stream()` can render them progressively in the browser.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI (main.py)                 │
│                                                           │
│  Sidebar: [Load Model]  [Upload PDF]  [Status]           │
│  Main:    [Chat input]  →  [Streamed answer]             │
└─────────────────────────┬────────────────────────────────┘
                          │
             ┌─────────────▼──────────────┐
             │        PDF Ingestion        │
             │  PdfReader: page text →     │
             │  clean paragraph list       │
             └─────────────┬──────────────┘
                          │
             ┌─────────────▼──────────────┐
             │       LocalEmbedding        │
             │  all-MiniLM-L6-v2          │
             │  batch embed → 384-dim     │
             │  L2-normalised vectors      │
             └─────────────┬──────────────┘
                          │
             ┌─────────────▼──────────────┐
             │        VectorIndex          │
             │  in-memory cosine store     │
             │  (pure Python stdlib)       │
             └─────────────┬──────────────┘
                           │ top-k chunks
             ┌─────────────▼──────────────┐
             │          AiModel            │
             │  Qwen2.5-3B-Instruct        │
             │  RAG prompt assembly        │
             │  model.generate() in thread │
             └─────────────┬──────────────┘
                           │ token stream
             ┌─────────────▼──────────────┐
             │    TextIteratorStreamer      │
             │  + st.write_stream()        │
             │  → live tokens in browser   │
             └─────────────────────────────┘
```

**Model weight acquisition** (first run only):

```
HuggingFace Hub
  ├── sentence-transformers/all-MiniLM-L6-v2   → embedding model
  └── Qwen/Qwen2.5-3B-Instruct                 → generation model (~cached locally)
```

---

## Project Structure

```
RAG_AI_SYSTEM/
├── main.py                  # Streamlit UI — entry point
├── local_llm.py             # AiModel: loads Qwen, orchestrates RAG, streams output
├── local_embedding.py       # LocalEmbedding: MiniLM wrapper + index interface
├── vector_index.py          # VectorIndex: pure-stdlib cosine/Euclidean vector store
├── pdf_reader.py            # PdfReader: PDF → clean paragraph list
├── pdfs/                    # Drop your PDFs here (gitignored)
├── .env                     # HF_TOKEN (gitignored — create this yourself)
└── CLAUDE.md                # Guidance for Claude Code in this repo
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Earlier versions not tested |
| CUDA-capable GPU | Recommended; CPU inference works but is slow |
| Hugging Face account | Free — needed for `HF_TOKEN` |

### Setup

```bash
# Clone the repository
git clone https://github.com/Mohamad-Hachem/RAG_AI_SYSTEM.git
cd RAG_AI_SYSTEM

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS / Linux

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install transformers streamlit pypdf python-dotenv huggingface_hub
```

> **GPU note:** the `cu121` wheel targets CUDA 12.1. Find the right wheel for your GPU at [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For CPU-only: `pip install torch torchvision torchaudio`.

### Add your Hugging Face token

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Read access is sufficient — neither model is gated, but the hub login step requires a token.

---

## Usage

```bash
streamlit run main.py
```

Streamlit will print a local URL (default `http://localhost:8501`). Open it in your browser.

> **First run:** both models are downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/`. This takes a few minutes depending on your connection. Subsequent runs start in seconds.

### Basic workflow

1. Wait for **"LLM ready"** in the sidebar — the Qwen model has finished loading.
2. Drag and drop any PDF onto the uploader, or click **Browse**.
3. Wait for **"Document ready — N paragraphs indexed"** in the sidebar.
4. Type your question in the chat input at the bottom and press Enter.
5. The answer streams token-by-token. Ask follow-up questions freely — the index is cached.
6. Upload a new PDF to start a fresh conversation.

---

## Tech Stack

<p align="left">

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--3B--Instruct-7C3AED?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![MiniLM](https://img.shields.io/badge/Embeddings-MiniLM-0F766E?style=for-the-badge)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Transformers](https://img.shields.io/badge/Inference-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PyTorch](https://img.shields.io/badge/Compute-PyTorch%20%7C%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pypdf](https://img.shields.io/badge/PDF-pypdf-64748B?style=for-the-badge)](https://pypi.org/project/pypdf/)
[![Python](https://img.shields.io/badge/Language-Python%203.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

</p>

| Layer | Technology | Role |
|---|---|---|
| **UI** | Streamlit | Chat interface, file upload, live streaming |
| **LLM** | Qwen2.5-3B-Instruct | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim semantic search vectors |
| **Inference** | HuggingFace Transformers | Model loading and generation |
| **Compute** | PyTorch (CUDA 12.1) | GPU-accelerated inference |
| **PDF parsing** | pypdf | Text extraction |
| **Vector store** | Custom `VectorIndex` | In-memory cosine search — no external DB |
| **Streaming** | `TextIteratorStreamer` | Non-blocking token delivery to UI |
| **Config** | python-dotenv | `.env`-based HF token loading |
| **Hub access** | huggingface_hub | Model download and authentication |

---

## Limitations

- **Single document at a time** — the index holds one PDF; there is no multi-document or cross-document Q&A.
- **In-memory index only** — `VectorIndex` is not persisted to disk; re-uploading the same PDF re-embeds it from scratch on every run.
- **Linear scan** — similarity search scans every stored vector; performance degrades on very large documents with thousands of paragraphs.
- **Paragraph chunking only** — the splitter uses whitespace-based paragraph breaks; fixed-token sliding-window chunking is not implemented.
- **Context window cap** — the top-k chunks must fit within Qwen's context window; very long paragraphs or a high `k` value can exceed it.
- **CPU inference is slow** — without a CUDA device, generation at 3B parameters takes significantly longer than real-time.
- **Cold start** — first run downloads both models from HuggingFace Hub, which can take several minutes depending on network speed.
- **Single-turn grounding** — the RAG prompt is rebuilt from scratch on every question; there is no conversation memory carried across turns.

---

## Future Improvements

- **Persistent index** — serialize `VectorIndex.vectors` and `.documents` to disk so documents do not need to be re-embedded on every startup.
- **Multi-document support** — merge indices from several PDFs into one `VectorIndex` for cross-document Q&A.
- **Sliding-window chunking** — replace paragraph splits with fixed-token overlapping windows for more uniform chunk sizes and better boundary handling.
- **ANN indexing** — replace linear scan with an approximate nearest-neighbour structure (e.g. HNSW) for sub-linear search at scale.
- **Quantized inference** — add `load_in_4bit=True` via `bitsandbytes` to run larger models on smaller GPUs.
- **Larger LLM support** — swap `model_name` in `AiModel.__init__` to any HuggingFace causal model (e.g. `meta-llama/Llama-3.2-3B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`).
- **Conversation memory** — accumulate prior Q&A turns in the RAG prompt for contextually aware follow-up answers.
- **Embedding progress bar** — show per-paragraph indexing progress during PDF ingestion rather than a single blocking wait.
