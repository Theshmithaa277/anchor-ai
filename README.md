# ⚓ Anchor AI

**Multimodal RAG System with Security Guardrails**

Anchor AI is a retrieval-augmented generation (RAG) system that answers questions grounded in your own documents, images, and videos — with built-in safety checks to ensure every answer is accurate, cited, and faithful to the source material.

---

## 🚀 Features

- **PDF Question Answering** — Ask questions directly against uploaded documents
- **Image Understanding** — Extract and query information from images
- **YouTube Knowledge Retrieval** — Ingest video transcripts (with auto-translation and Whisper fallback for non-captioned videos) and query their content
- **Semantic Search using FAISS** — Fast, accurate vector-based retrieval over embedded documents
- **Cross-Encoder Reranking** — Reranks retrieved chunks for higher relevance before generation
- **Citation Generation** — Every answer is traceable back to its source
- **Faithfulness Verification** — Flags and prevents hallucinated or unsupported claims
- **Security Guardrails** — Fail-closed safety checks to block unsafe or out-of-scope queries
- **Evaluation Framework** — Automated harness for measuring retrieval accuracy and guardrail performance

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Backend** | Python, Flask |
| **RAG Framework** | LangChain |
| **Vector Store** | FAISS |
| **LLM** | Groq API (Llama 3) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📂 Project Structure

```
anchor-ai/
├── app.py                 # Flask server — serves frontend & exposes /ask, /ingest routes
├── quest.py                # Core RAG pipeline — retrieval, reranking, guardrails, generation
├── load_to_faiss.py         # Document ingestion & FAISS indexing pipeline
├── static/                 # Frontend assets (CSS, JS)
├── templates/               # HTML templates
├── eval/                   # Evaluation harness & test cases
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

```bash
# Clone the repository
git clone https://github.com/Theshmithaa277/anchor-ai.git
cd anchor-ai

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file in the root directory with:
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶ Usage

```bash
# Start the Flask server
python app.py
```

Then open your browser at `http://localhost:5000`

1. **Ingest content** — upload PDFs, images, or paste a YouTube URL via the `/ingest` route
2. **Ask questions** — query your ingested knowledge base via the chat interface
3. **View citations** — every response includes source references so you can verify the answer

---

## 📸 Screenshots

<!-- Add screenshots of your UI here -->
| Home / Chat Interface | Citation View |
|---|---|
| _screenshot_ | _screenshot_ |

---

## 📈 Future Improvements

- [ ] Multi-user session support
- [ ] Support for additional document formats (Word, PPTX)
- [ ] Streaming responses for faster perceived latency
- [ ] Expanded evaluation suite with adversarial test cases
- [ ] Deployment guide (Docker + cloud hosting)

---

## 📄 License

This project is licensed under the MIT License.
