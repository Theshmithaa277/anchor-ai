import os
import shutil
import tempfile
import time

from flask import Flask, request, jsonify, send_file
from flask_login import login_required

from load_to_faiss import load_pdf, load_image, load_youtube, add_documents_to_index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from quest import get_client, run_query
from cache import ResponseCache, make_cache_key

from models import db
from auth import auth_bp, bcrypt, login_manager


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

FAISS_INDEX_PATH = "faiss_index"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = Flask(__name__, instance_relative_config=True)

# SECRET_KEY signs the session cookie used by Flask-Login — set a real
# value in .env for production. The fallback here is only so local dev
# doesn't crash on a missing env var; it must NOT be relied on outside dev.
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-only-insecure-key")

os.makedirs(app.instance_path, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    app.instance_path, "anchor_users.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
app.register_blueprint(auth_bp, url_prefix="/auth")

with app.app_context():
    db.create_all()


# ---------------------------------------------------------
# LOAD ONCE AT STARTUP
# ---------------------------------------------------------

client = get_client()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# IMPORTANT:
# Start with NO document loaded.
#
# This prevents an old PDF from being automatically loaded
# when Flask starts.
vectorstore = None

# Bumped on every successful /ingest — response_cache keys include this,
# so a cached answer from before the latest upload is never served again.
index_version = 0

response_cache = ResponseCache(ttl_seconds=600, max_entries=500)


# ---------------------------------------------------------
# DELETE OLD FAISS INDEX
# ---------------------------------------------------------

def clear_saved_index():
    """
    Completely remove the previously saved FAISS index.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)


# Start every Flask session with a clean knowledge base.
clear_saved_index()


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
@login_required
def home():
    return send_file("anchor_ai.html")


# ---------------------------------------------------------
# ASK
# ---------------------------------------------------------

@app.route("/ask", methods=["POST"])
@login_required
def ask():

    global vectorstore

    # -----------------------------------------------------
    # DO NOT ANSWER IF NO DOCUMENT HAS BEEN LOADED
    # -----------------------------------------------------

    if vectorstore is None:
        return jsonify({
            "answer": "Please upload a PDF/image or load a YouTube video first.",
            "blocked": False,
            "num_sources": 0,
            "faithfulness": 0,
            "grounded": False,
            "total_ms": 0
        }), 400

    data = request.get_json(silent=True) or {}

    query = data.get("query", "").strip()

    if not query:
        return jsonify({
            "answer": "Please enter a question.",
            "blocked": False,
            "num_sources": 0,
            "faithfulness": 0,
            "grounded": False,
            "total_ms": 0
        }), 400

    # -----------------------------------------------------
    # RESPONSE CACHE (repeated queries against the same
    # active index skip the whole pipeline — no retrieval,
    # no rerank, no Groq calls at all).
    # -----------------------------------------------------

    cache_key = make_cache_key(query, index_version)
    cached = response_cache.get(cache_key)
    if cached is not None:
        return jsonify({**cached, "cached": True})

    # -----------------------------------------------------
    # RUN RAG QUERY
    # -----------------------------------------------------

    t0 = time.perf_counter()

    try:

        result = run_query(
            query,
            client=client,
            vectorstore=vectorstore,
            verbose=False
        )

    except Exception as e:

        print("ERROR WHILE RUNNING QUERY:")
        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    t1 = time.perf_counter()

    total_ms = round((t1 - t0) * 1000)

    response_payload = {
        "answer": result["answer"],
        "blocked": result["blocked"],
        "num_sources": len(result["citations"]),
        "citations": result["citations"],
        "faithfulness": (
            result["faithfulness"]["confidence"]
            if result["faithfulness"]
            else 0
        ),
        "grounded": (
            result["faithfulness"]["grounded"]
            if result["faithfulness"]
            else True
        ),
        "total_ms": total_ms
    }

    # Don't cache blocked (guardrail-refused) answers or the "no
    # vectorstore" early-return above — only cache genuine, successful
    # RAG answers, so a transient safety-classifier hiccup never gets
    # permanently baked into the cache for that query.
    if not result["blocked"]:
        response_cache.set(cache_key, response_payload)

    return jsonify({**response_payload, "cached": False})


# ---------------------------------------------------------
# INGEST
# ---------------------------------------------------------

@app.route("/ingest", methods=["POST"])
@login_required
def ingest():

    global vectorstore, index_version

    try:

        # -------------------------------------------------
        # INCREMENTAL INDEXING
        #
        # Previously this route wiped the entire index before
        # loading each new source (PDF1 -> PDF2 meant PDF1
        # disappeared completely). That's now replaced with
        # incremental indexing: new uploads are ADDED to the
        # existing knowledge base via add_documents_to_index(),
        # which only embeds chunks that aren't already present
        # (by content hash) rather than rebuilding the whole
        # FAISS index from scratch every time.
        #
        # The index still starts empty each time the Flask
        # process itself restarts (clear_saved_index() at
        # startup, below, is unchanged).
        # -------------------------------------------------

        print("\n==============================")
        print("NEW SOURCE REQUESTED")
        print("==============================")

        docs = []

        # -------------------------------------------------
        # FILE UPLOAD
        # -------------------------------------------------

        if "file" in request.files:

            file = request.files["file"]

            if not file.filename:
                return jsonify({
                    "status": "error",
                    "message": "No file selected."
                }), 400

            suffix = os.path.splitext(
                file.filename
            )[1].lower()

            print("FILE:", file.filename)
            print("TYPE:", suffix)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                file.save(tmp.name)
                tmp_path = tmp.name

            try:

                if suffix == ".pdf":

                    print("Loading PDF...")
                    docs = load_pdf(tmp_path)

                    # load_pdf() only sees the random temp filename, so
                    # overwrite source_name with the user's real filename
                    # here — otherwise citations show "tmpAbCdEf.pdf"
                    # instead of the document's actual name.
                    for d in docs:
                        d.metadata["source_name"] = file.filename

                elif suffix in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp"
                ]:

                    print("Loading image...")
                    docs = load_image(tmp_path)

                    for d in docs:
                        d.metadata["source_name"] = file.filename

                else:

                    return jsonify({
                        "status": "error",
                        "message": f"Unsupported file type: {suffix}"
                    }), 400

            finally:

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # -------------------------------------------------
        # YOUTUBE
        # -------------------------------------------------

        elif request.is_json and request.json.get("youtube_url"):

            youtube_url = request.json[
                "youtube_url"
            ].strip()

            print("YOUTUBE URL:")
            print(youtube_url)

            print("Loading YouTube...")
            docs = load_youtube(youtube_url)

        # -------------------------------------------------
        # NOTHING PROVIDED
        # -------------------------------------------------

        else:

            return jsonify({
                "status": "error",
                "message": "No file or YouTube URL provided."
            }), 400

        # -------------------------------------------------
        # CHECK EXTRACTION
        # -------------------------------------------------

        if not docs:

            print("NO DOCUMENTS EXTRACTED")

            return jsonify({
                "status": "error",
                "message": (
                    "No text could be extracted from this source."
                )
            }), 400

        print("DOCUMENTS EXTRACTED:", len(docs))

        # -------------------------------------------------
        # SPLIT INTO CHUNKS
        # -------------------------------------------------

        chunks = splitter.split_documents(docs)

        if not chunks:

            return jsonify({
                "status": "error",
                "message": "Text was found, but no chunks were created."
            }), 400

        print("CHUNKS CREATED:", len(chunks))

        # -------------------------------------------------
        # INCREMENTAL ADD TO FAISS
        #
        # Only chunks that aren't already in the index (by
        # content hash) get embedded and added — a re-upload
        # of the same file, or overlapping content between
        # sources, doesn't pay to re-embed identical text.
        # -------------------------------------------------

        print("Adding to FAISS index (incremental)...")

        new_vectorstore, num_added, num_skipped = add_documents_to_index(
            chunks,
            embeddings,
            index_path=FAISS_INDEX_PATH,
            existing_vectorstore=vectorstore,
        )

        # -------------------------------------------------
        # ONLY AFTER EVERYTHING SUCCEEDS,
        # MAKE THE UPDATED VECTORSTORE ACTIVE
        # -------------------------------------------------

        vectorstore = new_vectorstore
        index_version += 1
        response_cache.clear()  # the knowledge base changed; old cached answers are stale

        print(f"VECTORSTORE READY ({num_added} chunks added, {num_skipped} duplicate chunks skipped)")
        print("==============================\n")

        return jsonify({
            "status": "ok",
            "chunks_added": num_added,
            "chunks_skipped_duplicate": num_skipped
        })

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    except Exception as e:

        print("\n!!!!!!!!!!!!!!!!!!!!!!!!")
        print("INGEST ERROR")
        print("!!!!!!!!!!!!!!!!!!!!!!!!")

        import traceback
        traceback.print_exc()

        # IMPORTANT (changed for incremental indexing):
        # Do NOT wipe the existing knowledge base just because THIS
        # upload failed — previously-ingested documents must survive
        # a failed later upload. `vectorstore` still points at the
        # last successfully-saved index (add_documents_to_index only
        # reassigns it after a successful save_local), so there's
        # nothing to roll back here.

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000,
        use_reloader=False
    )