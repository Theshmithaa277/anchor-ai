import os
import glob
import hashlib
import base64
import uuid
from datetime import datetime, timezone
import imageio_ffmpeg
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from PIL import Image, ImageEnhance
import pytesseract

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

import yt_dlp
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Constants ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "whisper-large-v3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OCR_PSM_MODE = "--psm 11"  # sparse text, no particular layout — good for scattered diagram labels


def _now_iso() -> str:
    """UTC ISO-8601 timestamp used as ingestion metadata across all loaders."""
    return datetime.now(timezone.utc).isoformat()


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Upscale small images and boost contrast to improve Tesseract accuracy."""
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def load_pdf(path: str) -> list[Document]:
    """Load a PDF file into LangChain Documents, tagged with source metadata.
    PyPDFLoader already attaches a 0-indexed "page" field to each Document
    automatically — that's what citations.py's page_number field reads."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    ingested_at = _now_iso()
    for d in docs:
        d.metadata["source_type"] = "pdf"
        d.metadata["source_name"] = os.path.basename(path)
        d.metadata["ingested_at"] = ingested_at
    return docs


VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _describe_image_with_vision_model(path: str) -> str | None:
    """
    Fallback for images where OCR finds no machine-readable text (e.g. a
    diagram, chart, or illustration) — ask a vision-capable Groq model to
    describe what's in the image instead, so it still becomes searchable
    rather than being silently dropped.
    """
    try:
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = _groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe all visible objects, labels, diagrams, "
                                 "and important information in this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠ Vision-model fallback failed for {path}: {e}")
        return None


def load_image(path: str) -> list[Document]:
    """
    Run OCR on an image and wrap the extracted text as a Document. If OCR
    finds nothing (common for diagrams/illustrations with little
    machine-readable text), fall back to describing the image with a
    vision model instead of dropping it entirely.
    """
    img = Image.open(path)
    img = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(img, config=OCR_PSM_MODE).strip()

    if not text:
        print(f"ℹ No OCR text detected in {path}, trying vision-model fallback...")
        text = _describe_image_with_vision_model(path)

    if not text:
        print(f"⚠ No text or description available for image: {path}")
        return []

    return [Document(
        page_content=text,
        metadata={
            "source_type": "image",
            "source_name": os.path.basename(path),
            "ingested_at": _now_iso(),
        }
    )]


def _transcribe_with_whisper(url: str) -> str | None:
    """
    Download audio via yt-dlp and transcribe with Groq's Whisper endpoint.

    Uses a unique filename per call (safe for concurrent use, unlike the
    previous fixed "temp_audio" filename), and imageio_ffmpeg's bundled
    ffmpeg binary instead of a hardcoded filesystem path — this works
    identically on any machine with no install or PATH setup required.
    """
    unique_prefix = f"temp_audio_{uuid.uuid4().hex}"
    audio_path = f"{unique_prefix}.mp3"

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{unique_prefix}.%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "quiet": True,
            "ffmpeg_location": os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()),
            # YouTube's anti-bot throttling (HTTP 403s) is one of the known
            # failure modes here — a transient block is often recoverable
            # on retry, so don't give up after a single attempt.
            "retries": 3,
            "fragment_retries": 3,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(audio_path):
            print(f"⚠ Expected audio file not found after download for {url} "
                  f"(download likely failed or was interrupted).")
            return None

        with open(audio_path, "rb") as f:
            # translations.create (not transcriptions.create) — this outputs
            # English regardless of the spoken language in the audio, so the
            # index stays English-normalized even for caption-less foreign-
            # language videos.
            translation = _groq_client.audio.translations.create(
                file=f,
                model=WHISPER_MODEL,
            )
        return translation.text

    except Exception as e:
        print(f"⚠ Whisper transcription failed for {url}: {e}")
        return None
    finally:
        # Clean up EVERYTHING matching this call's unique prefix, not just
        # the one exact filename we expected — a failed/interrupted
        # download can leave behind a differently-named partial file
        # (e.g. "<prefix>.mp4.part" if ffmpeg's conversion step never
        # ran), and that used to linger on disk indefinitely.
        for leftover in glob.glob(f"{unique_prefix}*"):
            try:
                os.remove(leftover)
            except OSError:
                pass


def load_youtube(url: str, use_whisper_fallback: bool = True) -> list[Document]:
    """
    Fetch a YouTube video's transcript, trying in order:
      1. An existing English transcript (manual or auto-generated)
      2. Any existing transcript, translated to English (if translatable) —
         checks ALL available tracks, not just the first one in the list,
         and prefers a manually-created transcript over an auto-generated one.
      3. Whisper transcription (if captions are unusable/untranslatable)

    Whichever path succeeds, the stored text ends up in English:
    - Caption path: explicit .translate("en") call
    - Whisper path: _transcribe_with_whisper uses Groq's *translations*
      endpoint (not transcriptions), which outputs English regardless of
      the spoken language in the audio.
    """
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url

    full_text = None
    source_method = "captions"

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        transcript = None

        # 1. Try a direct English transcript first (manual or auto-generated)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = None

        # 2. No English track — check EVERY available transcript (not just
        # the first one in the list, which could be any language, e.g. an
        # auto-generated Korean track) and translate the best candidate.
        if transcript is None:
            candidates = list(transcript_list)
            translatable = [t for t in candidates if t.is_translatable]

            if translatable:
                # Prefer a manually-created transcript over auto-generated
                # if both exist — manual ones are usually more accurate.
                translatable.sort(key=lambda t: t.is_generated)
                chosen = translatable[0]
                print(f"ℹ Found '{chosen.language_code}' transcript for {url}, translating to English...")
                transcript = chosen.translate("en")
            else:
                print(f"⚠ No translatable transcript found for {url} ({len(candidates)} track(s) "
                      f"available, none translatable) — falling back to Whisper.")
                raise NoTranscriptFound(video_id, ["en"], transcript_list)

        fetched_transcript = transcript.fetch()
        full_text = " ".join([snippet.text for snippet in fetched_transcript])
        source_method = f"captions_{transcript.language_code}"

    except (TranscriptsDisabled, NoTranscriptFound):
        if use_whisper_fallback:
            print(f"ℹ No usable captions for {url}, falling back to Whisper transcription...")
            full_text = _transcribe_with_whisper(url)
            source_method = "whisper"
        else:
            print(f"⚠ No transcript available for {url} — skipping (whisper fallback disabled).")
            return []
    except Exception as e:
        # NOTE: generic network/API errors land here and do NOT trigger Whisper
        # fallback by design — only missing-captions cases do.
        print(f"⚠ Failed to fetch transcript for {url}: {e} — skipping.")
        return []

    if not full_text:
        print(f"⚠ Could not obtain any text for {url} — skipping.")
        return []

    return [Document(
        page_content=full_text,
        metadata={
            "source_type": "youtube",
            "source_name": url,
            "transcript_method": source_method,
            "ingested_at": _now_iso(),
        }
    )]


def _chunk_hash(text: str) -> str:
    """Content hash used to detect duplicate chunks during incremental
    indexing (e.g. re-uploading the same PDF, or two sources that share
    a boilerplate paragraph) so we never re-embed the same text twice."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_index(all_documents: list[Document], index_path: str = "faiss_index"):
    """Chunk documents, embed them, and save a brand-new FAISS index to
    disk — this is a full (re)build, used by the CLI dev entrypoint
    below. For adding documents to an ALREADY-RUNNING app without
    rebuilding everything, use add_documents_to_index() instead."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_documents)
    for c in chunks:
        c.metadata["content_hash"] = _chunk_hash(c.page_content)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Saved FAISS index to '{index_path}'")
    return vectorstore


def add_documents_to_index(
    chunks: list[Document],
    embeddings,
    index_path: str = "faiss_index",
    existing_vectorstore: FAISS | None = None,
) -> tuple[FAISS, int, int]:
    """
    Incremental indexing: embed and add ONLY chunks that aren't already
    in the index (by content hash), instead of re-embedding and
    rebuilding the entire FAISS index on every upload.

    Pass `existing_vectorstore` when the caller already holds the live
    in-memory index (this is how app.py uses it — avoids an extra
    disk round-trip). If it's None, this will try to load one from
    `index_path` on disk, and fall back to creating a fresh index if
    none exists yet.

    Returns (vectorstore, num_chunks_added, num_chunks_skipped_as_duplicate).
    """
    for c in chunks:
        c.metadata.setdefault("content_hash", _chunk_hash(c.page_content))

    vectorstore = existing_vectorstore
    if vectorstore is None and os.path.isdir(index_path):
        try:
            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"⚠ Could not load existing index at '{index_path}' ({e}); starting fresh.")
            vectorstore = None

    if vectorstore is None:
        # Nothing to add to yet — this IS the first build.
        unique_chunks = list({c.metadata["content_hash"]: c for c in chunks}.values())
        vectorstore = FAISS.from_documents(unique_chunks, embeddings)
        vectorstore.save_local(index_path)
        skipped = len(chunks) - len(unique_chunks)
        return vectorstore, len(unique_chunks), skipped

    existing_hashes = {
        doc.metadata.get("content_hash")
        for doc in vectorstore.docstore._dict.values()
        if doc.metadata.get("content_hash")
    }

    new_chunks = []
    seen_in_batch = set()
    skipped = 0
    for c in chunks:
        h = c.metadata["content_hash"]
        if h in existing_hashes or h in seen_in_batch:
            skipped += 1
            continue
        seen_in_batch.add(h)
        new_chunks.append(c)

    if new_chunks:
        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(index_path)

    return vectorstore, len(new_chunks), skipped


def main():
    """Dev/test entry point — ingest a fixed set of sample files. Edit paths below as needed."""
    all_docs = []

    pdf_files = ["data/dsa_guide.pdf"]
    for f in pdf_files:
        if os.path.exists(f):
            all_docs.extend(load_pdf(f))
        else:
            print(f"⚠ PDF not found: {f}")

    image_files = ["data/neuron_diagram.jpg", "data/text.jpeg"]
    for f in image_files:
        if os.path.exists(f):
            all_docs.extend(load_image(f))
        else:
            print(f"⚠ Image not found: {f}")

    youtube_urls = ["https://youtu.be/gpVA5mx73UE?si=iomY5WN-js0Pkqe9"]
    for url in youtube_urls:
        all_docs.extend(load_youtube(url))

    if not all_docs:
        print("❌ No documents loaded — check your file paths / URLs above.")
    else:
        build_index(all_docs)


if __name__ == "__main__":
    main()