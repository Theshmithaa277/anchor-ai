import os
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


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Upscale small images and boost contrast to improve Tesseract accuracy."""
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def load_pdf(path: str) -> list[Document]:
    """Load a PDF file into LangChain Documents, tagged with source metadata."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "pdf"
        d.metadata["source_name"] = os.path.basename(path)
    return docs


def load_image(path: str) -> list[Document]:
    """
    Run OCR on an image and wrap the extracted text as a Document.
    Known limitation: works well on text-dense images, fails on small/
    stylized illustrations where there's little machine-readable text.
    """
    img = Image.open(path)
    img = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(img, config=OCR_PSM_MODE)
    if not text.strip():
        print(f"⚠ No text detected in image: {path}")
        return []
    return [Document(
        page_content=text,
        metadata={"source_type": "image", "source_name": os.path.basename(path)}
    )]


def _transcribe_with_whisper(url: str) -> str | None:
    """
    Download audio via yt-dlp and transcribe with Groq's Whisper endpoint.
    Note: uses a fixed temp filename, so this function is not safe to call
    concurrently for multiple URLs at once (sequential use only).
    """
    audio_path = "temp_audio.mp3"
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "temp_audio.%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "quiet": True,
            # Point yt-dlp directly at ffmpeg's location instead of relying
            # on system PATH — avoids PATH/environment-variable issues
            # entirely. Update this if ffmpeg is installed elsewhere.
            "ffmpeg_location": r"C:\Users\Shruthi\Downloads\ffmpeg\ffmpeg-8.1.2-essentials_build\bin",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

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
        if os.path.exists(audio_path):
            os.remove(audio_path)


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
        metadata={"source_type": "youtube", "source_name": url, "transcript_method": source_method}
    )]


def build_index(all_documents: list[Document], index_path: str = "faiss_index"):
    """Chunk documents, embed them, and save a FAISS index to disk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_documents)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Saved FAISS index to '{index_path}'")
    return vectorstore


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