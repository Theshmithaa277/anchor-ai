"""
Standalone ingestion test harness for Anchor AI.

Purpose: test load_youtube() and load_image() in isolation, with real
content, WITHOUT running the FAISS build step. This lets us see exactly
what text was extracted, which code path was used (captions vs Whisper,
OCR success/failure), and catch silent failures before they get buried
inside a 500-chunk index.

Usage:
    python test_ingestion.py --youtube <url> [<url2> ...]
    python test_ingestion.py --image <path> [<path2> ...]
    python test_ingestion.py --youtube <url> --image <path>

Assumes load_to_faiss.py is in the same directory (or on PYTHONPATH).
"""

import argparse
import sys
import time

from load_to_faiss import load_youtube, load_image


SEPARATOR = "=" * 70


def preview(text: str, n: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def test_youtube(url: str):
    print(SEPARATOR)
    print(f"[YOUTUBE] {url}")
    print(SEPARATOR)

    start = time.time()
    try:
        docs = load_youtube(url)
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED — unexpected exception (took {elapsed:.1f}s): {e}")
        return False
    elapsed = time.time() - start

    if not docs:
        print(f"❌ FAILED — no documents returned (took {elapsed:.1f}s)")
        print("   Check: captions disabled AND whisper fallback failed?")
        print("   Check: yt-dlp / ffmpeg installed and on PATH?")
        print("   Check: GROQ_API_KEY set in .env?")
        return False

    doc = docs[0]
    method = doc.metadata.get("transcript_method", "unknown")
    char_count = len(doc.page_content)

    print(f"✅ SUCCESS via '{method}' (took {elapsed:.1f}s)")
    print(f"   Characters extracted: {char_count}")
    print(f"   Preview: {preview(doc.page_content)}")

    if char_count < 50:
        print("⚠  WARNING: very little text extracted — verify this isn't junk/empty")

    return True


def test_image(path: str):
    print(SEPARATOR)
    print(f"[IMAGE] {path}")
    print(SEPARATOR)

    start = time.time()
    try:
        docs = load_image(path)
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ FAILED — unexpected exception (took {elapsed:.1f}s): {e}")
        return False
    elapsed = time.time() - start

    if not docs:
        print(f"❌ FAILED — no text detected (took {elapsed:.1f}s)")
        print("   Check: is Tesseract installed and on PATH? (tesseract --version)")
        print("   Check: is the image legible / high enough resolution?")
        print("   Check: does pytesseract.pytesseract.tesseract_cmd need to be set manually on Windows?")
        return False

    doc = docs[0]
    char_count = len(doc.page_content)

    print(f"✅ SUCCESS (took {elapsed:.1f}s)")
    print(f"   Characters extracted: {char_count}")
    print(f"   Preview: {preview(doc.page_content)}")

    if char_count < 20:
        print("⚠  WARNING: very little text extracted — OCR may have mostly failed")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Anchor AI ingestion in isolation")
    parser.add_argument("--youtube", nargs="+", default=[], help="YouTube URL(s) to test")
    parser.add_argument("--image", nargs="+", default=[], help="Image path(s) to test")
    args = parser.parse_args()

    if not args.youtube and not args.image:
        print("Nothing to test. Pass --youtube <url> and/or --image <path>.")
        sys.exit(1)

    results = []

    for url in args.youtube:
        results.append(("youtube", url, test_youtube(url)))
        print()

    for path in args.image:
        results.append(("image", path, test_image(path)))
        print()

    print(SEPARATOR)
    print("SUMMARY")
    print(SEPARATOR)
    passed = sum(1 for _, _, ok in results if ok)
    for kind, ref, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}  [{kind}] {ref}")
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    main()