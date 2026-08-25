from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from sentence_transformers import SentenceTransformer
from docx import Document
from pypdf import PdfReader, PdfWriter
from markitdown import MarkItDown
import numpy as np
import os
import base64
import tempfile
import requests

# ——— Image captioning via Groq vision model ———
# Uses the same GROQ_API_KEY env var as orchestrator.py's text LLM calls.
# NOTE: verify GROQ_VISION_MODEL is still current at
# https://console.groq.com/docs/models before your demo — Groq's
# available vision models change over time.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
IMAGE_CAPTION_PROMPT = (
    "Describe this lecture slide image or diagram in 1-2 sentences. "
    "Focus on the educational content: labels, steps, relationships, or "
    "data shown. If it's purely decorative (logo, background), say so briefly."
)

# ——— Global state (loaded once, reused across requests) ———
_model = None
_slide_texts = []   # each item also carries "source_file"
_vectors = None
_loaded_files = []  # list of file paths currently loaded (supports multiple)
_conversation_history = []  # list of {"question", "answer", "chunks"} for this class session

_FOLLOWUP_PHRASES = [
    "explain again", "explain that again", "explain one more time",
    "one more time", "say that again", "say again", "repeat that", "repeat it",
    "please repeat", "please repeat that",
    "didn't understand", "didnt understand", "didn't get it", "didnt get it",
    "not clear", "not clear sir", "not understood", "unclear",
    "i didn't get it", "i didnt get it", "still confused", "i m confused",
    "i am confused", "im confused", "confused",
    "come again", "come again sir", "once more", "again please", "again sir",
    "explain more", "can you explain more", "i'm lost", "im lost",
]


_markitdown = None


def _get_model():
    global _model
    if _model is None:
        print("[RAG] Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_markitdown():
    global _markitdown
    if _markitdown is None:
        _markitdown = MarkItDown()
    return _markitdown


def _caption_image_bytes(image_bytes, mime_type="image/png"):
    """
    Sends raw image bytes to a Groq vision model and returns a short
    caption. Fails soft (returns "") if no API key is set or the call
    errors — a missing caption should never block a file upload.
    """
    if not GROQ_API_KEY:
        return ""
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64}"
        resp = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[RAG] Image captioning failed (skipping image): {e}")
        return ""


def _caption_pptx_slide_images(slide):
    """Returns a list of '[Image: ...]' caption strings for every picture
    shape on this slide."""
    captions = []
    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            try:
                caption = _caption_image_bytes(
                    shape.image.blob, shape.image.content_type or "image/png"
                )
            except Exception as e:
                print(f"[RAG] Could not read pptx image: {e}")
                caption = ""
            if caption:
                captions.append(f"[Image: {caption}]")
    return captions


def _strip_markitdown_comments(md_text):
    """Removes MarkItDown's own '<!-- Slide number: N -->' comments AND
    its bare '![filename](Picture2.jpg)' image placeholders — neither
    carries real information; real image captions (if any) are generated
    separately via _caption_image_bytes() and appended by the caller."""
    lines = []
    for l in md_text.split("\n"):
        stripped = l.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("!["):
            continue
        lines.append(l)
    return "\n".join(lines).strip()


def _markdown_to_heading_and_points(md_text):
    """
    Splits a single chunk's markdown into (heading, points):
    - heading = first '#' line found, if any (title placeholder text)
    - points  = every other non-empty line (bullets, table rows, body text)
    """
    heading = ""
    points = []
    for raw_line in md_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # MarkItDown inserts its own "<!-- Slide number: N -->" comment per
        # file; since we convert one slide/page at a time this is always
        # "1" and meaningless — our own slide_number field is authoritative.
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        # Drop MarkItDown's bare "![filename](Picture2.jpg)" placeholder —
        # meaningless without a real caption; real captions (if any) are
        # generated separately via _caption_image_bytes() and appended by
        # the caller.
        if line.startswith("!["):
            continue
        if line.startswith("#") and not heading:
            heading = line.lstrip("#").strip()
        else:
            points.append(line)
    return heading, points


def _extract_from_pptx(path):
    """
    Converts each slide individually via MarkItDown (keeps slide_number
    intact for navigation/"repeat slide N", while gaining markdown table
    support that the old python-pptx-only extractor didn't have).
    """
    converter = _get_markitdown()
    prs = Presentation(path)
    num_slides = len(prs.slides)
    chunks = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(num_slides):
            # Build a temp single-slide pptx by stripping all other slide
            # references from the slide list (fast — no re-encoding of media).
            single = Presentation(path)
            slide_id_list = single.slides._sldIdLst
            all_ids = list(slide_id_list)
            for j, sld in enumerate(all_ids):
                if j != i:
                    slide_id_list.remove(sld)
            slide_path = os.path.join(tmp_dir, f"_slide_{i}.pptx")
            single.save(slide_path)

            result = converter.convert(slide_path)
            md_text = _strip_markitdown_comments(result.text_content or "")

            # Caption any pictures on the real (original) slide object —
            # markitdown itself only emits a meaningless filename placeholder.
            image_captions = _caption_pptx_slide_images(prs.slides[i])
            if image_captions:
                md_text = (md_text + "\n" + "\n".join(image_captions)).strip()

            if not md_text:
                continue

            heading, points = _markdown_to_heading_and_points(md_text)
            chunks.append({
                "slide_number": i + 1,
                "text": md_text,
                "heading": heading,
                "points": points,
            })
    return chunks


def _extract_from_pdf(path):
    """
    Converts each page individually via MarkItDown (keeps slide_number =
    page number for navigation, while gaining proper markdown table
    extraction that pdfplumber's raw text often garbled).
    """
    converter = _get_markitdown()
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    chunks = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(num_pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            page_path = os.path.join(tmp_dir, f"_page_{i}.pdf")
            with open(page_path, "wb") as f:
                writer.write(f)

            result = converter.convert(page_path)
            md_text = _strip_markitdown_comments(result.text_content or "")

            # MarkItDown's PDF converter is text/table-only — it does not
            # touch embedded images at all, so we extract + caption them
            # ourselves here.
            image_captions = []
            for img in reader.pages[i].images:
                try:
                    caption = _caption_image_bytes(img.data, "image/png")
                except Exception as e:
                    print(f"[RAG] Could not read pdf image: {e}")
                    caption = ""
                if caption:
                    image_captions.append(f"[Image: {caption}]")
            if image_captions:
                md_text = (md_text + "\n" + "\n".join(image_captions)).strip()

            if not md_text:
                continue

            _, points = _markdown_to_heading_and_points(md_text)
            chunks.append({
                "slide_number": i + 1,
                "text": md_text,
                "points": points,
            })
    return chunks


def _extract_from_docx(path):
    """
    DOCX has no page/slide concept, so we keep the original heuristic:
    start a new chunk at each heading. MarkItDown converts the whole
    file once (fast, single call) and now also captures tables, which
    the old python-docx-paragraphs-only extractor silently skipped.
    """
    converter = _get_markitdown()
    result = converter.convert(path)
    md_text = _strip_markitdown_comments(result.text_content or "")
    if not md_text:
        return []

    chunks = []
    current_lines = []
    chunk_number = 1

    def flush():
        nonlocal current_lines, chunk_number
        text_block = "\n".join(l.strip() for l in current_lines if l.strip())
        if text_block:
            _, points = _markdown_to_heading_and_points(text_block)
            chunks.append({
                "slide_number": chunk_number,
                "text": text_block,
                "points": points,
            })
            chunk_number += 1
        current_lines = []

    for raw_line in md_text.split("\n"):
        line = raw_line.strip()
        if line.startswith("#") and current_lines:
            flush()
        current_lines.append(line)
    flush()

    # python-docx has no reliable way to know WHERE in the document an
    # image sits relative to headings, so — unlike pptx/pdf, where each
    # image is captioned into its own numbered slide/page — docx images
    # are captioned and appended as one trailing chunk rather than being
    # placed inline near the right section.
    doc = Document(path)
    image_captions = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                caption = _caption_image_bytes(rel.target_part.blob, rel.target_part.content_type)
            except Exception as e:
                print(f"[RAG] Could not read docx image: {e}")
                caption = ""
            if caption:
                image_captions.append(f"[Image: {caption}]")
    if image_captions:
        text_block = "Images in this document:\n" + "\n".join(image_captions)
        chunks.append({
            "slide_number": chunk_number,
            "text": text_block,
            "points": image_captions,
        })

    return chunks


def _extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pptx":
        return _extract_from_pptx(path)
    elif ext == ".pdf":
        return _extract_from_pdf(path)
    elif ext == ".docx":
        return _extract_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_lecture(file_path, append=False):
    """
    Public interface: processes a lecture file (pptx/pdf/docx) into
    embeddings, held in memory for retrieval.

    append=False: replaces whatever was loaded before.
    append=True: adds this file's content alongside whatever is
        already loaded (multiple files in one class session).
    """
    global _slide_texts, _vectors, _loaded_files, _conversation_history

    print(f"[RAG] Loading lecture: {file_path} (append={append})")
    chunks = _extract_text(file_path)

    if not chunks:
        raise ValueError("No text could be extracted from this file.")

    source_name = os.path.basename(file_path)
    for c in chunks:
        c["source_file"] = source_name

    model = _get_model()
    texts = [c["text"] for c in chunks]
    new_vectors = model.encode(texts, show_progress_bar=False)

    if append and _vectors is not None:
        _slide_texts = _slide_texts + chunks
        _vectors = np.vstack([_vectors, new_vectors])
        _loaded_files.append(file_path)
    else:
        _slide_texts = chunks
        _vectors = new_vectors
        _loaded_files = [file_path]
        _conversation_history = []  # fresh class, fresh memory (only on non-append load)

    print(f"[RAG] Loaded {len(chunks)} chunks from {source_name} "
          f"(total chunks now: {len(_slide_texts)})")
    return len(chunks)


def remove_file(file_path):
    """
    Public interface: removes a single loaded file's chunks/vectors
    from memory without touching any other loaded files. Does NOT
    touch conversation history (a past answer stays valid even if
    the source file is later removed).

    Returns True if the file was found and removed, False otherwise.
    """
    global _slide_texts, _vectors, _loaded_files

    if file_path not in _loaded_files:
        return False

    source_name = os.path.basename(file_path)

    # build a mask of which chunks/vectors to KEEP (i.e. NOT from this file)
    keep_indices = [
        i for i, c in enumerate(_slide_texts)
        if c.get("source_file") != source_name
    ]

    if keep_indices:
        _slide_texts = [_slide_texts[i] for i in keep_indices]
        _vectors = _vectors[keep_indices]
    else:
        _slide_texts = []
        _vectors = None

    _loaded_files = [f for f in _loaded_files if f != file_path]

    print(f"[RAG] Removed file: {file_path} (remaining chunks: {len(_slide_texts)})")
    return True


def retrieve_relevant_slides(question, top_k=3):
    """
    Public interface: given a student's question, returns the top_k
    most relevant chunks across all currently loaded files, as a list
    of {"slide_number": int, "text": str, "score": float, "source_file": str}.
    Returns an empty list if no lecture is loaded yet.
    """
    if _vectors is None or not _slide_texts:
        return []

    model = _get_model()
    question_vector = model.encode([question])[0]

    q_norm = question_vector / np.linalg.norm(question_vector)
    v_norm = _vectors / np.linalg.norm(_vectors, axis=1, keepdims=True)
    similarities = np.dot(v_norm, q_norm)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "slide_number": _slide_texts[idx]["slide_number"],
            "text": _slide_texts[idx]["text"],
            "score": float(similarities[idx]),
            "source_file": _slide_texts[idx].get("source_file", "unknown"),
        })
    return results


def _is_followup(question):
    q = question.lower().strip()

    if any(phrase in q for phrase in _FOLLOWUP_PHRASES):
        return True

    word_count = len(q.split())
    if word_count <= 3 and _conversation_history:
        quick_check = retrieve_relevant_slides(question, top_k=1)
        if quick_check and quick_check[0]["score"] < 0.35:
            return True

    return False


def _build_lecture_text(chunks):
    """
    Joins retrieved chunks into a single pre-formatted block, ready to
    drop straight into an LLM prompt — so callers (e.g. Nithesh's
    get_llm_answer()) don't need to duplicate this formatting logic.
    """
    return "\n\n".join(
        f"[{c['source_file']}, slide {c['slide_number']}]: {c['text']}"
        for c in chunks
    )


def get_context_for_query(question, top_k=3):
    """
    Public interface: the 'smart' entry point the orchestrator should
    call instead of retrieve_relevant_slides() directly.
    """
    if _is_followup(question) and _conversation_history:
        last = _conversation_history[-1]
        return {
            "is_followup": True,
            "chunks": last["chunks"],
            "previous_question": last["question"],
            "previous_answer": last["answer"],
            "lecture_text": _build_lecture_text(last["chunks"]),
        }

    chunks = retrieve_relevant_slides(question, top_k=top_k)
    return {
        "is_followup": False,
        "chunks": chunks,
        "previous_question": None,
        "previous_answer": None,
        "lecture_text": _build_lecture_text(chunks),
    }


# Alias — the integration handoff doc refers to this function as
# retrieve_context(); both names now work so either the orchestrator
# or this module can use its preferred name without breaking the other.
retrieve_context = get_context_for_query


def add_to_history(question, answer, chunks):
    """
    Public interface: call this after Groq generates an answer, so
    future follow-ups ("explain again") can reuse this turn.
    """
    _conversation_history.append({
        "question": question,
        "answer": answer,
        "chunks": chunks,
    })


def clear_lecture():
    """Public interface: wipes ALL currently loaded lecture files AND conversation history."""
    global _slide_texts, _vectors, _loaded_files, _conversation_history
    _slide_texts = []
    _vectors = None
    _loaded_files = []
    _conversation_history = []
    print("[RAG] Lecture(s) and conversation history cleared from memory.")


def has_lecture_loaded():
    return _vectors is not None


def get_loaded_files():
    """Public interface: returns list of currently loaded file paths."""
    return list(_loaded_files)


def get_ordered_chunks():
    """
    Public interface: returns all currently loaded chunks in their
    original slide/page order, for the automatic 'teach the class'
    walkthrough (as opposed to retrieve_relevant_slides(), which
    returns only the top-k matches for a specific question).
    """
    return list(_slide_texts)