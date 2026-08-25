from pptx import Presentation
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
import numpy as np
import os

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


def _get_model():
    global _model
    if _model is None:
        print("[RAG] Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _extract_from_pptx(path):
    prs = Presentation(path)
    chunks = []
    for i, slide in enumerate(prs.slides, start=1):
        heading = ""
        title_shape = None
        try:
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                title_shape = slide.shapes.title
                heading = title_shape.text_frame.text.strip()
        except Exception:
            title_shape = None

        texts = []
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in paragraph.runs)
                    if line.strip():
                        texts.append(line.strip())

        full_text = (heading + " " if heading else "") + " ".join(texts)
        full_text = full_text.strip()
        if full_text:
            # "heading" is the slide's actual title placeholder text, kept
            # separate from "points" (the real body bullets) so the
            # orchestrator doesn't treat titles or footer/metadata text as
            # if they were content points to individually explain.
            chunks.append({
                "slide_number": i,
                "text": full_text,
                "heading": heading,
                "points": texts,
            })
    return chunks


def _extract_from_pdf(path):
    chunks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # PDFs don't have a clean bullet structure like pptx, so
                # each non-empty line is treated as one point.
                lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
                chunks.append({"slide_number": i, "text": text.strip(), "points": lines})
    return chunks


def _extract_from_docx(path):
    doc = Document(path)
    chunks = []
    current = []
    chunk_number = 1
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = ""
        if para.style is not None:
            style_name = getattr(para.style, "name", "") or ""

        if style_name.startswith("Heading") and current:
            chunks.append({"slide_number": chunk_number, "text": " ".join(current), "points": list(current)})
            chunk_number += 1
            current = []
        current.append(text)
    if current:
        chunks.append({"slide_number": chunk_number, "text": " ".join(current), "points": list(current)})
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

