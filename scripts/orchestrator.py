"""
AI Professor Robot — LangGraph Orchestrator
---------------------------------------------
Replaces the old linear teach_class() loop with a hub-and-spoke LangGraph
state machine. One graph.invoke() call = one unit of work (speak one
slide's worth of sentences, OR handle one interruption). An outer loop in
main() keeps re-invoking with the returned state, so the graph never needs
an unbounded recursion_limit and can jump to any node on any turn.

    dispatch (router) ──▶ speak_slide_node
                      ──▶ listen_question_node
                      ──▶ relevance_check_node
                      ──▶ answer_node
                      ──▶ post_answer_node
                      ──▶ END   (state["mode"] == "done")

INTERRUPT SOURCE (barge-in)
----------------------------
Keyboard (Space or Shift via pynput, with fallback to standard input).
Isolated behind InterruptSource so that swapping in Agilan's camera/YOLO
hand-raise module later means writing ONE new class with the same
wait()/is_set()/clear() interface and changing one line in main().

TTS
---
Supports edge-tts (online Azure) with automatic fallback to Piper ONNX
(offline Jetson).
"""

import os
import re
import io
import sys
import json
import time
import wave
import asyncio
import tempfile
import threading
import subprocess
from typing import TypedDict, Optional, List, Dict, Any, Tuple

import requests
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import rag_engine

# =====================================================================
# CONFIG
# =====================================================================
# Support both explicit and standard GROQ API key names
GROQ_LLM_API_KEY = os.environ.get("GROQ_LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_WHISPER_API_KEY = os.environ.get("GROQ_WHISPER_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_LLM_MODEL = "openai/gpt-oss-120b"
GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000")  # app.py's Flask server

EDGE_TTS_VOICE = "en-US-GuyNeural"

# Offline Piper TTS fallback
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_VOICE = os.path.join(_PROJECT_ROOT, "voices", "en_US-ryan-medium.onnx")
PIPER_MODEL = os.environ.get("PIPER_MODEL", _LOCAL_VOICE if os.path.exists(_LOCAL_VOICE) else "/home/jetson/ai-robot-project/voices/en_US-ryan-medium.onnx")
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "plughw:0,3")

MIC_NAME_HINT = "Microphone Array"
RECORD_SAMPLE_RATE = 48000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.02
SILENCE_SECONDS = 1.5
MAX_WAIT_FOR_SPEECH_SECONDS = 6
MAX_UTTERANCE_SECONDS = 20
CONSECUTIVE_CHUNKS_TO_CONFIRM_SPEECH = 3

RELEVANCE_THRESHOLD = 0.30  # below this cosine score, treat question as off-topic

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_checkpoint.json")


# =====================================================================
# INTERRUPT SOURCE — keyboard now, camera later, same interface
# =====================================================================
class InterruptSource:
    """Abstract shape: wait_for_trigger() blocks (poll-friendly) and
    returns True once triggered; clear() resets after being handled."""
    def is_set(self) -> bool:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass


class KeyboardInterruptSource(InterruptSource):
    """Fires on Space or Shift, anywhere in the X11 session, via pynput."""
    def __init__(self):
        self._event = threading.Event()
        self._listener = None

    def _on_press(self, key):
        try:
            from pynput.keyboard import Key
            if key in (Key.space, Key.shift, Key.shift_r):
                self._event.set()
        except Exception:
            pass

    def start(self):
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.daemon = True
            self._listener.start()
            print("[BARGE-IN] Keyboard listener active — press Space or Shift to interrupt.")
        except Exception as e:
            print(f"[BARGE-IN] Note: pynput keyboard listener unavailable ({e}). Running without global hotkey.")

    def stop(self):
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self):
        self._event.clear()


# =====================================================================
# SLIDE CLIENT — talks to app.py's Flask process over HTTP.
# =====================================================================
class SlideClient:
    def __init__(self, base_url=APP_BASE_URL):
        self.base_url = base_url

    def next(self):
        try:
            requests.post(f"{self.base_url}/api/slide/command", json={"command": "next"}, timeout=5)
        except Exception as e:
            print(f"[SLIDE] Could not advance projector slide: {e}")

    def goto(self, slide_number: int):
        try:
            requests.post(f"{self.base_url}/api/slide/command",
                           json={"command": "goto", "slide": slide_number}, timeout=5)
        except Exception as e:
            print(f"[SLIDE] Could not jump projector to slide {slide_number}: {e}")


# =====================================================================
# TTS — edge-tts with Piper fallback, interruptible playback
# =====================================================================
def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    return sentences or [text.strip()]


async def _edge_tts_to_file(text: str, out_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(out_path)


def synthesize_to_file(text: str, out_path: str):
    """Synthesizes text using edge-tts (online) or Piper (offline)."""
    try:
        asyncio.run(_edge_tts_to_file(text, out_path))
        return
    except Exception as e:
        print(f"[SPEAKING] edge-tts error ({e}), trying Piper offline fallback...")

    # Offline Piper TTS fallback
    if os.path.exists(PIPER_MODEL):
        try:
            from piper import PiperVoice, SynthesisConfig
            voice = PiperVoice.load(PIPER_MODEL)
            wav_path = out_path.replace(".mp3", ".wav")
            with wave.open(wav_path, "wb") as wf:
                voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.15))
            if os.path.exists(wav_path):
                # If mp3 was requested, keep as wav
                if out_path.endswith(".mp3"):
                    shutil_path = out_path
                    try:
                        import shutil
                        shutil.move(wav_path, shutil_path)
                    except Exception:
                        pass
                return
        except Exception as pe:
            print(f"[SPEAKING] Piper fallback error: {pe}")


def play_interruptible(audio_path: str, interrupt: InterruptSource) -> bool:
    """Plays audio_path via ffplay, aplay, or winsound, polling interrupt."""
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        # Fallback if ffplay is not installed
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(audio_path, winsound.SND_FILENAME)
                return True
            except Exception:
                return True
        else:
            try:
                subprocess.run(["aplay", "-D", AUDIO_DEVICE, audio_path], check=True)
                return True
            except Exception:
                return True

    try:
        while True:
            if proc.poll() is not None:
                return True
            if interrupt.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()


def speak_sentences_interruptible(sentences: List[str], start_index: int,
                                   interrupt: InterruptSource) -> int:
    """Speaks sentences[start_index:] one at a time. Returns stop index."""
    tmp_dir = tempfile.mkdtemp(prefix="prof_tts_")
    try:
        for i in range(start_index, len(sentences)):
            sentence = sentences[i]
            if not sentence.strip():
                continue
            print(f"[SPEAKING] {sentence}")
            out_path = os.path.join(tmp_dir, f"s{i}.mp3")
            try:
                synthesize_to_file(sentence, out_path)
            except Exception as e:
                print(f"[SPEAKING] Synthesis failed ({e}), skipping sentence.")
                continue

            if not os.path.exists(out_path):
                continue

            interrupt.clear()
            completed = play_interruptible(out_path, interrupt)
            if not completed:
                print(f"[SPEAKING] Interrupted at sentence {i}: {sentence[:50]!r}")
                return i
        return len(sentences)
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def speak_text(text: str, interrupt: Optional[InterruptSource] = None):
    """Simple non-resumable speak for prompts and answers."""
    if not text or not text.strip():
        return
    print(f"[SPEAKING] {text}")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        synthesize_to_file(text, tmp.name)
        if os.path.exists(tmp.name):
            dummy_interrupt = interrupt or KeyboardInterruptSource()
            play_interruptible(tmp.name, dummy_interrupt)
    finally:
        try:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
        except Exception:
            pass


# =====================================================================
# STT — Groq-hosted Whisper API (or Local Fallback)
# =====================================================================
def find_mic_device(name_hint=MIC_NAME_HINT):
    import sounddevice as sd
    devices = sd.query_devices()
    matches = [i for i, d in enumerate(devices)
               if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower()]
    if matches:
        idx = matches[0]
        print(f"[SETUP] Using mic: [{idx}] {devices[idx]['name']}")
        return idx
    any_input = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if any_input:
        idx = any_input[0][0]
        print(f"[SETUP] Note: hint '{name_hint}' not matched, using default mic: [{idx}] {devices[idx]['name']}")
        return idx
    print("[SETUP] No input-capable audio devices found at all.")
    raise RuntimeError("No usable microphone found.")


def record(mic_index) -> Tuple[np.ndarray, bool]:
    import sounddevice as sd
    silence_chunks = int(SILENCE_SECONDS * RECORD_SAMPLE_RATE / CHUNK_SIZE)
    max_wait_chunks = int(MAX_WAIT_FOR_SPEECH_SECONDS * RECORD_SAMPLE_RATE / CHUNK_SIZE)
    max_speech_chunks = int(MAX_UTTERANCE_SECONDS * RECORD_SAMPLE_RATE / CHUNK_SIZE)
    print("[LISTENING] listening...")
    chunks, silent_count, speech_started, loud_streak = [], 0, False, 0
    waited_chunks, speech_chunks = 0, 0
    with sd.InputStream(samplerate=RECORD_SAMPLE_RATE, channels=2, dtype='float32',
                         device=mic_index, blocksize=CHUNK_SIZE) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunks.append(chunk.copy())
            volume = np.abs(chunk).mean()
            if volume > SILENCE_THRESHOLD:
                loud_streak += 1
                if loud_streak >= CONSECUTIVE_CHUNKS_TO_CONFIRM_SPEECH:
                    speech_started = True
                    silent_count = 0
            elif speech_started:
                silent_count += 1
                if silent_count >= silence_chunks:
                    break
            if not speech_started:
                waited_chunks += 1
                if waited_chunks >= max_wait_chunks:
                    break
            else:
                speech_chunks += 1
                if speech_chunks >= max_speech_chunks:
                    break
    if not chunks:
        return np.array([], dtype='float32'), False
    return np.concatenate(chunks, axis=0), speech_started


def groq_transcribe(audio: np.ndarray) -> str:
    if not GROQ_WHISPER_API_KEY:
        print("[LISTENING] Note: GROQ_WHISPER_API_KEY not set.")
        return ""
    audio_mono = audio.mean(axis=1).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RECORD_SAMPLE_RATE)
        pcm16 = (np.clip(audio_mono, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    try:
        resp = requests.post(
            GROQ_WHISPER_URL,
            headers={"Authorization": f"Bearer {GROQ_WHISPER_API_KEY}"},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={"model": GROQ_WHISPER_MODEL, "language": "en"},
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        print(f"[LISTENING] Heard: {text}")
        return text
    except Exception as e:
        print(f"[LISTENING] Groq Whisper error: {e}")
        return ""


def listen_and_transcribe(mic_index) -> str:
    audio, speech_started = record(mic_index)
    if len(audio) == 0 or not speech_started:
        return ""
    return groq_transcribe(audio)


# =====================================================================
# LLM helpers
# =====================================================================
def clean_for_speech(text: str) -> str:
    text = (text.replace('\u2011', '-')
                .replace('\u2013', '-')
                .replace('\u2014', '-')
                .replace('\u2018', "'")
                .replace('\u2019', "'")
                .replace('\u201c', '"')
                .replace('\u201d', '"')
                .replace('\u00a0', ' '))
    text = re.sub(r'^[\*\-\•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.encode('ascii', 'ignore').decode('ascii').strip()


def _groq_chat(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.7) -> str:
    if not GROQ_LLM_API_KEY:
        return "Sorry, I can't reach my brain right now. Please check the GROQ_API_KEY setting."
    headers = {"Authorization": f"Bearer {GROQ_LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(GROQ_LLM_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return clean_for_speech(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print(f"[THINKING] Groq error: {e}")
        return "Sorry, I had trouble reaching the answer service."


SLIDE_SYSTEM_PROMPT = (
    "You are a friendly professor teaching a live class out loud. You will be given "
    "the raw content of one lecture slide. State the actual point(s) using the slide's "
    "own key terms — never invent facts. You may add one short relatable example. "
    "Aim for approximately 120-150 words (about 45-60 seconds of spoken explanation). "
    "Never use bullet points, markdown, or headers — speak naturally in flowing sentences. "
    "No greetings, explain the slide directly."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a friendly professor speaking out loud to a student in a classroom. "
    "Answer in natural, spoken sentences — never bullet points, numbers, or markdown. "
    "When explaining maths, write symbols out in plain spoken words "
    "(e.g. 'x squared', not 'x^2'). Keep it to 2-4 short sentences unless the question "
    "truly needs more depth. Be warm and encouraging."
)


def generate_slide_explanation(slide_text: str) -> str:
    return _groq_chat(SLIDE_SYSTEM_PROMPT, f"Slide content:\n{slide_text}", max_tokens=600)


def generate_answer(question: str, context: dict, already_spoken: str, is_irrelevant: bool) -> str:
    lecture_text = "" if is_irrelevant else context.get("lecture_text", "")
    prefix = ""
    if is_irrelevant:
        prefix = ("This question doesn't seem to relate to today's slide content. "
                  "Briefly and politely note that before answering from general knowledge. ")
    spoken_note = (
        f"For context, here is exactly what you just said out loud to the class on the "
        f"current slide, so you don't blindly repeat it and can refer back to it if useful:\n"
        f"\"{already_spoken}\"\n\n"
    ) if already_spoken else ""
    if context.get("is_followup"):
        user_prompt = (
            f"{prefix}A student didn't understand your previous explanation and asked you to "
            f"explain again. Do NOT repeat the same wording — use a different, simpler analogy.\n"
            f"{spoken_note}"
            f"Original question: {context.get('previous_question')}\n"
            f"Your previous explanation: {context.get('previous_answer')}\n"
            f"Relevant lecture material:\n{lecture_text}\n\nExplain it again, differently."
        )
    elif lecture_text:
        user_prompt = (
            f"{prefix}{spoken_note}Lecture material:\n{lecture_text}\n\n"
            f"Student's question: {question}\n"
            f"Answer using the lecture material above. If it doesn't fully cover the question, "
            f"say so honestly rather than making things up."
        )
    else:
        user_prompt = (
            f"{prefix}{spoken_note}No matching lecture material is available. "
            f"Answer the student's question from general knowledge.\n"
            f"Student's question: {question}"
        )
    return _groq_chat(ANSWER_SYSTEM_PROMPT, user_prompt)


# =====================================================================
# Intent detection
# =====================================================================
NEXT_SLIDE_PHRASES = ["next slide", "move on", "continue", "no doubts", "no doubt", "go ahead", "carry on"]
PREVIOUS_SLIDE_PHRASES = ["previous slide", "go back", "last slide", "slide before"]
_SLIDE_NUM_RE = re.compile(r'slide\s*(?:number\s*|#\s*)?(\d+)', re.IGNORECASE)


def is_next_slide_intent(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in NEXT_SLIDE_PHRASES)


def is_previous_slide_intent(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in PREVIOUS_SLIDE_PHRASES)


def extract_requested_slide_number(text: str) -> Optional[int]:
    m = _SLIDE_NUM_RE.search(text)
    return int(m.group(1)) if m else None


# =====================================================================
# GRAPH STATE
# =====================================================================
class ProfessorState(TypedDict):
    mode: str
    slide_index: int
    sentence_index: int
    slide_sentences: List[str]
    total_slides: int
    interrupt_reason: Optional[str]
    question_text: Optional[str]
    rag_context: Optional[dict]
    is_irrelevant: bool
    last_answer: Optional[str]
    spoken_log: Dict[int, str]
    qa_log: Dict[int, List[Tuple[str, str]]]


_interrupt: Optional[InterruptSource] = None
_mic_index: Optional[int] = None
_slide_client: Optional[SlideClient] = None


def _get_slides():
    return rag_engine.get_ordered_chunks()


# =====================================================================
# NODES
# =====================================================================
def dispatch_node(state: ProfessorState) -> ProfessorState:
    return state


def route(state: ProfessorState) -> str:
    slides = _get_slides()
    if state["mode"] == "lecturing" and state["slide_index"] >= len(slides):
        return "done"
    return state["mode"]


def speak_slide_node(state: ProfessorState) -> ProfessorState:
    slides = _get_slides()
    if state["slide_index"] >= len(slides):
        state["mode"] = "done"
        return state

    slide = slides[state["slide_index"]]
    slide_number = slide["slide_number"]

    if not state["slide_sentences"]:
        print(f"\n[TEACHING] Slide {slide_number} ({state['slide_index'] + 1}/{len(slides)})")
        explanation = generate_slide_explanation(slide["text"])
        state["slide_sentences"] = _split_sentences(explanation)
        state["spoken_log"][slide_number] = explanation
        state["sentence_index"] = 0

    _interrupt.clear()
    stop_index = speak_sentences_interruptible(
        state["slide_sentences"], state["sentence_index"], _interrupt
    )

    if stop_index < len(state["slide_sentences"]):
        # interrupted mid-sentence
        state["sentence_index"] = stop_index
        state["interrupt_reason"] = "barge_in"
        state["mode"] = "listen_question"
        return state

    # slide fully spoken — advance
    _slide_client.next()
    rag_engine.set_lecture_progress(slide_number + 1, max_slide=slide_number + 1)
    state["slide_index"] += 1
    state["sentence_index"] = 0
    state["slide_sentences"] = []
    state["mode"] = "lecturing"
    return state


def listen_question_node(state: ProfessorState) -> ProfessorState:
    question = listen_and_transcribe(_mic_index)
    if not question.strip():
        # resume where left off
        state["mode"] = "lecturing"
        state["interrupt_reason"] = None
        return state

    slides = _get_slides()
    current_slide_number = slides[state["slide_index"]]["slide_number"] if state["slide_index"] < len(slides) else None

    if is_next_slide_intent(question):
        state["mode"] = "lecturing"
        state["slide_sentences"] = []
        state["sentence_index"] = 0
        _slide_client.next()
        if current_slide_number:
            rag_engine.set_lecture_progress(current_slide_number + 1, max_slide=current_slide_number + 1)
        state["slide_index"] += 1
        return state

    if is_previous_slide_intent(question) and state["slide_index"] > 0:
        prev_number = slides[state["slide_index"] - 1]["slide_number"]
        state["slide_index"] -= 1
        state["sentence_index"] = 0
        if prev_number in state["spoken_log"]:
            state["slide_sentences"] = _split_sentences(state["spoken_log"][prev_number])
        else:
            state["slide_sentences"] = []
        _slide_client.goto(prev_number)
        state["mode"] = "lecturing"
        return state

    requested = extract_requested_slide_number(question)
    if requested is not None:
        target = next((s for s in slides if s["slide_number"] == requested), None)
        if target:
            state["slide_index"] = slides.index(target)
            state["sentence_index"] = 0
            state["slide_sentences"] = (_split_sentences(state["spoken_log"][requested])
                                         if requested in state["spoken_log"] else [])
            _slide_client.goto(requested)
            state["mode"] = "lecturing"
            return state

    state["question_text"] = question
    state["mode"] = "relevance_check"
    return state


def relevance_check_node(state: ProfessorState) -> ProfessorState:
    question = state["question_text"]
    context = rag_engine.get_context_for_query(question)
    chunks = context.get("chunks", [])
    top_score = chunks[0]["score"] if chunks else -1.0
    is_irrelevant = (not context.get("is_followup")) and (not chunks or top_score < RELEVANCE_THRESHOLD)
    state["rag_context"] = context
    state["is_irrelevant"] = is_irrelevant
    if is_irrelevant:
        print(f"[RELEVANCE] Question scored {top_score:.2f} (< {RELEVANCE_THRESHOLD}) — treating as off-topic.")
    state["mode"] = "answering"
    return state


def answer_node(state: ProfessorState) -> ProfessorState:
    slides = _get_slides()
    current_slide_number = slides[state["slide_index"]]["slide_number"] if state["slide_index"] < len(slides) else None
    already_spoken = state["spoken_log"].get(current_slide_number, "")

    answer = generate_answer(
        state["question_text"], state["rag_context"], already_spoken, state["is_irrelevant"]
    )
    speak_text(answer)

    rag_engine.add_to_history(state["question_text"], answer, state["rag_context"].get("chunks", []))
    if current_slide_number is not None:
        state["qa_log"].setdefault(current_slide_number, []).append((state["question_text"], answer))

    state["last_answer"] = answer
    state["mode"] = "post_answer"
    return state


def post_answer_node(state: ProfessorState) -> ProfessorState:
    state["question_text"] = None
    state["rag_context"] = None
    state["is_irrelevant"] = False
    state["interrupt_reason"] = None
    state["mode"] = "lecturing"
    return state


# =====================================================================
# GRAPH BUILD
# =====================================================================
def build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("[SETUP] langgraph not installed. Run: pip install langgraph")
        raise

    graph = StateGraph(ProfessorState)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("lecturing", speak_slide_node)
    graph.add_node("listen_question", listen_question_node)
    graph.add_node("relevance_check", relevance_check_node)
    graph.add_node("answering", answer_node)
    graph.add_node("post_answer", post_answer_node)

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges("dispatch", route, {
        "lecturing": "lecturing",
        "listen_question": "listen_question",
        "relevance_check": "relevance_check",
        "answering": "answering",
        "post_answer": "post_answer",
        "done": END,
    })
    for node in ["lecturing", "listen_question", "relevance_check", "answering", "post_answer"]:
        graph.add_edge(node, END)

    return graph.compile()


# =====================================================================
# MAIN
# =====================================================================
def find_latest_lecture_file():
    candidates = []
    for folder in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inbox"),
    ]:
        if os.path.isdir(folder):
            candidates.extend([
                os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in (".pptx", ".pdf", ".docx")
            ])
    return max(candidates, key=os.path.getmtime) if candidates else None


def main():
    global _interrupt, _mic_index, _slide_client

    _mic_index = find_mic_device()
    _slide_client = SlideClient()

    _interrupt = KeyboardInterruptSource()
    _interrupt.start()

    lecture_path = find_latest_lecture_file()
    if lecture_path:
        n_chunks = rag_engine.load_lecture(lecture_path)
        print(f"[SETUP] Loaded lecture: {os.path.basename(lecture_path)} — {n_chunks} chunks.")
    else:
        print("[SETUP] No lecture file in inbox — nothing to teach yet.")
        return

    graph = build_graph()

    state: ProfessorState = {
        "mode": "lecturing",
        "slide_index": 0,
        "sentence_index": 0,
        "slide_sentences": [],
        "total_slides": len(rag_engine.get_ordered_chunks()),
        "interrupt_reason": None,
        "question_text": None,
        "rag_context": None,
        "is_irrelevant": False,
        "last_answer": None,
        "spoken_log": {},
        "qa_log": {},
    }

    print("AI Professor orchestrator (LangGraph) started. Ctrl+C to stop.")
    try:
        while True:
            state = graph.invoke(state, config={"recursion_limit": 10})
            slides = _get_slides()
            if state["mode"] == "lecturing" and state["slide_index"] >= len(slides):
                speak_text("That's the end of today's lecture. Let me know if you have any final questions.")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _interrupt.stop()


if __name__ == "__main__":
    main()
