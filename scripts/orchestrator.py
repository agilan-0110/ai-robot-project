"""
AI Professor Robot — Unified Orchestrator & State Machine
----------------------------------------------------------
Features:
- Complete autonomous classroom teaching loop (teach_class) with doubt detour flow
- Hub-and-spoke LangGraph state machine for sentence-level barge-in interruptions
- Bidirectional SlideClient with automatic retry and backoff
- Multi-engine TTS (edge-tts + Piper ONNX) with cross-platform audio playback (ffplay / aplay / winsound)
- Math-to-speech normalization and vocabulary biasing
- Image-only caption slide support and 120-150 word elaborate lecture explanations
- Checkpoint persistence and crash recovery
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
from unittest.mock import MagicMock

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
GROQ_API_KEY = os.environ.get("GROQ_LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_LLM_API_KEY = GROQ_API_KEY
GROQ_WHISPER_API_KEY = os.environ.get("GROQ_WHISPER_API_KEY") or GROQ_API_KEY
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_LLM_MODEL = GROQ_MODEL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_LLM_URL = GROQ_URL
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000")

EDGE_TTS_VOICE = "en-US-GuyNeural"

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

RELEVANCE_THRESHOLD = 0.30

INBOX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_checkpoint.json")


# =====================================================================
# VOCABULARY & CORRECTIONS
# =====================================================================
SUBJECT_VOCAB = {
    "general": (
        "lecture, slide, slide number, syllabus, exam, question, answer, doubt, "
        "topic, concept, diagram, definition, formula, derivation, example, "
        "step, proof, summary, recall, repeat, explain again, previous slide"
    ),
    "physics": (
        "force, mass, acceleration, velocity, momentum, inertia, gravity, "
        "friction, tension, torque, work, energy, kinetic energy, potential energy, "
        "power, conservation, Newton, second law, first law, third law, normal force, "
        "free body diagram, vector, scalar, displacement, projectile, circular motion, "
        "centripetal, spring constant, Hooke, Joule, Watt, Newton metre, impulse, "
        "kilogram, metre per second squared, radians, angular velocity, equilibrium"
    ),
    "computer science": (
        "algorithm, complexity, big O, array, linked list, stack, queue, tree, "
        "binary search, graph, recursion, iteration, dynamic programming, sorting, "
        "pointer, memory, CPU, cache, thread, process, deadlock, bandwidth, latency"
    )
}

COMMON_CORRECTIONS = {
    "f is equal to m a": "F equals m a",
    "f equals m a": "F equals m a",
    "f=ma": "F equals m a",
    "newton second law": "Newton's second law",
    "newtons second law": "Newton's second law",
}

MATH_REPLACEMENTS = [
    (r'(\w+)\^2\b', r'\1 squared'),
    (r'(\w+)\^3\b', r'\1 cubed'),
    (r'(\w+)\^(\w+)', r'\1 to the power of \2'),
    (r'√(\w+)', r'square root of \1'),
    (r'∑', ' sum of '),
    (r'∏', ' product of '),
    (r'∞', ' infinity '),
    (r'π', ' pi '),
    (r'θ', ' theta '),
    (r'∆|Δ', ' delta '),
    (r'±', ' plus or minus '),
    (r'×', ' times '),
    (r'÷', ' divided by '),
    (r'=', ' equals '),
    (r'°', ' degrees '),
    (r'\s+', ' '),
]


def maths_to_speech(text):
    for pattern, repl in MATH_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    return text


def build_initial_prompt(subject):
    terms = SUBJECT_VOCAB.get(subject.lower().strip(), SUBJECT_VOCAB["general"])
    return f"This is a {subject} classroom in India. Students ask about: {terms}."


def watch_for_hand_raise(previously_selected_id=None):
    print("\n[IDLE] Waiting for hand raise... (press Enter to simulate)")
    input()
    print("[IDLE] Hand raise detected (simulated).")
    return True, None


# =====================================================================
# CHECKPOINTS
# =====================================================================
def load_checkpoint(lecture_path: str):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, {}, {}
    try:
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        if data.get("lecture_path") == lecture_path:
            idx = data.get("slide_index", 0)
            explanations = {int(k): v for k, v in data.get("explanations", {}).items()}
            qa = {int(k): v for k, v in data.get("qa_history", {}).items()}
            return idx, explanations, qa
    except Exception as e:
        print(f"[CHECKPOINT] Could not load checkpoint: {e}")
    return 0, {}, {}


def save_checkpoint(lecture_path: str, slide_index: int, explanations: dict, qa_history: dict):
    try:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({
                "lecture_path": lecture_path,
                "slide_index": slide_index,
                "explanations": explanations,
                "qa_history": qa_history,
                "timestamp": time.time()
            }, f)
    except Exception as e:
        print(f"[CHECKPOINT] Could not save checkpoint: {e}")


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except Exception:
            pass


# =====================================================================
# INTERRUPT SOURCE
# =====================================================================
class InterruptSource:
    def is_set(self) -> bool:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass


class KeyboardInterruptSource(InterruptSource):
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
            print(f"[BARGE-IN] Note: pynput keyboard listener unavailable ({e}).")

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
# SLIDE CLIENT — with retry, backoff, and full backward compatibility
# =====================================================================
class SlideClient:
    def __init__(self, host="127.0.0.1", port=5000, base_url=None, timeout=2.0, max_retries=3):
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries

    def send_command(self, command: str, slide_number: Optional[int] = None) -> bool:
        payload = {"command": command}
        if slide_number is not None:
            payload["slide"] = slide_number

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(f"{self.base_url}/api/slide/command", json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") in ("done", "ok", "success"):
                        return True
                    return True
            except Exception:
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    time.sleep(backoff)
        return False

    def next(self) -> bool:
        return self.send_command("next")

    def next_slide(self) -> bool:
        return self.send_command("next")

    def goto(self, slide_number: int) -> bool:
        return self.send_command("goto", slide_number)

    def goto_slide(self, slide_number: int) -> bool:
        return self.send_command("goto", slide_number)

    def previous(self) -> bool:
        return self.send_command("prev")

    def prev_slide(self) -> bool:
        return self.send_command("prev")

    def get_status(self) -> Optional[dict]:
        try:
            resp = requests.get(f"{self.base_url}/api/slide/status", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def is_connected(self) -> bool:
        return self.get_status() is not None


# =====================================================================
# TTS ENGINE & PLAYBACK
# =====================================================================
def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    return sentences or [text.strip()]


async def _edge_tts_to_file(text: str, out_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(out_path)


def synthesize_to_file(text: str, out_path: str):
    try:
        asyncio.run(_edge_tts_to_file(text, out_path))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            return
    except Exception:
        pass

    if os.path.exists(PIPER_MODEL):
        try:
            from piper import PiperVoice, SynthesisConfig
            voice = PiperVoice.load(PIPER_MODEL)
            wav_path = out_path.replace(".mp3", ".wav")
            with wave.open(wav_path, "wb") as wf:
                voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.15))
            if os.path.exists(wav_path):
                if out_path.endswith(".mp3"):
                    try:
                        import shutil
                        shutil.move(wav_path, out_path)
                    except Exception:
                        pass
        except Exception as pe:
            print(f"[SPEAKING] Piper fallback error: {pe}")


def play_interruptible(audio_path: str, interrupt: Optional[InterruptSource] = None) -> bool:
    dummy_interrupt = interrupt or KeyboardInterruptSource()
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
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
            if dummy_interrupt.is_set():
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
    if not text or not text.strip():
        return
    print(f"[SPEAKING] {text}")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        synthesize_to_file(text, tmp.name)
        if os.path.exists(tmp.name):
            play_interruptible(tmp.name, interrupt)
    finally:
        try:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
        except Exception:
            pass


def speak_answer(piper_voice, text: str):
    if not text or not text.strip():
        return
    print(f"[SPEAKING] {text}")
    if piper_voice and not isinstance(piper_voice, MagicMock) and hasattr(piper_voice, "synthesize_wav"):
        try:
            from piper import SynthesisConfig
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            with wave.open(tmp_wav.name, "wb") as wf:
                piper_voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.15))
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(tmp_wav.name, winsound.SND_FILENAME)
            else:
                subprocess.run(["aplay", "-D", AUDIO_DEVICE, tmp_wav.name], check=True)
            try:
                os.remove(tmp_wav.name)
            except Exception:
                pass
            return
        except Exception:
            pass
    speak_text(text)


# =====================================================================
# STT — Whisper
# =====================================================================
class WhisperSession:
    def __init__(self, model_size="base.en", reload_every=10):
        self.model_size = model_size
        self.reload_every = reload_every
        self.count = 0
        self.model = None

    def transcribe(self, audio: np.ndarray) -> str:
        self.count += 1
        return groq_transcribe(audio)


def find_mic_device(name_hint=MIC_NAME_HINT):
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        matches = [i for i, d in enumerate(devices)
                   if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower()]
        if matches:
            return matches[0]
        any_input = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if any_input:
            return any_input[0][0]
    except Exception:
        pass
    return 0


def record(mic_index) -> Tuple[np.ndarray, bool]:
    try:
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
    except Exception:
        return np.array([], dtype='float32'), False


def groq_transcribe(audio: np.ndarray) -> str:
    if not GROQ_WHISPER_API_KEY or len(audio) == 0:
        return ""
    audio_mono = audio.mean(axis=1).astype(np.float32) if len(audio.shape) > 1 else audio.astype(np.float32)
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
    except Exception:
        return ""


def listen_and_transcribe(session_or_mic=None, initial_prompt=None, mic_index=None) -> str:
    actual_mic = mic_index if mic_index is not None else (session_or_mic if isinstance(session_or_mic, int) else 0)
    audio, speech_started = record(actual_mic)
    if len(audio) == 0 or not speech_started:
        return ""
    return groq_transcribe(audio)


# =====================================================================
# LLM HELPERS & PROMPT BUILDERS
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


def normalize_unicode_text(text):
    return clean_for_speech(text)


def extract_image_captions_if_image_only(slide_text):
    lines = [l.strip() for l in slide_text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    captions = []
    has_image = False
    has_text_bullets = False

    for l in lines:
        if l.startswith("#"):
            continue
        m = re.match(r'^\[Image:\s*(.*)\]$', l)
        if m:
            has_image = True
            captions.append(m.group(1).strip())
        else:
            has_text_bullets = True

    if has_image and not has_text_bullets:
        return captions
    return None


def build_slide_explanation_prompt(slide_text):
    system_prompt = (
        "You are a friendly professor teaching a live class out loud. "
        "Teach and elaborate on the slide content thoroughly in natural, spoken prose, "
        "the way a great teacher explains concepts face-to-face to students. "
        "Unpack each key idea with necessary context, explain why it matters, "
        "and provide a clear, relatable example to help the concept land. "
        "Use the slide's own key terms and facts — do not invent claims or contradict the material. "
        "Aim for approximately 120-150 words (about 45-60 seconds of spoken explanation) "
        "so students get a complete, well-elaborated understanding. "
        "Never use bullet points, numbered lists, markdown formatting, headers, or bold text. "
        "When explaining maths, never use math symbols (like ^, √, ≠, ÷, π, ∫) or fractions written as a/b — "
        "always write them out in plain spoken words (e.g. 'x squared', 'the square root of sixteen'). "
        "Do not greet or introduce yourself, just begin teaching and elaborating on the topic directly."
    )

    captions = extract_image_captions_if_image_only(slide_text)
    if captions:
        caption_desc = " ".join(captions)
        heading_match = re.search(r'^\s*#+\s*(.+)$', slide_text, re.MULTILINE)
        heading_prefix = f"Slide Topic: {heading_match.group(1).strip()}\n" if heading_match else ""
        user_content = (
            f"{heading_prefix}This slide has no text content, only an image. "
            f"Here is a description of that image: {caption_desc}. "
            f"Explain the slide's topic based on this description."
        )
    else:
        user_content = f"Slide content:\n{slide_text}"

    return system_prompt, user_content


def _groq_chat(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.7) -> str:
    key = GROQ_API_KEY or GROQ_LLM_API_KEY
    if not key:
        return "Sorry, I can't reach my brain right now. Please check the GROQ_API_KEY setting."
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return clean_for_speech(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception as e:
        print(f"[THINKING] Groq error: {e}")
        return "Sorry, I had trouble reaching the answer service."


def generate_slide_explanation(slide_text: str) -> str:
    system_prompt, user_content = build_slide_explanation_prompt(slide_text)
    return _groq_chat(system_prompt, user_content, max_tokens=600)


def answer_doubt_about_slide(slide: dict, doubt_question: str) -> str:
    prompt = f"Slide content:\n{slide['text']}\n\nStudent's doubt:\n{doubt_question}"
    return _groq_chat(ANSWER_SYSTEM_PROMPT, prompt, max_tokens=400)


def get_llm_answer(question: str, context: dict) -> str:
    if context.get("not_covered_yet", False):
        slide_num = context.get("requested_slide")
        slide_ref = f"slide {slide_num}" if slide_num else "that slide"
        ans = f"We haven't covered {slide_ref} yet in today's lecture. Let's continue with our current topic first, and we will get to that soon."
        print(f"[THINKING] Answer (not covered yet): {ans}")
        return ans

    lecture_text = context.get("lecture_text", "")
    is_followup = context.get("is_followup", False)

    if is_followup:
        user_content = (
            f"A student didn't understand your previous explanation and asked you to explain again.\n"
            f"Original question: {context.get('previous_question')}\n"
            f"Previous explanation: {context.get('previous_answer')}\n"
            f"Lecture material:\n{lecture_text}\n\nExplain it again differently."
        )
    elif lecture_text:
        user_content = f"Lecture material:\n{lecture_text}\n\nStudent's question: {question}"
    else:
        user_content = f"Student's question: {question}"

    return _groq_chat(ANSWER_SYSTEM_PROMPT, user_content)


ANSWER_SYSTEM_PROMPT = (
    "You are a friendly professor speaking out loud to a student in a classroom. "
    "Answer in natural, spoken sentences — never bullet points, numbers, or markdown. "
    "When explaining maths, write symbols out in plain spoken words "
    "(e.g. 'x squared', not 'x^2'). Keep it to 2-4 short sentences unless the question "
    "truly needs more depth. Be warm and encouraging."
)


def generate_answer(question: str, context: dict, already_spoken: str, is_irrelevant: bool) -> str:
    lecture_text = "" if is_irrelevant else context.get("lecture_text", "")
    prefix = ""
    if is_irrelevant:
        prefix = ("This question doesn't seem to relate to today's slide content. "
                  "Briefly and politely note that before answering from general knowledge. ")
    spoken_note = (
        f"For context, here is what you just said to the class:\n\"{already_spoken}\"\n\n"
    ) if already_spoken else ""

    if context.get("is_followup"):
        user_prompt = (
            f"{prefix}A student asked you to explain again differently.\n"
            f"{spoken_note}"
            f"Original question: {context.get('previous_question')}\n"
            f"Your previous explanation: {context.get('previous_answer')}\n"
            f"Relevant lecture material:\n{lecture_text}\n\nExplain it again."
        )
    elif lecture_text:
        user_prompt = (
            f"{prefix}{spoken_note}Lecture material:\n{lecture_text}\n\n"
            f"Student's question: {question}"
        )
    else:
        user_prompt = (
            f"{prefix}{spoken_note}Student's question: {question}"
        )
    return _groq_chat(ANSWER_SYSTEM_PROMPT, user_prompt)


# =====================================================================
# INTENT DETECTION
# =====================================================================
NEXT_SLIDE_PHRASES = ["next slide", "move on", "continue", "no doubts", "no doubt", "go ahead", "carry on"]
PREVIOUS_SLIDE_PHRASES = ["previous slide", "go back", "last slide", "slide before"]
PAST_QUESTION_PHRASES = ["what did you say", "what was the doubt", "what did i ask", "previous question", "last question", "what was asked"]
_SLIDE_NUM_RE = re.compile(r'slide\s*(?:number\s*|#\s*)?(\d+)', re.IGNORECASE)


def is_next_slide_intent(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in NEXT_SLIDE_PHRASES)


def is_previous_slide_intent(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in PREVIOUS_SLIDE_PHRASES)


def is_asking_about_past_questions(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in PAST_QUESTION_PHRASES)


def extract_requested_slide_number(text: str) -> Optional[int]:
    m = _SLIDE_NUM_RE.search(text)
    return int(m.group(1)) if m else None


# =====================================================================
# AUTONOMOUS CLASSROOM TEACHING (Legacy Loop & Detour Logic)
# =====================================================================
def teach_class(whisper_session, piper_voice, initial_prompt, mic_index, lecture_path, slide_client=None):
    slides = rag_engine.get_ordered_chunks()
    if not slides:
        print("[TEACHING] No slides found in loaded lecture.")
        return

    if slide_client is None:
        slide_client = SlideClient()

    start_index, slide_explanations, slide_qa_history = load_checkpoint(lecture_path)
    if start_index > 0:
        print(f"[TEACHING] Resuming lecture from slide {start_index + 1}/{len(slides)}")

    current_lecture_slide = 0
    max_slide_reached = 0

    for i in range(start_index, len(slides)):
        slide = slides[i]
        current_number = slide["slide_number"]
        current_lecture_slide = current_number
        max_slide_reached = max(max_slide_reached, current_number)
        rag_engine.set_lecture_progress(current_lecture_slide, max_slide_reached)

        # 1. Advance slide first
        slide_client.next_slide()

        # 2. Generate and speak explanation
        print(f"\n[TEACHING] Slide {current_number} ({i + 1}/{len(slides)})")
        if current_number in slide_explanations:
            explanation = slide_explanations[current_number]
        else:
            explanation = generate_slide_explanation(slide["text"])
            slide_explanations[current_number] = explanation
            save_checkpoint(lecture_path, i, slide_explanations, slide_qa_history)

        speak_answer(piper_voice, explanation)

        # 3. Question answering loop for this slide
        while True:
            speak_answer(piper_voice, "Any questions on this slide?")
            response = listen_and_transcribe(whisper_session, initial_prompt, mic_index)

            if not response.strip() or is_next_slide_intent(response):
                print("[TEACHING] Moving to next slide.")
                break

            if is_previous_slide_intent(response):
                if i > 0:
                    prev_number = slides[i - 1]["slide_number"]
                    slide_client.goto_slide(prev_number)
                    if prev_number in slide_explanations:
                        speak_answer(piper_voice, slide_explanations[prev_number])
                    else:
                        explanation = generate_slide_explanation(slides[i - 1]["text"])
                        slide_explanations[prev_number] = explanation
                        speak_answer(piper_voice, explanation)
                    slide_client.goto_slide(current_lecture_slide)
                else:
                    speak_answer(piper_voice, "This is already the first slide.")
                continue

            if is_asking_about_past_questions(response):
                target_number = extract_requested_slide_number(response) or current_number
                history = slide_qa_history.get(target_number, [])
                if history:
                    q, a = history[-1]
                    speak_answer(piper_voice, f"On slide {target_number}, you asked: {q}. I answered: {a}")
                else:
                    speak_answer(piper_voice, f"You haven't asked anything on slide {target_number} yet.")
                continue

            requested_slide = extract_requested_slide_number(response)
            if requested_slide is not None:
                if requested_slide == current_lecture_slide:
                    if requested_slide in slide_explanations:
                        speak_answer(piper_voice, slide_explanations[requested_slide])
                    else:
                        explanation = generate_slide_explanation(slide["text"])
                        slide_explanations[requested_slide] = explanation
                        speak_answer(piper_voice, explanation)
                elif requested_slide <= max_slide_reached:
                    slide_client.goto_slide(requested_slide)
                    if requested_slide in slide_explanations:
                        speak_answer(piper_voice, slide_explanations[requested_slide])
                    else:
                        target = next((s for s in slides if s["slide_number"] == requested_slide), None)
                        if target:
                            explanation = generate_slide_explanation(target["text"])
                            slide_explanations[requested_slide] = explanation
                            speak_answer(piper_voice, explanation)
                    slide_client.goto_slide(current_lecture_slide)
                else:
                    target = next((s for s in slides if s["slide_number"] == requested_slide), None)
                    if target:
                        print(f"[TEACHING] Slide {requested_slide} requested but not covered yet.")
                        speak_answer(piper_voice, f"We haven't covered slide {requested_slide} yet. Let's continue with today's lecture first.")
                    else:
                        speak_answer(piper_voice, f"I don't have slide {requested_slide} in this lecture.")
                continue

            answer = answer_doubt_about_slide(slide, response)
            slide_qa_history.setdefault(current_number, []).append((response, answer))
            rag_engine.add_to_history(response, answer, [])
            speak_answer(piper_voice, answer)
            save_checkpoint(lecture_path, i, slide_explanations, slide_qa_history)

    speak_answer(piper_voice, "That's the end of today's lecture. Let me know if you have any final questions.")
    clear_checkpoint()
    print("[TEACHING] Class complete.")


# =====================================================================
# GRAPH STATE MACHINE (LangGraph Engine)
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

    if _interrupt:
        _interrupt.clear()
    stop_index = speak_sentences_interruptible(
        state["slide_sentences"], state["sentence_index"], _interrupt or KeyboardInterruptSource()
    )

    if stop_index < len(state["slide_sentences"]):
        state["sentence_index"] = stop_index
        state["interrupt_reason"] = "barge_in"
        state["mode"] = "listen_question"
        return state

    if _slide_client:
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
        state["mode"] = "lecturing"
        state["interrupt_reason"] = None
        return state

    slides = _get_slides()
    current_slide_number = slides[state["slide_index"]]["slide_number"] if state["slide_index"] < len(slides) else None

    if is_next_slide_intent(question):
        state["mode"] = "lecturing"
        state["slide_sentences"] = []
        state["sentence_index"] = 0
        if _slide_client:
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
        if _slide_client:
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
            if _slide_client:
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
