"""
AI Professor Robot — Integrated Orchestrator
---------------------------------------------
Combines:
  - Nithesh's voice pipeline (Whisper STT -> Groq LLM -> Piper TTS)
  - Subash's RAG module (rag_engine.py) for lecture-grounded answers
  - A placeholder for Agilan's hand-raise/camera module (see the
    CAMERA INTEGRATION section below for exactly how to plug it in
    once the CSI camera + adapter are connected)

Run this file directly on the Jetson:
    python orchestrator.py

Requires rag_engine.py in the same folder (or on PYTHONPATH), and
GROQ_API_KEY set as an environment variable.
"""
import os
import sys
import time
import re
import subprocess
import wave
import traceback
import requests
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import rag_engine

# =====================================================================
# MIC CONFIG — auto-detected at startup instead of a hardcoded index,
# since USB device indices can shift after every reboot/reconnect
# (this was the root cause of the "Invalid number of channels" bug).
# =====================================================================
MIC_NAME_HINT = "Microphone Array"   # substring to look for in the mic's device name;
                         # update this once you confirm the mic's real
                         # name from `python -c "import sounddevice as sd; print(sd.query_devices())"`
RECORD_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000

# ——— RECORDING / SILENCE SETTINGS ———
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.02
SILENCE_SECONDS = 1.5
MAX_WAIT_FOR_SPEECH_SECONDS = 5
DOUBT_WAIT_SECONDS = 4  # shorter timeout used only for the "Any doubts?" check during teaching
MAX_UTTERANCE_SECONDS = 20
CONSECUTIVE_CHUNKS_TO_CONFIRM_SPEECH = 3

HALLUCINATION_PHRASES = {
    "thank you",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "please subscribe",
}

SUBJECT_VOCAB = {
    "computer science": "variables, algorithms, DSA, data structures, Python, Java, GPU, CPU, machine learning, AI",
    "mathematics": "algebra, calculus, geometry, equations, derivatives, integrals, matrices, probability",
    "physics": "force, velocity, acceleration, energy, momentum, electricity, magnetism, Newton, gravity",
    "chemistry": "atoms, molecules, reactions, elements, compounds, acids, bases, periodic table",
    "biology": "cells, DNA, genetics, organisms, evolution, photosynthesis, ecosystem",
    "general": "school, class, students, exam, homework, project",
}

COMMON_CORRECTIONS = {
    "unvariable": "a variable",
    "data such as": "DSA",
    "1 in nano": "orin nano",
    "ai gp jetson": "ai gpu jetson",
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"   # confirmed production model — 20b is unavailable
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ——— TTS / SPEAKER SETTINGS ———
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_VOICE = os.path.join(_PROJECT_ROOT, "voices", "en_US-ryan-medium.onnx")
PIPER_MODEL = os.environ.get("PIPER_MODEL", _LOCAL_VOICE if os.path.exists(_LOCAL_VOICE) else "/home/jetson/ai-robot-project/voices/en_US-ryan-medium.onnx")
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "plughw:0,3")
SPEECH_WAV = os.environ.get("SPEECH_WAV", os.path.join(_PROJECT_ROOT, "last_answer.wav") if sys.platform == "win32" else "/home/jetson/ai-professor/last_answer.wav")
PIPER_LENGTH_SCALE = 1.2

# ——— SLIDE COMPANION / PROJECTOR SETTINGS (Feature 1) ———
SLIDE_COMPANION_HOST = os.environ.get("SLIDE_COMPANION_HOST", "127.0.0.1")
SLIDE_COMPANION_PORT = int(os.environ.get("SLIDE_COMPANION_PORT", "5000"))


class SlideClient:
    """
    Client for autonomous slide control.
    Supports:
      1. Browser Live Viewer on Jetson (Flask app.py on port 5000 -> /api/slide/command) [Default, 0 software on laptop]
      2. Companion script on laptop (slide_companion.py on port 5055 -> /command)
    Retries up to 3 times with 1s backoff if no 'done' acknowledgment is received.
    """
    def __init__(self, host=SLIDE_COMPANION_HOST, port=SLIDE_COMPANION_PORT, timeout=3.0, max_retries=3):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        if int(self.port) == 5000:
            self.url = f"http://{self.host}:{self.port}/api/slide/command"
        else:
            self.url = f"http://{self.host}:{self.port}/command"

    def send_command(self, command, slide_number=None):
        payload = {"command": command}
        if slide_number is not None:
            payload["slide"] = int(slide_number)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "done":
                        print(f"[SLIDES] Successfully executed '{command}' (ack: done)")
                        return True
                print(f"[SLIDES] Attempt {attempt} returned status {resp.status_code}")
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"[SLIDES] Slide command '{command}' attempt {attempt} failed ({e}), retrying in 1s...")
                    time.sleep(1)
                else:
                    print(f"[SLIDES] Slide command '{command}' failed after {self.max_retries} attempts ({e}).")
        return False

    def next_slide(self):
        """Advances projector by exactly one slide."""
        return self.send_command("next")

    def goto_slide(self, slide_number):
        """Jumps projector directly to slide_number."""
        return self.send_command("goto", slide_number)


# ——— MATHS NOTATION -> SPOKEN WORDS ———
MATH_REPLACEMENTS = [
    (r'√\(([^)]+)\)', r'the square root of \1'),
    (r'√(\w+)', r'the square root of \1'),
    (r'√', 'square root of'),
    (r'(\w+)\^2\b', r'\1 squared'),
    (r'(\w+)\^3\b', r'\1 cubed'),
    (r'(\w+)\^(\w+)', r'\1 to the power of \2'),
    (r'\b(\d+)/(\d+)\b', r'\1 over \2'),
    (r'≠', ' is not equal to '),
    (r'≈', ' is approximately equal to '),
    (r'≤', ' is less than or equal to '),
    (r'≥', ' is greater than or equal to '),
    (r'∫', ' integral of '),
    (r'∑', ' sum of '),
    (r'∏', ' product of '),
    (r'∞', ' infinity '),
    (r'π', ' pi '),
    (r'θ', ' theta '),
    (r'∆|Δ', ' delta '),
    (r'±', ' plus or minus '),
    (r'×', ' times '),
    (r'÷', ' divided by '),
    (r'∂', ' partial derivative of '),
    (r'∈', ' belongs to '),
    (r'∀', ' for all '),
    (r'∃', ' there exists '),
    (r'=', ' equals '),
    (r'(?<=\w)\s*\+\s*(?=\w)', ' plus '),
    (r'°', ' degrees '),
    (r'\s+', ' '),
]


def build_initial_prompt(subject):
    terms = SUBJECT_VOCAB.get(subject.lower().strip(), SUBJECT_VOCAB["general"])
    return f"This is a {subject} classroom in India. Students ask about: {terms}."


def apply_corrections(text):
    for wrong, right in COMMON_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def maths_to_speech(text):
    for pattern, repl in MATH_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    return text


# =====================================================================
# MIC DISCOVERY — replaces the old hardcoded DEVICE_INDEX.
# Finds a real input-capable device by name at startup, every time,
# so a shifted USB index (or a reboot with no RTC/clock issues) never
# silently breaks recording again.
# =====================================================================
def find_mic_device(name_hint=MIC_NAME_HINT):
    import sounddevice as sd
    devices = sd.query_devices()

    matches = [
        i for i, d in enumerate(devices)
        if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower()
    ]
    if matches:
        idx = matches[0]
        print(f"[SETUP] Using mic: [{idx}] {devices[idx]['name']} "
              f"({devices[idx]['max_input_channels']} in)")
        return idx

    # No name match — pick default or first available input device
    any_input = [
        (i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0
    ]
    if any_input:
        default_idx = any_input[0][0]
        print(f"[SETUP] Note: name hint '{name_hint}' not matched, falling back to default mic: [{default_idx}] {devices[default_idx]['name']}")
        return default_idx
    else:
        print("[SETUP] No input-capable audio devices found at all.")
        print("[SETUP] Check that the mic is physically connected/powered on, then run:")
        print('    python -c "import sounddevice as sd; print(sd.query_devices())"')
        raise RuntimeError("No usable microphone found — see device list above.")


# =====================================================================
# CAMERA / HAND-RAISE INTEGRATION POINT
# ---------------------------------------------------------------------
# This is a placeholder until the CSI camera + adapter are connected
# and Agilan's YOLOv8-pose module is wired in. It currently simulates
# a hand raise with an Enter keypress so the rest of the pipeline can
# be tested end-to-end right now.
#
# TO INTEGRATE THE REAL CAMERA MODULE LATER:
#   1. `import hand_raise` (Agilan's module file)
#   2. In main(), once at startup:
#          yolo_model = YOLO("yolov8n-pose.pt")
#          cap = cv2.VideoCapture(<camera source>)
#   3. Replace the call to watch_for_hand_raise() below with:
#          triggered, selected_id = hand_raise.watch_for_hand_raise(
#              yolo_model, cap, previously_selected_id=selected_id
#          )
#      (his function already returns this exact (bool, id) shape, so
#      the loop in main() below needs no other changes)
# =====================================================================
def watch_for_hand_raise(previously_selected_id=None):
    """
    STUB — simulates a hand-raise trigger by waiting for Enter.
    Returns (triggered: bool, selected_id: None) to match the shape
    Agilan's real watch_for_hand_raise() will return, so main()'s loop
    doesn't need to change when the camera module is wired in.
    """
    print("\n[IDLE] Waiting for hand raise... (press Enter to simulate)")
    input()
    print("[IDLE] Hand raise detected (simulated).")
    return True, None


def record(mic_index, max_wait_override=None):
    import sounddevice as sd

    wait_seconds = max_wait_override if max_wait_override is not None else MAX_WAIT_FOR_SPEECH_SECONDS
    silence_chunks = int(SILENCE_SECONDS * RECORD_SAMPLE_RATE / CHUNK_SIZE)
    max_wait_chunks = int(wait_seconds * RECORD_SAMPLE_RATE / CHUNK_SIZE)
    max_speech_chunks = int(MAX_UTTERANCE_SECONDS * RECORD_SAMPLE_RATE / CHUNK_SIZE)

    print("[LISTENING] (listening...)")
    recorded_chunks = []
    silent_count = 0
    speech_started = False
    loud_streak = 0
    waited_chunks = 0
    speech_chunks = 0

    with sd.InputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=2,
        dtype='float32',
        device=mic_index,
        blocksize=CHUNK_SIZE
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            recorded_chunks.append(chunk.copy())
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

    if not recorded_chunks:
        return np.array([], dtype='float32'), False

    audio = np.concatenate(recorded_chunks, axis=0)
    duration = len(audio) / RECORD_SAMPLE_RATE

    if speech_started and silent_count >= silence_chunks:
        reason = "hit silence after speech"
    elif not speech_started:
        reason = "gave up waiting for speech"
    else:
        reason = "hit max utterance cap"

    print(f"[LISTENING] recorded {duration:.1f}s | speech detected: {speech_started} | stopped: {reason}")
    return audio, speech_started


class WhisperSession:
    """
    Wraps the Whisper model and periodically reloads it after a number
    of transcriptions. faster-whisper's CUDA backend (ctranslate2) has
    no manual 'defragment' call, and long sessions of repeated
    transcribe() calls have been observed to fragment GPU memory until
    it runs out, even with nothing else competing for it. Reloading
    the model periodically forces CUDA to release that fragmented
    memory and start clean.
    """
    def __init__(self, reload_every=10):
        self.reload_every = reload_every
        self.count = 0
        self._load()

    def _load(self):
        from faster_whisper import WhisperModel
        print("[SETUP] Loading Whisper model...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")

    def transcribe(self, audio, initial_prompt):
        self.count += 1
        if self.count > self.reload_every:
            print(f"[SETUP] Reloading Whisper after {self.reload_every} transcriptions to clear CUDA fragmentation...")
            del self.model
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._load()
            self.count = 1
        return self.model.transcribe(audio, language="en", initial_prompt=initial_prompt)


def listen_and_transcribe(whisper_session, initial_prompt, mic_index, max_wait_override=None):
    from scipy.signal import resample

    audio, speech_started = record(mic_index, max_wait_override=max_wait_override)
    if len(audio) == 0 or not speech_started:
        # No real speech was detected in this window — skip Whisper
        # entirely. Feeding near-silent audio through Whisper alongside
        # an initial_prompt can cause it to hallucinate the prompt text
        # itself back as if it were transcribed speech, which is exactly
        # what was happening here before this check existed.
        return ""

    audio_mono = audio.mean(axis=1)
    num_samples = int(len(audio_mono) * WHISPER_SAMPLE_RATE / RECORD_SAMPLE_RATE)
    audio_16k = resample(audio_mono, num_samples).astype(np.float32)

    segments, info = whisper_session.transcribe(audio_16k, initial_prompt)
    text = " ".join(seg.text for seg in segments).strip().lower()
    text = apply_corrections(text)

    if text in HALLUCINATION_PHRASES:
        print(f"[LISTENING] discarded as hallucination: '{text}'")
        return ""

    print(f"[LISTENING] Heard: {text}")
    return text


def clean_for_speech(text):
    text = re.sub(r'^[\*\-\•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\n+', ' ', text)

    text = maths_to_speech(text)

    replacements = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': ', ',
        '\u2026': '...',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    leftover = re.findall(r'[^\x20-\x7E]', text)
    if leftover:
        print(f"[clean_for_speech] Unmapped symbols stripped: {set(leftover)} "
              f"in text: {text[:80]!r}")
    text = re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =====================================================================
# get_llm_answer() — now takes the RAG context alongside the question,
# and grounds the answer in lecture_text when it's available. If no
# lecture is loaded (or the RAG chunks come back empty), lecture_text
# is just an empty string and the system prompt below instructs the
# model to fall back on general knowledge and say so honestly —
# matching the edge-case behaviour Subash already validated.
# =====================================================================
def get_llm_answer(question, context):
    if context.get("not_covered_yet", False):
        slide_num = context.get("requested_slide")
        slide_ref = f"slide {slide_num}" if slide_num else "that slide"
        ans = f"We haven't covered {slide_ref} yet in today's lecture. Let's continue with our current topic first, and we will get to that soon."
        print(f"[THINKING] Answer (not covered yet): {ans}")
        return ans

    if not GROQ_API_KEY:
        print("[THINKING] ERROR: GROQ_API_KEY not set.")
        return "Sorry, I can't reach my brain right now."

    print("[THINKING] Sending question to Groq...")

    lecture_text = context.get("lecture_text", "")
    is_followup = context.get("is_followup", False)

    if is_followup:
        user_content = f"""A student didn't understand your previous explanation and asked you
to explain again. Do NOT repeat the same wording — explain it differently,
using a simpler analogy or a new example a 10th grade student would relate to.

Original question: {context.get('previous_question')}
Your previous explanation: {context.get('previous_answer')}

Relevant lecture material (may be empty if nothing matched):
{lecture_text}

Now explain it again, differently."""
    elif lecture_text:
        user_content = f"""Lecture material:
{lecture_text}

Student's question: {question}

Answer using the lecture material above. If it doesn't fully cover the
question, say so honestly rather than making things up."""
    else:
        user_content = f"""No lecture material is loaded right now. Answer the student's
question from general knowledge, and briefly mention that this isn't
from today's lecture material.

Student's question: {question}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly professor speaking out loud to a student in a classroom. "
                    "Answer in natural, spoken, conversational sentences, the way a teacher would "
                    "explain something to a student face-to-face, not a written document. "
                    "Teach and elaborate on the answer thoroughly — unpack each key point with context, "
                    "a brief relatable example, or why it matters. "
                    "Aim for approximately 120-150 words (about 45-60 seconds of spoken explanation) "
                    "so the student receives a complete, clear understanding. "
                    "Never use bullet points, numbered lists, markdown formatting, headers, or bold text. "
                    "Just plain flowing sentences, as if you were talking. "
                    "When explaining maths, never use math symbols like ^, √, ≠, ÷, π, ∫, or fractions "
                    "written as a/b — always write them out in plain spoken words instead "
                    "(e.g. 'x squared', 'the square root of 16', 'three over four', 'pi'), "
                    "exactly the way you'd say them out loud to a student. "
                    "Be warm and encouraging, like a good teacher."
                )
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 600,
    }

    answer = None
    try:
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                choice = response.json()["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    print("[THINKING] WARNING: answer was cut off by max_tokens limit")
                answer = choice["message"]["content"].strip()
                answer = clean_for_speech(answer)
                break
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                print(f"[THINKING] Groq connection attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
        if answer is None:
            raise last_error if last_error else Exception("Groq call failed after retries")
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
        print("[THINKING] Groq unreachable after retries — network issue")
        answer = "Sorry, I'm having trouble with my internet connection right now. Could you ask that again in a moment?"
    except Exception as e:
        print(f"[THINKING] Groq API error: {e}")
        answer = "Sorry, I had trouble reaching the answer service."

    print(f"[THINKING] Answer: {answer}")
    if not answer or not answer.strip():
        print("[THINKING] WARNING: answer came back empty, using fallback")
        answer = "Sorry, I need a moment to think about that one — could you ask it again, maybe a bit more specifically?"
    return answer


def synthesize_with_pauses(voice, text, syn_config, pause_ms=350):
    """
    Synthesizes text sentence-by-sentence and stitches the results
    together with a short silence gap between each sentence. Piper has
    no pause/SSML support, so this is the mechanical way to make
    multi-point answers sound like they're pausing between ideas
    instead of reading everything in one flat run.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    combined_frames = b''
    params = None
    for i, sentence in enumerate(sentences):
        part_path = SPEECH_WAV + f".part{i}.wav"
        with wave.open(part_path, "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config=syn_config)
        with wave.open(part_path, 'rb') as wf:
            p = wf.getparams()
            frames = wf.readframes(wf.getnframes())
        if params is None:
            params = p
        combined_frames += frames
        if i < len(sentences) - 1:
            silence_frames = int(params.framerate * pause_ms / 1000)
            combined_frames += b'\x00' * (silence_frames * params.sampwidth * params.nchannels)
        os.remove(part_path)

    with wave.open(SPEECH_WAV, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(combined_frames)


def pad_wav_silence(path, lead_ms=400, trail_ms=400):
    with wave.open(path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    frame_size = params.sampwidth * params.nchannels
    lead_frames = int(params.framerate * lead_ms / 1000)
    trail_frames = int(params.framerate * trail_ms / 1000)
    silence_lead = b'\x00' * (lead_frames * frame_size)
    silence_trail = b'\x00' * (trail_frames * frame_size)
    with wave.open(path, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(silence_lead + frames + silence_trail)


def speak_answer(voice, answer):
    print(f"[SPEAKING] {answer}")
    if not answer or not answer.strip():
        print("[SPEAKING] Skipping — empty text, nothing to speak.")
        return

    from piper import SynthesisConfig
    t0 = time.time()
    try:
        syn_config = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
        synthesize_with_pauses(voice, answer, syn_config, pause_ms=350)
        pad_wav_silence(SPEECH_WAV, lead_ms=400, trail_ms=400)
        t1 = time.time()
        print(f"[SPEAKING] synthesis took {t1 - t0:.2f}s")
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(SPEECH_WAV, winsound.SND_FILENAME)
            except Exception as pe:
                print(f"[SPEAKING] Windows audio playback note: {pe}")
        else:
            subprocess.run(["aplay", "-D", AUDIO_DEVICE, SPEECH_WAV], check=True)
        t2 = time.time()
        print(f"[SPEAKING] playback took {t2 - t1:.2f}s")
    except Exception as e:
        print(f"[SPEAKING] Error during TTS/playback: {e}")
        traceback.print_exc()


NEXT_SLIDE_PHRASES = [
    "next slide", "next", "move on", "continue", "no doubts", "no doubt",
    "i have no doubts", "go ahead", "proceed", "carry on", "that's clear",
    "its clear", "understood", "got it", "no questions", "no",
]


def is_next_slide_intent(text):
    t = text.lower().strip()
    if not t:
        return True
    return any(p in t for p in NEXT_SLIDE_PHRASES)


def extract_image_captions_if_image_only(slide_text):
    """
    Returns a list of caption strings if the slide contains ONLY image caption(s)
    (and optional slide heading, but NO real bullet text).
    Returns None if the slide contains regular bullet points or text.
    """
    lines = [l.strip() for l in slide_text.strip().split("\n") if l.strip()]
    if not lines:
        return None

    captions = []
    has_image = False
    has_text_bullets = False

    for l in lines:
        if l.startswith("#"):
            continue  # slide heading is allowed
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
    """
    Builds the (system_prompt, user_content) pair for slide explanation.
    Handles image-only caption slides (Feature 3) and elaborate explanation framing (Feature 4).
    """
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


def normalize_unicode_text(text):
    if not text:
        return ""
    text = (text.replace('\u2011', '-')
                .replace('\u2013', '-')
                .replace('\u2014', '-')
                .replace('\u2018', "'")
                .replace('\u2019', "'")
                .replace('\u201c', '"')
                .replace('\u201d', '"')
                .replace('\u00a0', ' '))
    return text.encode('ascii', 'ignore').decode('ascii')


def generate_slide_explanation(slide_text):
    if not GROQ_API_KEY:
        return "Sorry, I can't reach my brain right now."

    slide_text = normalize_unicode_text(slide_text)
    system_prompt, user_content = build_slide_explanation_prompt(slide_text)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 600,
    }
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        answer = normalize_unicode_text(clean_for_speech(answer))
        return answer
    except Exception as e:
        print(f"[TEACHING] Error generating slide explanation: {e}")
        return "Let's move on to the next point."


SLIDE_NUMBER_PATTERN = re.compile(r'slide\s*(?:number\s*|#\s*)?(\d+)', re.IGNORECASE)


def extract_requested_slide_number(text):
    match = SLIDE_NUMBER_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return None


CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_checkpoint.json")


def save_checkpoint(lecture_path, next_slide_index, slide_explanations, slide_qa_history=None):
    """
    Saves teaching progress to a small JSON file after each slide, so
    that if the process crashes (e.g. the CUDA OOM issue) and gets
    restarted, it can resume from where it left off instead of
    starting the whole lecture over from slide 1. This is deliberately
    a plain file, not a database — there's no query/scale need here,
    just "remember where we were" across a restart.
    """
    import json
    try:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({
                "lecture_path": lecture_path,
                "next_slide_index": next_slide_index,
                "slide_explanations": slide_explanations,
                "slide_qa_history": {str(k): v for k, v in (slide_qa_history or {}).items()},
            }, f)
    except Exception as e:
        print(f"[TEACHING] Could not save checkpoint: {e}")


def load_checkpoint(lecture_path):
    """
    Returns (next_slide_index, slide_explanations, slide_qa_history) if
    a checkpoint exists for this exact lecture file, otherwise (0, {}, {}).
    """
    import json
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, {}, {}
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        if data.get("lecture_path") != lecture_path:
            return 0, {}, {}
        slide_explanations = {int(k): v for k, v in data.get("slide_explanations", {}).items()}
        slide_qa_history = {int(k): [tuple(pair) for pair in v] for k, v in data.get("slide_qa_history", {}).items()}
        print(f"[TEACHING] Resuming from checkpoint: slide index {data.get('next_slide_index', 0)}.")
        return data.get("next_slide_index", 0), slide_explanations, slide_qa_history
    except Exception as e:
        print(f"[TEACHING] Could not read checkpoint ({e}), starting from the beginning.")
        return 0, {}, {}


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def generate_point_explanation(point_text):
    """
    Gives a short, simple explanation of ONE point that has already
    been read aloud verbatim — this is deliberately lighter/shorter
    than generate_slide_explanation(), since its job is just to
    clarify one line, not summarize the whole slide.
    """
    if not GROQ_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a professor. The student just heard ONE point from a lecture "
                "slide read aloud, word for word. Now give a short, simple, spoken "
                "explanation of what it means in plain everyday language — one to two "
                "sentences, optionally with a very brief example. Do not repeat the "
                "point itself, just explain it. No bullet points, no markdown."
            )},
            {"role": "user", "content": point_text},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return clean_for_speech(answer)
    except Exception as e:
        print(f"[TEACHING] Error generating point explanation: {e}")
        return ""


PREVIOUS_SLIDE_PHRASES = [
    "previous slide", "go back", "last slide", "slide before",
    "back to previous", "go to the previous slide", "before this slide",
]


def is_previous_slide_request(text):
    t = text.lower()
    return any(p in t for p in PREVIOUS_SLIDE_PHRASES)


def is_asking_about_past_questions(text):
    t = text.lower()
    return ("slide" in t) and ("ask" in t or "question" in t)


def answer_doubt_about_slide(slide, question):
    """
    Answers a doubt raised during the current slide using ONLY that
    slide's own content as context — not a semantic search across the
    whole deck. This keeps doubt answers relevant to what's actually
    on screen right now, instead of occasionally pulling in unrelated
    material from elsewhere in the lecture.
    """
    context = {
        "is_followup": False,
        "chunks": [],
        "previous_question": None,
        "previous_answer": None,
        "lecture_text": slide["text"],
    }
    return get_llm_answer(question, context)


def teach_class(whisper_session, piper_voice, initial_prompt, mic_index, lecture_path, slide_client=None):
    if slide_client is None:
        slide_client = SlideClient()

    slides = rag_engine.get_ordered_chunks()
    if not slides:
        print("[TEACHING] No lecture loaded, nothing to teach.")
        return

    # Tracks the exact wording spoken for each slide, so "repeat slide N"
    # replays the same explanation verbatim instead of Groq generating a
    # brand new (and possibly different, or wrong-slide) answer each time.
    start_index, slide_explanations, slide_qa_history = load_checkpoint(lecture_path)

    current_lecture_slide = 0
    max_slide_reached = 0
    if start_index > 0 and start_index <= len(slides):
        current_lecture_slide = slides[start_index - 1]["slide_number"]
        max_slide_reached = current_lecture_slide
        rag_engine.set_lecture_progress(current_lecture_slide, max_slide_reached)

    print(f"[TEACHING] Starting class, {len(slides)} slides queued.")
    for i, slide in enumerate(slides):
        if i < start_index:
            continue

        current_number = slide["slide_number"]
        print(f"[TEACHING] Slide {current_number} ({i+1}/{len(slides)})")

        # FEATURE 1: Timing — 'next' must fire BEFORE the robot starts narrating the new slide
        slide_client.next_slide()

        # Update main lecture position (only changed by 'next', never by doubt detours)
        current_lecture_slide = current_number
        max_slide_reached = max(max_slide_reached, current_number)
        rag_engine.set_lecture_progress(current_lecture_slide, max_slide_reached)

        explanation = generate_slide_explanation(slide["text"])
        speak_answer(piper_voice, explanation)

        slide_explanations[current_number] = explanation
        save_checkpoint(lecture_path, i + 1, slide_explanations, slide_qa_history)

        while True:
            speak_answer(piper_voice, "Any doubts?")
            response = listen_and_transcribe(
                whisper_session, initial_prompt, mic_index, max_wait_override=DOUBT_WAIT_SECONDS
            )

            if not response.strip():
                # Silence within the short wait window — move on without
                # narrating it, rather than announcing "no doubts heard".
                break

            if is_next_slide_intent(response):
                break

            if is_previous_slide_request(response):
                if i > 0:
                    prev_number = slides[i - 1]["slide_number"]
                    # Doubt detour to previous slide
                    slide_client.goto_slide(prev_number)
                    if prev_number in slide_explanations:
                        print(f"[TEACHING] Replaying previous slide {prev_number} verbatim.")
                        speak_answer(piper_voice, slide_explanations[prev_number])
                    else:
                        prev_slide = slides[i - 1]
                        explanation = generate_slide_explanation(prev_slide["text"])
                        slide_explanations[prev_number] = explanation
                        speak_answer(piper_voice, explanation)
                    # Return to current slide after detour
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
                    # On current slide: replay or explain without moving projector
                    if requested_slide in slide_explanations:
                        print(f"[TEACHING] Replaying stored explanation for slide {requested_slide} verbatim.")
                        speak_answer(piper_voice, slide_explanations[requested_slide])
                    else:
                        explanation = generate_slide_explanation(slide["text"])
                        slide_explanations[requested_slide] = explanation
                        speak_answer(piper_voice, explanation)
                elif requested_slide <= max_slide_reached:
                    # Detour to earlier covered slide: goto N -> explain -> goto current_lecture_slide
                    slide_client.goto_slide(requested_slide)
                    if requested_slide in slide_explanations:
                        print(f"[TEACHING] Replaying stored explanation for slide {requested_slide} verbatim.")
                        speak_answer(piper_voice, slide_explanations[requested_slide])
                    else:
                        target = next((s for s in slides if s["slide_number"] == requested_slide), None)
                        if target:
                            explanation = generate_slide_explanation(target["text"])
                            slide_explanations[requested_slide] = explanation
                            speak_answer(piper_voice, explanation)
                    # Return to current slide after detour
                    slide_client.goto_slide(current_lecture_slide)
                else:
                    # N > max_slide_reached: do NOT send goto command
                    target = next((s for s in slides if s["slide_number"] == requested_slide), None)
                    if target:
                        print(f"[TEACHING] Slide {requested_slide} requested but not covered yet (max reached: {max_slide_reached}).")
                        speak_answer(piper_voice, f"We haven't covered slide {requested_slide} yet. Let's continue with today's lecture first.")
                    else:
                        speak_answer(piper_voice, f"I don't have slide {requested_slide} in this lecture.")
                continue

            # General doubt — answered using only the current slide's own
            # content, so the answer stays relevant to what's on screen.
            answer = answer_doubt_about_slide(slide, response)
            slide_qa_history.setdefault(current_number, []).append((response, answer))
            rag_engine.add_to_history(response, answer, [])
            speak_answer(piper_voice, answer)
            save_checkpoint(lecture_path, i, slide_explanations, slide_qa_history)

    speak_answer(piper_voice, "That's the end of today's lecture. Let me know if you have any final questions.")
    clear_checkpoint()
    print("[TEACHING] Class complete.")


INBOX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox")


def find_latest_lecture_file():
    if not os.path.isdir(INBOX_FOLDER):
        return None
    candidates = [
        os.path.join(INBOX_FOLDER, f) for f in os.listdir(INBOX_FOLDER)
        if os.path.splitext(f)[1].lower() in (".pptx", ".pdf", ".docx")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def auto_detect_subject(slides):
    if not slides or not GROQ_API_KEY:
        return "general"
    sample_text = " ".join(s["text"] for s in slides[:2])[:800]
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Reply with ONLY one or two words naming the school subject this content belongs to (e.g. 'computer science', 'physics'). No punctuation, no explanation."},
            {"role": "user", "content": sample_text},
        ],
        "temperature": 0,
        "max_tokens": 10,
    }
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        subject = r.json()["choices"][0]["message"]["content"].strip().lower()
        if not subject:
            print("[SETUP] Auto-detect returned empty, using general.")
            return "general"
        print(f"[SETUP] Auto-detected subject: {subject}")
        return subject
    except Exception as e:
        print(f"[SETUP] Subject auto-detect failed ({e}), using general.")
        return "general"


def main():
    from piper import PiperVoice

    # --- mic setup (auto-detected, not hardcoded) ---
    mic_index = find_mic_device()

    # --- lecture file auto-loaded from the inbox folder (same folder app.py saves uploads into) ---
    lecture_path = find_latest_lecture_file()
    if lecture_path:
        try:
            n_chunks = rag_engine.load_lecture(lecture_path)
            print(f"[SETUP] Auto-loaded lecture: {os.path.basename(lecture_path)} — {n_chunks} chunks.")
        except Exception as e:
            print(f"[SETUP] Could not load lecture ({e}) — continuing without one.")
    else:
        print("[SETUP] No lecture file found in inbox — answers will use general knowledge only.")

    subject = auto_detect_subject(rag_engine.get_ordered_chunks())
    initial_prompt = build_initial_prompt(subject)
    print(f"[SETUP] Using vocabulary bias for: {subject}")

    # Whisper is wrapped in a session object that reloads itself every
    # 10 transcriptions to clear CUDA memory fragmentation — see
    # WhisperSession above for why this is necessary on long sessions.
    whisper_session = WhisperSession(reload_every=10)

    print("Loading Piper voice (once, staying resident for the session)...")
    piper_voice = PiperVoice.load(PIPER_MODEL)

    print("AI Professor orchestrator started. Ctrl+C to stop.")
    selected_id = None
    try:
        if rag_engine.has_lecture_loaded():
            teach_class(whisper_session, piper_voice, initial_prompt, mic_index, lecture_path)

        while True:
            # See "CAMERA / HAND-RAISE INTEGRATION POINT" above for how
            # to swap this stub for Agilan's real module later.
            triggered, selected_id = watch_for_hand_raise(previously_selected_id=selected_id)
            if not triggered:
                continue

            question = listen_and_transcribe(whisper_session, initial_prompt, mic_index)
            if not question:
                print("No question heard, going back to idle.")
                continue

            context = rag_engine.get_context_for_query(question)

            t0 = time.time()
            answer = get_llm_answer(question, context)
            print(f"[THINKING] Groq call took {time.time() - t0:.2f}s")

            rag_engine.add_to_history(question, answer, context["chunks"])

            speak_answer(piper_voice, answer)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
