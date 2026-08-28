"""
Unit Tests for Autonomous Slide Control & Progress Tracking (Feature 1)
-----------------------------------------------------------------------
Verifies:
  1. Slide companion server receives 'next' and 'goto <N>' commands and returns 'done'.
  2. SlideClient successfully completes 'next' and 'goto <N>' command round-trips.
  3. SlideClient retries with backoff upon network/HTTP error or missing 'done' ack.
  4. Monotonic tracking of current_lecture_slide and max_slide_reached in rag_engine.
  5. Doubt-detour flow:
     - N <= max_slide_reached sends 'goto N', narrates, and sends 'goto <current_lecture_slide>'
     - N > max_slide_reached sends NO goto command and informs student it's not covered yet
     - current_lecture_slide and max_slide_reached are not modified by doubt detours
"""

import os
import sys
import time
import json
import threading
from unittest.mock import MagicMock, patch
from http.server import HTTPServer

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import rag_engine
import orchestrator
from http.server import BaseHTTPRequestHandler


class MockSlideServerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body.decode("utf-8")) if post_body else {}
        cmd = data.get("command")
        slide = data.get("slide")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {"status": "done", "command": cmd}
        if slide is not None:
            resp["slide"] = slide
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress test logs


def run_test(name, condition, details=""):
    if condition:
        print(f"  [PASS] {name}")
        return True
    else:
        print(f"  [FAIL] {name} - {details}")
        return False


def test_companion_server_and_client():
    print("\n--- Test Suite 1A: Slide Server & Client Round-Trip ---")
    all_passed = True
    test_port = 5098
    server = HTTPServer(("127.0.0.1", test_port), MockSlideServerHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        client = orchestrator.SlideClient(host="127.0.0.1", port=test_port, timeout=2.0)

        # 1. Test next command
        res_next = client.next_slide()
        all_passed &= run_test("Client 'next' command returns True on 'done' ack", res_next is True)

        # 2. Test goto command
        res_goto = client.goto_slide(5)
        all_passed &= run_test("Client 'goto 5' command returns True on 'done' ack", res_goto is True)

    finally:
        server.shutdown()
        server.server_close()

    return all_passed


def test_client_retry_behavior():
    print("\n--- Test Suite 1B: SlideClient Retry with Backoff ---")
    all_passed = True

    # Test failure retry: server unreachable (bad port)
    bad_client = orchestrator.SlideClient(host="127.0.0.1", port=59999, timeout=0.2, max_retries=3)

    t0 = time.time()
    res_fail = bad_client.next_slide()
    elapsed = time.time() - t0

    all_passed &= run_test("Client returns False when all 3 attempts fail", res_fail is False)
    all_passed &= run_test("Client performed retries with backoff delay (took >= 2s)", elapsed >= 1.9, f"took {elapsed:.2f}s")

    # Test transient failure: 1 failure then success
    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {"status": "done"}

    with patch("requests.post") as mock_post:
        import requests
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Temporary blip"),
            mock_resp_success
        ]
        t_client = orchestrator.SlideClient(host="127.0.0.1", port=5055, timeout=1.0, max_retries=3)
        res_recovered = t_client.goto_slide(3)
        all_passed &= run_test("Client recovers after transient network error", res_recovered is True)
        all_passed &= run_test("Client made exactly 2 attempts before success", mock_post.call_count == 2)

    return all_passed


def test_rag_engine_progress_state():
    print("\n--- Test Suite 1C: rag_engine Lecture Progress State ---")
    all_passed = True

    rag_engine.clear_lecture()
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("Initial progress is (0, 0)", c == 0 and m == 0)

    # Advance to slide 1
    rag_engine.set_lecture_progress(1)
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("Slide 1 progress is (1, 1)", c == 1 and m == 1)

    # Advance to slide 3
    rag_engine.set_lecture_progress(3)
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("Slide 3 progress is (3, 3)", c == 3 and m == 3)

    # Detour progress does not reduce max_slide_reached
    rag_engine.set_lecture_progress(2, max_slide=3)
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("max_slide_reached does not decrease on detour", m == 3)

    rag_engine.clear_lecture()
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("clear_lecture() resets progress to (0, 0)", c == 0 and m == 0)

    return all_passed


def test_doubt_detour_flow():
    print("\n--- Test Suite 1D: teach_class Doubt Detour Flow ---")
    all_passed = True

    # Setup mock slides in rag_engine
    mock_slides = [
        {"slide_number": 1, "text": "Slide 1 Content: Introduction to Physics", "heading": "Intro", "points": ["Intro point"]},
        {"slide_number": 2, "text": "Slide 2 Content: Newton's Laws", "heading": "Laws", "points": ["First law", "Second law"]},
        {"slide_number": 3, "text": "Slide 3 Content: Work and Energy", "heading": "Energy", "points": ["Work", "Energy"]},
        {"slide_number": 4, "text": "Slide 4 Content: Thermodynamics", "heading": "Thermo", "points": ["Heat"]},
    ]

    # Mock slide client to record sent commands
    sent_commands = []
    class MockSlideClient:
        def send_command(self, cmd, slide_number=None):
            sent_commands.append((cmd, slide_number))
            return True
        def next_slide(self):
            return self.send_command("next")
        def goto_slide(self, slide_number):
            return self.send_command("goto", slide_number)

    mock_client = MockSlideClient()

    # Mock piper_voice and whisper_session
    mock_voice = MagicMock()
    mock_whisper = MagicMock()

    # Mock responses for teach_class:
    # On slide 1: doubt "explain slide 1 again" -> stays on slide 1
    # On slide 2: doubt "can you explain slide 1?" -> detour: goto 1, explain, goto 2
    #             doubt "what about slide 4?" -> future slide: NO goto, says not covered yet
    #             "next slide" -> moves on
    # On slide 3: "next slide" -> moves on
    # On slide 4: "no doubts" -> ends
    doubt_responses = [
        "",  # slide 1: silence -> next slide
        "can you explain slide 1?",  # slide 2: detour to covered slide 1
        "what about slide 4?",       # slide 2: request for future slide 4
        "next slide",                # slide 2: proceed
        "",                          # slide 3: silence -> next
        "no doubts",                 # slide 4: finish
    ]
    resp_idx = 0
    def mock_listen(*args, **kwargs):
        nonlocal resp_idx
        if resp_idx < len(doubt_responses):
            val = doubt_responses[resp_idx]
            resp_idx += 1
            return val
        return ""

    with patch("rag_engine.get_ordered_chunks", return_value=mock_slides), \
         patch("orchestrator.generate_slide_explanation", return_value="Test explanation"), \
         patch("orchestrator.speak_answer") as mock_speak, \
         patch("orchestrator.listen_and_transcribe", side_effect=mock_listen), \
         patch("orchestrator.load_checkpoint", return_value=(0, {}, {})), \
         patch("orchestrator.save_checkpoint"):

        orchestrator.teach_class(
            whisper_session=mock_whisper,
            piper_voice=mock_voice,
            initial_prompt="physics",
            mic_index=0,
            lecture_path="dummy.pptx",
            slide_client=mock_client
        )

    # Analyze commands executed:
    # Expected sequence:
    # Slide 1 start: ('next', None)
    # Slide 2 start: ('next', None)
    # Slide 2 doubt detour to 1: ('goto', 1) followed by ('goto', 2)
    # Slide 2 doubt future 4: NO goto commands!
    # Slide 3 start: ('next', None)
    # Slide 4 start: ('next', None)
    all_passed &= run_test(
        "First command sent is 'next' before slide 1 narration",
        len(sent_commands) >= 1 and sent_commands[0] == ("next", None)
    )

    detour_goto_pairs = [c for c in sent_commands if c[0] == "goto"]
    all_passed &= run_test(
        "Detour sent 'goto 1' then 'goto 2' to return to current position",
        detour_goto_pairs == [("goto", 1), ("goto", 2)]
    )

    no_goto_4 = not any(c == ("goto", 4) for c in sent_commands)
    all_passed &= run_test("Future slide 4 was NOT sent via 'goto'", no_goto_4)

    # Check spoken response for future slide
    spoken_texts = [call[0][1] for call in mock_speak.call_args_list]
    future_slide_warned = any("haven't covered slide 4 yet" in text.lower() for text in spoken_texts)
    all_passed &= run_test(
        "Robot spoke warning that slide 4 has not been covered yet",
        future_slide_warned
    )

    # Ensure max_slide_reached reached 4 at the end of class
    _, final_max = rag_engine.get_lecture_progress()
    all_passed &= run_test("Final max_slide_reached reached 4", final_max == 4)

    return all_passed


def test_not_covered_yet_filtering():
    print("\n--- Test Suite 2: Feature 2 Not Covered Yet Filtering ---")
    all_passed = True

    rag_engine.clear_lecture()

    # Create 5 mock slide chunks with simple 4-dimensional normalized vectors
    import numpy as np
    mock_chunks = [
        {"slide_number": 1, "text": "Newton's First Law of Motion", "heading": "First Law", "points": ["Inertia"], "source_file": "physics.pptx"},
        {"slide_number": 2, "text": "Newton's Second Law: F = ma", "heading": "Second Law", "points": ["Acceleration"], "source_file": "physics.pptx"},
        {"slide_number": 3, "text": "Newton's Third Law: Action and Reaction", "heading": "Third Law", "points": ["Pairs"], "source_file": "physics.pptx"},
        {"slide_number": 4, "text": "Gravitational Field and Universal Gravitation", "heading": "Gravity", "points": ["G constant"], "source_file": "physics.pptx"},
        {"slide_number": 5, "text": "Electromagnetism and Maxwell Equations", "heading": "EM Waves", "points": ["Fields"], "source_file": "physics.pptx"},
    ]
    rag_engine._slide_texts = mock_chunks
    # 5 orthogonal or distinct vectors
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    rag_engine._vectors = vectors

    # Mock _get_model to return a dummy vector on encode
    mock_model = MagicMock()
    # Query vector closest to slide 4 [0.0, 0.0, 1.0, 0.0]
    mock_model.encode.return_value = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    with patch.object(rag_engine, "_get_model", return_value=mock_model):
        # 1. When max_slide_reached = 2 (only slides 1 and 2 covered)
        rag_engine.set_max_slide_reached(2)

        # Exact lookup for covered slide 1
        ctx_1 = rag_engine.get_context_for_query("Can you explain slide 1?")
        all_passed &= run_test(
            "Exact lookup for covered slide 1 succeeds",
            not ctx_1.get("not_covered_yet", False) and len(ctx_1["chunks"]) == 1 and ctx_1["chunks"][0]["slide_number"] == 1
        )

        # Exact lookup for covered slide 2
        ctx_2 = rag_engine.get_context_for_query("Can you explain slide 2?")
        all_passed &= run_test(
            "Exact lookup for covered slide 2 succeeds",
            not ctx_2.get("not_covered_yet", False) and len(ctx_2["chunks"]) == 1 and ctx_2["chunks"][0]["slide_number"] == 2
        )

        # Exact lookup for uncovered future slide 4
        ctx_4 = rag_engine.get_context_for_query("Can you explain slide 4?")
        all_passed &= run_test(
            "Exact lookup for uncovered slide 4 returns not_covered_yet=True and empty chunks",
            ctx_4.get("not_covered_yet", False) is True and len(ctx_4["chunks"]) == 0 and ctx_4.get("requested_slide") == 4
        )

        # Test get_llm_answer with not_covered_yet context
        ans_4 = orchestrator.get_llm_answer("Can you explain slide 4?", ctx_4)
        all_passed &= run_test(
            "get_llm_answer returns clear not covered message for slide 4",
            "haven't covered slide 4 yet" in ans_4.lower()
        )

        # Semantic search filtering: query matches slide 4 best, but slide 4 > max_slide_reached (2)
        # Therefore semantic search must ONLY return slides 1 or 2, never slide 4 or 5
        semantic_results = rag_engine.retrieve_relevant_slides("gravity and fields", top_k=3)
        all_passed &= run_test(
            "Semantic search returns results",
            len(semantic_results) > 0
        )
        all_under_or_equal_2 = all(r["slide_number"] <= 2 for r in semantic_results)
        all_passed &= run_test(
            "Semantic search candidates are strictly <= max_slide_reached (<= 2)",
            all_under_or_equal_2
        )

        # Non-existent slide 999 falls back to semantic search, which is also filtered to <= 2
        ctx_999 = rag_engine.get_context_for_query("Explain slide 999")
        fallback_chunks = ctx_999["chunks"]
        all_passed &= run_test(
            "Non-existent slide 999 fallback respects max_slide_reached filter",
            len(fallback_chunks) > 0 and all(c["slide_number"] <= 2 for c in fallback_chunks)
        )

        # 2. When max_slide_reached = 0 (unconstrained / testing mode)
        rag_engine.set_max_slide_reached(0)
        ctx_unconstrained = rag_engine.get_context_for_query("Can you explain slide 4?")
        all_passed &= run_test(
            "Unconstrained mode (max_slide_reached=0) permits accessing slide 4",
            not ctx_unconstrained.get("not_covered_yet", False) and len(ctx_unconstrained["chunks"]) == 1 and ctx_unconstrained["chunks"][0]["slide_number"] == 4
        )

    rag_engine.clear_lecture()
    return all_passed


def test_image_only_and_elaborate_prompts():
    print("\n--- Test Suite 3: Features 3 & 4 (Image-Only Captions & Elaborate Explanations) ---")
    all_passed = True

    # 1. Test image-only detection logic
    slide_image_only_bare = "[Image: A schematic of a DC electric motor with magnetic poles and rotor.]"
    caps1 = orchestrator.extract_image_captions_if_image_only(slide_image_only_bare)
    all_passed &= run_test("Bare [Image: ...] slide is detected as image-only", caps1 is not None and len(caps1) == 1)
    all_passed &= run_test("Caption text extracted correctly", caps1[0] == "A schematic of a DC electric motor with magnetic poles and rotor.")

    slide_image_with_heading = "# DC Motor Working Principle\n[Image: A diagram showing Lorentz force on a current loop.]"
    caps2 = orchestrator.extract_image_captions_if_image_only(slide_image_with_heading)
    all_passed &= run_test("Heading + [Image: ...] slide is detected as image-only", caps2 is not None and len(caps2) == 1)

    slide_with_bullets_and_image = "# DC Motor\n- Converts electrical energy to mechanical\n[Image: Motor diagram]"
    caps3 = orchestrator.extract_image_captions_if_image_only(slide_with_bullets_and_image)
    all_passed &= run_test("Slide with text bullets and image is NOT detected as image-only", caps3 is None)

    slide_text_only = "# Newton's Law\nEvery action has an equal and opposite reaction."
    caps4 = orchestrator.extract_image_captions_if_image_only(slide_text_only)
    all_passed &= run_test("Text-only slide is NOT detected as image-only", caps4 is None)

    # 2. Test prompt construction for image-only slide (Feature 3)
    sys_prompt_img, user_prompt_img = orchestrator.build_slide_explanation_prompt(slide_image_with_heading)
    all_passed &= run_test(
        "Image-only prompt contains 'This slide has no text content, only an image'",
        "This slide has no text content, only an image" in user_prompt_img
    )
    all_passed &= run_test(
        "Image-only prompt contains image caption description",
        "A diagram showing Lorentz force on a current loop" in user_prompt_img
    )
    all_passed &= run_test(
        "Image-only prompt contains slide topic heading",
        "Slide Topic: DC Motor Working Principle" in user_prompt_img
    )

    # 3. Test elaborate explanation constraints (Feature 4)
    all_passed &= run_test(
        "System prompt contains instruction to 'Teach and elaborate on'",
        "Teach and elaborate on" in sys_prompt_img
    )
    all_passed &= run_test(
        "System prompt sets target of '120-150 words'",
        "120-150 words" in sys_prompt_img
    )
    all_passed &= run_test(
        "System prompt references '45-60 seconds'",
        "45-60 seconds" in sys_prompt_img
    )
    all_passed &= run_test(
        "System prompt prohibits markdown/bullet points",
        "Never use bullet points" in sys_prompt_img
    )

    # 4. Test generate_slide_explanation payload and max_tokens
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "This diagram illustrates the working of an electric motor where current flows..."}}]
        }
        mock_post.return_value = mock_resp

        orig_key = orchestrator.GROQ_API_KEY
        orchestrator.GROQ_API_KEY = "dummy_key"
        try:
            explanation = orchestrator.generate_slide_explanation(slide_image_with_heading)
            all_passed &= run_test("generate_slide_explanation calls API and returns cleaned speech", len(explanation) > 0)

            # Inspect payload sent to Groq
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            all_passed &= run_test("Payload max_tokens is 600", payload["max_tokens"] == 600)
            all_passed &= run_test("Payload user content contains image caption path", "This slide has no text content, only an image" in payload["messages"][1]["content"])
            all_passed &= run_test("Payload system message has 120-150 words elaboration target", "120-150 words" in payload["messages"][0]["content"])
        finally:
            orchestrator.GROQ_API_KEY = orig_key

    return all_passed


def test_flask_browser_live_viewer():
    print("\n--- Test Suite 4: Flask Browser Live Viewer Endpoints ---")
    all_passed = True

    import app as flask_app
    client = flask_app.app.test_client()

    # Setup dummy slides in rag_engine
    rag_engine.clear_lecture()
    rag_engine._slide_texts = [
        {"slide_number": 1, "text": "Slide 1: Intro", "heading": "Intro", "points": ["Welcome"], "source_file": "lecture.pptx"},
        {"slide_number": 2, "text": "Slide 2: Physics", "heading": "Physics", "points": ["Force", "Motion"], "source_file": "lecture.pptx"},
    ]
    rag_engine.set_lecture_progress(1)

    # 1. Test GET /presentation
    res_page = client.get("/presentation")
    all_passed &= run_test("GET /presentation returns 200 OK", res_page.status_code == 200)
    all_passed &= run_test("GET /presentation contains live sync script", b"EventSource" in res_page.data)

    # 2. Test GET /api/slide/status
    res_status = client.get("/api/slide/status")
    all_passed &= run_test("GET /api/slide/status returns 200 OK", res_status.status_code == 200)
    status_data = res_status.get_json()
    all_passed &= run_test("Status data reflects slide 1", status_data["current_slide"] == 1 and status_data["total_slides"] == 2)
    all_passed &= run_test("Status slide heading is 'Intro'", status_data["slide"]["heading"] == "Intro")

    # 3. Test POST /api/slide/command 'next'
    res_cmd_next = client.post("/api/slide/command", json={"command": "next"})
    all_passed &= run_test("POST /api/slide/command 'next' returns 200 OK", res_cmd_next.status_code == 200)
    cmd_next_data = res_cmd_next.get_json()
    all_passed &= run_test("Next command response has status 'done' and advances to slide 2", cmd_next_data["status"] == "done" and cmd_next_data["current_slide"] == 2)
    c, m = rag_engine.get_lecture_progress()
    all_passed &= run_test("rag_engine updated progress to slide 2", c == 2 and m == 2)

    # 4. Test POST /api/slide/command 'goto' (detour to slide 1)
    res_cmd_goto = client.post("/api/slide/command", json={"command": "goto", "slide": 1})
    all_passed &= run_test("POST /api/slide/command 'goto' returns 200 OK", res_cmd_goto.status_code == 200)
    c_detour, m_detour = rag_engine.get_lecture_progress()
    all_passed &= run_test("Detour goto sets current_slide=1 while preserving max_slide_reached=2", c_detour == 1 and m_detour == 2)

    rag_engine.clear_lecture()
    return all_passed


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUTONOMOUS SLIDES & RAG TEST SUITE")
    print("=" * 70)

    t1 = test_companion_server_and_client()
    t2 = test_client_retry_behavior()
    t3 = test_rag_engine_progress_state()
    t4 = test_doubt_detour_flow()
    t5 = test_not_covered_yet_filtering()
    t6 = test_image_only_and_elaborate_prompts()
    t7 = test_flask_browser_live_viewer()

    print("\n" + "=" * 70)
    if t1 and t2 and t3 and t4 and t5 and t6 and t7:
        print("ALL TESTS PASSED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        print("=" * 70)
        sys.exit(1)



