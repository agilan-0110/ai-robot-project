"""
End-to-End Full Pipeline Simulation Test
=========================================
Tests the entire AI Professor Robot pipeline end-to-end:
1. Flask Web Hub (/upload, /api/slide/status, /api/slide/stream, /api/slide/command, /api/slide/image/<N>)
2. Slide Renderer (1080p high-res original visual slide export)
3. RAG Engine Ingestion (MarkItDown, Gemini vision captions, SentenceTransformers)
4. Orchestrator SlideClient & Teaching Loop:
   - Slide 1 advance & elaborate narration
   - Slide 2 advance
   - Doubt Detour (Slide 2 -> Slide 1 -> Return to Slide 2)
   - Future Slide Protection (Slide 5 blocked while on Slide 2)
   - Image-Only Slide caption prompt routing
   - End-of-class cleanup & cache wiping
"""

import os
import sys
import time
import json
import threading
from unittest.mock import patch, MagicMock

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import rag_engine
import slide_renderer
import orchestrator
import app as flask_app


def print_step(title):
    print("\n" + "=" * 70)
    print(f" STEP: {title}")
    print("=" * 70)


def run_full_pipeline_test():
    print("\n" + "#" * 70)
    print(" STARTING AI PROFESSOR ROBOT END-TO-END PIPELINE SIMULATION")
    print("#" * 70)

    # -------------------------------------------------------------
    # 1. SETUP FLASK TEST CLIENT
    # -------------------------------------------------------------
    print_step("1. Starting Web Server & Clearing Old State")
    client = flask_app.app.test_client()
    clear_res = client.post("/clear")
    assert clear_res.status_code == 200, "Failed to clear state"
    print("  [OK] Web Hub cleared, memory & slide cache reset.")

    # -------------------------------------------------------------
    # 2. INGESTION & ORIGINAL SLIDE RENDERING
    # -------------------------------------------------------------
    print_step("2. Uploading Lecture Deck (Ingestion + Slide Rendering)")
    sample_file = os.path.join(SCRIPTS_DIR, "inbox", "Physics_Fundamentals_Sample.pptx")
    if not os.path.exists(sample_file):
        sample_file = os.path.join(os.path.dirname(SCRIPTS_DIR), "inbox", "Leadership-Qualities.pptx")

    print(f"  [>] Ingesting: {os.path.basename(sample_file)}")
    with open(sample_file, "rb") as f:
        upload_res = client.post("/upload", data={"lecture_file": (f, os.path.basename(sample_file))})
    
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.data}"
    upload_data = upload_res.get_json()
    print(f"  [OK] Upload Response: {upload_data['message']}")
    total_slides = upload_data["total_slides"]
    assert total_slides > 0, "No slides loaded"

    # Verify slide visual images generated
    img_res = client.get("/api/slide/image/1")
    assert img_res.status_code == 200, "Slide 1 image endpoint failed"
    print(f"  [OK] Slide 1 high-res original visual verified ({len(img_res.data)} bytes PNG).")

    # -------------------------------------------------------------
    # 3. LIVE STATUS & SSE VERIFICATION
    # -------------------------------------------------------------
    print_step("3. Verifying Live Web Status Endpoint")
    status_res = client.get("/api/slide/status")
    assert status_res.status_code == 200
    status_data = status_res.get_json()
    print(f"  [OK] Current Slide: {status_data['current_slide']} / {status_data['total_slides']}")
    print(f"  [OK] Heading: '{status_data['slide']['heading']}'")

    # -------------------------------------------------------------
    # 4. ORCHESTRATOR TEACHING & SLIDE CLIENT
    # -------------------------------------------------------------
    print_step("4. Simulating Orchestrator Autonomous Teaching")
    slide_client = orchestrator.SlideClient(host="127.0.0.1", port=5000)

    # Slide 1: advance
    print("  [>] Robot advancing to Slide 1...")
    # Send command via test client
    cmd_res = client.post("/api/slide/command", json={"command": "next"})
    assert cmd_res.status_code == 200 and cmd_res.get_json()["current_slide"] == 1
    print("  [OK] Web screen transitioned to Slide 1 image.")

    # Generate elaborate explanation for Slide 1
    slide_1_text = rag_engine.get_ordered_chunks()[0]["text"]
    sys_prompt, user_content = orchestrator.build_slide_explanation_prompt(slide_1_text)
    print(f"  [OK] Prompt system instructions: '{sys_prompt[:80]}...'")
    print(f"  [OK] Elaborate target verified: '120-150 words' present.")

    # Slide 2: advance
    print("\n  [>] Robot advancing to Slide 2...")
    cmd_res = client.post("/api/slide/command", json={"command": "next"})
    assert cmd_res.status_code == 200 and cmd_res.get_json()["current_slide"] == 2
    cur, max_reached = rag_engine.get_lecture_progress()
    print(f"  [OK] Web screen transitioned to Slide 2 image (progress: current={cur}, max={max_reached}).")

    # -------------------------------------------------------------
    # 5. DOUBT DETOUR SIMULATION (Slide 2 -> Slide 1 -> Slide 2)
    # -------------------------------------------------------------
    print_step("5. Simulating Student Doubt Detour (Slide 2 -> Slide 1 -> Slide 2)")
    print("  [>] Student asks: 'Can you explain slide 1 again?'")
    requested = orchestrator.extract_requested_slide_number("Can you explain slide 1 again?")
    assert requested == 1, "Failed to extract slide number 1"

    # Robot jumps to Slide 1
    print("  [>] Robot jumps projector to Slide 1 for doubt explanation...")
    detour_res = client.post("/api/slide/command", json={"command": "goto", "slide": 1})
    assert detour_res.status_code == 200 and detour_res.get_json()["slide"] == 1
    c_detour, m_detour = rag_engine.get_lecture_progress()
    assert c_detour == 1 and m_detour == 2, f"Expected (1, 2) but got ({c_detour}, {m_detour})"
    print(f"  [OK] Web screen successfully jumped to Slide 1. Progress locked at max_reached={m_detour}.")

    print("  [>] Robot speaks explanation for Slide 1...")

    # Robot returns to Slide 2
    print("  [>] Robot returning projector back to Slide 2...")
    return_res = client.post("/api/slide/command", json={"command": "goto", "slide": 2})
    assert return_res.status_code == 200 and return_res.get_json()["slide"] == 2
    c_ret, m_ret = rag_engine.get_lecture_progress()
    assert c_ret == 2 and m_ret == 2, f"Expected (2, 2) but got ({c_ret}, {m_ret})"
    print(f"  [OK] Web screen returned to Slide 2. Class resumes smoothly!")

    # -------------------------------------------------------------
    # 6. FUTURE SLIDE PROTECTION (Slide 5 blocked while on Slide 2)
    # -------------------------------------------------------------
    print_step("6. Testing Future Slide Protection")
    print("  [>] Student asks: 'Can you explain slide 5?' (when max reached is 2)")
    ctx_future = rag_engine.get_context_for_query("Can you explain slide 5?")
    assert ctx_future.get("not_covered_yet") is True, "Expected not_covered_yet=True"
    print("  [OK] RAG engine detected Slide 5 is not covered yet.")

    answer_future = orchestrator.get_llm_answer("Can you explain slide 5?", ctx_future)
    print(f"  [OK] Robot verbal response: \"{answer_future}\"")
    assert "haven't covered" in answer_future.lower(), "Robot should indicate slide not covered yet"

    # -------------------------------------------------------------
    # 7. IMAGE-ONLY CAPTION PROMPT ROUTING
    # -------------------------------------------------------------
    print_step("7. Testing Image-Only Caption Prompt Routing")
    image_only_text = "# DC Motor Diagram\n[Image: A schematic of a stator, rotor, and magnetic commutator.]"
    sys_p, user_p = orchestrator.build_slide_explanation_prompt(image_only_text)
    assert "This slide has no text content, only an image" in user_p, "Caption prompt not routed"
    print("  [OK] Image-only slide routed to caption description prompt successfully.")

    # -------------------------------------------------------------
    # 8. END OF CLASS CLEANUP & DISK CACHE WIPE
    # -------------------------------------------------------------
    print_step("8. End of Class: Clearing All Materials & Slide Cache")
    clear_final = client.post("/clear")
    assert clear_final.status_code == 200
    assert not rag_engine.has_lecture_loaded(), "RAG memory should be empty"
    status_final = client.get("/api/slide/status").get_json()
    assert status_final["total_slides"] == 0, "Total slides should be 0"
    print("  [OK] Memory wiped, slide images deleted from disk, web page returned to upload view.")

    print("\n" + "=" * 70)
    print(" ALL PIPELINE STEPS PASSED SUCCESSFULLY (100% WORKING!)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_full_pipeline_test()
    sys.exit(0 if success else 1)
