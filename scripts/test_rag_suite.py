import os
import sys
import numpy as np

# Ensure scripts dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_engine

def run_tests():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE RAG ENGINE TEST SUITE FOR NVIDIA JETSON")
    print("=" * 70)
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {details}")

    # -------------------------------------------------------------
    # Test Suite 1: Unicode & Normalization Safety
    # -------------------------------------------------------------
    print("\n--- Suite 1: Unicode & Text Normalization ---")
    dirty_text = "Leadership \u2013 decision\u2011making is key. \u201cGreat leaders\u201d inspire\u2026 \u00a0\u2022 Act now!"
    cleaned = rag_engine.normalize_text(dirty_text)
    assert_test("Non-breaking hyphen \\u2011 normalized to -", "-" in cleaned and "\u2011" not in cleaned)
    assert_test("Smart quotes normalized to ASCII", '"' in cleaned and "\u201c" not in cleaned)
    assert_test("Non-breaking space normalized to space", "\u00a0" not in cleaned)
    assert_test("Bullet symbol \\u2022 stripped cleanly", "\u2022" not in cleaned)
    assert_test("Control characters stripped", rag_engine.normalize_text("Hello\x00World\x07!") == "HelloWorld!")

    # -------------------------------------------------------------
    # Test Suite 2: Generic Heading & Administrative Filter
    # -------------------------------------------------------------
    print("\n--- Suite 2: Generic Heading & Administrative Clutter Filter ---")
    assert_test("Detects 'Notes:' as generic heading", rag_engine._is_generic_heading("Notes:"))
    assert_test("Detects '### Notes:' with hash prefix as generic heading", rag_engine._is_generic_heading("### Notes:"))
    assert_test("Detects '## Speaker Notes' as generic heading", rag_engine._is_generic_heading("## Speaker Notes"))
    assert_test("Detects 'Slide 4' as generic heading", rag_engine._is_generic_heading("Slide 4"))
    assert_test("Allows valid concept heading 'Decision Making'", not rag_engine._is_generic_heading("Decision Making"))
    
    # Test markdown conversion when # Notes: is at the top
    md_with_notes = "# Notes:\nLeadership Principles\nIntegrity\nAccountability"
    h, pts = rag_engine._markdown_to_heading_and_points(md_with_notes)
    assert_test("Notes: suppressed as candidate heading when first", h == "" or h != "Notes:")
    assert_test("Notes: not leaked into body points", "Notes:" not in pts and "notes:" not in [p.lower() for p in pts])

    # Test markdown conversion when real heading is first and ### Notes: appears later
    md_subsequent_notes = "# Decision Making\nPoint 1\n### Notes:\nPoint 2\n### Accountability"
    h2, pts2 = rag_engine._markdown_to_heading_and_points(md_subsequent_notes)
    assert_test("Real heading preserved when ### Notes: is subsequent", h2 == "Decision Making")
    assert_test("### Notes: dropped from points when subsequent", not any("notes" in p.lower() for p in pts2))
    assert_test("Valid sub-heading ### Accountability kept as clean point", "Accountability" in pts2)

    # -------------------------------------------------------------
    # Test Suite 2B: Vision API Network Retry Loop Verification
    # -------------------------------------------------------------
    print("\n--- Suite 2B: Vision Captioning Retry Verification ---")
    from unittest.mock import patch, MagicMock
    import requests

    # Test that transient network errors retry up to 3 times before succeeding
    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Diagram showing architecture"}]}}]
    }

    with patch("requests.post") as mock_post:
        # First call raises ConnectionError, second succeeds
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection reset"),
            mock_resp_success
        ]
        # Temporarily ensure GEMINI_API_KEY is truthy for the test
        orig_key = rag_engine.GEMINI_API_KEY
        rag_engine.GEMINI_API_KEY = "test_key"
        try:
            caption = rag_engine._caption_image_bytes(b"dummy_bytes")
            assert_test("Retries on network exception and gets caption", caption == "Diagram showing architecture")
            assert_test("Made exactly 2 attempts before succeeding", mock_post.call_count == 2)
        finally:
            rag_engine.GEMINI_API_KEY = orig_key

    # Test that when all 3 attempts fail, it exhausts retries without premature exit
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Read timeout")
        orig_key = rag_engine.GEMINI_API_KEY
        rag_engine.GEMINI_API_KEY = "test_key"
        try:
            caption = rag_engine._caption_image_bytes(b"dummy_bytes")
            assert_test("Returns empty string after exhausting all 3 attempts", caption == "")
            assert_test("Made all 3 attempts on continuous failure", mock_post.call_count == 3)
        finally:
            rag_engine.GEMINI_API_KEY = orig_key


    # -------------------------------------------------------------
    # Test Suite 3: Vector Math & Edge-Case Safety
    # -------------------------------------------------------------
    print("\n--- Suite 3: Vector Similarity & Edge-Case Safety ---")
    rag_engine.clear_lecture()
    assert_test("Empty query returns empty list without error", rag_engine.retrieve_relevant_slides("") == [])
    assert_test("Whitespace query returns empty list", rag_engine.retrieve_relevant_slides("   ") == [])
    assert_test("Punctuation-only query handles safely", rag_engine.retrieve_relevant_slides("???!!!") == [])
    assert_test("top_k <= 0 returns empty list", rag_engine.retrieve_relevant_slides("test", top_k=0) == [])
    assert_test("Unloaded state has_lecture_loaded() is False", not rag_engine.has_lecture_loaded())

    # -------------------------------------------------------------
    # Test Suite 4: Real Document Loading & Schema Consistency
    # -------------------------------------------------------------
    print("\n--- Suite 4: Document Ingestion & Schema Consistency ---")
    sample_pptx = os.path.join(os.path.dirname(__file__), "..", "scripts", "inbox", "Physics_Fundamentals_Sample.pptx")
    if not os.path.exists(sample_pptx):
        sample_pptx = os.path.join(os.path.dirname(__file__), "..", "inbox", "Leadership-Qualities.pptx")

    n_chunks = rag_engine.load_lecture(sample_pptx, append=False)
    assert_test("Lecture loaded successfully", n_chunks > 0)
    assert_test("has_lecture_loaded() is True", rag_engine.has_lecture_loaded())
    
    loaded_chunks = rag_engine.get_ordered_chunks()
    required_keys = {"slide_number", "text", "heading", "points", "source_file"}
    all_keys_present = all(required_keys.issubset(c.keys()) for c in loaded_chunks)
    assert_test("All chunks have consistent schema (including 'heading')", all_keys_present)
    assert_test("No NaN in embeddings", not np.isnan(rag_engine._vectors).any())

    # -------------------------------------------------------------
    # Test Suite 5: Smart Retrieval & Exact Slide Navigation
    # -------------------------------------------------------------
    print("\n--- Suite 5: Smart Retrieval & Exact Slide Lookup ---")
    # Conceptual retrieval
    results = rag_engine.retrieve_relevant_slides("What is Newton's first law or basic concepts?", top_k=3)
    assert_test("Semantic retrieval returns top_k results", len(results) > 0)
    assert_test("Result carries heading and source_file", "heading" in results[0] and "source_file" in results[0])
    
    # Exact slide retrieval (digits and words)
    ctx_digit = rag_engine.get_context_for_query("Can you explain slide 2?")
    assert_test("Exact slide 2 lookup detected", ctx_digit["chunks"][0]["slide_number"] == 2)
    assert_test("Exact slide score is 1.0", ctx_digit["chunks"][0]["score"] == 1.0)

    ctx_word = rag_engine.get_context_for_query("Please explain slide three")
    assert_test("Word-form 'slide three' resolves to slide 3", ctx_word["chunks"][0]["slide_number"] == 3)

    # Out of range slide fallback
    ctx_out_of_range = rag_engine.get_context_for_query("Explain slide 999")
    assert_test("Out-of-range slide falls back gracefully to semantic search", len(ctx_out_of_range["chunks"]) > 0)

    # -------------------------------------------------------------
    # Test Suite 6: Follow-up Detection & Conversation Memory
    # -------------------------------------------------------------
    print("\n--- Suite 6: Multi-Turn Conversation & Follow-up Memory ---")
    rag_engine.add_to_history("What is force?", "Force is an interaction that changes motion.", ctx_digit["chunks"])
    assert_test("Follow-up 'explain again' recognized", rag_engine._is_followup("explain again"))
    assert_test("Follow-up 'i didn't understand' recognized", rag_engine._is_followup("i didn't understand"))
    assert_test("Follow-up 'not clear' recognized", rag_engine._is_followup("not clear sir"))
    
    followup_ctx = rag_engine.get_context_for_query("explain that again")
    assert_test("Follow-up context flags is_followup=True", followup_ctx["is_followup"] is True)
    assert_test("Follow-up context retains previous question", followup_ctx["previous_question"] == "What is force?")

    # -------------------------------------------------------------
    # Test Suite 7: Multi-Deck Ingestion & File Removal
    # -------------------------------------------------------------
    print("\n--- Suite 7: Multi-Deck Ingestion & File Removal ---")
    deck2_path = os.path.join(os.path.dirname(__file__), "..", "inbox", "Leadership-Qualities.pptx")
    if os.path.exists(deck2_path):
        initial_count = len(rag_engine.get_ordered_chunks())
        n2 = rag_engine.load_lecture(deck2_path, append=True)
        assert_test("Appended second lecture deck", len(rag_engine.get_loaded_files()) == 2)
        assert_test("Vector matrix stacked correctly", len(rag_engine.get_ordered_chunks()) == initial_count + n2)

        # Disambiguation: question mentions specific deck name
        disambig_ctx = rag_engine.get_context_for_query("Explain slide 1 of Physics")
        assert_test("Disambiguates slide 1 to Physics deck", "physics" in disambig_ctx["chunks"][0]["source_file"].lower())

        # Remove single file
        removed = rag_engine.remove_file(deck2_path)
        assert_test("Successfully removed second deck", removed is True)
        assert_test("Remaining chunk count restored", len(rag_engine.get_ordered_chunks()) == initial_count)
        assert_test("Loaded files list updated", len(rag_engine.get_loaded_files()) == 1)

    # -------------------------------------------------------------
    # Test Suite 8: Complete Cleanup & Memory Reset
    # -------------------------------------------------------------
    print("\n--- Suite 8: Complete Cleanup & Memory Reset ---")
    rag_engine.clear_lecture()
    assert_test("clear_lecture() empties _slide_texts", len(rag_engine.get_ordered_chunks()) == 0)
    assert_test("clear_lecture() wipes vectors", rag_engine._vectors is None)
    assert_test("clear_lecture() wipes conversation history", len(rag_engine._conversation_history) == 0)
    assert_test("has_lecture_loaded() returns False after clear", not rag_engine.has_lecture_loaded())

    rate = (passed / total * 100) if total > 0 else 0.0
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed}/{total} TESTS PASSED ({rate:.1f}% SUCCESS RATE)")
    print("=" * 70)
    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
