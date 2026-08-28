import os
import json
import time
from flask import Flask, request, render_template_string, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import rag_engine
import slide_renderer

app = Flask(__name__)

INBOX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox")
os.makedirs(INBOX_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pptx", ".pdf", ".docx"}

UNIFIED_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Professor Robot - Classroom Presentation & Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background: #000000;
            color: #f8fafc;
            font-family: 'Outfit', sans-serif;
        }

        /* ——— 1. UPLOAD VIEW (STATE: NO LECTURE LOADED) ——— */
        #uploadView {
            display: flex;
            width: 100vw;
            height: 100vh;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background: radial-gradient(circle at top right, #1e1b4b, #090d16 65%);
        }
        .upload-card {
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(71, 85, 105, 0.6);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 540px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }
        .upload-card h1 {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .upload-card p.subtitle {
            color: #94a3b8;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            margin-bottom: 28px;
        }
        .dropzone {
            border: 2px dashed #475569;
            border-radius: 14px;
            padding: 36px 20px;
            background: rgba(15, 23, 42, 0.6);
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 20px;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: #38bdf8;
            background: rgba(56, 189, 248, 0.08);
            transform: scale(1.01);
        }
        .dropzone-icon {
            font-size: 2.8rem;
            margin-bottom: 12px;
            display: block;
        }
        .dropzone-text {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 4px;
        }
        .dropzone-hint {
            font-size: 0.82rem;
            color: #64748b;
            font-family: 'Inter', sans-serif;
        }
        input[type="file"] { display: none; }
        .btn-upload {
            width: 100%;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            font-family: 'Outfit', sans-serif;
        }
        .btn-upload:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
        }
        .btn-upload:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .upload-status {
            margin-top: 16px;
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif;
            color: #38bdf8;
            min-height: 22px;
        }

        /* ——— 2. PRESENTATION VIEW (ORIGINAL VISUAL SLIDES) ——— */
        #presentationView {
            display: none;
            position: relative;
            width: 100vw;
            height: 100vh;
            background: #000000;
            overflow: hidden;
        }

        /* Fullscreen Slide Image Viewport */
        .slide-viewport {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #000000;
            z-index: 10;
        }
        .slide-visual-img {
            max-width: 100vw;
            max-height: 100vh;
            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: opacity 0.25s ease, transform 0.25s ease;
            user-select: none;
        }
        .slide-visual-img.fading {
            opacity: 0;
            transform: scale(0.99);
        }

        /* Floating Top Bar (Discreet & Projector Optimized) */
        .status-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 54px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 28px;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 0, 0) 100%);
            z-index: 100;
            opacity: 0.85;
            transition: opacity 0.3s ease;
        }
        .status-bar:hover {
            opacity: 1;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(10px);
        }
        .status-left {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.5);
            color: #34d399;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }
        .deck-title {
            color: #cbd5e1;
            font-size: 0.9rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        }
        .status-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .slide-counter {
            font-size: 0.95rem;
            font-weight: 700;
            color: #38bdf8;
            background: rgba(15, 23, 42, 0.7);
            padding: 4px 14px;
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .btn-ctrl {
            background: rgba(30, 41, 59, 0.7);
            color: #cbd5e1;
            border: 1px solid #475569;
            padding: 5px 12px;
            border-radius: 7px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: 'Inter', sans-serif;
        }
        .btn-ctrl:hover {
            background: #334155;
            color: white;
        }

        /* Bottom Progress Bar */
        .progress-bar-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: rgba(0, 0, 0, 0.5);
            z-index: 100;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            width: 0%;
            transition: width 0.35s ease;
        }

        /* ——— 3. SLIDE-OVER MANAGEMENT DRAWER ——— */
        .drawer-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 200;
        }
        .drawer {
            position: fixed;
            top: 0; right: -420px;
            width: 400px;
            height: 100vh;
            background: #1e293b;
            border-left: 1px solid #334155;
            padding: 30px 24px;
            box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
            z-index: 201;
            transition: right 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
        }
        .drawer.open { right: 0; }
        .drawer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #334155;
        }
        .drawer-header h2 { font-size: 1.3rem; font-weight: 700; color: #f8fafc; }
        .btn-close {
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 1.5rem;
            cursor: pointer;
        }
        .drawer-body { flex: 1; overflow-y: auto; }
        .loaded-section h3 {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }
        .file-pill {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.88rem;
            color: #e2e8f0;
            border: 1px solid #334155;
        }
        .btn-pill-remove {
            background: #ef4444;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
        }
        .drawer-upload-box {
            margin-top: 20px;
            border-top: 1px solid #334155;
            padding-top: 20px;
        }
        .drawer-upload-box h3 {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
        }
        .btn-clear-class {
            margin-top: auto;
            width: 100%;
            background: #dc2626;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .btn-clear-class:hover { background: #b91c1c; }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.98); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>
    <!-- 1. UPLOAD VIEW (Default if no lecture loaded) -->
    <div id="uploadView">
        <div class="upload-card">
            <h1>AI Professor Robot</h1>
            <p class="subtitle">Autonomous Classroom Teaching & Live Presentation</p>

            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <span class="dropzone-icon">📁</span>
                <div class="dropzone-text" id="dropzoneText">Click or Drag Lecture File Here</div>
                <div class="dropzone-hint">PowerPoint (.pptx) or PDF (.pdf)</div>
                <input type="file" id="fileInput" accept=".pptx,.pdf,.docx" onchange="handleFileSelected(this)">
            </div>

            <button class="btn-upload" id="btnUpload" onclick="uploadSelectedFile()" disabled>
                🚀 Load & Start Live Presentation
            </button>
            <div class="upload-status" id="uploadStatus"></div>
        </div>
    </div>

    <!-- 2. PRESENTATION VIEW (ORIGINAL VISUAL SLIDES) -->
    <div id="presentationView">
        <div class="status-bar">
            <div class="status-left">
                <div class="live-badge">
                    <div class="live-dot"></div>
                    <span>LIVE AUTONOMOUS SYNC</span>
                </div>
                <span class="deck-title" id="deckTitle">Lecture Material</span>
            </div>
            <div class="status-right">
                <div class="slide-counter" id="slideCounter">Slide 1 / 1</div>
                <button class="btn-ctrl" onclick="toggleDrawer()">⚙️ Manage Decks</button>
                <button class="btn-ctrl" onclick="toggleFullScreen()">⛶ Full Screen (F11)</button>
            </div>
        </div>

        <div class="slide-viewport">
            <img id="slideVisualImg" class="slide-visual-img" src="" alt="Slide Visual">
        </div>

        <div class="progress-bar-container">
            <div class="progress-bar-fill" id="progressBar"></div>
        </div>
    </div>

    <!-- 3. SLIDE-OVER MANAGEMENT DRAWER -->
    <div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
    <div class="drawer" id="drawer">
        <div class="drawer-header">
            <h2>Lecture Management</h2>
            <button class="btn-close" onclick="toggleDrawer()">&times;</button>
        </div>
        <div class="drawer-body">
            <div class="loaded-section">
                <h3>Loaded Files</h3>
                <div id="loadedFilesList"></div>
            </div>

            <div class="drawer-upload-box">
                <h3>Append Another Lecture</h3>
                <input type="file" id="drawerFileInput" accept=".pptx,.pdf,.docx" style="display:block; margin-bottom:10px; color:#cbd5e1; font-size:0.85rem;" onchange="uploadFromDrawer(this)">
            </div>
        </div>
        <button class="btn-clear-class" onclick="clearClass()">🗑️ Clear All & End Class</button>
    </div>

    <script>
        let currentSlideNum = -1;
        let selectedFile = null;
        let isPresenting = false;

        // ——— File Drag and Drop / Selection ———
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }, false);
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }, false);
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                selectedFile = e.dataTransfer.files[0];
                onFileChosen();
            }
        });

        function handleFileSelected(input) {
            if (input.files.length > 0) {
                selectedFile = input.files[0];
                onFileChosen();
            }
        }

        function onFileChosen() {
            document.getElementById('dropzoneText').innerText = `Selected: ${selectedFile.name}`;
            document.getElementById('btnUpload').disabled = false;
        }

        // ——— Upload Action ———
        async function uploadSelectedFile() {
            if (!selectedFile) return;
            const btn = document.getElementById('btnUpload');
            const status = document.getElementById('uploadStatus');
            btn.disabled = true;
            btn.innerText = '⏳ Rendering Original Slides & Preparing Brain...';
            status.innerText = 'Converting slides to high-res visuals and indexing...';

            const formData = new FormData();
            formData.append('lecture_file', selectedFile);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    status.innerText = 'Success! Starting live presentation...';
                    setTimeout(pollStatus, 400);
                } else {
                    status.innerText = `Error: ${data.message || 'Failed to upload'}`;
                    btn.disabled = false;
                    btn.innerText = '🚀 Load & Start Live Presentation';
                }
            } catch (e) {
                status.innerText = `Network error: ${e.message}`;
                btn.disabled = false;
                btn.innerText = '🚀 Load & Start Live Presentation';
            }
        }

        async function uploadFromDrawer(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const formData = new FormData();
            formData.append('lecture_file', file);
            try {
                await fetch('/upload', { method: 'POST', body: formData });
                input.value = '';
                pollStatus();
            } catch (e) {
                alert(`Upload failed: ${e.message}`);
            }
        }

        // ——— View Switcher & Slide Rendering ———
        function showPresentationView() {
            document.getElementById('uploadView').style.display = 'none';
            document.getElementById('presentationView').style.display = 'block';
            isPresenting = true;
        }

        function showUploadView() {
            document.getElementById('presentationView').style.display = 'none';
            document.getElementById('uploadView').style.display = 'flex';
            document.getElementById('btnUpload').disabled = true;
            document.getElementById('btnUpload').innerText = '🚀 Load & Start Live Presentation';
            document.getElementById('dropzoneText').innerText = 'Click or Drag Lecture File Here';
            document.getElementById('uploadStatus').innerText = '';
            selectedFile = null;
            isPresenting = false;
            currentSlideNum = -1;
        }

        function updateSlideUI(data) {
            if (!data || data.total_slides === 0) {
                if (isPresenting) showUploadView();
                return;
            }

            if (!isPresenting) showPresentationView();

            const cur = data.current_slide || 1;
            const total = data.total_slides || 1;

            // Render drawer file pills
            if (data.loaded_files) {
                const list = document.getElementById('loadedFilesList');
                list.innerHTML = '';
                data.loaded_files.forEach((f, idx) => {
                    const pill = document.createElement('div');
                    pill.className = 'file-pill';
                    pill.innerHTML = `<span>📄 ${f}</span><button class="btn-pill-remove" onclick="removeFile(${idx})">Remove</button>`;
                    list.appendChild(pill);
                });
            }

            if (cur === currentSlideNum) return;
            currentSlideNum = cur;

            // Update title & counter
            if (data.slide && data.slide.source_file) {
                document.getElementById('deckTitle').innerText = data.slide.source_file;
            }
            document.getElementById('slideCounter').innerText = `Slide ${cur} / ${total}`;
            const pct = Math.min(100, Math.max(0, (cur / total) * 100));
            document.getElementById('progressBar').style.width = `${pct}%`;

            // Transition Slide Visual Image
            const img = document.getElementById('slideVisualImg');
            img.classList.add('fading');
            setTimeout(() => {
                img.src = `/api/slide/image/${cur}?t=${Date.now()}`;
                img.onload = () => { img.classList.remove('fading'); };
            }, 100);
        }

        // ——— SSE Stream & Polling ———
        function startSSE() {
            const evtSource = new EventSource('/api/slide/stream');
            evtSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    updateSlideUI(data);
                } catch (e) {
                    console.error('Error parsing SSE event:', e);
                }
            };
            evtSource.onerror = function() {
                pollStatus();
            };
        }

        async function pollStatus() {
            try {
                const res = await fetch('/api/slide/status');
                if (res.ok) {
                    const data = await res.json();
                    updateSlideUI(data);
                }
            } catch (e) {
                console.warn('Status poll error:', e);
            }
        }

        // ——— Drawer & File Management ———
        function toggleDrawer() {
            const drawer = document.getElementById('drawer');
            const overlay = document.getElementById('drawerOverlay');
            if (drawer.classList.contains('open')) {
                drawer.classList.remove('open');
                overlay.style.display = 'none';
            } else {
                drawer.classList.add('open');
                overlay.style.display = 'block';
            }
        }

        async function removeFile(index) {
            try {
                await fetch(`/remove/${index}`, { method: 'POST' });
                pollStatus();
            } catch (e) {
                alert(`Error removing file: ${e.message}`);
            }
        }

        async function clearClass() {
            if (!confirm('Are you sure you want to clear all loaded materials and end this class session?')) return;
            try {
                await fetch('/clear', { method: 'POST' });
                toggleDrawer();
                showUploadView();
            } catch (e) {
                alert(`Error clearing class: ${e.message}`);
            }
        }

        function toggleFullScreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => {
                    console.error(`Fullscreen error: ${err.message}`);
                });
            } else {
                if (document.exitFullscreen) document.exitFullscreen();
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            pollStatus();
            startSSE();
            setInterval(pollStatus, 1500);
        });
    </script>
</body>
</html>
"""


def _loaded_file_names():
    return [os.path.basename(p) for p in rag_engine.get_loaded_files()]


@app.route("/")
def index():
    """Single unified endpoint for both upload and live original slide presentation."""
    return render_template_string(UNIFIED_PAGE)


@app.route("/presentation")
def presentation():
    return render_template_string(UNIFIED_PAGE)


@app.route("/viewer")
def viewer():
    return render_template_string(UNIFIED_PAGE)


@app.route("/api/slide/image/<int:slide_number>")
def get_slide_image(slide_number):
    """
    Returns the high-resolution rendered original slide image for the requested slide number.
    """
    ordered = rag_engine.get_ordered_chunks()
    if not ordered or slide_number < 1 or slide_number > len(ordered):
        return jsonify({"error": "Slide number not found"}), 404

    target_chunk = ordered[slide_number - 1]
    source_file = target_chunk.get("source_file")

    # Look up full filepath from loaded files
    full_path = None
    for loaded in rag_engine.get_loaded_files():
        if os.path.basename(loaded) == source_file or loaded == source_file:
            full_path = loaded
            break

    if not full_path:
        # Check inbox folder directly
        candidate = os.path.join(INBOX_FOLDER, source_file)
        if os.path.exists(candidate):
            full_path = candidate

    if full_path:
        # Determine internal slide number within this specific file
        file_chunks = [c for c in ordered if c.get("source_file") == source_file]
        try:
            internal_num = file_chunks.index(target_chunk) + 1
        except ValueError:
            internal_num = target_chunk.get("slide_number", slide_number)

        img_path = slide_renderer.get_slide_image_path(full_path, internal_num)
        if img_path and os.path.exists(img_path):
            return send_file(img_path, mimetype="image/png")

        # If cache is missing, render on the fly
        slide_renderer.render_deck_slides(full_path)
        img_path = slide_renderer.get_slide_image_path(full_path, internal_num)
        if img_path and os.path.exists(img_path):
            return send_file(img_path, mimetype="image/png")

    return jsonify({"error": "Slide visual image unavailable"}), 404


@app.route("/api/slide/status")
def slide_status():
    """Returns the current slide position, max reached, and full slide content."""
    current, max_reached = rag_engine.get_lecture_progress()
    ordered = rag_engine.get_ordered_chunks()

    current_chunk = None
    if ordered:
        for c in ordered:
            if c["slide_number"] == current:
                current_chunk = c
                break
        if current_chunk is None and current > 0 and current <= len(ordered):
            current_chunk = ordered[current - 1]
        elif current_chunk is None:
            current_chunk = ordered[0]

    return jsonify({
        "current_slide": current if current > 0 else (1 if ordered else 0),
        "max_slide_reached": max_reached,
        "total_slides": len(ordered),
        "slide": current_chunk,
        "loaded_files": _loaded_file_names(),
    })


@app.route("/api/slide/stream")
def slide_stream():
    """Server-Sent Events stream for instant real-time slide transitions."""
    def event_stream():
        last_seen = None
        while True:
            current, max_reached = rag_engine.get_lecture_progress()
            ordered = rag_engine.get_ordered_chunks()
            state_id = (current, max_reached, len(ordered))

            if state_id != last_seen:
                last_seen = state_id
                current_chunk = None
                if ordered:
                    for c in ordered:
                        if c["slide_number"] == current:
                            current_chunk = c
                            break
                    if current_chunk is None and current > 0 and current <= len(ordered):
                        current_chunk = ordered[current - 1]
                    elif current_chunk is None:
                        current_chunk = ordered[0]

                data = {
                    "current_slide": current if current > 0 else (1 if ordered else 0),
                    "max_slide_reached": max_reached,
                    "total_slides": len(ordered),
                    "slide": current_chunk,
                    "loaded_files": _loaded_file_names(),
                }
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/slide/command", methods=["POST"])
def slide_command():
    """
    HTTP endpoint allowing orchestrator.py SlideClient to advance or jump slides.
    Supports 'next' and 'goto <N>' (detours).
    """
    data = request.get_json(silent=True) or {}
    cmd = data.get("command", "").lower().strip()
    current, max_reached = rag_engine.get_lecture_progress()
    ordered = rag_engine.get_ordered_chunks()

    if cmd == "next":
        next_num = current + 1 if current < len(ordered) else len(ordered)
        rag_engine.set_lecture_progress(next_num)
        return jsonify({"status": "done", "command": "next", "current_slide": next_num})
    elif cmd == "goto":
        slide = data.get("slide")
        if slide is None:
            return jsonify({"error": "Missing 'slide' field"}), 400
        try:
            slide_int = int(slide)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid slide number"}), 400

        # Detour goto preserves max_slide_reached
        rag_engine.set_lecture_progress(slide_int, max_slide=max_reached)
        return jsonify({"status": "done", "command": "goto", "slide": slide_int})
    else:
        return jsonify({"error": f"Unknown command '{cmd}'. Supported: 'next', 'goto'"}), 400


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("lecture_file")
    if not file or file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "message": f"Unsupported file type: {ext}"}), 400

    safe_name = secure_filename(file.filename)
    save_path = os.path.join(INBOX_FOLDER, safe_name)

    if os.path.exists(save_path) and save_path not in rag_engine.get_loaded_files():
        base, extension = os.path.splitext(safe_name)
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(INBOX_FOLDER, f"{base}_{counter}{extension}")
            counter += 1

    file.save(save_path)

    try:
        already_had_files = bool(rag_engine.get_loaded_files())
        num_chunks = rag_engine.load_lecture(save_path, append=already_had_files)

        # Render original visual slide images (PNGs)
        try:
            slide_renderer.render_deck_slides(save_path)
        except Exception as render_err:
            print(f"[APP] Slide visual rendering notice: {render_err}")

        return jsonify({
            "success": True,
            "message": f"Success! Added '{os.path.basename(save_path)}' ({num_chunks} slides ready).",
            "filename": os.path.basename(save_path),
            "total_slides": len(rag_engine.get_ordered_chunks())
        })
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"success": False, "message": f"Error processing file: {e}"}), 500


@app.route("/remove/<int:index>", methods=["POST"])
def remove_one(index):
    files = rag_engine.get_loaded_files()

    if index < 0 or index >= len(files):
        return jsonify({"success": False, "message": "File not found."}), 404

    target_path = files[index]
    target_name = os.path.basename(target_path)

    rag_engine.remove_file(target_path)
    slide_renderer.clear_deck_cache(target_path)

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            print(f"[APP] Deleted file from disk: {target_path}")
        except Exception as e:
            print(f"[APP] Could not delete file: {e}")

    return jsonify({"success": True, "message": f"Removed '{target_name}'."})


@app.route("/clear", methods=["POST"])
def clear():
    files_to_delete = rag_engine.get_loaded_files()
    rag_engine.clear_lecture()
    slide_renderer.clear_deck_cache()

    for path in files_to_delete:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[APP] Deleted file from disk: {path}")
            except Exception as e:
                print(f"[APP] Could not delete file: {e}")

    return jsonify({"success": True, "message": "All lectures cleared."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
