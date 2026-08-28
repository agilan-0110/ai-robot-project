import os
import json
import time
from flask import Flask, request, render_template_string, jsonify, Response
from werkzeug.utils import secure_filename
import rag_engine

app = Flask(__name__)

INBOX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox")
os.makedirs(INBOX_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pptx", ".pdf", ".docx"}

UPLOAD_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Professor Robot - Dashboard & Upload</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: #0f172a;
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 36px;
            width: 100%;
            max-width: 560px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }
        h1 {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        p.subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; }
        .btn-live {
            display: block;
            text-align: center;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            padding: 14px 20px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 28px;
            box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .btn-live:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5);
        }
        .upload-section {
            border-top: 1px solid #334155;
            padding-top: 24px;
            margin-top: 8px;
        }
        .file-input-wrapper {
            position: relative;
            margin-bottom: 16px;
        }
        input[type="file"] {
            width: 100%;
            padding: 12px;
            background: #0f172a;
            border: 1px dashed #475569;
            border-radius: 8px;
            color: #cbd5e1;
            font-size: 0.9rem;
            cursor: pointer;
        }
        button.btn-submit {
            width: 100%;
            background: #0284c7;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        button.btn-submit:hover { background: #0369a1; }
        .msg {
            margin-top: 16px;
            padding: 12px;
            border-radius: 8px;
            background: rgba(14, 165, 233, 0.15);
            border: 1px solid rgba(14, 165, 233, 0.3);
            color: #38bdf8;
            font-size: 0.9rem;
        }
        .files-list {
            margin-top: 24px;
            border-top: 1px solid #334155;
            padding-top: 20px;
        }
        .files-list h3 { font-size: 1rem; color: #cbd5e1; margin-bottom: 12px; }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.88rem;
            color: #e2e8f0;
        }
        .btn-remove {
            background: #ef4444;
            color: white;
            border: none;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-clear-all {
            width: 100%;
            background: #dc2626;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-top: 12px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>AI Professor Robot</h1>
        <p class="subtitle">Autonomous Classroom Teaching & Live Presentation</p>

        <a href="/presentation" class="btn-live">📽️ Launch Live Presentation Viewer</a>

        <div class="upload-section">
            <h3 style="font-size:1rem; margin-bottom:12px; color:#e2e8f0;">Upload Lecture Material</h3>
            <form method="POST" action="/upload" enctype="multipart/form-data">
                <div class="file-input-wrapper">
                    <input type="file" name="lecture_file" accept=".pptx,.pdf,.docx" required>
                </div>
                <button type="submit" class="btn-submit">Ingest & Prepare Lecture</button>
            </form>

            {% if message %}
                <div class="msg">{{ message }}</div>
            {% endif %}

            {% if loaded_files %}
                <div class="files-list">
                    <h3>Currently Loaded Materials ({{ loaded_files|length }}):</h3>
                    {% for f in loaded_files %}
                        <div class="file-item">
                            <span>📄 {{ f }}</span>
                            <form method="POST" action="/remove/{{ loop.index0 }}" style="display:inline;">
                                <button type="submit" class="btn-remove">Remove</button>
                            </form>
                        </div>
                    {% endfor %}
                    <form method="POST" action="/clear">
                        <button type="submit" class="btn-clear-all">Clear All Materials (End of Class)</button>
                    </form>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

PRESENTATION_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Professor - Live Presentation</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background: #090d16;
            color: #f8fafc;
            font-family: 'Outfit', sans-serif;
            user-select: none;
        }

        /* Top Status Bar */
        .status-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 40px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(51, 65, 85, 0.6);
            z-index: 100;
        }
        .status-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .live-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }
        .deck-title {
            color: #94a3b8;
            font-size: 0.95rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
        }
        .status-right {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .slide-counter {
            font-size: 1.1rem;
            font-weight: 700;
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.1);
            padding: 6px 16px;
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .btn-fullscreen {
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #475569;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-fullscreen:hover {
            background: #334155;
            color: white;
        }

        /* Main Slide Stage */
        .stage {
            position: absolute;
            top: 60px;
            bottom: 12px;
            left: 0;
            right: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px 80px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .slide-card {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            opacity: 1;
            transition: opacity 0.3s ease, transform 0.3s ease;
        }
        .slide-card.entering {
            animation: slideEnter 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes slideEnter {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .slide-heading {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 32px;
            background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .points-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-width: 1100px;
        }
        .points-list li {
            position: relative;
            padding-left: 36px;
            font-size: 1.8rem;
            line-height: 1.45;
            color: #e2e8f0;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
        }
        .points-list li::before {
            content: "";
            position: absolute;
            left: 4px;
            top: 14px;
            width: 12px;
            height: 12px;
            border-radius: 4px;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.7);
        }

        /* Image / Diagram Card */
        .image-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 20px;
            padding: 32px 40px;
            max-width: 1000px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            margin-top: 16px;
        }
        .image-card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            color: #818cf8;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 14px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .image-card-desc {
            font-size: 1.6rem;
            line-height: 1.5;
            color: #cbd5e1;
            font-family: 'Inter', sans-serif;
            font-style: italic;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            max-width: 600px;
        }
        .empty-state h2 {
            font-size: 2.2rem;
            margin-bottom: 16px;
            color: #cbd5e1;
        }
        .empty-state p {
            font-size: 1.2rem;
            color: #64748b;
            margin-bottom: 28px;
        }

        /* Progress Bar */
        .progress-bar-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: rgba(15, 23, 42, 0.8);
            z-index: 100;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            width: 0%;
            transition: width 0.4s ease;
        }
    </style>
</head>
<body>
    <div class="status-bar">
        <div class="status-left">
            <div class="live-badge" id="liveBadge">
                <div class="live-dot"></div>
                <span id="liveStatusText">LIVE AUTONOMOUS SYNC</span>
            </div>
            <span class="deck-title" id="deckTitle">Loading presentation...</span>
        </div>
        <div class="status-right">
            <div class="slide-counter" id="slideCounter">Slide 1 / 1</div>
            <button class="btn-fullscreen" onclick="toggleFullScreen()">⛶ Full Screen (F11)</button>
        </div>
    </div>

    <div class="stage">
        <div class="slide-card entering" id="slideCard">
            <h1 class="slide-heading" id="slideHeading">Waiting for Lecture...</h1>
            <ul class="points-list" id="pointsList"></ul>
            <div class="image-card" id="imageCard" style="display:none;">
                <div class="image-card-header">
                    <span>🖼️ Diagram / Image Visual</span>
                </div>
                <div class="image-card-desc" id="imageCardDesc"></div>
            </div>
        </div>
    </div>

    <div class="progress-bar-container">
        <div class="progress-bar-fill" id="progressBar"></div>
    </div>

    <script>
        let currentSlideNum = -1;

        function updateSlideUI(data) {
            if (!data || !data.slide) {
                document.getElementById('slideHeading').innerText = 'No Lecture Loaded';
                document.getElementById('pointsList').innerHTML = '<li>Upload a presentation from the dashboard to begin.</li>';
                document.getElementById('imageCard').style.display = 'none';
                document.getElementById('slideCounter').innerText = '0 / 0';
                document.getElementById('progressBar').style.width = '0%';
                return;
            }

            const slide = data.slide;
            const cur = data.current_slide || slide.slide_number;
            const total = data.total_slides || 1;

            if (cur === currentSlideNum) {
                return; // already showing this slide
            }
            currentSlideNum = cur;

            // Trigger enter animation
            const card = document.getElementById('slideCard');
            card.classList.remove('entering');
            void card.offsetWidth; // trigger reflow
            card.classList.add('entering');

            // Source file title
            if (slide.source_file) {
                document.getElementById('deckTitle').innerText = slide.source_file;
            }

            // Counter & progress
            document.getElementById('slideCounter').innerText = `Slide ${cur} / ${total}`;
            const pct = Math.min(100, Math.max(0, (cur / total) * 100));
            document.getElementById('progressBar').style.width = `${pct}%`;

            // Heading
            const heading = slide.heading || (slide.text.startsWith('#') ? slide.text.split('\\n')[0].replace(/^#+\\s*/, '') : `Slide ${cur}`);
            document.getElementById('slideHeading').innerText = heading;

            // Points & Image Captions
            const pointsList = document.getElementById('pointsList');
            pointsList.innerHTML = '';

            const imageCaptions = [];
            const textBullets = [];

            if (slide.points && slide.points.length > 0) {
                slide.points.forEach(p => {
                    const imgMatch = p.match(/^\\[Image:\\s*(.*)\\]$/);
                    if (imgMatch) {
                        imageCaptions.push(imgMatch[1]);
                    } else if (p.trim() && p.trim() !== heading) {
                        textBullets.push(p);
                    }
                });
            } else if (slide.text) {
                slide.text.split('\\n').forEach(line => {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed.startsWith('#')) return;
                    const imgMatch = trimmed.match(/^\\[Image:\\s*(.*)\\]$/);
                    if (imgMatch) {
                        imageCaptions.push(imgMatch[1]);
                    } else {
                        textBullets.push(trimmed);
                    }
                });
            }

            // Render text bullets
            textBullets.forEach(b => {
                const li = document.createElement('li');
                li.innerText = b;
                pointsList.appendChild(li);
            });

            // Render image caption card if present
            const imageCard = document.getElementById('imageCard');
            if (imageCaptions.length > 0) {
                imageCard.style.display = 'block';
                document.getElementById('imageCardDesc').innerText = imageCaptions.join(' ');
            } else {
                imageCard.style.display = 'none';
            }
        }

        // 1. Connect to live SSE stream
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
                // Fall back to polling if SSE drops
                console.warn('SSE connection interrupted, using status poll fallback...');
                pollStatus();
            };
        }

        // 2. Periodic status polling fallback
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

        // Fullscreen toggle
        function toggleFullScreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(err => {
                    console.error(`Error attempting fullscreen: ${err.message}`);
                });
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                }
            }
        }

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {
            pollStatus();
            startSSE();
            setInterval(pollStatus, 1500); // safety fallback poll every 1.5s
        });
    </script>
</body>
</html>
"""


def _loaded_file_names():
    return [os.path.basename(p) for p in rag_engine.get_loaded_files()]


@app.route("/")
def index():
    return render_template_string(
        UPLOAD_PAGE,
        message=None,
        loaded_files=_loaded_file_names(),
    )


@app.route("/presentation")
def presentation():
    """Live classroom projector slide viewer (zero software on teacher laptop)."""
    return render_template_string(PRESENTATION_PAGE)


@app.route("/viewer")
def viewer():
    """Alias for /presentation."""
    return render_template_string(PRESENTATION_PAGE)


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
        return render_template_string(UPLOAD_PAGE, message="No file selected.", loaded_files=_loaded_file_names())

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template_string(UPLOAD_PAGE, message=f"Unsupported file type: {ext}", loaded_files=_loaded_file_names())

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
        message = f"Success! Added '{os.path.basename(save_path)}' — {num_chunks} chunks ready."
    except Exception as e:
        message = f"Error processing file: {e}"
        if os.path.exists(save_path):
            os.remove(save_path)

    return render_template_string(UPLOAD_PAGE, message=message, loaded_files=_loaded_file_names())


@app.route("/remove/<int:index>", methods=["POST"])
def remove_one(index):
    files = rag_engine.get_loaded_files()

    if index < 0 or index >= len(files):
        return render_template_string(UPLOAD_PAGE, message="File not found (already removed?).", loaded_files=_loaded_file_names())

    target_path = files[index]
    target_name = os.path.basename(target_path)

    rag_engine.remove_file(target_path)

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            print(f"[APP] Deleted file from disk: {target_path}")
        except Exception as e:
            print(f"[APP] Could not delete file: {e}")

    return render_template_string(UPLOAD_PAGE, message=f"Removed '{target_name}'.", loaded_files=_loaded_file_names())


@app.route("/clear", methods=["POST"])
def clear():
    files_to_delete = rag_engine.get_loaded_files()

    rag_engine.clear_lecture()

    for path in files_to_delete:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[APP] Deleted file from disk: {path}")
            except Exception as e:
                print(f"[APP] Could not delete file: {e}")

    return render_template_string(UPLOAD_PAGE, message="All lectures cleared and files deleted.", loaded_files=_loaded_file_names())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
