from flask import Flask, request, render_template_string
import os
import rag_engine
from werkzeug.utils import secure_filename

app = Flask(__name__)

INBOX_FOLDER = "inbox"
os.makedirs(INBOX_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pptx", ".pdf", ".docx"}

UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head><title>AI Professor - Lecture Upload</title></head>
<body style="font-family: sans-serif; max-width: 500px; margin: 60px auto;">
    <h2>Upload Lecture File</h2>
    <p>Accepted formats: .pptx, .pdf, .docx</p>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="lecture_file" required><br><br>
        <button type="submit">Add File</button>
    </form>
    {% if message %}
        <p><strong>{{ message }}</strong></p>
    {% endif %}
    {% if loaded_files %}
        <p style="color: green;">Currently loaded files:</p>
        <ul>
        {% for f in loaded_files %}
            <li>
                {{ f }}
                <form method="POST" action="/remove/{{ loop.index0 }}" style="display:inline;">
                    <button type="submit" style="background:#e67e22;color:white;font-size:0.8em;">Remove</button>
                </form>
            </li>
        {% endfor %}
        </ul>
        <form method="POST" action="/clear">
            <button type="submit" style="background:#c0392b;color:white;">Clear All (end of class)</button>
        </form>
    {% endif %}
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
