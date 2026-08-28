"""
Slide Companion Script (Teacher's Laptop)
------------------------------------------
Standalone HTTP server that runs on the teacher's laptop to control presentation
slides (PowerPoint, Google Slides, PDF viewers, etc.) via network commands
sent from the AI Professor Robot (Jetson).

Commands:
  - POST /command {"command": "next"}      -> Simulates Right Arrow key
  - POST /command {"command": "goto", "slide": <N>} -> Simulates typing digits <N> + Enter
  - GET  /health                           -> Returns {"status": "ok"}

Usage:
  python slide_companion.py [--port 5055] [--host 0.0.0.0]
"""

import os
import sys
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Allow disabling actual keystrokes for testing/mocking environments
ENABLE_KEYSTROKES = os.environ.get("SLIDE_COMPANION_ENABLE_KEYS", "1") != "0"

if sys.platform == "win32" and ENABLE_KEYSTROKES:
    import ctypes
    user32 = ctypes.windll.user32
    KEYEVENTF_KEYUP = 0x0002
    VK_RIGHT = 0x27
    VK_RETURN = 0x0D
    VK_0 = 0x30
else:
    user32 = None


def press_key(vk_code):
    """Simulates pressing and releasing a virtual key with a slight delay."""
    if user32 is not None and ENABLE_KEYSTROKES:
        user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.05)
    else:
        # Mock/simulated mode or non-Windows fallback
        print(f"[COMPANION] (Simulated keypress vk=0x{vk_code:02X})")


def simulate_next():
    """Simulates Right Arrow keypress to advance to the next slide."""
    print("[COMPANION] Executing 'next' -> pressing Right Arrow")
    press_key(0x27 if user32 is not None else 0x27)


def simulate_goto(slide_number):
    """
    Simulates typing the digits of slide_number followed by Enter.
    This is the native PowerPoint Slide Show shortcut to jump directly to a slide.
    """
    s_num = str(int(slide_number))
    print(f"[COMPANION] Executing 'goto {s_num}' -> typing digits + Enter")
    for digit in s_num:
        vk = 0x30 + int(digit)  # 0x30 is '0'
        press_key(vk)
    # Press Enter
    press_key(0x0D)


class SlideCommandHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to keep console output clean
        print(f"[COMPANION SERVER] {self.address_string()} - {format % args}")

    def _send_json_response(self, status_code, data):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        if self.path == "/health" or self.path == "/ping":
            self._send_json_response(200, {"status": "ok"})
        else:
            self._send_json_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/command":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_json_response(400, {"error": "Invalid JSON"})
                return

            cmd = payload.get("command", "").lower().strip()
            if cmd == "next":
                simulate_next()
                self._send_json_response(200, {"status": "done", "command": "next"})
            elif cmd == "goto":
                slide = payload.get("slide")
                if slide is None:
                    self._send_json_response(400, {"error": "Missing 'slide' field for goto command"})
                    return
                try:
                    slide_int = int(slide)
                except (ValueError, TypeError):
                    self._send_json_response(400, {"error": "Invalid slide number"})
                    return
                simulate_goto(slide_int)
                self._send_json_response(200, {"status": "done", "command": "goto", "slide": slide_int})
            else:
                self._send_json_response(400, {"error": f"Unknown command '{cmd}'. Supported: 'next', 'goto'"})
        else:
            self._send_json_response(404, {"error": "Not found"})


def get_local_ip():
    """Returns the primary local IPv4 address of this machine."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def run_companion_server(host="0.0.0.0", port=5055):
    server = HTTPServer((host, port), SlideCommandHandler)
    local_ip = get_local_ip()
    print("=" * 64)
    print(" AI PROFESSOR ROBOT - SLIDE COMPANION SERVER")
    print("=" * 64)
    print(f" Teacher Laptop IPv4: {local_ip}")
    print(f" Port:                {port}")
    print(f" Status:              READY & LISTENING")
    print(f" Keystrokes Enabled:  {ENABLE_KEYSTROKES}")
    print("-" * 64)
    print(f" [!] On Robot Jetson, configure:")
    print(f"     export SLIDE_COMPANION_HOST=\"{local_ip}\"")
    print(f" [!] In PowerPoint, press F5 for Slide Show mode before class.")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[COMPANION] Stopping slide companion server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slide Companion Server for AI Professor Robot")
    parser.add_argument("--host", default=os.environ.get("SLIDE_COMPANION_HOST", "0.0.0.0"), help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLIDE_COMPANION_PORT", "5055")), help="Port to listen on")
    args = parser.parse_args()
    run_companion_server(host=args.host, port=args.port)
