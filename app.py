import base64
import json
import os
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = os.getenv("HANDY_AI_HOST", "127.0.0.1")
PORT = int(os.getenv("HANDY_AI_PORT", "5000"))
OPEN_BROWSER = os.getenv("HANDY_AI_OPEN_BROWSER", "1") == "1"
USE_SERVER_CAMERA = os.getenv("HANDY_AI_USE_SERVER_CAMERA", "0") == "1"
CLIENT_FRAME_TIMEOUT_SECONDS = float(
    os.getenv("HANDY_AI_CLIENT_FRAME_TIMEOUT_SECONDS", "3")
)
LANDMARK_SMOOTHING = float(os.getenv("HANDY_AI_LANDMARK_SMOOTHING", "0.35"))
AI_COOLDOWN_SECONDS = float(os.getenv("HANDY_AI_COOLDOWN_SECONDS", "4"))

# Serve files from the project root so index.html and /static work when app.py runs.
os.chdir(BASE_DIR)

# Configure Gemini only when a real API key is present. The UI can still run without it.
api_key = os.getenv("GOOGLE_API_KEY")
model = None
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
else:
    print("GOOGLE_API_KEY is not set. AI solving is disabled until you add it.")

# Hand Detector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.7,
    minTrackCon=0.5,
)

# Global state
prev_pos = None
smoothed_lmList = None
canvas = None
output_text = ""
latest_frame = None
latest_status = "Waiting for camera frames..."
last_client_frame_at = 0.0
frame_lock = threading.Lock()
frame_processing_lock = threading.Lock()
ai_solve_lock = threading.Lock()
ai_solve_in_progress = False
last_ai_request_at = 0.0
server_camera_error = ""
server_camera = None


def drawLowPolyHand(img, lmList):
    """
    Draws a low-poly style mesh by filling triangles between landmarks.
    """
    palm_triangles = [
        (0, 1, 5),
        (0, 5, 9),
        (0, 9, 13),
        (0, 13, 17),
    ]

    mesh_color = (180, 180, 180)
    line_color = (50, 50, 50)

    for p1, p2, p3 in palm_triangles:
        pt1 = lmList[p1][0:2]
        pt2 = lmList[p2][0:2]
        pt3 = lmList[p3][0:2]
        triangle_cnt = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(img, [triangle_cnt], mesh_color)
        cv2.polylines(img, [triangle_cnt], True, line_color, 2)

    fingers_indices = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    for finger in fingers_indices:
        for i in range(len(finger) - 1):
            p1 = finger[i]
            p2 = finger[i + 1]
            pt1 = lmList[p1][0:2]
            pt2 = lmList[p2][0:2]

            cv2.line(img, pt1, pt2, mesh_color, 15)
            cv2.line(img, pt1, pt2, line_color, 3)
            cv2.circle(img, pt1, 8, line_color, cv2.FILLED)

        cv2.circle(img, lmList[finger[-1]][0:2], 8, line_color, cv2.FILLED)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None, None


def smooth_landmarks(lmList):
    global smoothed_lmList

    if not lmList:
        smoothed_lmList = None
        return None

    if smoothed_lmList is None or len(smoothed_lmList) != len(lmList):
        smoothed_lmList = [list(point) for point in lmList]
        return smoothed_lmList

    smoothed = []
    for index, point in enumerate(lmList):
        previous_point = smoothed_lmList[index]
        next_point = list(point)
        next_point[0] = int(
            previous_point[0] + (point[0] - previous_point[0]) * LANDMARK_SMOOTHING
        )
        next_point[1] = int(
            previous_point[1] + (point[1] - previous_point[1]) * LANDMARK_SMOOTHING
        )

        if len(point) > 2:
            next_point[2] = int(
                previous_point[2]
                + (point[2] - previous_point[2]) * LANDMARK_SMOOTHING
            )

        smoothed.append(next_point)

    smoothed_lmList = smoothed
    return smoothed_lmList


def ensure_canvas_shape(img):
    global canvas

    if canvas is None or canvas.shape != img.shape:
        canvas = np.zeros_like(img)


def clear_session():
    global canvas, prev_pos, smoothed_lmList, output_text, latest_frame, latest_status

    prev_pos = None
    smoothed_lmList = None
    output_text = ""
    latest_frame = None
    latest_status = "Canvas cleared."

    if canvas is None:
        return

    canvas = np.zeros_like(canvas)


def draw(info, previous_position, current_canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if previous_position is None:
            previous_position = current_pos
        cv2.line(current_canvas, previous_position, current_pos, (255, 0, 255), 10)
    elif fingers == [1, 1, 0, 0, 0]:
        previous_position = None
    elif fingers == [0, 0, 0, 0, 0]:
        current_canvas = np.zeros_like(current_canvas)
        previous_position = None

    if fingers == [0, 1, 0, 0, 0]:
        return current_pos, current_canvas

    return None, current_canvas


def clean_ai_text(text):
    for token in ("**", "*", "$", "#", "`", "_"):
        text = text.replace(token, "")
    return text.strip()


def sendToAI(current_canvas, fingers):
    if fingers != [1, 1, 1, 1, 1]:
        return None

    if model is None:
        return "Set GOOGLE_API_KEY to enable AI solving."

    pil_image = Image.fromarray(current_canvas)
    try:
        response = model.generate_content(
            ["Solve this math problem exactly and show work", pil_image]
        )
        return clean_ai_text(response.text)
    except Exception as exc:
        return f"Error: {exc}"


def solve_canvas_snapshot(canvas_snapshot):
    global ai_solve_in_progress, output_text

    try:
        ai_response = sendToAI(canvas_snapshot, [1, 1, 1, 1, 1])
        if ai_response:
            output_text = ai_response
    finally:
        with ai_solve_lock:
            ai_solve_in_progress = False


def request_ai_solution(current_canvas, fingers):
    global ai_solve_in_progress, last_ai_request_at, output_text

    if fingers != [1, 1, 1, 1, 1]:
        return

    now = time.time()
    with ai_solve_lock:
        if ai_solve_in_progress or now - last_ai_request_at < AI_COOLDOWN_SECONDS:
            return

        ai_solve_in_progress = True
        last_ai_request_at = now
        output_text = "Solving..."

    snapshot = current_canvas.copy()
    threading.Thread(
        target=solve_canvas_snapshot,
        args=(snapshot,),
        daemon=True,
    ).start()


def get_status_text(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return "Writing..."
    if fingers == [1, 1, 0, 0, 0]:
        return "Pause"
    if fingers == [0, 0, 0, 0, 0]:
        return "Cleared!"
    if fingers == [1, 1, 1, 1, 1]:
        return "Solving..."
    return "Show your hand to start drawing."


def draw_status_banner(img, fingers):
    status_text = get_status_text(fingers)
    status_color = (255, 255, 255)

    if fingers == [0, 1, 0, 0, 0]:
        status_color = (255, 0, 255)
    elif fingers == [1, 1, 0, 0, 0]:
        status_color = (0, 255, 255)
    elif fingers == [0, 0, 0, 0, 0]:
        status_color = (0, 0, 255)
    elif fingers == [1, 1, 1, 1, 1]:
        status_color = (0, 255, 0)

    cv2.putText(
        img,
        status_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2,
    )
    return status_text


def store_latest_frame(img, status_text):
    global latest_frame, latest_status

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        return

    latest_status = status_text
    latest_frame = base64.b64encode(buffer).decode()


def process_frame(img):
    global canvas, prev_pos, smoothed_lmList

    img = cv2.flip(img, 1)
    ensure_canvas_shape(img)

    black_board = np.full_like(img, 50)
    info = getHandInfo(img)
    fingers, lmList = info

    if fingers:
        lmList = smooth_landmarks(lmList)
        info = fingers, lmList
        drawLowPolyHand(black_board, lmList)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        request_ai_solution(canvas, fingers)
    else:
        prev_pos = None
        smoothed_lmList = None

    combined_image = cv2.addWeighted(black_board, 1, canvas, 1, 0)
    status_text = draw_status_banner(combined_image, fingers)

    with frame_lock:
        store_latest_frame(combined_image, status_text)


def process_client_frame(img):
    global last_client_frame_at

    last_client_frame_at = time.time()
    if not frame_processing_lock.acquire(blocking=False):
        return

    try:
        process_frame(img)
    finally:
        frame_processing_lock.release()


def process_video_stream():
    global server_camera, server_camera_error

    server_camera = cv2.VideoCapture(0)
    server_camera.set(3, 1280)
    server_camera.set(4, 720)

    if not server_camera.isOpened():
        server_camera_error = "Server camera could not be opened."
        print(server_camera_error)
        return

    while True:
        if time.time() - last_client_frame_at < CLIENT_FRAME_TIMEOUT_SECONDS:
            time.sleep(0.05)
            continue

        success, img = server_camera.read()
        if not success:
            time.sleep(0.05)
            continue

        if not frame_processing_lock.acquire(blocking=False):
            continue

        try:
            process_frame(img)
        finally:
            frame_processing_lock.release()


class APIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory or BASE_DIR, **kwargs)

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_cors_headers()
            self.end_headers()
            with open(os.path.join(BASE_DIR, "index.html"), "rb") as file_handle:
                self.wfile.write(file_handle.read())
            return

        if parsed_path.path == "/api/health":
            self.send_json(
                {
                    "status": "ok",
                    "backend": "app.py",
                    "aiConfigured": model is not None,
                    "serverCameraEnabled": USE_SERVER_CAMERA,
                    "serverCameraAvailable": bool(
                        server_camera is not None and server_camera.isOpened()
                    ),
                    "serverCameraError": server_camera_error,
                    "frameReady": latest_frame is not None,
                }
            )
            return

        if parsed_path.path == "/api/frame":
            with frame_lock:
                payload = {"frame": latest_frame, "statusText": latest_status}
            self.send_json(payload)
            return

        if parsed_path.path == "/api/output":
            self.send_json({"text": output_text})
            return

        if parsed_path.path == "/api/clear":
            clear_session()
            self.send_json({"status": "cleared"})
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/clear":
            clear_session()
            self.send_json({"status": "cleared"})
            return

        if self.path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode())
                b64 = data.get("frame", "")
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]

                if not b64:
                    raise ValueError("Missing frame data")

                img_data = base64.b64decode(b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Invalid image data")

                process_client_frame(img)
                self.send_json({"status": "ok"})
                return
            except Exception as exc:
                print(f"Error handling upload: {exc}")
                self.send_json({"status": "error"}, status=400)
                return

        self.send_json({"status": "not-found"}, status=404)

    def log_message(self, format, *args):
        pass


def main():
    if USE_SERVER_CAMERA:
        video_thread = threading.Thread(target=process_video_stream, daemon=True)
        video_thread.start()
        print("Server camera mode is enabled.")
    else:
        print("Browser camera mode is enabled.")

    handler = partial(APIHandler, directory=BASE_DIR)
    httpd = ThreadingHTTPServer((HOST, PORT), handler)

    print(f"Server running at http://{HOST}:{PORT}")
    if OPEN_BROWSER:
        webbrowser.open(f"http://{HOST}:{PORT}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        httpd.server_close()
        if server_camera is not None and server_camera.isOpened():
            server_camera.release()


if __name__ == "__main__":
    main()
