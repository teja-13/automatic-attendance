import base64
import csv
import io
import threading
from datetime import datetime

import cv2
import face_recognition
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

from main import configure_optional_cuda_dll_path, get_runtime_config, load_known_faces


app = Flask(__name__)
state_lock = threading.Lock()

runtime_config = {}
known_face_encodings = []
known_face_names = []
attendance_state = {}
attendance_events = []


def now_local_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_attendance_rows():
    rows = []
    for name in sorted(attendance_state.keys()):
        item = attendance_state[name]
        rows.append(
            {
                "name": name,
                "status": item["status"],
                "last_seen": item["last_seen"],
                "last_location": item["last_location"],
                "confidence": item["confidence"],
            }
        )
    return rows


def reset_attendance_state():
    global attendance_state
    attendance_state = {
        name: {
            "status": "Absent",
            "last_seen": "-",
            "last_location": "-",
            "confidence": 0.0,
        }
        for name in sorted(set(known_face_names))
    }


def initialize_system():
    global runtime_config
    global known_face_encodings
    global known_face_names
    global attendance_events

    configure_optional_cuda_dll_path()
    runtime_config = get_runtime_config()
    known_face_encodings, known_face_names = load_known_faces(runtime_config, "known_faces")
    reset_attendance_state()
    attendance_events = []


def decode_data_url_to_bgr(data_url):
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image payload")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Image decode failed")

    return frame


def detect_known_faces(frame_bgr, tolerance=0.48):
    detection_model = runtime_config.get("detection_model", "hog")
    live_scale = runtime_config.get("live_scale", 0.25)

    small_frame = cv2.resize(frame_bgr, (0, 0), fx=live_scale, fy=live_scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=detection_model)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    best_hits = {}
    unknown_count = 0

    for face_encoding in face_encodings:
        if not known_face_encodings:
            unknown_count += 1
            continue

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) == 0:
            unknown_count += 1
            continue

        best_match_index = int(np.argmin(face_distances))
        distance = float(face_distances[best_match_index])

        if distance <= tolerance:
            name = known_face_names[best_match_index]
            confidence = round((1.0 - distance) * 100.0, 2)

            previous = best_hits.get(name)
            if previous is None or confidence > previous["confidence"]:
                best_hits[name] = {"name": name, "confidence": confidence}
        else:
            unknown_count += 1

    return list(best_hits.values()), unknown_count


def apply_attendance_rules(location, recognized_people):
    timestamp = now_local_iso()
    location_label = "Classroom" if location == "classroom" else "Elsewhere"
    status = "Present" if location == "classroom" else "Absent"

    for person in recognized_people:
        name = person["name"]
        confidence = person["confidence"]

        if name not in attendance_state:
            attendance_state[name] = {
                "status": "Absent",
                "last_seen": "-",
                "last_location": "-",
                "confidence": 0.0,
            }

        attendance_state[name]["status"] = status
        attendance_state[name]["last_seen"] = timestamp
        attendance_state[name]["last_location"] = location_label
        attendance_state[name]["confidence"] = confidence

        attendance_events.append(
            {
                "time": timestamp,
                "name": name,
                "location": location_label,
                "status": status,
                "confidence": confidence,
            }
        )

    if len(attendance_events) > 200:
        del attendance_events[:-200]


def state_payload():
    return {
        "runtime": {
            "label": runtime_config.get("label", "CPU"),
            "detection_model": runtime_config.get("detection_model", "hog"),
            "gpu_available": runtime_config.get("gpu_available", False),
            "gpu_devices": runtime_config.get("gpu_devices", 0),
        },
        "known_people": sorted(set(known_face_names)),
        "attendance": build_attendance_rows(),
        "events": list(reversed(attendance_events[-30:])),
    }


@app.get("/")
def index():
    payload = state_payload()
    return render_template_string(HTML_TEMPLATE, initial_payload=payload)


@app.get("/api/state")
def api_state():
    with state_lock:
        return jsonify(state_payload())


@app.post("/api/process")
def api_process():
    body = request.get_json(silent=True) or {}
    image_data = body.get("image")
    location = body.get("location", "classroom")

    if location not in {"classroom", "elsewhere"}:
        return jsonify({"error": "location must be classroom or elsewhere"}), 400

    try:
        frame = decode_data_url_to_bgr(image_data)
        recognized_people, unknown_count = detect_known_faces(frame)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    with state_lock:
        apply_attendance_rules(location, recognized_people)
        payload = state_payload()

    payload["recognized"] = recognized_people
    payload["unknown_count"] = unknown_count
    payload["processed_location"] = location
    return jsonify(payload)


@app.post("/api/clear")
def api_clear():
    with state_lock:
        reset_attendance_state()
        attendance_events.clear()
        return jsonify(state_payload())


@app.post("/api/reload")
def api_reload():
    global known_face_encodings
    global known_face_names

    with state_lock:
        known_face_encodings, known_face_names = load_known_faces(runtime_config, "known_faces")
        reset_attendance_state()
        attendance_events.clear()
        return jsonify(state_payload())


@app.get("/api/export")
def api_export():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Status", "Last Seen", "Last Location", "Confidence"])

    for row in build_attendance_rows():
        writer.writerow(
            [
                row["name"],
                row["status"],
                row["last_seen"],
                row["last_location"],
                row["confidence"],
            ]
        )

    buffer = io.BytesIO(output.getvalue().encode("utf-8"))
    buffer.seek(0)

    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Automatic Attendance Dashboard</title>
  <style>
    :root {
      --bg: #f3f8ef;
      --panel: #ffffff;
      --ink: #1d2a28;
      --muted: #52625f;
      --line: #dce7e2;
      --accent: #117a65;
      --accent-soft: #daf3ec;
      --danger: #c23b2d;
      --danger-soft: #fde6e3;
      --amber: #ba6f0a;
      --amber-soft: #fff1df;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% -10%, #e4f6ee 0%, transparent 35%),
        radial-gradient(circle at 100% 0%, #fff0db 0%, transparent 40%),
        var(--bg);
      min-height: 100vh;
    }

    .shell {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      background: linear-gradient(120deg, #0f8b6f, #1f6f8b 65%, #f2a154 140%);
      color: #ffffff;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 14px 38px rgba(16, 70, 61, 0.25);
    }

    .hero h1 {
      margin: 0;
      font-size: 2rem;
      letter-spacing: 0.4px;
    }

    .hero p {
      margin: 10px 0 0;
      opacity: 0.95;
      font-size: 0.98rem;
    }

    .grid {
      margin-top: 20px;
      display: grid;
      gap: 16px;
      grid-template-columns: 1.2fr 1fr;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(24, 53, 45, 0.08);
    }

    .card h2 {
      margin: 0 0 10px;
      font-size: 1.15rem;
    }

    .runtime {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }

    .badge {
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.82rem;
      font-weight: 600;
    }

    .badge-accent {
      background: var(--accent-soft);
      color: #0d6a57;
    }

    .badge-amber {
      background: var(--amber-soft);
      color: #92590a;
    }

    .camera-wrap {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr;
    }

    video {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #111;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }

    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    button,
    .link-btn {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    button:hover,
    .link-btn:hover {
      transform: translateY(-1px);
      opacity: 0.92;
    }

    .btn-primary { background: var(--accent); color: white; }
    .btn-danger { background: var(--danger); color: white; }
    .btn-neutral { background: #edf3f0; color: #193630; }

    .auto-row {
      margin-top: 10px;
      padding: 10px;
      border-radius: 10px;
      border: 1px dashed #b8ccc4;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 0.9rem;
      color: var(--muted);
    }

    select,
    input[type="number"] {
      border: 1px solid #b9ccc5;
      border-radius: 8px;
      padding: 6px 8px;
      background: white;
      color: var(--ink);
    }

    .recognition-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .recognition-list li {
      border-radius: 999px;
      padding: 7px 10px;
      background: #eef6f2;
      color: #1b4f42;
      border: 1px solid #d5e8e0;
      font-size: 0.84rem;
      font-weight: 600;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    th,
    td {
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid #e3ece8;
    }

    th {
      color: #3f5b54;
      font-size: 0.8rem;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .status-present {
      color: #0b7a51;
      font-weight: 700;
    }

    .status-absent {
      color: #af2e22;
      font-weight: 700;
    }

    .events {
      max-height: 260px;
      overflow: auto;
      border: 1px solid #e1ebe7;
      border-radius: 10px;
    }

    .event-row {
      display: grid;
      grid-template-columns: 110px 1fr 90px;
      gap: 10px;
      padding: 9px 10px;
      border-bottom: 1px solid #edf3f0;
      font-size: 0.86rem;
      align-items: center;
    }

    .event-row:last-child {
      border-bottom: none;
    }

    .event-status-present { color: #0e7b53; font-weight: 700; }
    .event-status-absent { color: #b03124; font-weight: 700; }

    .notice {
      margin-top: 10px;
      font-size: 0.88rem;
      color: #2f6157;
      min-height: 18px;
    }

    canvas { display: none; }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Automatic Attendance Dashboard</h1>
      <p>
        Rule: detected in Classroom => Present, detected Elsewhere => Absent.
      </p>
      <div class="runtime" id="runtimeBadges"></div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Camera Capture</h2>
        <div class="camera-wrap">
          <video id="cameraFeed" autoplay muted playsinline></video>
          <canvas id="captureCanvas"></canvas>

          <div class="buttons">
            <button class="btn-primary" id="startCameraBtn" type="button">Start Camera</button>
            <button class="btn-neutral" id="captureClassBtn" type="button">Capture Classroom</button>
            <button class="btn-danger" id="captureElseBtn" type="button">Capture Elsewhere</button>
            <button class="btn-neutral" id="stopCameraBtn" type="button">Stop Camera</button>
          </div>

          <div class="auto-row">
            <label>
              <input type="checkbox" id="autoMode" />
              Auto capture
            </label>
            <label>
              Every
              <input id="intervalSeconds" type="number" min="1" max="20" value="3" />
              sec
            </label>
            <label>
              Location
              <select id="autoLocation">
                <option value="classroom">Classroom</option>
                <option value="elsewhere">Elsewhere</option>
              </select>
            </label>
          </div>

          <p class="notice" id="captureNotice"></p>
        </div>
      </article>

      <article class="card">
        <h2>Latest Recognition</h2>
        <ul class="recognition-list" id="recognitionList"></ul>
        <p class="notice" id="recognitionNotice">No captures yet.</p>

        <div class="buttons" style="margin-top: 14px;">
          <button class="btn-neutral" id="reloadFacesBtn" type="button">Reload Known Faces</button>
          <button class="btn-neutral" id="clearSessionBtn" type="button">Clear Session</button>
          <a class="link-btn btn-neutral" href="/api/export" target="_blank" rel="noopener">Export CSV</a>
        </div>
      </article>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Attendance Board</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Last Seen</th>
              <th>Location</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody id="attendanceBody"></tbody>
        </table>
      </article>

      <article class="card">
        <h2>Event Log</h2>
        <div class="events" id="eventsList"></div>
      </article>
    </section>
  </div>

  <script>
    const initialPayload = {{ initial_payload | tojson }};

    const cameraFeed = document.getElementById("cameraFeed");
    const captureCanvas = document.getElementById("captureCanvas");
    const runtimeBadges = document.getElementById("runtimeBadges");
    const captureNotice = document.getElementById("captureNotice");
    const recognitionList = document.getElementById("recognitionList");
    const recognitionNotice = document.getElementById("recognitionNotice");
    const attendanceBody = document.getElementById("attendanceBody");
    const eventsList = document.getElementById("eventsList");

    const autoMode = document.getElementById("autoMode");
    const intervalSeconds = document.getElementById("intervalSeconds");
    const autoLocation = document.getElementById("autoLocation");

    let mediaStream = null;
    let autoTimer = null;

    function setNotice(message) {
      captureNotice.textContent = message;
    }

    function runtimeBadge(label, className) {
      return `<span class="badge ${className}">${label}</span>`;
    }

    function renderRuntime(payload) {
      const runtime = payload.runtime || {};
      const knownPeople = payload.known_people || [];
      const modeLabel = runtime.gpu_available
        ? `Mode: ${runtime.label}`
        : `Mode: ${runtime.label}`;

      runtimeBadges.innerHTML = [
        runtimeBadge(modeLabel, "badge-accent"),
        runtimeBadge(`Detector: ${(runtime.detection_model || "hog").toUpperCase()}`, "badge-amber"),
        runtimeBadge(`Known People: ${knownPeople.length}`, "badge-accent"),
      ].join("");
    }

    function renderRecognition(recognized, unknownCount, location) {
      recognitionList.innerHTML = "";

      if (!recognized || recognized.length === 0) {
        recognitionNotice.textContent = `No known faces recognized (${unknownCount || 0} unknown) in ${location}.`;
        return;
      }

      recognitionNotice.textContent = `${recognized.length} known face(s) recognized in ${location}. Unknown: ${unknownCount || 0}.`;
      for (const person of recognized) {
        const item = document.createElement("li");
        item.textContent = `${person.name} (${person.confidence.toFixed(1)}%)`;
        recognitionList.appendChild(item);
      }
    }

    function renderAttendance(payload) {
      const attendance = payload.attendance || [];
      attendanceBody.innerHTML = "";

      for (const row of attendance) {
        const tr = document.createElement("tr");
        const statusClass = row.status === "Present" ? "status-present" : "status-absent";

        tr.innerHTML = `
          <td>${row.name}</td>
          <td class="${statusClass}">${row.status}</td>
          <td>${row.last_seen}</td>
          <td>${row.last_location}</td>
          <td>${Number(row.confidence || 0).toFixed(2)}%</td>
        `;
        attendanceBody.appendChild(tr);
      }
    }

    function renderEvents(payload) {
      const events = payload.events || [];
      eventsList.innerHTML = "";

      if (events.length === 0) {
        const empty = document.createElement("div");
        empty.className = "event-row";
        empty.innerHTML = "<div>-</div><div>No attendance events yet.</div><div>-</div>";
        eventsList.appendChild(empty);
        return;
      }

      for (const event of events) {
        const row = document.createElement("div");
        const statusClass = event.status === "Present" ? "event-status-present" : "event-status-absent";
        row.className = "event-row";
        row.innerHTML = `
          <div>${event.time}</div>
          <div>${event.name} @ ${event.location}</div>
          <div class="${statusClass}">${event.status}</div>
        `;
        eventsList.appendChild(row);
      }
    }

    function renderAll(payload) {
      renderRuntime(payload);
      renderAttendance(payload);
      renderEvents(payload);
    }

    async function startCamera() {
      if (mediaStream) {
        setNotice("Camera is already running.");
        return;
      }

      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        });

        cameraFeed.srcObject = mediaStream;
        setNotice("Camera started.");
      } catch (error) {
        setNotice(`Unable to access camera: ${error.message}`);
      }
    }

    function stopCamera() {
      if (!mediaStream) {
        setNotice("Camera is not running.");
        return;
      }

      for (const track of mediaStream.getTracks()) {
        track.stop();
      }
      mediaStream = null;
      cameraFeed.srcObject = null;
      setNotice("Camera stopped.");
    }

    async function captureAndSend(location) {
      if (!mediaStream) {
        setNotice("Start the camera first.");
        return;
      }

      const width = cameraFeed.videoWidth;
      const height = cameraFeed.videoHeight;
      if (!width || !height) {
        setNotice("Camera feed is not ready yet.");
        return;
      }

      captureCanvas.width = width;
      captureCanvas.height = height;
      const ctx = captureCanvas.getContext("2d");
      ctx.drawImage(cameraFeed, 0, 0, width, height);

      const imageData = captureCanvas.toDataURL("image/jpeg", 0.9);
      setNotice(`Processing ${location} snapshot...`);

      try {
        const response = await fetch("/api/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageData, location }),
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Unknown server error");
        }

        renderAll(payload);
        renderRecognition(payload.recognized, payload.unknown_count, payload.processed_location);
        setNotice(`Snapshot processed: ${location}.`);
      } catch (error) {
        setNotice(`Capture failed: ${error.message}`);
      }
    }

    function stopAutoMode() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
      }
    }

    function applyAutoMode() {
      stopAutoMode();

      if (!autoMode.checked) {
        setNotice("Auto capture disabled.");
        return;
      }

      const seconds = Number(intervalSeconds.value || 3);
      const safeSeconds = Math.max(1, Math.min(20, seconds));
      const location = autoLocation.value;

      autoTimer = setInterval(() => {
        captureAndSend(location);
      }, safeSeconds * 1000);

      setNotice(`Auto capture enabled: every ${safeSeconds}s at ${location}.`);
    }

    async function clearSession() {
      const response = await fetch("/api/clear", { method: "POST" });
      const payload = await response.json();
      renderAll(payload);
      renderRecognition([], 0, "-");
      setNotice("Attendance session cleared.");
    }

    async function reloadKnownFaces() {
      setNotice("Reloading known faces...");
      const response = await fetch("/api/reload", { method: "POST" });
      const payload = await response.json();
      renderAll(payload);
      renderRecognition([], 0, "-");
      setNotice("Known faces reloaded.");
    }

    document.getElementById("startCameraBtn").addEventListener("click", startCamera);
    document.getElementById("stopCameraBtn").addEventListener("click", stopCamera);
    document.getElementById("captureClassBtn").addEventListener("click", () => captureAndSend("classroom"));
    document.getElementById("captureElseBtn").addEventListener("click", () => captureAndSend("elsewhere"));
    document.getElementById("clearSessionBtn").addEventListener("click", clearSession);
    document.getElementById("reloadFacesBtn").addEventListener("click", reloadKnownFaces);

    autoMode.addEventListener("change", applyAutoMode);
    intervalSeconds.addEventListener("change", applyAutoMode);
    autoLocation.addEventListener("change", applyAutoMode);

    renderAll(initialPayload);
    renderRecognition([], 0, "-");
  </script>
</body>
</html>
"""


initialize_system()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
