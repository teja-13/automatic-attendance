# Automatic Attendance System

This project is a face-recognition-based attendance system with two run modes:

1. Desktop webcam mode (`main.py`) for live recognition with on-screen overlays.
2. Web dashboard mode (`attendance_web.py`) for camera capture, attendance tracking, student registration, and CSV export.

## What The Project Does

- Loads known faces from the `known_faces` folder.
- Detects and recognizes people from camera frames.
- Applies attendance rule:
  - Captured in `classroom` => `Present`
  - Captured in `elsewhere` => `Absent`
- Maintains live attendance board and recent event log.
- Lets you register new students from the web UI.
- Exports attendance state to CSV.

## Project Structure

```text
attendance_web.py     # Flask web app and dashboard APIs/UI
main.py               # Desktop live webcam recognition app
render.yaml           # Render deployment config
requirements.txt      # Python dependencies
known_faces/
  bhuvan/
  teja/
```

Known-face dataset format supported:

- Preferred: `known_faces/person_name/image.jpg`
- Backward-compatible: `known_faces/person_name.jpg`

## Prerequisites

- Python 3.11 (recommended, matches Render config)
- Webcam access
- pip

## Local Setup

### 1. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Runtime Mode (CUDA vs CPU)

The web app is CPU-safe by default and does not require CUDA.

Optional runtime tuning:

Windows PowerShell:

```powershell
$env:FACE_DETECTION_MODEL="hog"
```

CMD:

```cmd
set FACE_DETECTION_MODEL=hog
```

## Run The Web Dashboard

```powershell
python attendance_web.py
```

Open: `http://127.0.0.1:5000`

### Web workflow

1. Click **Start Camera**.
2. Capture **Classroom** or **Elsewhere** snapshots.
3. Attendance board updates automatically.
4. Register student from **Student Registration** form.
5. Click **Reload Known Faces** after dataset updates (if needed).
6. Export CSV from **Export CSV**.

## Run Desktop Live Recognition

```powershell
python main.py
```

- Press `q` to quit webcam window.
- Shows recognition overlays and a session-recognized list.

## Key API Endpoints (Web App)

- `GET /api/state` - current runtime/attendance/events
- `POST /api/process` - process one camera snapshot
- `POST /api/register` - register new student image
- `POST /api/reload` - reload known faces from disk
- `POST /api/clear` - clear current attendance session
- `GET /api/export` - export CSV

## Deploy On Render

This repo already includes `render.yaml` for a Python web service with `gunicorn`.

Important for Render:

1. Use `runtime.txt` (`python-3.11.8`) to pin Python to a supported version.
2. `Procfile` uses `web: gunicorn attendance_web:app`.
3. `render.yaml` handles build/install and binds with `--bind 0.0.0.0:$PORT`.
4. Render uses CPU mode (`FACE_DETECTION_MODEL=hog`) and low thread env settings.

## Troubleshooting

- Error about CUDA/CNN required:
  - Use `FACE_DETECTION_MODEL=hog`.
- `ModuleNotFoundError: face_recognition`:
  - Re-run `pip install -r requirements.txt`.
- No known faces found:
  - Add images under `known_faces/<person_name>/` and reload.
- Registration fails:
  - Use JPG/PNG with one clear front-facing face only.
