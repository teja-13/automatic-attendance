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

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install `face-recognition` package explicitly (required because imports use `face_recognition`):

Windows:

```powershell
pip install dlib-bin==19.24.6
pip install face-recognition==1.3.0 --no-deps
pip install face-recognition-models==0.3.0
```

Linux/macOS (if needed):

```bash
pip install dlib==19.24.2
pip install face-recognition==1.3.0 --no-deps
pip install face-recognition-models==0.3.0
```

## Runtime Mode (CUDA vs CPU)

By default, the app expects CUDA (`REQUIRE_CUDA=1`). On most local systems and on Render, use CPU fallback:

Windows PowerShell:

```powershell
$env:REQUIRE_CUDA="0"
```

CMD:

```cmd
set REQUIRE_CUDA=0
```

Optional CUDA DLL path settings on Windows (only if you actually use CUDA):

- `CUDA_DLL_DIR`
- `CUDA_PATH`

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

1. Set environment variable `REQUIRE_CUDA=0` (Render instances are CPU).
2. Deploy from repository root.
3. Render uses `buildCommand` and `startCommand` from `render.yaml`.

## Troubleshooting

- Error about CUDA/CNN required:
  - Set `REQUIRE_CUDA=0` before running.
- `ModuleNotFoundError: face_recognition`:
  - Run the explicit install commands for `face-recognition` shown above.
- No known faces found:
  - Add images under `known_faces/<person_name>/` and reload.
- Registration fails:
  - Use JPG/PNG with one clear front-facing face only.
