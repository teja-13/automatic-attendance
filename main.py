import os
import collections
import hashlib

import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def configure_optional_cuda_dll_path():
    if os.name != "nt":
        return

    candidate_dirs = []
    env_cuda_dll_dir = os.environ.get("CUDA_DLL_DIR")
    if env_cuda_dll_dir:
        candidate_dirs.append(env_cuda_dll_dir)

    env_cuda_path = os.environ.get("CUDA_PATH")
    if env_cuda_path:
        candidate_dirs.append(os.path.join(env_cuda_path, "bin"))

    candidate_dirs.append(r"E:\CUDA_tools\bin")

    for dll_dir in candidate_dirs:
        if dll_dir and os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            print(f"[INFO] Added CUDA DLL directory: {dll_dir}")
            return

    print("[INFO] CUDA DLL directory not found. Continuing with default library paths.")


def get_runtime_config():
    config = {
        "gpu_available": False,
        "gpu_devices": 0,
        "detection_model": "hog",
        "enrollment_scale": 0.5,
        "live_scale": 0.25,
        "label": "CPU",
    }

    try:
        cuda_enabled_in_dlib = bool(getattr(dlib, "DLIB_USE_CUDA", False))
        gpu_devices = dlib.cuda.get_num_devices() if hasattr(dlib, "cuda") else 0

        if cuda_enabled_in_dlib and gpu_devices > 0:
            config.update(
                gpu_available=True,
                gpu_devices=gpu_devices,
                detection_model="cnn",
                enrollment_scale=0.25,
                live_scale=0.5,
                label=f"GPU/CUDA ({gpu_devices} device(s))",
            )
    except Exception as exc:
        print(f"[WARN] CUDA probe failed: {exc}")

    return config

def get_color_for_name(name):
    if name == "Unknown": return (255, 200, 0) 
    hex_dig = hashlib.sha256(name.encode()).hexdigest()
    return (int(hex_dig[0:2], 16), int(hex_dig[2:4], 16), int(hex_dig[4:6], 16))

def load_known_faces(runtime_config, known_faces_dir="known_faces"):
    known_face_encodings = []
    known_face_names = []
    detection_model = runtime_config["detection_model"]
    enrollment_scale = runtime_config["enrollment_scale"]
    image_extensions = (".jpg", ".jpeg", ".png")
    
    print(f"--- FACE ENCODING START [{runtime_config['label']}] ---")
    if not os.path.exists(known_faces_dir): 
        os.makedirs(known_faces_dir)
        return [], []

    image_paths = []
    for root, _, files in os.walk(known_faces_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, filename))

    if not image_paths:
        print(f"[INFO] No images found in {known_faces_dir}. Add .jpg/.jpeg/.png files to continue.")
        print(f"--- Pre-processing Complete: {len(known_face_names)} faces ---\n")
        return known_face_encodings, known_face_names

    for image_path in sorted(image_paths):
        relative_path = os.path.relpath(image_path, known_faces_dir)
        path_parts = relative_path.split(os.sep)
        if len(path_parts) > 1:
            # Folder-based dataset: known_faces/person_name/image.jpg
            person_name = path_parts[0].replace("_", " ").title()
        else:
            # Backward-compatible flat dataset: known_faces/person_name.jpg
            person_name = os.path.splitext(path_parts[0])[0].replace("_", " ").title()

        display_path = relative_path.replace("\\", "/")

        try:
            img = face_recognition.load_image_file(image_path)

            small_img = cv2.resize(img, (0, 0), fx=enrollment_scale, fy=enrollment_scale)

            tmp_locations = face_recognition.face_locations(small_img, model=detection_model)

            if enrollment_scale != 1.0:
                scale_back = 1.0 / enrollment_scale
                face_locations = [
                    (int(t * scale_back), int(r * scale_back), int(b * scale_back), int(l * scale_back))
                    for (t, r, b, l) in tmp_locations
                ]
            else:
                face_locations = tmp_locations

            encodings = face_recognition.face_encodings(img, face_locations, model="large")

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                print(f"[ENCODED] {person_name} <- {display_path}")
            elif detection_model != "hog":
                print(f"[RETRY] {display_path}: trying HOG fallback...")
                face_locations = face_recognition.face_locations(img, model="hog")
                encodings = face_recognition.face_encodings(img, face_locations, model="large")
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
                    print(f"[ENCODED] {person_name} (HOG fallback) <- {display_path}")
                else:
                    print(f"[SKIPPED] No usable face found in {display_path}")
            else:
                print(f"[SKIPPED] No usable face found in {display_path}")

        except Exception as e:
            print(f"[ERROR] {display_path}: {e}")
                
    print(f"--- Pre-processing Complete: {len(known_face_names)} faces ---\n")
    return known_face_encodings, known_face_names

def draw_fancy_box(draw, top, right, bottom, left, color, name, percentage):
    line_thickness, corner_length = 3, 20 
    draw.line([(left, top), (left + corner_length, top)], fill=color, width=line_thickness)
    draw.line([(left, top), (left, top + corner_length)], fill=color, width=line_thickness)
    draw.line([(right - corner_length, top), (right, top)], fill=color, width=line_thickness)
    draw.line([(right, top), (right, top + corner_length)], fill=color, width=line_thickness)
    draw.line([(left, bottom - corner_length), (left, bottom)], fill=color, width=line_thickness)
    draw.line([(left, bottom), (left + corner_length, bottom)], fill=color, width=line_thickness)
    draw.line([(right - corner_length, bottom), (right, bottom)], fill=color, width=line_thickness)
    draw.line([(right, bottom - corner_length), (right, bottom)], fill=color, width=line_thickness)

    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    display_text = f"{name}" + (f" ({percentage:.0f}%)" if percentage else "")
    text_bbox = draw.textbbox((0,0), display_text, font=font)
    text_w, text_h = text_bbox[2]-text_bbox[0], text_bbox[3]-text_bbox[1]
    
    label_bg = Image.new('RGBA', (text_w + 10, text_h + 10), color + (128,))
    ImageDraw.Draw(label_bg).text((5, 5), display_text, font=font, fill=(255, 255, 255, 255))
    return left + (right - left - (text_w + 10)) // 2, bottom + 5, label_bg

def run_live_face_recognition(known_face_encodings, known_face_names, runtime_config):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    detection_model = runtime_config["detection_model"]
    live_scale = runtime_config["live_scale"]
    scale_back = 1.0 / live_scale

    TOLERANCE = 0.48  
    
    name_occurrence_count = collections.defaultdict(int)
    recognized_in_session = set()
    REQUIRED_STABILITY = 10 

    print(f"LIVE FEED ACTIVE [{runtime_config['label']}] with {detection_model.upper()} detector.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=live_scale, fy=live_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model=detection_model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        labels_to_paste, current_frame_names = [], []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = (
                int(top * scale_back),
                int(right * scale_back),
                int(bottom * scale_back),
                int(left * scale_back),
            )
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name, percentage = "Unknown", None

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                distance = face_distances[best_match_index]
                
                if distance <= TOLERANCE:
                    name = known_face_names[best_match_index]
                    percentage = (1.0 - distance) * 100
                    current_frame_names.append(name)
                    
                    name_occurrence_count[name] += 1
                    if name_occurrence_count[name] >= REQUIRED_STABILITY:
                        if name not in recognized_in_session:
                            print(f"[LOGGED] {name} verified.")
                            recognized_in_session.add(name)

            label_info = draw_fancy_box(draw, top, right, bottom, left, get_color_for_name(name), name, percentage)
            labels_to_paste.append(label_info)
        
        for name in list(name_occurrence_count.keys()):
            if name not in current_frame_names and name not in recognized_in_session:
                name_occurrence_count[name] = max(0, name_occurrence_count[name] - 1)

        for x, y, label_img in labels_to_paste:
            pil_image.paste(label_img, (x, y), label_img)

        # UI: Session List
        log_y = 10
        try: font = ImageFont.truetype("arial.ttf", 18)
        except: font = ImageFont.load_default()
        
        draw.text((10, log_y), "Recognized in session:", font=font, fill=(255,255,255))
        for person_name in sorted(list(recognized_in_session)):
            log_y += 25
            draw.rectangle([(10, log_y + 4), (25, log_y + 19)], fill=get_color_for_name(person_name))
            draw.text((35, log_y), person_name, font=font, fill=get_color_for_name(person_name))

        result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Face Recognition - {runtime_config['label']}", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    configure_optional_cuda_dll_path()
    runtime_config = get_runtime_config()
    print(
        f"[INFO] Runtime selected: {runtime_config['label']} "
        f"(model={runtime_config['detection_model']}, live_scale={runtime_config['live_scale']})"
    )

    known_encodings, known_names = load_known_faces(runtime_config, "known_faces")
    if not known_encodings:
        print("[INFO] No known faces enrolled. Starting in detection-only mode.")

    run_live_face_recognition(known_encodings, known_names, runtime_config)