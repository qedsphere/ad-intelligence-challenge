from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame

SAMPLE_STEP = 5


def _detect_faces(img_bgr) -> Tuple[int, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    h, w = gray.shape[:2]
    max_area_ratio = 0.0
    for (x, y, fw, fh) in faces:
        ar = (fw * fh) / float(max(1, w * h))
        if ar > max_area_ratio:
            max_area_ratio = ar
    return len(faces), max_area_ratio


def compute(frames: List[Frame]) -> Dict[str, Any]:
    t_first_face = None
    max_face_area_ratio = 0.0
    face_frames = 0
    samples = 0
    for i in range(0, len(frames), max(1, SAMPLE_STEP)):
        samples += 1
        f = frames[i]
        count, area = _detect_faces(f.img)
        if count > 0:
            face_frames += 1
            if t_first_face is None:
                t_first_face = f.t
            if area > max_face_area_ratio:
                max_face_area_ratio = area
    face_present = face_frames > 0
    face_time_ratio = float(face_frames) / float(max(1, samples))
    return {
        "face_present": bool(face_present),
        "t_first_face": t_first_face,
        "max_face_area_ratio": float(max_face_area_ratio),
        "face_time_ratio": float(face_time_ratio),
    }
