"""
Run MediaPipe FaceLandmarker over every image in a person's folder.

Works on any folder of face-crop images — a train/val person folder
(data/raw/clips_videos/<split>/<video_id>/<entity_id>/) or a flat eval folder
that has no video/person naming convention at all; the folder is only ever
used as-is, never parsed for structure.

For each input folder this writes one self-contained output folder under
LANDMARKS_DIR (config.py, project-root-level "landmarks/"), named after the
input folder itself:
    <LANDMARKS_DIR>/<person_id>/
        landmarks.json
        annotated/
            <image_name>.jpg  (one per input image)

  - annotated/ holds one annotated copy of every input image (raw copy-through
    if no face was detected — never resized or upscaled to help detection)
  - landmarks.json holds the raw face landmarks, mouth-only landmarks, and
    Mouth Aspect Ratio (MAR) for every image, null where no face was found

Usage:
    python scripts/features/extract_landmarks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image as PILImage
from PIL import ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ANNOTATED_SUBDIR_NAME,
    ANNOTATION_MESH_COLOR,
    ANNOTATION_MOUTH_COLOR,
    ANNOTATION_POINT_RADIUS,
    EXTRACT_LANDMARKS_PERSON_DIR,
    FACE_LANDMARKER_MODEL_PATH,
    IMAGE_EXTENSIONS,
    JSON_INDENT,
    LANDMARKS_DIR,
    LANDMARKS_JSON_FILENAME,
    MAR_HORIZONTAL_PAIR,
    MAR_VERTICAL_PAIRS,
    MEDIAPIPE_DELEGATE,
    MEDIAPIPE_MIN_FACE_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_FACE_PRESENCE_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_NUM_FACES,
    MOUTH_LANDMARK_INDICES,
    NORMALIZATION_CENTER_INDEX,
    NORMALIZATION_SCALE_PAIR,
    NORMALIZE_LANDMARKS,
    RAW_CLIPS_VIDEOS_DIR,
    SAVE_LANDMARKS_JSON,
)


class PersonLandmarkExtractor:
    """Detects face + mouth landmarks and MAR for every image in a folder,
    and writes annotated images plus a landmarks.json into one output folder.
    """

    _DELEGATES = {"CPU": BaseOptions.Delegate.CPU, "GPU": BaseOptions.Delegate.GPU}

    def __init__(
        self,
        model_path: Path = FACE_LANDMARKER_MODEL_PATH,
        num_faces: int = MEDIAPIPE_NUM_FACES,
        min_face_detection_confidence: float = MEDIAPIPE_MIN_FACE_DETECTION_CONFIDENCE,
        min_face_presence_confidence: float = MEDIAPIPE_MIN_FACE_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        delegate: str = MEDIAPIPE_DELEGATE,
        mouth_indices: list[int] = MOUTH_LANDMARK_INDICES,
        mar_vertical_pairs: list[tuple[int, int]] = MAR_VERTICAL_PAIRS,
        mar_horizontal_pair: tuple[int, int] = MAR_HORIZONTAL_PAIR,
        image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
        landmarks_dir: Path = LANDMARKS_DIR,
        save_annotated: bool = True,
        save_json: bool = SAVE_LANDMARKS_JSON,
        normalize: bool = NORMALIZE_LANDMARKS,
        normalization_center_index: int = NORMALIZATION_CENTER_INDEX,
        normalization_scale_pair: tuple[int, int] = NORMALIZATION_SCALE_PAIR,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            if model_path == FACE_LANDMARKER_MODEL_PATH:
                from download_model import download_if_missing

                download_if_missing()
            else:
                raise FileNotFoundError(f"FaceLandmarker model not found: {model_path}")
        if delegate not in self._DELEGATES:
            raise ValueError(f"delegate must be one of {sorted(self._DELEGATES)}, got {delegate!r}")

        self.mouth_indices = mouth_indices
        self.mar_vertical_pairs = mar_vertical_pairs
        self.mar_horizontal_pair = mar_horizontal_pair
        self.image_extensions = image_extensions
        self.landmarks_dir = Path(landmarks_dir)
        self.save_annotated = save_annotated
        self.save_json = save_json
        self.normalize = normalize
        self.normalization_center_index = normalization_center_index
        self.normalization_scale_pair = normalization_scale_pair

        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=self._DELEGATES[delegate],
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=num_faces,
                min_face_detection_confidence=min_face_detection_confidence,
                min_face_presence_confidence=min_face_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    # ---- public API ----

    def process_person(
        self,
        person_dir: Path,
        person_id: str | None = None,
        output_dir: Path | None = None,
    ) -> dict:
        """Extract landmarks for every image in person_dir. Writes an output
        folder (annotated images and/or landmarks.json, whichever is enabled)
        only if save_annotated or save_json is set — otherwise this is a pure
        in-memory extraction with no disk output for the person. Returns the
        same dict that would be written to landmarks.json.
        """
        person_dir = Path(person_dir)
        person_id = person_id or person_dir.name

        needs_output_dir = self.save_annotated or self.save_json
        output_dir = None
        annotated_dir = None
        if needs_output_dir:
            output_dir = Path(output_dir) if output_dir is not None else self.landmarks_dir / person_id
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_annotated:
                annotated_dir = output_dir / ANNOTATED_SUBDIR_NAME
                annotated_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for image_path in self._list_images(person_dir):
            image, raw_record = self._extract_image(image_path)
            if self.save_annotated:
                self._save_annotated_image(image, raw_record, annotated_dir / image_path.name)
            records.append(self._normalize_record(raw_record) if self.normalize else raw_record)

        person_record = {
            "person_id": person_id,
            "folder": str(person_dir),
            "normalized": self.normalize,
            "images": records,
        }
        if self.save_json:
            self._write_json(person_record, output_dir / LANDMARKS_JSON_FILENAME)
        return person_record

    def extract_image(self, image_path: Path) -> dict:
        """Run detection on a single image and return its record only
        (no file output) — for ad-hoc use outside process_person.
        """
        _, raw_record = self._extract_image(Path(image_path))
        return self._normalize_record(raw_record) if self.normalize else raw_record

    # ---- internals ----

    def _list_images(self, folder: Path) -> list[Path]:
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in self.image_extensions)

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        return np.array(PILImage.open(image_path).convert("RGB"))

    def _detect(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def _extract_image(self, image_path: Path) -> tuple[np.ndarray, dict]:
        image = self._load_image(image_path)
        landmarks = self._detect(image)

        if landmarks is None:
            return image, {
                "image_name": image_path.name,
                "detected": False,
                "face_landmarks": None,
                "mouth_landmarks": None,
                "mar": None,
            }

        h, w = image.shape[:2]
        face_xyz = self._landmarks_to_list(landmarks, w, h)
        mouth_xyz = [face_xyz[i] for i in self.mouth_indices]
        mar = self._compute_mar(landmarks, w, h)

        return image, {
            "image_name": image_path.name,
            "detected": True,
            "face_landmarks": face_xyz,
            "mouth_landmarks": mouth_xyz,
            "mar": mar,
        }

    @staticmethod
    def _landmarks_to_list(landmarks, w: int, h: int) -> list[list[float]]:
        return [[p.x * w, p.y * h, p.z * w] for p in landmarks]

    def _compute_mar(self, landmarks, w: int, h: int) -> float:
        def dist(i: int, j: int) -> float:
            a, b = landmarks[i], landmarks[j]
            return float(np.hypot((a.x - b.x) * w, (a.y - b.y) * h))

        vertical = float(np.mean([dist(i, j) for i, j in self.mar_vertical_pairs]))
        horizontal = dist(*self.mar_horizontal_pair)
        return vertical / horizontal if horizontal > 0 else 0.0

    def _normalize_record(self, record: dict) -> dict:
        """Center face/mouth landmarks on the nose tip and scale by interocular
        distance. mar is left untouched (already scale-invariant). No-op when
        no face was detected.
        """
        if not record["detected"]:
            return record

        face = record["face_landmarks"]
        cx, cy, cz = face[self.normalization_center_index]
        i, j = self.normalization_scale_pair
        scale = float(np.hypot(face[i][0] - face[j][0], face[i][1] - face[j][1]))
        if scale == 0:
            return record

        def normalize_points(points: list[list[float]]) -> list[list[float]]:
            return [[(x - cx) / scale, (y - cy) / scale, (z - cz) / scale] for x, y, z in points]

        return {
            **record,
            "face_landmarks": normalize_points(face),
            "mouth_landmarks": normalize_points(record["mouth_landmarks"]),
        }

    @staticmethod
    def _save_annotated_image(image: np.ndarray, record: dict, out_path: Path) -> None:
        pil_image = PILImage.fromarray(image)
        if record["detected"]:
            draw = ImageDraw.Draw(pil_image)
            r = ANNOTATION_POINT_RADIUS
            for x, y, _ in record["face_landmarks"]:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=ANNOTATION_MESH_COLOR)
            for x, y, _ in record["mouth_landmarks"]:
                draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=ANNOTATION_MOUTH_COLOR)
        pil_image.save(out_path)

    @staticmethod
    def _write_json(person_record: dict, out_path: Path) -> None:
        with open(out_path, "w") as f:
            json.dump(person_record, f, indent=JSON_INDENT)


if __name__ == "__main__":
    person_dir = (
        Path(EXTRACT_LANDMARKS_PERSON_DIR)
        if EXTRACT_LANDMARKS_PERSON_DIR
        else next((RAW_CLIPS_VIDEOS_DIR / "train").glob("*/*"))
    )

    extractor = PersonLandmarkExtractor()
    result = extractor.process_person(person_dir)

    total = len(result["images"])
    detected = sum(r["detected"] for r in result["images"])
    print(f"[{result['person_id']}] {detected}/{total} images with a detected face")
    print(f"output written to: {LANDMARKS_DIR / result['person_id']}")

