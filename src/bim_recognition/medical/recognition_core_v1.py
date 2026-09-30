"""
SignBridge Medical Recognizer V1
MCIE-facing inference module for the balanced 50/50 + relational geometry model.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CLASS_MAP = {
    0: ("Badan","Body"), 1: ("Bahu","Shoulder"), 2: ("Jururawat","Nurse"),
    3: ("Jari","Fingers"), 4: ("Kaki","Feet"), 5: ("Kulit","Skin"),
    6: ("Leher","Neck"), 7: ("Doktor","Doctor"), 8: ("Bibir","Lips"),
    9: ("Gigi","Teeth"), 10: ("Hidung","Nose"), 11: ("Lidah","Tongue"),
    12: ("Mata","Eyes"), 13: ("Mulut","Mouth"), 14: ("Lemah","Weak"),
    15: ("Tulang","Bones"), 16: ("Sinar-X","X-ray"), 17: ("Tulang Rusuk","Rib Cage"),
    18: ("Darah","Blood"), 19: ("Jantung","Heart"), 20: ("Kesakitan","Pain"),
    21: ("Buah Pinggang","Kidney"), 22: ("Asma","Asthma"), 23: ("Batuk","Cough"),
    24: ("Bengkak","Swollen"), 25: ("Gatal-Gatal","Itchy"), 26: ("Muntah","Vomit"),
    27: ("Selesema","Flu"), 28: ("Demam","Fever"), 29: ("Sakit Kepala","Headache"),
    30: ("Sakit Perut","Stomach pain"), 31: ("Sakit Pinggang","Waist pain"),
    32: ("Sakit Tekak","Sore throat"), 33: ("Bedah","Surgery"), 34: ("Kecemasan","Emergency"),
    35: ("Krim","Cream"), 36: ("Pil","Pill"), 37: ("Mengandung","Pregnant"),
    38: ("Penat","Tired"), 39: ("Perut","Stomach"),
}

FACE_INDICES = [33,133,362,263,70,300,1,4,61,291,13,14,152,234,454]
SELECTED_HAND_POINTS = [0,4,8,12,16,20]


class SignBridgeMedicalRecognizer:
    def __init__(self, model_path, scaler_path, pose_task_path, hand_task_path, face_task_path):
        for p in [model_path, scaler_path, pose_task_path, hand_task_path, face_task_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.pose_task_path = pose_task_path
        self.hand_task_path = hand_task_path
        self.face_task_path = face_task_path

        self.model = tf.keras.models.load_model(model_path, compile=False)
        scaler = np.load(scaler_path)

        self.pose_mean = scaler["pose_mean"]
        self.pose_std = scaler["pose_std"]
        self.face_mean = scaler["face_mean"]
        self.face_std = scaler["face_std"]
        self.geo_mean = scaler["geo_mean"]
        self.geo_std = scaler["geo_std"]

    def _create_pose_landmarker(self):
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.pose_task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def _create_hand_landmarker(self):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.hand_task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def _create_face_landmarker(self):
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.face_task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
        )

    @staticmethod
    def _extract_pose_array(result):
        out = np.zeros((33,4), dtype=np.float32)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            for i, lm in enumerate(result.pose_landmarks[0][:33]):
                vis = getattr(lm, "visibility", 1.0)
                if vis is None:
                    vis = 1.0
                out[i] = [lm.x, lm.y, lm.z, vis]
        return out

    @staticmethod
    def _extract_hand_arrays(result):
        left = np.zeros((21,4), dtype=np.float32)
        right = np.zeros((21,4), dtype=np.float32)

        if not result.hand_landmarks:
            return left, right

        for idx, landmarks in enumerate(result.hand_landmarks):
            arr = np.zeros((21,4), dtype=np.float32)
            for i, lm in enumerate(landmarks[:21]):
                arr[i] = [lm.x, lm.y, lm.z, 1.0]

            handedness = None
            try:
                handedness = result.handedness[idx][0].category_name
            except Exception:
                pass

            if handedness and handedness.lower() == "left":
                left = arr
            elif handedness and handedness.lower() == "right":
                right = arr

        return left, right

    @staticmethod
    def _extract_face_array(result):
        out = np.zeros((15,4), dtype=np.float32)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            lmks = result.face_landmarks[0]
            for i, landmark_id in enumerate(FACE_INDICES):
                lm = lmks[landmark_id]
                out[i] = [lm.x, lm.y, lm.z, 1.0]
        return out

    def _extract_frame_features(self, pose_result, hand_result, face_result):
        pose = self._extract_pose_array(pose_result)
        left, right = self._extract_hand_arrays(hand_result)
        face = self._extract_face_array(face_result)
        out = np.concatenate([pose, left, right, face], axis=0)
        if out.shape != (90,4):
            raise ValueError(f"Unexpected frame shape {out.shape}")
        return out

    @staticmethod
    def _trim_to_active_sign(sequence, padding_frames=4, smooth_window=5, activity_threshold=0.4):
        if len(sequence) == 0:
            return sequence

        hand_validity = sequence[:,33:75,3]
        hand_active = (np.max(hand_validity, axis=1) > 0).astype(np.float32)

        if np.sum(hand_active) == 0:
            return sequence

        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        smoothed = np.convolve(hand_active, kernel, mode="same")
        active = np.where(smoothed >= activity_threshold)[0]

        if len(active) == 0:
            return sequence

        start = max(0, int(active[0]) - padding_frames)
        end = min(len(sequence), int(active[-1]) + padding_frames + 1)
        return sequence[start:end]

    @staticmethod
    def _temporal_normalize(sequence, target_frames=64):
        if len(sequence) == 0:
            return np.zeros((target_frames,90,4), dtype=np.float32)

        if len(sequence) == 1:
            return np.repeat(sequence, target_frames, axis=0)

        old_pos = np.linspace(0,1,len(sequence))
        new_pos = np.linspace(0,1,target_frames)
        out = np.zeros((target_frames,90,4), dtype=np.float32)

        for landmark in range(90):
            for feature in range(4):
                out[:,landmark,feature] = np.interp(
                    new_pos,
                    old_pos,
                    sequence[:,landmark,feature]
                )

        return out.astype(np.float32)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 30.0

        sequence = []
        frame_index = 0
        last_timestamp = -1

        pose_landmarker = self._create_pose_landmarker()
        hand_landmarker = self._create_hand_landmarker()
        face_landmarker = self._create_face_landmarker()

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(frame_index * 1000.0 / fps)
                timestamp_ms = max(timestamp_ms, last_timestamp + 1)
                last_timestamp = timestamp_ms

                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                sequence.append(
                    self._extract_frame_features(
                        pose_result, hand_result, face_result
                    )
                )
                frame_index += 1

        finally:
            cap.release()
            pose_landmarker.close()
            hand_landmarker.close()
            face_landmarker.close()

        sequence = np.asarray(sequence, dtype=np.float32)
        raw_frames = len(sequence)

        if raw_frames == 0:
            raise RuntimeError("No readable frames found.")

        trimmed = self._trim_to_active_sign(sequence)
        normalized = self._temporal_normalize(trimmed, target_frames=64)

        hand_presence = float(
            np.mean(
                np.max(normalized[:,33:75,3], axis=1) > 0
            )
        )
        face_presence = float(
            np.mean(
                np.max(normalized[:,75:90,3], axis=1) > 0
            )
        )

        info = {
            "raw_frames": int(raw_frames),
            "trimmed_frames": int(len(trimmed)),
            "hand_presence": hand_presence,
            "face_presence": face_presence,
        }

        return normalized, info

    @staticmethod
    def _build_face_anchors(face):
        xyz = face[...,:3]
        valid = face[...,3]

        left_eye_xyz = (xyz[...,0,:] + xyz[...,1,:]) / 2.0
        left_eye_valid = ((valid[...,0] > 0) & (valid[...,1] > 0)).astype(np.float32)

        right_eye_xyz = (xyz[...,2,:] + xyz[...,3,:]) / 2.0
        right_eye_valid = ((valid[...,2] > 0) & (valid[...,3] > 0)).astype(np.float32)

        nose_xyz = (xyz[...,6,:] + xyz[...,7,:]) / 2.0
        nose_valid = ((valid[...,6] > 0) & (valid[...,7] > 0)).astype(np.float32)

        mouth_xyz = (
            xyz[...,8,:] + xyz[...,9,:] + xyz[...,10,:] + xyz[...,11,:]
        ) / 4.0
        mouth_valid = (
            (valid[...,8] > 0)
            & (valid[...,9] > 0)
            & (valid[...,10] > 0)
            & (valid[...,11] > 0)
        ).astype(np.float32)

        chin_xyz = xyz[...,12,:]
        chin_valid = (valid[...,12] > 0).astype(np.float32)

        anchor_xyz = np.stack(
            [left_eye_xyz, right_eye_xyz, nose_xyz, mouth_xyz, chin_xyz],
            axis=-2
        )
        anchor_valid = np.stack(
            [left_eye_valid, right_eye_valid, nose_valid, mouth_valid, chin_valid],
            axis=-1
        )

        return anchor_xyz.astype(np.float32), anchor_valid.astype(np.float32)

    @staticmethod
    def _hand_face_geometry(hand, anchor_xyz, anchor_valid):
        selected = hand[...,SELECTED_HAND_POINTS,:]
        hand_xyz = selected[...,:3]
        hand_valid = (selected[...,3] > 0).astype(np.float32)

        vectors = hand_xyz[..., :, None, :] - anchor_xyz[..., None, :, :]
        distances = np.linalg.norm(vectors, axis=-1, keepdims=True)

        pair_valid = (
            hand_valid[..., :, None]
            * anchor_valid[..., None, :]
        )[...,None]

        vectors = vectors * pair_valid
        distances = distances * pair_valid

        features = np.concatenate([vectors, distances], axis=-1)
        return features.reshape(*features.shape[:-3], 120).astype(np.float32)

    def _extract_geometry(self, sequence):
        if sequence.shape != (64,90,4):
            raise ValueError(f"Expected (64,90,4), got {sequence.shape}")

        left = sequence[:,33:54,:]
        right = sequence[:,54:75,:]
        face = sequence[:,75:90,:]

        anchor_xyz, anchor_valid = self._build_face_anchors(face)

        left_geo = self._hand_face_geometry(left, anchor_xyz, anchor_valid)
        right_geo = self._hand_face_geometry(right, anchor_xyz, anchor_valid)

        geometry = np.concatenate([left_geo, right_geo], axis=-1)

        if geometry.shape != (64,240):
            raise ValueError(f"Unexpected geometry shape {geometry.shape}")

        return geometry.astype(np.float32)

    def _prepare_model_inputs(self, sequence):
        pose = sequence[:,:75,:].reshape(1,64,300)
        face = sequence[:,75:90,:].reshape(1,64,60)
        geometry = self._extract_geometry(sequence).reshape(1,64,240)

        pose = ((pose - self.pose_mean) / self.pose_std).astype(np.float32)
        face = ((face - self.face_mean) / self.face_std).astype(np.float32)
        geometry = ((geometry - self.geo_mean) / self.geo_std).astype(np.float32)

        for name, arr in {
            "pose": pose,
            "face": face,
            "geometry": geometry
        }.items():
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"{name} input contains NaN or Inf")

        return pose, face, geometry

    def predict_sign(self, sequence, top_k=3):
        pose_input, face_input, geometry_input = self._prepare_model_inputs(sequence)

        probabilities = self.model.predict(
            {
                "pose_hand_input": pose_input,
                "face_input": face_input,
                "geometry_input": geometry_input
            },
            verbose=0
        )[0]

        ranked_ids = np.argsort(probabilities)[::-1]
        predictions = []

        for rank, class_id in enumerate(ranked_ids[:top_k], start=1):
            class_id = int(class_id)
            malay, english = CLASS_MAP[class_id]
            predictions.append(
                {
                    "rank": rank,
                    "class_id": class_id,
                    "gloss": malay,
                    "english": english,
                    "confidence": float(probabilities[class_id]),
                }
            )

        return predictions, probabilities

    def recognize_bim(self, video_path, top_k=3):
        sequence, info = self.preprocess_video(video_path)
        predictions, _ = self.predict_sign(sequence, top_k=top_k)

        top1_conf = predictions[0]["confidence"]
        top2_conf = predictions[1]["confidence"] if len(predictions) >= 2 else 0.0

        return {
            "recognizer": "SignBridge_Medical_Recognizer_V1",
            "top1": {
                "class_id": int(predictions[0]["class_id"]),
                "gloss": predictions[0]["gloss"],
                "english": predictions[0]["english"],
                "confidence": float(top1_conf),
            },
            "top_k": predictions,
            "confidence_margin": float(top1_conf - top2_conf),
            "quality": {
                "hand_presence": float(info["hand_presence"]),
                "face_presence": float(info["face_presence"]),
                "raw_frames": int(info["raw_frames"]),
                "trimmed_frames": int(info["trimmed_frames"]),
            },
        }
