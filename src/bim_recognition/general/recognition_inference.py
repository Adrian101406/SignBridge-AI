"""
SignBridge General Recognizer V1
================================

MCIE-facing inference module for the general BIM recognizer.

Expected model input:
    (1, 64, 300)

Feature layout per frame:
    33 pose + 21 left hand + 21 right hand = 75 landmarks
    each landmark: [x, y, z, validity]
    75 * 4 = 300 features

Main API:
    recognizer = SignBridgeGeneralRecognizer(...)
    result = recognizer.recognize_bim(video_path, top_k=3)

The module expects:
- best_zenodo_lstm.keras
- zenodo_xyz_scaling.npz
- general_label_mapping.json
- pose_landmarker.task
- hand_landmarker.task
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SignBridgeGeneralRecognizer:

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        label_mapping_path: str,
        pose_task_path: str,
        hand_task_path: str,
    ) -> None:

        required = {
            "model": model_path,
            "scaler": scaler_path,
            "label mapping": label_mapping_path,
            "pose task": pose_task_path,
            "hand task": hand_task_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required SignBridge general-recognizer files:\n"
                + "\n".join(missing)
            )

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_mapping_path = label_mapping_path
        self.pose_task_path = pose_task_path
        self.hand_task_path = hand_task_path

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False
        )

        self.mean, self.std = self._load_scaler(
            self.scaler_path
        )

        self.class_map = self._load_label_mapping(
            self.label_mapping_path
        )

        self._validate_model()

    # ========================================================
    # LOADERS
    # ========================================================

    @staticmethod
    def _load_scaler(path: str) -> Tuple[np.ndarray, np.ndarray]:

        scaler = np.load(path)

        keys = set(
            scaler.files
        )

        possible_pairs = [
            ("mean", "std"),
            ("feature_mean", "feature_std"),
            ("xyz_mean", "xyz_std"),
            ("pose_mean", "pose_std"),
        ]

        for mean_key, std_key in possible_pairs:

            if (
                mean_key in keys
                and
                std_key in keys
            ):

                mean = scaler[
                    mean_key
                ]

                std = scaler[
                    std_key
                ]

                return (
                    mean.astype(
                        np.float32
                    ),
                    std.astype(
                        np.float32
                    )
                )

        raise KeyError(
            "Could not identify mean/std arrays in scaler. "
            f"Available keys: {sorted(keys)}"
        )

    @staticmethod
    def _load_label_mapping(
        path: str
    ) -> Dict[int, Dict[str, str]]:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            raw = json.load(
                f
            )

        class_map = {}

        # ----------------------------------------------------
        # Case A:
        # {"0": "Saya", "1": "Awak", ...}
        # ----------------------------------------------------

        if isinstance(
            raw,
            dict
        ):

            numeric_key_count = sum(
                str(k).isdigit()
                for k in raw.keys()
            )

            if numeric_key_count > 0:

                for key, value in raw.items():

                    if not str(
                        key
                    ).isdigit():
                        continue

                    class_id = int(
                        key
                    )

                    if isinstance(
                        value,
                        str
                    ):

                        class_map[
                            class_id
                        ] = {
                            "gloss":
                                value,

                            "english":
                                value,
                        }

                    elif isinstance(
                        value,
                        dict
                    ):

                        gloss = (
                            value.get(
                                "malay"
                            )
                            or
                            value.get(
                                "gloss"
                            )
                            or
                            value.get(
                                "label"
                            )
                            or
                            value.get(
                                "name"
                            )
                            or
                            str(
                                class_id
                            )
                        )

                        english = (
                            value.get(
                                "english"
                            )
                            or
                            gloss
                        )

                        class_map[
                            class_id
                        ] = {
                            "gloss":
                                str(
                                    gloss
                                ),

                            "english":
                                str(
                                    english
                                ),
                        }

                if class_map:
                    return class_map

            # ------------------------------------------------
            # Case B:
            # {"Saya": 0, "Awak": 1, ...}
            # ------------------------------------------------

            numeric_value_count = sum(
                isinstance(
                    v,
                    int
                )
                for v in raw.values()
            )

            if numeric_value_count > 0:

                for gloss, class_id in raw.items():

                    if not isinstance(
                        class_id,
                        int
                    ):
                        continue

                    class_map[
                        int(
                            class_id
                        )
                    ] = {
                        "gloss":
                            str(
                                gloss
                            ),

                        "english":
                            str(
                                gloss
                            ),
                    }

                if class_map:
                    return class_map

            # ------------------------------------------------
            # Case C:
            # {"class_names": [...]} or {"labels": [...]}
            # ------------------------------------------------

            for list_key in [
                "class_names",
                "labels",
                "classes"
            ]:

                if (
                    list_key in raw
                    and
                    isinstance(
                        raw[
                            list_key
                        ],
                        list
                    )
                ):

                    for class_id, value in enumerate(
                        raw[
                            list_key
                        ]
                    ):

                        if isinstance(
                            value,
                            str
                        ):

                            class_map[
                                class_id
                            ] = {
                                "gloss":
                                    value,

                                "english":
                                    value,
                            }

                        elif isinstance(
                            value,
                            dict
                        ):

                            gloss = (
                                value.get(
                                    "malay"
                                )
                                or
                                value.get(
                                    "gloss"
                                )
                                or
                                value.get(
                                    "label"
                                )
                                or
                                value.get(
                                    "name"
                                )
                                or
                                str(
                                    class_id
                                )
                            )

                            english = (
                                value.get(
                                    "english"
                                )
                                or
                                gloss
                            )

                            class_map[
                                class_id
                            ] = {
                                "gloss":
                                    str(
                                        gloss
                                    ),

                                "english":
                                    str(
                                        english
                                    ),
                            }

                    if class_map:
                        return class_map

        # ----------------------------------------------------
        # Case D:
        # ["Saya", "Awak", ...]
        # ----------------------------------------------------

        if isinstance(
            raw,
            list
        ):

            for class_id, value in enumerate(
                raw
            ):

                if isinstance(
                    value,
                    str
                ):

                    class_map[
                        class_id
                    ] = {
                        "gloss":
                            value,

                        "english":
                            value,
                    }

                elif isinstance(
                    value,
                    dict
                ):

                    gloss = (
                        value.get(
                            "malay"
                        )
                        or
                        value.get(
                            "gloss"
                        )
                        or
                        value.get(
                            "label"
                        )
                        or
                        value.get(
                            "name"
                        )
                        or
                        str(
                            class_id
                        )
                    )

                    english = (
                        value.get(
                            "english"
                        )
                        or
                        gloss
                    )

                    class_map[
                        class_id
                    ] = {
                        "gloss":
                            str(
                                gloss
                            ),

                        "english":
                            str(
                                english
                            ),
                    }

            if class_map:
                return class_map

        raise ValueError(
            "Unsupported label_mapping.json format."
        )

    def _validate_model(
        self
    ) -> None:

        output_classes = int(
            self.model.output_shape[
                -1
            ]
        )

        if len(
            self.class_map
        ) != output_classes:

            raise ValueError(
                "Label mapping size does not match model output. "
                f"Mapping: {len(self.class_map)}, "
                f"model output: {output_classes}"
            )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    def _create_pose_landmarker(
        self
    ):

        options = (
            vision.PoseLandmarkerOptions(

                base_options=
                    python.BaseOptions(
                        model_asset_path=
                            self.pose_task_path
                    ),

                running_mode=
                    vision.RunningMode.VIDEO,

                num_poses=1,

                min_pose_detection_confidence=
                    0.5,

                min_pose_presence_confidence=
                    0.5,

                min_tracking_confidence=
                    0.5,
            )
        )

        return (
            vision.PoseLandmarker
            .create_from_options(
                options
            )
        )

    def _create_hand_landmarker(
        self
    ):

        options = (
            vision.HandLandmarkerOptions(

                base_options=
                    python.BaseOptions(
                        model_asset_path=
                            self.hand_task_path
                    ),

                running_mode=
                    vision.RunningMode.VIDEO,

                num_hands=2,

                min_hand_detection_confidence=
                    0.5,

                min_hand_presence_confidence=
                    0.5,

                min_tracking_confidence=
                    0.5,
            )
        )

        return (
            vision.HandLandmarker
            .create_from_options(
                options
            )
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_pose_array(
        result
    ) -> np.ndarray:

        output = np.zeros(
            (
                33,
                4
            ),
            dtype=np.float32
        )

        if (
            result.pose_landmarks
            and
            len(
                result.pose_landmarks
            ) > 0
        ):

            landmarks = (
                result.pose_landmarks[
                    0
                ]
            )

            for i, lm in enumerate(
                landmarks[
                    :33
                ]
            ):

                visibility = getattr(
                    lm,
                    "visibility",
                    1.0
                )

                if visibility is None:
                    visibility = 1.0

                output[
                    i
                ] = [
                    lm.x,
                    lm.y,
                    lm.z,
                    visibility,
                ]

        return output

    @staticmethod
    def _extract_hand_arrays(
        result
    ) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:

        left_hand = np.zeros(
            (
                21,
                4
            ),
            dtype=np.float32
        )

        right_hand = np.zeros(
            (
                21,
                4
            ),
            dtype=np.float32
        )

        if not result.hand_landmarks:
            return (
                left_hand,
                right_hand
            )

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            arr = np.zeros(
                (
                    21,
                    4
                ),
                dtype=np.float32
            )

            for i, lm in enumerate(
                landmarks[
                    :21
                ]
            ):

                arr[
                    i
                ] = [
                    lm.x,
                    lm.y,
                    lm.z,
                    1.0,
                ]

            handedness = None

            try:

                handedness = (
                    result.handedness[
                        hand_index
                    ][0]
                    .category_name
                )

            except Exception:
                pass

            if (
                handedness is not None
                and
                handedness.lower()
                ==
                "left"
            ):

                left_hand = arr

            elif (
                handedness is not None
                and
                handedness.lower()
                ==
                "right"
            ):

                right_hand = arr

        return (
            left_hand,
            right_hand
        )

    def _extract_frame_features(
        self,
        pose_result,
        hand_result,
    ) -> np.ndarray:

        pose = (
            self._extract_pose_array(
                pose_result
            )
        )

        left_hand, right_hand = (
            self._extract_hand_arrays(
                hand_result
            )
        )

        features = np.concatenate(
            [
                pose,
                left_hand,
                right_hand,
            ],
            axis=0
        )

        if features.shape != (
            75,
            4
        ):

            raise ValueError(
                f"Unexpected frame feature shape: "
                f"{features.shape}"
            )

        return features

    # ========================================================
    # TEMPORAL PREPROCESSING
    # ========================================================

    @staticmethod
    def _trim_to_active_sign(
        sequence: np.ndarray,
        padding_frames: int = 4,
        smooth_window: int = 5,
        activity_threshold: float = 0.4,
    ) -> np.ndarray:

        if len(
            sequence
        ) == 0:

            return sequence

        hand_validity = sequence[
            :,
            33:75,
            3
        ]

        hand_active = (
            np.max(
                hand_validity,
                axis=1
            )
            >
            0
        ).astype(
            np.float32
        )

        if np.sum(
            hand_active
        ) == 0:

            return sequence

        kernel = (
            np.ones(
                smooth_window,
                dtype=np.float32
            )
            /
            smooth_window
        )

        smoothed = np.convolve(
            hand_active,
            kernel,
            mode="same"
        )

        active_indices = np.where(
            smoothed
            >=
            activity_threshold
        )[0]

        if len(
            active_indices
        ) == 0:

            return sequence

        start = max(
            0,
            int(
                active_indices[
                    0
                ]
            )
            -
            padding_frames
        )

        end = min(
            len(
                sequence
            ),
            int(
                active_indices[
                    -1
                ]
            )
            +
            padding_frames
            +
            1
        )

        return sequence[
            start:end
        ]

    @staticmethod
    def _temporal_normalize(
        sequence: np.ndarray,
        target_frames: int = 64,
    ) -> np.ndarray:

        if len(
            sequence
        ) == 0:

            return np.zeros(
                (
                    target_frames,
                    75,
                    4
                ),
                dtype=np.float32
            )

        if len(
            sequence
        ) == 1:

            return np.repeat(
                sequence,
                target_frames,
                axis=0
            )

        old_positions = np.linspace(
            0,
            1,
            len(
                sequence
            )
        )

        new_positions = np.linspace(
            0,
            1,
            target_frames
        )

        output = np.zeros(
            (
                target_frames,
                75,
                4
            ),
            dtype=np.float32
        )

        for landmark in range(
            75
        ):

            for feature in range(
                4
            ):

                output[
                    :,
                    landmark,
                    feature
                ] = np.interp(
                    new_positions,
                    old_positions,
                    sequence[
                        :,
                        landmark,
                        feature
                    ]
                )

        return output.astype(
            np.float32
        )

    def preprocess_video(
        self,
        video_path: str,
    ) -> Tuple[
        np.ndarray,
        Dict
    ]:

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Cannot open video: "
                f"{video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if (
            fps is None
            or
            fps <= 0
            or
            np.isnan(
                fps
            )
        ):

            fps = 30.0

        sequence = []

        frame_index = 0
        last_timestamp = -1

        pose_landmarker = (
            self._create_pose_landmarker()
        )

        hand_landmarker = (
            self._create_hand_landmarker()
        )

        try:

            while True:

                success, frame = (
                    cap.read()
                )

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=
                        mp.ImageFormat.SRGB,

                    data=
                        rgb
                )

                timestamp_ms = int(
                    frame_index
                    *
                    1000.0
                    /
                    fps
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp + 1
                )

                last_timestamp = (
                    timestamp_ms
                )

                pose_result = (
                    pose_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                hand_result = (
                    hand_landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                frame_features = (
                    self._extract_frame_features(
                        pose_result,
                        hand_result,
                    )
                )

                sequence.append(
                    frame_features
                )

                frame_index += 1

        finally:

            cap.release()

            pose_landmarker.close()
            hand_landmarker.close()

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        raw_frames = len(
            sequence
        )

        if raw_frames == 0:

            raise RuntimeError(
                "No readable frames found."
            )

        trimmed = (
            self._trim_to_active_sign(
                sequence
            )
        )

        normalized = (
            self._temporal_normalize(
                trimmed,
                target_frames=64
            )
        )

        hand_presence = float(
            np.mean(
                np.max(
                    normalized[
                        :,
                        33:75,
                        3
                    ],
                    axis=1
                )
                >
                0
            )
        )

        info = {
            "raw_frames":
                int(
                    raw_frames
                ),

            "trimmed_frames":
                int(
                    len(
                        trimmed
                    )
                ),

            "hand_presence":
                hand_presence,
        }

        return (
            normalized,
            info
        )

    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    def _prepare_input(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        if sequence.shape != (
            64,
            75,
            4
        ):

            raise ValueError(
                "Expected sequence shape "
                f"(64,75,4), got "
                f"{sequence.shape}"
            )

        features = sequence.reshape(
            1,
            64,
            300
        ).astype(
            np.float32
        )

        features = (
            features
            -
            self.mean
        ) / self.std

        features = features.astype(
            np.float32
        )

        if (
            np.isnan(
                features
            ).any()
            or
            np.isinf(
                features
            ).any()
        ):

            raise ValueError(
                "General model input contains "
                "NaN or Inf."
            )

        return features

    def predict_sign(
        self,
        sequence: np.ndarray,
        top_k: int = 3,
    ) -> Tuple[
        List[
            Dict
        ],
        np.ndarray
    ]:

        if not (
            1
            <=
            top_k
            <=
            len(
                self.class_map
            )
        ):

            raise ValueError(
                "Invalid top_k."
            )

        model_input = (
            self._prepare_input(
                sequence
            )
        )

        probabilities = (
            self.model.predict(
                model_input,
                verbose=0
            )[0]
        )

        ranked_ids = np.argsort(
            probabilities
        )[::-1]

        predictions = []

        for rank, class_id in enumerate(
            ranked_ids[
                :top_k
            ],
            start=1
        ):

            class_id = int(
                class_id
            )

            label = (
                self.class_map[
                    class_id
                ]
            )

            predictions.append(
                {
                    "rank":
                        rank,

                    "class_id":
                        class_id,

                    "gloss":
                        label[
                            "gloss"
                        ],

                    "english":
                        label[
                            "english"
                        ],

                    "confidence":
                        float(
                            probabilities[
                                class_id
                            ]
                        ),
                }
            )

        return (
            predictions,
            probabilities
        )

    # ========================================================
    # MCIE-FACING API
    # ========================================================

    def recognize_bim(
        self,
        video_path: str,
        top_k: int = 3,
    ) -> Dict:

        sequence, info = (
            self.preprocess_video(
                video_path
            )
        )

        predictions, _ = (
            self.predict_sign(
                sequence,
                top_k=
                    top_k
            )
        )

        top1_confidence = (
            predictions[
                0
            ][
                "confidence"
            ]
        )

        if len(
            predictions
        ) >= 2:

            top2_confidence = (
                predictions[
                    1
                ][
                    "confidence"
                ]
            )

        else:

            top2_confidence = 0.0

        confidence_margin = (
            top1_confidence
            -
            top2_confidence
        )

        return {
            "recognizer":
                "SignBridge_General_Recognizer_V1",

            "domain":
                "general",

            "top1": {
                "class_id":
                    int(
                        predictions[
                            0
                        ][
                            "class_id"
                        ]
                    ),

                "gloss":
                    predictions[
                        0
                    ][
                        "gloss"
                    ],

                "english":
                    predictions[
                        0
                    ][
                        "english"
                    ],

                "confidence":
                    float(
                        top1_confidence
                    ),
            },

            "top_k":
                predictions,

            "confidence_margin":
                float(
                    confidence_margin
                ),

            "quality": {
                "hand_presence":
                    float(
                        info[
                            "hand_presence"
                        ]
                    ),

                "raw_frames":
                    int(
                        info[
                            "raw_frames"
                        ]
                    ),

                "trimmed_frames":
                    int(
                        info[
                            "trimmed_frames"
                        ]
                    ),
            },
        }
