import os
import sys
import platform

# =============================================================================
# POSE INFERENCE MODULE — MediaPipe + YOLOv8
# =============================================================================
# Runs both MediaPipe BlazePose (33 landmarks) and YOLOv8-Pose (17 keypoints)
# on video frames for comparative analysis.
#
# MediaPipe COMPATIBILITY NOTE:
#   MediaPipe v0.10.35 works on macOS Apple Silicon (M1–M4).
#   v1.0.x has a known Metal GPU crash in TensorsToDetectionsCalculator.
#   We pin to v0.10.35 in requirements.txt to ensure cross-platform support.
# =============================================================================

import cv2
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from typing import List, Optional

# Import our project settings — paths, joint definitions, etc.
# RESEARCH NOTE: §6.4.2 — We extract landmarks from BOTH models to compare
# how much the Gi degrades each one.
from src.config import PATHS, VIDEO_CONFIG, JOINTS_MP, JOINTS_YOLO


def run_mediapipe(frames: List[np.ndarray], image_width: int, image_height: int) -> pd.DataFrame:
    """
    Runs MediaPipe Pose inference on a list of video frames using the Tasks API.

    CONCEPT: MediaPipe is Google's AI framework for detecting body landmarks.
    Version 1.0 replaced the old 'mp.solutions.pose' style with a new
    'Tasks API'. We use the Tasks API here.

    RESEARCH NOTE: §6.4.2 — MediaPipe BlazePose extracts 33 landmarks.
    We only keep the joints defined in JOINTS_MP from config.py.

    Returns:
        DataFrame with columns [frame, joint_idx, joint_name, x, y, z, confidence, model]
        Returns an EMPTY DataFrame if MediaPipe crashes — pipeline continues with YOLOv8 only.
    """
    # CONCEPT: try/except is Python's way of handling errors gracefully.
    # Code inside 'try' runs normally. If anything crashes, execution jumps
    # to 'except' instead of aborting the whole program. This lets the pipeline
    # keep running with YOLOv8 even if MediaPipe fails unexpectedly.

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Path to the .task model file we downloaded
        model_path = PATHS["raw_data"].parent.parent / "pose_landmarker_heavy.task"

        if not model_path.exists():
            print(f"\n  ⚠️  MediaPipe model file not found at:\n     {model_path}")
            print("     Run this command to download it (29 MB):")
            print('     curl -L -o pose_landmarker_heavy.task \\')
            print('       "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"')
            return pd.DataFrame()

        # CONCEPT: BaseOptions tells MediaPipe where the model file is, and
        # which hardware to run on. We use CPU for consistent cross-platform
        # behaviour on both macOS (Apple Silicon) and Windows (CUDA).
        # On Windows, GPU delegate can be used for faster inference.
        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU  # Force CPU on Mac
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,        # Process one frame at a time
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        results_list = []
        print("  Running MediaPipe Pose inference...")

        # CONCEPT: 'with ... as landmarker' is a context manager.
        # It automatically sets up and tears down the landmarker safely,
        # even if an error occurs mid-loop.
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for frame_idx, frame in enumerate(tqdm(frames, desc="  MediaPipe")):

                # CONCEPT: OpenCV loads images in BGR (Blue-Green-Red) byte order.
                # MediaPipe expects RGB (Red-Green-Blue). We must swap the channels
                # or the AI will see wrong colours and detection accuracy will drop.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Wrap our numpy array in MediaPipe's own Image container
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Run pose estimation on this single frame
                result = landmarker.detect(mp_image)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    # result.pose_landmarks is a list (one entry per person detected).
                    # We always take index [0] — the first (and usually only) person.
                    for joint_idx, landmark in enumerate(result.pose_landmarks[0]):
                        if joint_idx in JOINTS_MP:
                            results_list.append({
                                "frame":      frame_idx,
                                "joint_idx":  joint_idx,
                                "joint_name": JOINTS_MP[joint_idx],
                                "x":          landmark.x,           # Normalised 0.0–1.0
                                "y":          landmark.y,           # Normalised 0.0–1.0
                                "z":          landmark.z,           # Depth estimate
                                # Tasks API 'visibility' tells us if the joint is occluded by the Gi
                                "confidence": getattr(landmark, "visibility", 0.0),
                                "model":      "mediapipe"
                            })
                else:
                    # No person detected in this frame — store zeros so the frame
                    # still exists in the DataFrame (prevents gaps in analysis)
                    for joint_idx, joint_name in JOINTS_MP.items():
                        results_list.append({
                            "frame":      frame_idx,
                            "joint_idx":  joint_idx,
                            "joint_name": joint_name,
                            "x":          0.0,
                            "y":          0.0,
                            "z":          0.0,
                            "confidence": 0.0,
                            "model":      "mediapipe"
                        })

        n_detected = sum(1 for r in results_list if r["confidence"] > 0)
        print(f"  ✅ MediaPipe done — {len(frames)} frames, {n_detected} detections")
        return pd.DataFrame(results_list)

    except Exception as e:
        # -------------------------------------------------------------------------
        # GRACEFUL FALLBACK
        # If MediaPipe crashes for any reason (Metal GPU bug, missing model file,
        # incompatible OS version), we print a clear warning and return an empty
        # DataFrame. The rest of the pipeline continues using YOLOv8 results only.
        #
        # RESEARCH NOTE: For the final published results, MediaPipe MUST run.
        # This fallback is only for dry-run testing on the MacBook.
        # On the ASUS ROG with CUDA + Windows, MediaPipe runs without issues.
        # -------------------------------------------------------------------------
        print(f"\n  ⚠️  MediaPipe SKIPPED — {type(e).__name__}: {e}")
        print("     Continuing with YOLOv8 results only.")
        print("     Check model file and MediaPipe installation.")
        return pd.DataFrame()   # Return empty — downstream code handles this safely


def run_yolov8(frames: List[np.ndarray], image_width: int, image_height: int) -> pd.DataFrame:
    """
    Runs YOLOv8x-Pose inference on a list of video frames.

    CONCEPT: YOLOv8 is Ultralytics' single-stage object detector.
    The '-Pose' variant also outputs 17 keypoints (COCO format) per detected person.
    It outputs pixel coordinates (not normalised), so we divide by width/height
    to convert to the same 0.0–1.0 range as MediaPipe.

    RESEARCH NOTE: §6.4.2 — YOLOv8 uses 17 COCO keypoints vs MediaPipe's 33.
    Both are mapped to the same 6 marker joints for fair comparison.

    CONCEPT: If you have an NVIDIA GPU (like the RTX 3060 in your ASUS ROG),
    YOLOv8 will automatically detect CUDA and use the GPU — no code changes needed.
    On the MacBook M4, it uses CPU (still fast enough for testing).
    """
    # Load YOLOv8x-pose model. Downloads automatically on first run (~130 MB).
    # 'x' = extra-large = highest accuracy. For real-time use, try 'yolov8n-pose.pt'.
    model = YOLO("yolov8x-pose.pt")

    results_list = []
    print("  Running YOLOv8-Pose inference...")

    for frame_idx, frame in enumerate(tqdm(frames, desc="  YOLOv8")):
        # Run inference. verbose=False stops YOLO printing stats for every frame.
        yolo_results = model(frame, verbose=False)

        # yolo_results is a list (one per image). We only sent one frame, so [0].
        r = yolo_results[0]

        # Check if any keypoints were detected
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            # r.keypoints.data shape: [num_people, 17, 3] — (x, y, confidence)
            # .cpu().numpy() moves data from GPU memory to CPU as a numpy array
            keypoints = r.keypoints.data[0].cpu().numpy()  # First person only

            for joint_idx, (x_px, y_px, conf) in enumerate(keypoints):
                if joint_idx in JOINTS_YOLO:
                    results_list.append({
                        "frame":      frame_idx,
                        "joint_idx":  joint_idx,
                        "joint_name": JOINTS_YOLO[joint_idx],
                        # Normalise pixel coords → 0.0–1.0 to match MediaPipe format
                        "x":          float(x_px) / image_width  if image_width  > 0 else 0.0,
                        "y":          float(y_px) / image_height if image_height > 0 else 0.0,
                        "z":          0.0,          # YOLOv8 is 2D only (no depth)
                        "confidence": float(conf),
                        "model":      "yolov8"
                    })
        else:
            # No person detected — fill zeros to maintain complete frame coverage
            for joint_idx, joint_name in JOINTS_YOLO.items():
                results_list.append({
                    "frame":      frame_idx,
                    "joint_idx":  joint_idx,
                    "joint_name": joint_name,
                    "x":          0.0,
                    "y":          0.0,
                    "z":          0.0,
                    "confidence": 0.0,
                    "model":      "yolov8"
                })

    n_detected = sum(1 for r in results_list if r["confidence"] > 0)
    print(f"  ✅ YOLOv8 done — {len(frames)} frames, {n_detected} detections")
    return pd.DataFrame(results_list)


def run_inference(video_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main entry point: opens a video, extracts frames, runs both models.

    RESEARCH NOTE: §6.4.2 — Both models run on the same frames from the same
    video to ensure a perfectly fair comparison.

    Returns:
        Combined DataFrame with results from both MediaPipe and YOLOv8.
    """
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    image_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames into memory as a list of numpy arrays
    frames = []
    print(f"  Reading frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Extracted {len(frames)} frames. Dimensions: {image_width}x{image_height}")

    # Run both models on the exact same frames
    df_mp   = run_mediapipe(frames, image_width, image_height)
    df_yolo = run_yolov8(frames, image_width, image_height)

    # Combine results. pd.concat stacks DataFrames vertically.
    # ignore_index=True resets the row numbers after combining.
    dfs_to_combine = [df for df in [df_mp, df_yolo] if not df.empty]
    if not dfs_to_combine:
        raise RuntimeError("Both MediaPipe and YOLOv8 returned no data!")
    df_combined = pd.concat(dfs_to_combine, ignore_index=True)

    if output_csv_path:
        save_landmarks_csv(df_combined, output_csv_path)

    return df_combined


def save_landmarks_csv(df: pd.DataFrame, output_path: str):
    """Saves landmark DataFrame to CSV."""
    print(f"  💾 Saving landmarks to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(df)} rows.")


def load_landmarks_csv(csv_path: str) -> pd.DataFrame:
    """Loads a previously saved landmarks CSV."""
    print(f"  📂 Loading landmarks from: {csv_path}")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    # Quick standalone test
    sample_video = str(PATHS["sample_data"] / "test_oizuki.mp4")
    output_csv   = str(PATHS["landmarks"]   / "test_oizuki_landmarks.csv")
    df = run_inference(sample_video, output_csv)
    print(f"\nDone! Shape: {df.shape}")
    print(df.head())
