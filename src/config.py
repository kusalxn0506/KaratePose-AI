# =============================================================================
# KaratePose-AI — Central Configuration
# =============================================================================
# PURPOSE:
#   This file is the SINGLE SOURCE OF TRUTH for every setting in the project.
#   Instead of scattering magic numbers and file paths across multiple files,
#   we define EVERYTHING here. Every other module imports from this file.
#
# WHY THIS MATTERS:
#   - Change a path once here → it updates everywhere automatically
#   - All research parameters (thresholds, FPS, etc.) are documented in one place
#   - Reviewers/supervisors can see exactly what settings you used
#
# HOW TO USE:
#   from src.config import CONFIG, JOINTS_MP, JOINTS_YOLO, DE_LEVA_WEIGHTS
# =============================================================================

from pathlib import Path  # Python's built-in way to handle file paths cross-platform


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
# Path.resolve() converts relative paths to absolute paths.
# Path(__file__) refers to THIS file (config.py), and .parent goes up one folder.

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # The "Karate Pose AI" folder

PATHS = {
    # --- Input Data ---
    "sample_data":      PROJECT_ROOT / "data" / "sample",          # Test videos go here
    "raw_data":         PROJECT_ROOT / "data" / "raw",             # Real filmed data
    "calibration":      PROJECT_ROOT / "data" / "raw" / "calibration",
    "set_a":            PROJECT_ROOT / "data" / "raw" / "set_a_control",
    "set_b":            PROJECT_ROOT / "data" / "raw" / "set_b_experimental",

    # --- Intermediate Data ---
    "landmarks":        PROJECT_ROOT / "data" / "landmarks",       # Extracted pose CSVs

    # --- Output Data ---
    "metrics":          PROJECT_ROOT / "output" / "metrics",       # Jitter / confidence CSVs
    "statistics":       PROJECT_ROOT / "output" / "statistics",    # Statistical test results
    "filtered":         PROJECT_ROOT / "output" / "filtered",      # EKF-filtered coordinates
    "cog":              PROJECT_ROOT / "output" / "cog",           # Center of Gravity data
    "plots":            PROJECT_ROOT / "output" / "plots",         # Generated figures
}


# =============================================================================
# 2. VIDEO / CAMERA SETTINGS
# =============================================================================
# These must match your actual recording settings from the data collection SOP.

VIDEO_CONFIG = {
    "primary_fps":      60,        # Frames per second for the main 4K stream
    "validation_fps":   240,       # Frames per second for the high-speed validation stream
    "primary_width":    3840,      # 4K UHD horizontal resolution (pixels)
    "primary_height":   2160,      # 4K UHD vertical resolution (pixels)

    # Calibration rod length in meters — this is the physical rod you place in the
    # frame before filming. We use it to calculate how many pixels = 1 meter.
    "calibration_rod_length_m": 1.00,

    # Video file extensions to look for when scanning directories
    "video_extensions": [".mov", ".mp4", ".avi", ".MOV", ".MP4"],
}


# =============================================================================
# 3. PRACTITIONER INFORMATION
# =============================================================================
# The N=3 cohort from the proposal

PRACTITIONERS = {
    "P01": {"grade": "1st Dan",     "description": "Primary investigator-participant"},
    "P02": {"grade": "Black Belt",  "description": "Asian Championship winner, National champion"},
    "P03": {"grade": "4th Dan",     "description": "Senior master instructor"},
}


# =============================================================================
# 4. TECHNIQUE DEFINITIONS
# =============================================================================
# The three standardised Shotokan techniques from the protocol

TECHNIQUES = {
    "OIZUKI":   {"name": "Oi-zuki",                         "reps": 10},
    "GEDAN":    {"name": "Gedan-barai",                     "reps": 10},
    "STANCE":   {"name": "Zenkutsu-dachi to Kiba-dachi",    "reps": 10},
}


# =============================================================================
# 5. MEDIAPIPE POSE LANDMARK MAP
# =============================================================================
# MediaPipe Pose outputs 33 landmarks. Each landmark has:
#   - x, y: normalised coordinates (0.0 to 1.0, relative to image size)
#   - z: depth estimate (relative to hip midpoint)
#   - visibility: confidence score (0.0 to 1.0)
#
# Below we map the landmark INDEX (integer) to a human-readable name.
# We only need the ones relevant to our study (the 6 marker joints + extras).
#
# Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

JOINTS_MP = {
    # --- Head & Torso (for CoG calculation) ---
    0:  "nose",
    11: "left_shoulder",        # Left Acromion
    12: "right_shoulder",       # Right Acromion

    # --- Upper Limbs ---
    13: "left_elbow",           # Left Lateral Epicondyle
    14: "right_elbow",          # Right Lateral Epicondyle
    15: "left_wrist",           # Left Ulnar Styloid
    16: "right_wrist",          # Right Ulnar Styloid

    # --- Hips ---
    23: "left_hip",             # Left Greater Trochanter
    24: "right_hip",            # Right Greater Trochanter

    # --- Lower Limbs ---
    25: "left_knee",            # Left Lateral Femoral Condyle
    26: "right_knee",           # Right Lateral Femoral Condyle
    27: "left_ankle",           # Left Lateral Malleolus
    28: "right_ankle",          # Right Lateral Malleolus

    # --- Feet (for stance depth) ---
    31: "left_foot_index",
    32: "right_foot_index",
}

# The 6 joints where we placed physical fluorescent markers (from the SOP).
# These are the PRIMARY joints for the benchmark analysis.
MARKER_JOINTS_MP = {
    "ankle":    [27, 28],       # Lateral Malleolus
    "knee":     [25, 26],       # Lateral Femoral Condyle
    "hip":      [23, 24],       # Greater Trochanter
    "shoulder": [11, 12],       # Acromion Process
    "elbow":    [13, 14],       # Lateral Epicondyle of Humerus
    "wrist":    [15, 16],       # Ulnar Styloid Process
}


# =============================================================================
# 6. YOLOv8-POSE KEYPOINT MAP
# =============================================================================
# YOLOv8-Pose uses the COCO keypoint format: 17 keypoints.
# Each keypoint has: x, y (pixel coordinates) and confidence (0.0 to 1.0).
#
# IMPORTANT DIFFERENCE from MediaPipe:
#   - MediaPipe: coordinates are NORMALISED (0.0 to 1.0)
#   - YOLOv8:    coordinates are in PIXELS (0 to image_width/height)
#   We normalise YOLOv8 outputs in the pose_inference module to make them
#   directly comparable.

JOINTS_YOLO = {
    0:  "nose",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Mapping from our common joint GROUP names to YOLOv8 indices
MARKER_JOINTS_YOLO = {
    "ankle":    [15, 16],
    "knee":     [13, 14],
    "hip":      [11, 12],
    "shoulder": [5, 6],
    "elbow":    [7, 8],
    "wrist":    [9, 10],
}


# =============================================================================
# 7. DE LEVA (1996) BODY SEGMENT MASS PROPORTIONS
# =============================================================================
# These are the relative mass weights for each body segment, from:
#   de Leva, P. (1996). "Adjustments to Zatsiorsky-Seluyanov's segment
#   inertia parameters." Journal of Biomechanics, 29(9), 1223-1230.
#
# WHAT THIS MEANS:
#   If a person weighs 70 kg, their trunk weighs approximately:
#   70 × 0.4346 = 30.4 kg
#
# HOW WE USE IT:
#   Center of Gravity (CoG) = weighted average of all segment midpoints:
#   X_CoG = Σ(weight_i × x_midpoint_i)
#
# NOTE: The "×2" segments (arms, legs) are listed as PER-LIMB weights.
# When computing CoG, we add left and right separately.

DE_LEVA_WEIGHTS = {
    "head_neck":    0.0681,     # Top of head → C7/Acromion midpoint
    "trunk":        0.4346,     # Acromion midpoint → Greater Trochanter midpoint
    "upper_arm":    0.0263,     # Acromion → Lateral Epicondyle (PER ARM)
    "forearm":      0.0150,     # Lateral Epicondyle → Ulnar Styloid (PER ARM)
    "hand":         0.0059,     # Ulnar Styloid → Fingertip (PER HAND)
    "thigh":        0.1416,     # Greater Trochanter → Femoral Condyle (PER LEG)
    "shank":        0.0433,     # Femoral Condyle → Lateral Malleolus (PER LEG)
    "foot":         0.0137,     # Lateral Malleolus → Toe tip (PER FOOT)
}

# Quick sanity check: all weights should sum to 1.0
# Head(0.0681) + Trunk(0.4346) + 2×UpperArm(0.0526) + 2×Forearm(0.0300)
# + 2×Hand(0.0118) + 2×Thigh(0.2832) + 2×Shank(0.0866) + 2×Foot(0.0274) = ~1.0
_total = (DE_LEVA_WEIGHTS["head_neck"] + DE_LEVA_WEIGHTS["trunk"]
          + 2 * DE_LEVA_WEIGHTS["upper_arm"] + 2 * DE_LEVA_WEIGHTS["forearm"]
          + 2 * DE_LEVA_WEIGHTS["hand"] + 2 * DE_LEVA_WEIGHTS["thigh"]
          + 2 * DE_LEVA_WEIGHTS["shank"] + 2 * DE_LEVA_WEIGHTS["foot"])
assert abs(_total - 1.0) < 0.01, f"de Leva weights don't sum to 1.0! Got {_total}"


# =============================================================================
# 8. STATISTICAL ANALYSIS PARAMETERS
# =============================================================================
# All pre-registered statistical settings from the proposal.

STATS_CONFIG = {
    "alpha":            0.05,       # Significance level for hypothesis testing
    "fdr_q":            0.05,       # False Discovery Rate threshold (Benjamini-Hochberg)
    "normality_alpha":  0.05,       # Shapiro-Wilk normality test threshold

    # Effect size interpretation thresholds (rank-biserial correlation)
    "effect_small":     0.10,
    "effect_medium":    0.30,
    "effect_large":     0.50,
}


# =============================================================================
# 9. EKF FILTER PARAMETERS (Phase B — Paper 2)
# =============================================================================
# These will be tuned on sample data before the real experiment.

EKF_CONFIG = {
    "confidence_threshold": 0.50,   # T_thresh: below this, reject the measurement
    "gamma":                5.0,    # γ: scaling factor for dynamic R_t
    "sigma_0":              0.01,   # σ₀²: base measurement noise variance
    "dt":                   1/60,   # Δt: time between frames (1/60 seconds at 60 FPS)
}


# =============================================================================
# 10. JOINT ANGLE DEFINITIONS
# =============================================================================
# Joint angles are calculated from THREE landmarks: (A, B, C)
# The angle is measured AT joint B, between segments BA and BC.
#
# Example: Knee angle = angle at the KNEE between hip-knee and knee-ankle.
#
# These use MediaPipe landmark indices.

JOINT_ANGLES_MP = {
    "left_knee_angle":   {"A": 23, "B": 25, "C": 27},  # Hip → Knee → Ankle
    "right_knee_angle":  {"A": 24, "B": 26, "C": 28},
    "left_hip_angle":    {"A": 11, "B": 23, "C": 25},   # Shoulder → Hip → Knee
    "right_hip_angle":   {"A": 12, "B": 24, "C": 26},
    "left_elbow_angle":  {"A": 11, "B": 13, "C": 15},   # Shoulder → Elbow → Wrist
    "right_elbow_angle": {"A": 12, "B": 14, "C": 16},
}

# Same angles but using YOLOv8 indices
JOINT_ANGLES_YOLO = {
    "left_knee_angle":   {"A": 11, "B": 13, "C": 15},
    "right_knee_angle":  {"A": 12, "B": 14, "C": 16},
    "left_hip_angle":    {"A": 5,  "B": 11, "C": 13},
    "right_hip_angle":   {"A": 6,  "B": 12, "C": 14},
    "left_elbow_angle":  {"A": 5,  "B": 7,  "C": 9},
    "right_elbow_angle": {"A": 6,  "B": 8,  "C": 10},
}


# =============================================================================
# 11. VISUALISATION SETTINGS
# =============================================================================
PLOT_CONFIG = {
    "figure_dpi":       300,        # High DPI for publication-quality figures
    "figure_format":    "png",      # Output format (png for raster, pdf for vector)
    "color_set_a":      "#2196F3",  # Blue for Set A (Control)
    "color_set_b":      "#F44336",  # Red for Set B (Experimental / Gi)
    "color_filtered":   "#4CAF50",  # Green for EKF-filtered output
    "font_size":        12,
    "style":            "seaborn-v0_8-whitegrid",  # Clean academic plot style
}


# =============================================================================
# QUICK REFERENCE: How this config connects to the research
# =============================================================================
#
#   Proposal Section    →   Config Section
#   ─────────────────────────────────────────
#   §6.3.1 Camera       →   VIDEO_CONFIG
#   §6.2   Participants →   PRACTITIONERS
#   §6.3.5 Techniques   →   TECHNIQUES
#   §6.4.2 MediaPipe    →   JOINTS_MP, MARKER_JOINTS_MP
#   §6.4.2 YOLOv8       →   JOINTS_YOLO, MARKER_JOINTS_YOLO
#   §6.4.3 Joint Angles →   JOINT_ANGLES_MP, JOINT_ANGLES_YOLO
#   §6.5   Statistics    →   STATS_CONFIG
#   §6.6   EKF Filter   →   EKF_CONFIG
#   §6.6.4 CoG Model    →   DE_LEVA_WEIGHTS
# =============================================================================
