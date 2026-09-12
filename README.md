# 🥋 KaratePose-AI

**BSc (Hons) Research Project — Comparative Analysis of AI Pose Estimation for Karate Biomechanics**

> Investigating how traditional Karate uniforms (Gi) affect the accuracy of AI-based
> pose estimation models, comparing **MediaPipe BlazePose** (33 landmarks) vs
> **YOLOv8-Pose** (17 COCO keypoints).

---

## 📋 Project Overview

This research quantifies the **degradation in pose estimation accuracy** caused by
loose-fitting clothing (Karate Gi) compared to tight-fitting sportswear (rashguard).

### Research Design
| | Set A (Control) | Set B (Experimental) |
|---|---|---|
| **Clothing** | Rashguard (tight-fit) | Karate Gi (loose-fit) |
| **Technique** | Oizuki (lunge punch) | Oizuki (lunge punch) |
| **Repetitions** | 10 per participant | 10 per participant |
| **Participants** | N = 3 practitioners | N = 3 practitioners |

### Metrics Computed
- **Spatial Jitter** — frame-to-frame Euclidean displacement of landmarks
- **Confidence Volatility** — stability of detection confidence scores
- **Joint Angles** — anatomical angles via dot product / arccos

### Statistical Tests
- Shapiro-Wilk normality test
- Wilcoxon signed-rank test (non-parametric)
- Paired t-test (parametric) with Cohen's d
- Benjamini-Hochberg FDR correction
- Linear Mixed-Effects Model (pooled analysis)

---

## 🏗️ Project Structure

```
Karate Pose AI/
├── src/                        # Source code modules
│   ├── config.py               # Central configuration (paths, joints, settings)
│   ├── 01_frame_extractor.py   # Video → frames extraction
│   ├── 02_pose_inference.py    # MediaPipe + YOLOv8 pose detection
│   ├── 03_metrics.py           # Jitter, confidence, joint angle calculations
│   ├── 04_aggregator.py        # Frame → repetition aggregation (anti-pseudoreplication)
│   ├── 05_statistics.py        # Inferential statistical tests
│   ├── 06_video_overlay.py     # Side-by-side skeleton visualisation
│   └── 08_visualiser.py        # Publication-ready plot generation
│
├── data/
│   └── sample/                 # Sample test videos
│       └── test_oizuki.mp4     # Sample Oizuki punch video
│
├── run_pipeline.py             # Main CLI pipeline runner
├── run_demo.py                 # Quick demo with synthetic data
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (tested on 3.14)
- macOS (Apple Silicon M1–M4) or Windows (CUDA optional)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/KaratePose-AI.git
cd KaratePose-AI

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download AI model weights (not included in repo)
# MediaPipe Pose Landmarker (Heavy — 29 MB)
curl -L -o pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# YOLOv8x-Pose (133 MB — auto-downloads on first run)
```

### Run
```bash
# Run the full pipeline on the sample video
python run_pipeline.py --sample

# Or run individual steps
python run_pipeline.py --step inference --sample
python run_pipeline.py --step metrics --sample

# Watch the skeleton overlay
python -m src.06_video_overlay
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mediapipe` | 0.10.35 | Pose estimation (33 landmarks) |
| `ultralytics` | ≥8.0.0 | YOLOv8-Pose (17 keypoints) |
| `opencv-python` | ≥4.8.0 | Video I/O and image processing |
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical computing |
| `scipy` | ≥1.11.0 | Statistical tests |
| `pingouin` | ≥0.5.3 | Effect sizes (rank-biserial) |
| `statsmodels` | ≥0.14.0 | Linear Mixed-Effects Models |
| `matplotlib` | ≥3.7.0 | Plotting |
| `seaborn` | ≥0.12.0 | Statistical visualisation |

> ⚠️ **macOS Note:** MediaPipe must be pinned to `v0.10.35`. Version 1.0.x crashes
> on Apple Silicon due to a Metal GPU bug in `TensorsToDetectionsCalculator`.

---

## 📄 License

This project is part of a BSc (Hons) research study. All rights reserved.

---

## 👤 Author

*BSc (Hons) Computing — Research Project*
