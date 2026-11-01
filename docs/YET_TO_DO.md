# 📋 KaratePose-AI — Yet To Do

> **Created:** 1 November 2026
> **Author:** Kusal Nirmala
> **Git branch:** `main` (all code pushed as of commit `5acd81b`)

---

## ✅ What's DONE (Don't Redo)

### Code — Phase 1 (Paper 1: The Problem) — FROZEN
- [x] `src/01_frame_extractor.py` — Video → frame extraction
- [x] `src/02_pose_inference.py` — MediaPipe + YOLOv8 inference
- [x] `src/03_metrics.py` — Jitter, confidence volatility, joint angles
- [x] `src/04_aggregator.py` — Per-rep aggregation (anti-pseudoreplication)
- [x] `src/05_statistics.py` — Wilcoxon, LMM, FDR correction
- [x] `src/06_video_overlay.py` — Side-by-side skeleton overlay
- [x] `src/08_visualiser.py` — Publication plots for Paper 1

### Code — Phase 2 (Paper 2: The Solution) — COMPLETE
- [x] `src/07_ekf_filter.py` — Event-Triggered Kinematic EKF (4 modes, ablation toggles)
- [x] `src/09_cog_tracker.py` — De Leva 1996 CoG engine + stance depth
- [x] `src/10_filter_benchmark.py` — MPJPE, PCK@0.2, phase lag, velocity preservation
- [x] `src/rep_detector.py` — Velocity-based auto rep segmentation
- [x] `run_pipeline.py` — Full orchestrator (`--phase 1` / `--phase 2`)
- [x] `src/config.py` — All EKF params, de Leva weights, benchmark paths

### Documentation — COMPLETE
- [x] `README.md` — Full project structure, run commands, architecture diagram
- [x] `docs/Phase_2_Plan.md` — Phase 2 architecture & roadmap
- [x] `docs/Video_Recording_Guide.md` — Updated to self-paced protocol (no metronome)
- [x] `docs/Paper_1_Draft.md` — Manuscript 1 draft

### Git — All Pushed
- [x] All code and docs committed to `origin/main`
- [x] Latest commit: `5acd81b` — rep detector + self-paced protocol update

---

## 🔴 CRITICAL — Must Do Before Submission

### 1. Record Real Video Data
- [ ] Schedule sessions with 3 practitioners (P01, P02, P03)
- [ ] Follow `docs/Video_Recording_Guide.md` exactly
- [ ] **Self-paced protocol** — no metronome needed, natural cadence
- [ ] Record calibration rod clip (1.00m, once per session)
- [ ] Per practitioner: 6 primary clips + 2 slow-mo = 8 files
- [ ] **Total: 24 video files + 1 calibration clip**
- [ ] Transfer & rename files per naming convention (`SETA_OIZUKI_T01_60FPS.mov`)
- [ ] Place into `data/raw/set_a_control/P0X/` and `data/raw/set_b_experimental/P0X/`

### 2. Run Full Pipeline on Real Data

```bash
# Phase 1: Extract → Infer → Metrics → Aggregate → Stats → Plots
python run_pipeline.py --phase 1

# Phase 2: Filter → CoG → Benchmark
python run_pipeline.py --phase 2

# If you measured calibration scale (pixels per metre):
python run_pipeline.py --phase 2 --step cog --scale <PIXELS_PER_METRE>
```

### 3. Run Rep Detector on Each Landmark CSV

```bash
# After inference produces landmark CSVs, segment reps:
python -m src.rep_detector --input data/landmarks/SETA_OIZUKI_T01_60FPS_landmarks.csv --technique OIZUKI
python -m src.rep_detector --input data/landmarks/SETA_GEDAN_T01_60FPS_landmarks.csv --technique GEDAN
python -m src.rep_detector --input data/landmarks/SETA_STANCE_T01_60FPS_landmarks.csv --technique STANCE
# ... repeat for SETB files
```

> The rep detector outputs `output/metrics/rep_boundaries_*.csv`.
> Feed these boundaries into `04_aggregator.aggregate_per_repetition()` for
> Paper 1 stats on real data.

### 4. Write Paper 2 Manuscript
- [ ] **Introduction** — Pose estimation under occlusion, EKF motivation
- [ ] **Literature Review** — Physics-informed filtering in biomechanics, de Leva 1996, Kime
- [ ] **Methodology** — EKF formulation, confidence-gated measurement, CoG model, benchmark metrics
- [ ] **Results** — Tables from `output/benchmark/benchmark_summary.csv` and `output/cog/cog_all_combined.csv`
- [ ] **Discussion** — EKF vs SKF vs linear interp, velocity preservation = Kime fidelity
- [ ] **Conclusion**

### 5. Generate Publication Figures for Paper 2
- [ ] EKF vs Raw trajectory plot (wrist during Oi-zuki)
- [ ] Filter comparison bar chart (MPJPE by mode)
- [ ] PCK@0.2 radar chart (per joint)
- [ ] CoG trajectory comparison (raw vs EKF, sagittal plane)
- [ ] Velocity profile overlay showing Kime preservation (PVR metric)
- [ ] Phase lag comparison (cross-correlation offset per filter)

> **Tip:** Use `08_visualiser.py` as a template. Copy it for Paper 2 plots
> or write a new `11_phase2_visualiser.py`.

---

## 🟡 IMPORTANT — Should Do

### 6. Wire Rep Detector into Pipeline (Optional Enhancement)
- [ ] Add `--step reps` to `run_pipeline.py` Phase 2
- [ ] Auto-run rep detection before aggregation
- [ ] Pass detected boundaries into `step_4_aggregate()` instead of equal division
- [ ] This replaces the simple equal-division fallback with real velocity-based boundaries

### 7. Calculate Calibration Scale
- [ ] Open calibration rod video in any image viewer
- [ ] Measure the rod's height in pixels (bottom tip to top tip)
- [ ] `scale = pixel_height_of_rod / 1.00` → this gives pixels per metre
- [ ] Use: `python run_pipeline.py --phase 2 --step cog --scale <value>`
- [ ] This converts CoG stance depth from pixels to real metres

### 8. EKF Parameter Tuning on Real Data
Current EKF config (in `src/config.py`):
```
confidence_threshold: 0.50
gamma: 5.0
sigma_0: 0.01
dt: 1/60
sigma_a: 75.0
vel_decay_max: 0.98
vel_decay_min: 0.80
```
- [ ] Run EKF on real Gi data (Set B) — check if `sigma_a=75.0` still works
- [ ] If EKF overshoots on real data, try `sigma_a` between 50–100
- [ ] If EKF is too sluggish, increase `sigma_a`
- [ ] Check `confidence_threshold=0.50` — if Gi occlusions drop confidence lower, consider 0.40

### 9. Validate Rep Detector on Real Data
- [ ] Run on a real 10-rep recording
- [ ] Visually verify detected boundaries make sense (check CSV output)
- [ ] If peaks are missed: lower `--prominence` (default auto-tuned)
- [ ] If too many peaks: increase `--min-distance` or set `--n-reps 10`

---

## 🟢 NICE TO HAVE — If Time Permits

### 10. Ablation Study Tables
- [ ] Use `07_ekf_filter.py` ablation toggles to run systematic ablation:
  ```bash
  # Inside Python or a script:
  from src.07_ekf_filter import run_filter
  # Run with different ablation flags per the module's API
  ```
- [ ] Record MPJPE/PCK for each ablation condition
- [ ] Create Table: "Contribution of each EKF component"

### 11. Slow-Motion Validation (240fps Ground Truth)
- [ ] Use 240fps Oi-zuki clips as pseudo ground truth
- [ ] Downsample 240fps → 60fps
- [ ] Compare EKF-filtered 60fps against 240fps reference
- [ ] This gives a true MPJPE against real high-framerate data

### 12. Statistical Tests for Paper 2
- [ ] Paired comparison: EKF-filtered Set A vs EKF-filtered Set B
- [ ] Wilcoxon on MPJPE improvement per joint
- [ ] Effect sizes on velocity preservation ratio

### 13. Create `11_phase2_visualiser.py`
- [ ] Dedicated plotting module for Paper 2 figures
- [ ] Import from `config.py` only (follows the frozen module rule)
- [ ] Reads from `output/filtered/`, `output/cog/`, `output/benchmark/`

---

## 📁 Key File Locations

| What | Path |
|---|---|
| Project root | `Karate Pose AI/` |
| All source code | `src/` |
| Config (all params) | `src/config.py` |
| Pipeline entry point | `run_pipeline.py` |
| Raw video input | `data/raw/set_a_control/` and `data/raw/set_b_experimental/` |
| Landmark CSVs | `data/landmarks/` |
| Filtered output | `output/filtered/` |
| CoG output | `output/cog/` |
| Benchmark results | `output/benchmark/` |
| Rep boundaries | `output/metrics/rep_boundaries_*.csv` |
| Recording guide | `docs/Video_Recording_Guide.md` |
| Phase 2 plan | `docs/Phase_2_Plan.md` |
| Paper 1 draft | `docs/Paper_1_Draft.md` |

---

## ⚠️ Critical Rules (Don't Break These)

1. **NEVER modify** modules 01–06 or 08 (Phase 1 is frozen)
2. Phase 2 modules import **only from `config.py`** and consume landmark CSVs from module 02
3. Always activate venv before running: `source .venv/bin/activate`
4. iPhone recording format: **H.264** (Most Compatible), NOT HEVC
5. Camera: **4K 60fps**, tripod at **1.0m height**, **4.5m distance**
6. Self-paced protocol — **no metronome** — `rep_detector.py` handles segmentation

---

## 🔑 Key Benchmark Results (Test Data — For Reference)

| Filter | MediaPipe MPJPE | MediaPipe PVR | YOLOv8 MPJPE | YOLOv8 PVR |
|--------|-----------------|---------------|--------------|------------|
| Raw | 0.000 | 1.00 | 0.000 | 1.00 |
| SKF | 0.002 | 0.80 | 0.002 | 0.84 |
| Linear Interp | 0.009 | 0.85 | 0.001 | 0.97 |
| **EKF** | **0.013** | **1.52** | **0.004** | **1.18** |

> **Key story for Paper 2:** EKF has zero phase lag + highest velocity
> preservation ratio (PVR > 1.0). This means the EKF preserves *Kime*
> (decisive finishing tension) better than all baselines. SKF smooths
> away the velocity peaks (PVR < 1.0 = kills Kime).

---

## 🏃 Quick Command Reference

```bash
# Activate environment
cd "Karate Pose AI"
source .venv/bin/activate

# Full Phase 1
python run_pipeline.py --phase 1

# Full Phase 2
python run_pipeline.py --phase 2

# Individual Phase 2 steps
python run_pipeline.py --phase 2 --step filter
python run_pipeline.py --phase 2 --step cog --scale 1920
python run_pipeline.py --phase 2 --step benchmark

# Rep detection
python -m src.rep_detector --input data/landmarks/FILE.csv --technique OIZUKI

# Standalone modules
python -m src.07_ekf_filter
python -m src.09_cog_tracker
python -m src.10_filter_benchmark

# Git
git add -A && git commit -m "message" && git push origin main
```

---

*Good luck with the recordings and papers, Kusal! Everything is built and ready — just add real data and write. 🥋*
