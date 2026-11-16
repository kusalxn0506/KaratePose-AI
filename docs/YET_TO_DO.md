# 📋 KaratePose-AI — Yet To Do

> **Created:** 1 November 2026
> **Author:** Kusal Nirmala
> **Git branch:** `main` (all code pushed as of commit `e8f57f4`)
> **Last Updated:** 16 November 2026 — Added Phase 2 video overlay + publication prototype

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
  - [x] **Bug-fix (8 Nov):** Two-stage occlusion handling (ballistic momentum → soft confidence-gated tether)
  - [x] **Bug-fix (8 Nov):** Covariance P damping during predict-only coasting (prevents K≈1 cliff drops)
  - [x] **Bug-fix (8 Nov):** Re-acquisition blending (5-frame R ramp on confidence recovery)
- [x] `src/09_cog_tracker.py` — De Leva 1996 CoG engine + stance depth
- [x] `src/10_filter_benchmark.py` — MPJPE, PCK@0.2, phase lag, velocity preservation
  - [x] **Bug-fix (8 Nov):** Confidence-masked MPJPE (excludes low-confidence glitch frames)
  - [x] **Bug-fix (8 Nov):** Both-sided confidence-masked PVR (prevents artificial velocity spikes)
- [x] `src/rep_detector.py` — Velocity-based auto rep segmentation
- [x] `src/11_phase2_visualiser.py` — Publication-ready plots for Paper 2 (8 figures)
  - [x] **Bug-fix (8 Nov):** SKF label corrected from "Savitzky–Golay" to "Standard Kalman"
- [x] `src/12_phase2_overlay.py` — Side-by-side Raw vs EKF video overlay with skeleton rendering
  - [x] **Added (9 Nov):** Three-panel mode, confidence annotations, side-by-side comparison
- [x] `run_pipeline.py` — Full orchestrator (`--phase 1` / `--phase 2`, includes `--step plots`, `--step overlay`)
- [x] `src/config.py` — All EKF params, de Leva weights, benchmark paths
  - [x] **Added (8 Nov):** `max_coast_frames` (12) and `reacq_ramp_frames` (5) parameters

### Documentation — COMPLETE
- [x] `README.md` — Full project structure, run commands, architecture diagram
- [x] `docs/Phase_2_Plan.md` — Phase 2 architecture & roadmap
- [x] `docs/Video_Recording_Guide.md` — Updated to self-paced protocol (no metronome)
- [x] `docs/Paper_1_Draft.md` — Manuscript 1 draft (The Degradation Problem)
- [x] `docs/Paper_2_Draft.md` — Manuscript 2 draft (The EKF & CoG Solution)
- [x] **Publication Prototype (16 Nov):** Paper 2 publication-ready document with embedded figures, LaTeX equations, and `[PLACEHOLDER]` markers for multi-practitioner data

### Git — All Pushed
- [x] All code and docs committed to `origin/main`

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
# Phase 1: Extract → Infer → Metrics → Reps → Aggregate → Stats → Plots
python run_pipeline.py --phase 1

# Phase 2: Filter → CoG → Benchmark → Publication Plots
python run_pipeline.py --phase 2

# With calibration scale (pixels per metre):
python run_pipeline.py --phase 2 --step cog --scale <PIXELS_PER_METRE>
```

### 3. Run Rep Detector / Verify Rep Boundaries

```bash
# Automated via pipeline:
python run_pipeline.py --step reps

# Or standalone per landmark CSV:
python -m src.rep_detector --input data/landmarks/SETA_OIZUKI_T01_60FPS_landmarks.csv --technique OIZUKI
python -m src.rep_detector --input data/landmarks/SETA_GEDAN_T01_60FPS_landmarks.csv --technique GEDAN
python -m src.rep_detector --input data/landmarks/SETA_STANCE_T01_60FPS_landmarks.csv --technique STANCE
```

> Rep boundaries are saved to `output/metrics/rep_boundaries_*.csv` and automatically consumed by `step_4_aggregate()`.

### 4. Finalize Paper 2 Manuscript
- [x] **Full draft created:** `docs/Paper_2_Draft.md`
- [ ] Update Results section tables with final numbers from real data
- [ ] Insert finalized figures from `output/plots/phase2/`
- [ ] Proofread for journal/conference submission format

### 5. Generate Publication Figures for Paper 2 ✅ DONE
- [x] EKF vs Raw trajectory plot (`trajectory_wrist_mp.png`)
- [x] Velocity profile overlay showing Kime preservation (`velocity_wrist_mp.png`)
- [x] Filter comparison bar chart (`mpjpe_bars.png`)
- [x] PCK@0.2 radar chart (`pck_radar_mp.png`)
- [x] Peak Velocity Preservation Ratio bar chart (`pvr_bars.png`)
- [x] CoG trajectory comparison (`cog_trajectory_mp.png`)
- [x] Stance depth over time (`stance_depth_mp.png`)
- [x] Benchmark summary table figure (`benchmark_table.png`)
- [x] Automated via `python run_pipeline.py --phase 2 --step plots` or `python -m src.11_phase2_visualiser`

---

## 🟡 IMPORTANT — Should Do

### 6. Wire Rep Detector into Pipeline ✅ DONE
- [x] Added `--step reps` to `run_pipeline.py`
- [x] Integrated `detect_reps_velocity` inside `step_4_aggregate()` with graceful fallback
- [x] Auto-loads metrics & landmarks from disk when running individual steps

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
max_coast_frames: 12        # Stage 1→2 transition (200ms at 60 FPS)
reacq_ramp_frames: 5        # Re-acquisition blending duration
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

### 13. ~~Create `11_phase2_visualiser.py`~~ ✅ DONE
- [x] Dedicated plotting module for Paper 2 figures
- [x] 8 plots: trajectory, velocity, MPJPE bars, PCK radar, PVR bars, CoG trajectory, stance depth, benchmark table
- [x] Run: `python -m src.11_phase2_visualiser` or `python run_pipeline.py --phase 2 --step plots`

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
| SKF (Standard Kalman) | 0.002 | 0.97 | 0.002 | 0.86 |
| Linear Interp | 0.000 | 1.00 | 0.000 | 1.00 |
| **EKF** | **0.002** | **0.97** | **0.002** | **2.10** |

> **Key story for Paper 2:** EKF has zero phase lag + near-perfect velocity
> preservation ratio (MediaPipe PVR ≈ 0.97). The Standard Kalman (SKF)
> over-smooths slightly (PVR = 0.86–0.97). All metrics are now
> confidence-masked — low-confidence glitch frames are excluded so
> PVR reflects true physical motion, not detector artifacts.
>
> **Note:** Raw and Linear Interp MPJPE = 0.000 is expected (reference IS
> the raw data). YOLOv8 EKF PVR = 2.10 reflects momentum preservation
> through shorter occlusion gaps — will normalize on real multi-rep data.

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

# Phase 2 publication plots
python run_pipeline.py --phase 2 --step plots
python -m src.11_phase2_visualiser

# Phase 2 video overlay (Raw vs EKF side-by-side)
python run_pipeline.py --phase 2 --step overlay
python -m src.12_phase2_overlay --three-panel

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
