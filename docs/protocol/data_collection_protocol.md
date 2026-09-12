# KaratePose-AI: Data Collection Standard Operating Procedure (SOP)

**Project:** Kinematic Refinement of Coordinate-Based Human Pose Estimation via a Physics-Informed Filter Engine for Elite Martial Arts Analytics  
**Institution:** BSc (Hons) Artificial Intelligence — University of Plymouth / NSBM Green University  
**Revision:** 2.0 (Standardized Experimental Protocol)

---

## 1. Experimental Objectives

This protocol establishes a standardized, reproducible video capture procedure to:
1. **Isolate Garment Occlusion**: Capture strictly matched kinematic performances under two garment conditions:
   - **Set A (Control / Ground Truth Proxy)**: Form-fitting compression apparel with high-contrast physical joint markers.
   - **Set B (Experimental)**: Traditional heavyweight 14oz rigid canvas Karate Gi (*dogi*).
2. **Standardize Kinetics & Execution**: Minimize inter-trial kinematic variation across Shotokan karate techniques (*Oi-zuki*, *Gedan-barai*, *Zenkutsu-dachi* to *Kiba-dachi* transitions) through cadence cues and fixed calibration.
3. **Ensure High-Fidelity Ingestion**: Provide synchronous video assets suitable for frame-by-frame 2D landmark extraction (MediaPipe Holistic and Ultralytics YOLOv8-Pose).

---

## 2. Hardware, Optical & Environmental Setup

### 2.1 Optical & Camera Specifications
| Parameter | Primary Inference Stream | Slow-Motion Validation Subsample |
| :--- | :--- | :--- |
| **Device** | iPhone 15 Pro Max (Primary 24mm f/1.78 lens) | iPhone 15 Pro Max (Telephoto 120mm / Main) |
| **Resolution** | 4K UHD ($3840 \times 2160$) | 1080p FHD ($1920 \times 1080$) |
| **Frame Rate** | 60.0 FPS (Constant Frame Rate) | 240.0 FPS (High-Speed Ground Truth) |
| **Shutter Speed** | Fixed at $1/500\text{ s}$ or faster (prevents motion blur) | Automatic high-speed |
| **Focus / Exposure** | Manual AE/AF Lock (Locked on practitioner center) | Manual AE/AF Lock |
| **Color Profile** | Standard / SDR Rec.709 (High dynamic range) | SDR Rec.709 |

### 2.2 Spatial & Geometric Calibration
- **Tripod Height:** $1.0\text{ m} \pm 0.02\text{ m}$ (aligned with the practitioner's approximate standing Center of Gravity / Iliac crest).
- **Camera-to-Subject Distance:** $4.5\text{ m}$ perpendicular to the sagittal/frontal execution plane.
- **Field of View (FOV):** Full body visible throughout full extension (minimum $0.5\text{ m}$ ground margin below feet and $0.5\text{ m}$ overhead margin at full height).
- **Orientation Angles:**
  - **Angle 1 (Primary Sagittal View):** $90^\circ$ perpendicular to linear step direction.
  - **Angle 2 (Frontal / $45^\circ$ Oblique View):** $45^\circ$ off-axis (captures hip rotation and stance depth).
- **Spatial Calibration Object:** A $1.00\text{ m}$ rigid calibration rod placed at the execution plane prior to filming for pixel-to-metric scaling ($S_{px} = \text{pixels} / \text{meter}$).

### 2.3 Lighting & Background Control
- **Illuminance:** Minimum $500\text{ lux}$ uniform, diffuse indoor lighting (no direct backlighting, no harsh overhead spotlights creating moving ground shadows).
- **Backdrop:** Matte, non-reflective neutral background (e.g., solid gray/dark wall) maximizing contrast against both white gi fabric and dark athletic wear.

---

## 3. Garment & Marker Protocols

```
      [ SET A: CONTROL ]                       [ SET B: EXPERIMENTAL ]
  Form-Fitting Compression Gear                14oz Rigid Canvas Karate Gi
+ High-Contrast Joint Markers               Heavy, Folding Silhouette Occlusion
          │                                              │
          ▼                                              ▼
   Flawless Baseline                              Target Degradation
  (Proxy Ground Truth)                         (False Boundaries & Jitter)
```

### 3.1 Condition A — Control (Form-Fitting + Contrast Markers)
- **Apparel:** Matte black compression shirt (long/short sleeve) and compression tights.
- **Physical Joint Markers:** Fluorescent adhesive circular markers ($25\text{ mm}$ diameter) affixed to anatomical landmarks:
  1. Lateral Malleolus (Ankle)
  2. Lateral Femoral Condyle (Knee)
  3. Greater Trochanter (Hip)
  4. Acromion Process (Shoulder)
  5. Lateral Epicondyle of Humerus (Elbow)
  6. Ulnar Styloid Process (Wrist)

### 3.2 Condition B — Experimental (14oz Heavyweight Canvas Gi)
- **Apparel:** Traditional 14oz (approx. $400\text{ g/m}^2$) unbleached/bleached 100% cotton canvas Karate Gi (Shotokan cut, long jacket, wide trousers).
- **Fastening:** Belt (*obi*) securely tied in standard square knot; jacket overlap secured to reflect real-world dojo/competition conditions.

---

## 4. Standardized Martial Arts Execution Protocol

To minimize kinematic variance, all takes follow a fixed 4-phase audio-cue protocol (60 BPM audible metronome or verbal count):

```
Time:     [0.0s] ────────── [3.0s] ────────── [4.0s] ────────── [5.0s] ────────── [8.0s]
Phase:    Static Calibration ── Stance Initiation ── Peak Kinetic Kime ── Isometric Hold ── Recovery
State:    (Yoi / T-Pose)        (Rapid Transition)    (Impact Point)      (Fixed Stance)     (Return to Yoi)
```

### 4.1 Target Techniques

#### Technique 1: *Oi-zuki* (Stepping Lunge Punch in *Zenkutsu-dachi*)
- **Starting Position:** Left *Zenkutsu-dachi* (front stance), *Gedan-barai* ready.
- **Kinetic Action:** Right leg drives forward $1.5\times$ shoulder width into right *Zenkutsu-dachi*; right *Chudan Oi-zuki* (middle punch) executes with full $180^\circ$ forearm rotation.
- **Kinematic Focus:** Ankle/Knee velocity along sagittal plane; hip translation; elbow extension.
- **Repetitions:** 10 repetitions (Set A) + 10 repetitions (Set B).

#### Technique 2: *Gedan-barai* (Downward Sweeping Block in *Zenkutsu-dachi*)
- **Starting Position:** Left *Zenkutsu-dachi*, right hand chambered at ear.
- **Kinetic Action:** Downward sweeping block with aggressive hip rotation (*Hanmi* to *Shomen* or vice-versa).
- **Kinematic Focus:** Severe canvas folding across torso and upper arm; rapid deceleration at block contact point.
- **Repetitions:** 10 repetitions (Set A) + 10 repetitions (Set B).

#### Technique 3: *Zenkutsu-dachi* to *Kiba-dachi* Transition (Stance Shift)
- **Starting Position:** Right *Zenkutsu-dachi* (deep front stance, $60\%$ weight forward).
- **Kinetic Action:** $90^\circ$ pivot into *Kiba-dachi* (horse-riding stance, $50/50$ weight distribution, wide knee abduction).
- **Kinematic Focus:** Major hip-joint occlusion by heavy trouser folds and jacket hem; Center of Gravity vertical and lateral shift.
- **Repetitions:** 10 repetitions (Set A) + 10 repetitions (Set B).

---

## 5. Step-by-Step Recording Procedure

1. **Pre-Flight Checklist:**
   - [ ] Clean camera lens.
   - [ ] Mount tripod securely at $1.0\text{ m}$ height, $4.5\text{ m}$ distance.
   - [ ] Confirm $500+\text{ lux}$ diffuse illumination.
   - [ ] Record a 5-second take of the $1.00\text{ m}$ calibration rod.
   - [ ] Start audio metronome at 60 BPM.

2. **Per-Take Execution Sequence:**
   - **Step 1:** Press Record.
   - **Step 2:** Slate / Audio Sync Clap (performer claps hands sharply in frame to generate an acoustic and visual sync marker).
   - **Step 3:** 3-second Static Calibration Pose (*Yoi-dachi* or neutral T-pose).
   - **Step 4:** Perform 10 consecutive repetitions of the designated technique with 3-second resets between reps.
   - **Step 5:** 3-second final static hold.
   - **Step 6:** Stop Record and log entry in `data_collection_sheet.csv`.

---

## 6. Data Integrity & File Management

### 6.1 Directory Structure
```
raw_data/
├── calibration/
│   └── CALIB_20260815_1m_rod.mov
├── set_a_control/
│   ├── SETA_OIZUKI_T01_60FPS.mov
│   ├── SETA_GEDAN_T01_60FPS.mov
│   └── SETA_STANCE_T01_60FPS.mov
└── set_b_experimental/
    ├── SETB_OIZUKI_T01_60FPS.mov
    ├── SETB_GEDAN_T01_60FPS.mov
    └── SETB_STANCE_T01_60FPS.mov
```

### 6.2 Naming Convention
`[SET]_[TECHNIQUE]_[TAKE_ID]_[FPS]FPS.[ext]`  
*Example:* `SETA_OIZUKI_T01_60FPS.mov`, `SETB_STANCE_T02_240FPS.mov`
