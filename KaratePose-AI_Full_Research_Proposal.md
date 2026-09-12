---
title: "Kinematic Refinement of Coordinate-Based Human Pose Estimation via a Physics-Informed Filter Engine for Elite Martial Arts Analytics"
author: "[Student Name Placeholder]"
date: "2026-09-02"
---

# 1. Title Page

**Full Title:** Kinematic Refinement of Coordinate-Based Human Pose Estimation via a Physics-Informed Filter Engine for Elite Martial Arts Analytics

**Short Title:** KaratePose-AI

**Student:** [Student Name / ID Placeholder]

**Institution:** BSc (Hons) Artificial Intelligence — University of Plymouth / NSBM Green University

**Supervisor:** [Supervisor Name Placeholder]

**Date:** September 2026

**Word Count:** [Word Count Placeholder]

---

# 2. Abstract

The application of monocular Human Pose Estimation (HPE) in sports biomechanics represents a paradigm shift toward democratized, ubiquitous motion capture. However, lightweight edge-deployable architectures such as MediaPipe Pose and YOLOv8-Pose suffer severe degradation when confronted with the compounding challenges of high-velocity ballistic movements and extreme rigid garment occlusion. This research addresses a fundamental gap in current computer vision literature: the uncharacterized degradation of coordinate-based HPE under the influence of heavyweight, rigid canvas—specifically the 14oz Shotokan Karate Gi. Unlike loose-fitting or draping fabrics, rigid canvas exhibits high structural stiffness, producing sharp creases and false limb boundaries that persistently trigger confidence collapse and spatial jitter in pose estimators.

This dissertation proposes a two-phase investigation. **Paper 1** will establish the first rigorously controlled benchmark quantifying the extent of this tracking degradation by comparing performance under a 14oz canvas Gi against a form-fitting control condition. Utilising a multi-practitioner cohort ($N=3$) comprising athletes spanning 1st Dan to 4th Dan—including an Asian Karate Championship winner—the study employs a within-subject, repeated-measures experimental design with skill-level stratification. Spatial jitter, confidence volatility, and joint angle error will be statistically modelled using non-parametric inference with False Discovery Rate correction to identify specific joint-level and velocity-dependent vulnerabilities. Building on this empirical foundation, **Paper 2** introduces a novel event-triggered, confidence-gated kinematic filter based on an Extended Kalman Filter (EKF) architecture, integrated with Newtonian momentum projection. This system actively mitigates coordinate noise during occlusion events while preserving the genuine ballistic kinematic extrema (*Kime*) characteristic of elite martial arts. Additionally, the filtered coordinates will be synthesised with de Leva's segmental biomechanical model to enable robust Center of Gravity tracking. The expected contribution is a computationally efficient, physics-informed pipeline that restores the viability of lightweight HPE for complex, high-velocity martial arts analytics, validated across multiple skill levels of elite Shotokan practitioners.

---

# 3. Introduction & Background

The evolution of computer vision, particularly in the domain of Human Pose Estimation (HPE), has catalysed profound advancements in sports analytics and biomechanical modeling. The transition from marker-based optoelectronic systems (e.g., Vicon) to markerless, monocular HPE offers unparalleled ecological validity, allowing performance capture outside the constrained laboratory environment. Contemporary lightweight models, notably MediaPipe Pose (BlazePose) and YOLOv8-Pose, provide real-time, edge-capable inference. However, their reliability precipitously declines when subjected to the dual perturbations of high-velocity kinetics and complex garment occlusion. 

The central problem addressed in this research is the failure of these lightweight models under concurrent high-velocity motion and rigid garment occlusion. While existing literature extensively investigates pose estimation in the context of standard clothing or uncontrolled garment variation, the specific challenge of rigid structural fabrics remains largely ignored. The Shotokan karate discipline serves as an optimal testbed due to the unique properties of the traditional attire. The 14oz canvas Karate Gi is a heavyweight, high-stiffness garment. Unlike casual wear or winter clothing which drape and conform dynamically to the body's contours, the rigid canvas forms persistent, geometric folds. These folds create sharp edge gradients and false limb boundaries that deceive convolutional and transformer-based feature extractors. 

This compounding effect—where ballistic velocities exceeding 8 m/s (characteristic of martial arts strikes) intersect with rigid canvas occlusion—precipitates systemic confidence collapse and extreme spatial jitter in pose estimation outputs. The algorithms struggle to differentiate between the true anatomical joint and the structural vertex of a canvas fold. 

The significance of this research lies in its commitment to real-world dojo deployment. While computationally heavy architectures such as Vision Transformers (e.g., ViTPose) or high-resolution networks (e.g., HRNet) might offer marginal improvements through brute-force parameter scaling, they are fundamentally incompatible with edge deployment constraints on consumer hardware (e.g., mobile devices). Therefore, the solution necessitates a post-processing algorithmic intervention—a physics-informed filter engine—that mathematically compensates for the perceptual failures of edge-deployable models without increasing inference overhead.

---

# 4. Literature Review

### 4.1 Human Pose Estimation Architectures
The landscape of HPE is broadly divided into top-down and bottom-up paradigms. MediaPipe Pose (Lugaresi et al., 2019) represents a state-of-the-art top-down pipeline optimised for real-time mobile inference, outputting 33 3D landmarks. Its underlying BlazePose architecture leverages a dual-network approach: a lightweight face/torso detector followed by a precise landmark regressor. Conversely, YOLOv8-Pose (Jocher et al., 2023) is a single-stage, anchor-free model that extends the YOLO object detection framework to predict 17 COCO keypoints simultaneously with bounding boxes, offering superior multi-person handling. Both architectures prioritise speed over absolute precision, making them susceptible to occlusion-induced jitter. Heavier alternatives, such as ViTPose (Xu et al., 2022) and HRNet, achieve superior accuracy through complex multi-scale feature fusion and global self-attention mechanisms. However, these models require significant GPU compute, violating the edge-deployment constraints critical for accessible sports analytics.

### 4.2 Computer Vision in Martial Arts
The application of CV in martial arts has seen growing interest. Echeverria & Santos (2021) utilised OpenPose on karate kumite footage to model psychomotor performance within the KUMITRON framework. Lu (2022) introduced AiShifu, leveraging HRNet for static karate pose coaching. Zhang et al. (2017) provided the Martial Arts, Dancing and Sports (MADS) dataset, offering a multi-view stereo benchmark. Recently, Dael et al. (2026) conducted comparative evaluations of MediaPipe and YOLOv8 across various dynamic activities including martial arts. Other recent works have explored CNN-LSTM architectures for automated kata scoring and UFC strike analysis. However, a critical gap pervades this literature: these studies operate under the implicit assumption of standard clothing or treat garment variation as an uncontrolled confounding variable, failing to isolate the specific degradation caused by martial arts attire.

### 4.3 Clothing-Induced Occlusion in Pose Estimation
The specific impact of clothing on HPE has only recently garnered rigorous attention. Yamaguchi and colleagues (2022, 2023) developed methodologies using high-frame-rate cameras and blinking LEDs to obtain ground truth posture under loose-fitting clothes. They quantified the degradation, finding that while tight clothing yields 1.5–3.5cm errors, loose garments inflate errors to 5–12+cm (a 30-150% increase). Furthermore, the Loose Inertial Poser (LIP) framework (CVPR 2024) attempted fabric-aware pose recovery. A vital distinction, however, separates loose/draping fabrics (low stiffness, dynamic conformity) from rigid canvas (high stiffness, structural creases). The 14oz Karate Gi behaves distinctly, creating persistent false limb boundaries that remain completely uncharacterized in current benchmarking literature.

### 4.4 Temporal Filtering & Physics-Informed Refinement
Standard temporal smoothing techniques, such as the classic Kalman filter (Welch & Bishop, 1995) or Butterworth low-pass/Savitzky-Golay filters, assume continuous signal noise and effectively "smooth away" the genuine kinematic spikes (Kime) that define martial arts strikes, resulting in unacceptable phase lag and amplitude attenuation. More advanced approaches, such as confidence-gated adaptive filtering, dynamically scale the measurement noise covariance $R_t$ based on detection confidence. Event-triggered architectures activate smoothing selectively during detected occlusion events. Recent works like PhysPose introduce differentiable physics priors to enforce biomechanical plausibility. Despite these advancements, there exists no event-triggered, confidence-gated filter specifically designed to handle the abrupt, non-linear occlusion events generated by rigid canvas during ballistic motion.

### 4.5 Biomechanical Modeling for Center of Gravity
Tracking the Center of Gravity (CoG) is paramount for analyzing balance and power generation in martial arts. While marker-based systems rely on precise anatomical landmarks, markerless CoG estimation requires robust segmental modeling. The primary standard remains de Leva (1996), which adjusted Zatsiorsky-Seluyanov's segment inertia parameters to derive accurate mass proportions for linked-segment models. Dumas et al. (2007) later provided modern ISB-compliant 3D inertia tensors as a valuable complement. Needham et al. (2021) successfully demonstrated markerless CoG tracking in linear sprinting. Applying de Leva's mass-weighted calculations to 2D coordinates provides a computationally viable pathway for CoG estimation, provided the underlying coordinate data can be sufficiently stabilized.

### 4.6 Research Gap Summary
The following taxonomy table contextualizes the proposed study against the existing state-of-the-art:

| Study | Application Domain | Kinetic Velocity | Garment Variable Isolated | Fabric Type Evaluated |
| :--- | :--- | :--- | :--- | :--- |
| Echeverria (2021) | Karate Kumite | High | No | Uncontrolled |
| Lu (2022) | Static Karate | Low | No | Uncontrolled |
| Yamaguchi (2022) | General Motion | Low/Medium | Yes | Loose (Low Stiffness) |
| LIP (2024) | General Motion | Medium | Yes | Loose (Low Stiffness) |
| **Proposed Study** | **Karate Kihon** | **High (>8m/s)** | **Yes** | **Rigid 14oz Canvas** |

**Conclusion:** No prior research systematically isolates rigid canvas as a controlled independent variable in ballistic martial arts HPE, nor provides a targeted physics-informed filtering solution.

---

# 5. Research Aim, Objectives & Questions

### 5.1 Aim
To investigate, quantify, and mitigate the degradation of monocular markerless human pose estimation caused by rigid heavyweight canvas garment occlusion during high-velocity Shotokan karate techniques.

### 5.2 Objectives
1. Design and execute a rigorously controlled dual-garment video capture protocol that isolates 14oz canvas occlusion as the sole independent variable.
2. Benchmark tracking degradation—specifically spatial jitter and confidence volatility—in MediaPipe and YOLOv8-Pose architectures.
3. Develop an event-triggered, confidence-gated kinematic Extended Kalman Filter (EKF) utilizing Newtonian momentum projection.
4. Implement biomechanical Center of Gravity (CoG) and stance depth tracking using de Leva's segmental anthropometric model.
5. Statistically validate the proposed filter's performance against raw network output, continuous standard Kalman filtering, and linear interpolation baselines.

### 5.3 Research Questions
- **RQ1:** To what extent does 14oz rigid canvas garment occlusion degrade spatial jitter and detection confidence in MediaPipe and YOLOv8-Pose during standardized Shotokan karate techniques?
- **RQ2:** Does an event-triggered, confidence-gated kinematic filter significantly reduce coordinate noise while preserving genuine ballistic kinematic extrema (Kime) compared to standard smoothing techniques?
- **RQ3:** Can filtered 2D coordinates, when synthesized with de Leva's segmental model, yield biomechanically valid and stable Center of Gravity trajectories under extreme garment occlusion?

### 5.4 Hypotheses
- **H1:** Spatial jitter at the hip and knee joints will be significantly greater under the 14oz Gi (Set B) compared to the form-fitting control condition (Set A).
- **H2:** The proposed event-triggered EKF will produce a lower Mean Per Joint Position Error (MPJPE) and higher Percentage of Correct Keypoints (PCK@0.2) than raw outputs, standard Kalman filters, and linear interpolation baselines.
- **H3:** CoG trajectories reconstructed from the EKF-filtered coordinates will exhibit significantly reduced artificial vertical and horizontal displacement compared to those derived from unfiltered occluded output.

---

# 6. Methodology

### 6.1 Research Philosophy & Design
The research adopts a positivist paradigm, employing a quantitative, experimental methodology. The core structure is a within-subject paired comparison across two controlled states: Set A (Control) and Set B (Experimental), replicated across a multi-practitioner cohort ($N=3$) with skill-level stratification. This design enables both within-subject (paired garment comparison) and between-subject (skill-level dependent degradation) analyses, providing a comprehensive characterisation of rigid-garment occlusion effects across elite Shotokan practitioners. The multi-practitioner approach is grounded in sports biomechanics best practices (Bates, 1996; Mullineaux et al., 2001), where small cohorts of elite athletes with high within-subject trial replicates yield superior signal clarity compared to larger samples of heterogeneous novices.

### 6.2 Participant Cohort & Ethical Considerations
Three expert Shotokan karate practitioners will form the study cohort, deliberately selected to span a range of competitive experience, technical maturity, and anthropometric variation:

| ID | Grade | Experience & Credentials | Role in Study |
| :--- | :--- | :--- | :--- |
| **P01** | 1st Dan Black Belt | Active practitioner; BSc AI researcher | Primary investigator-participant; protocol development & validation |
| **P02** | Black Belt (Elite) | **Asian Karate Championship winner; National champion** | Elite competition-level kinematics — highest velocity executions; maximal Gi stress |
| **P03** | 4th Dan Shotokan | Senior master instructor; 20+ years of teaching experience | Most refined, biomechanically "textbook" technique; gold-standard form reference |

**Justification for N=3:**
- **Statistical Power:** Each practitioner performs 10 repetitions × 3 techniques × 2 conditions = 60 trials. Across the cohort, this yields **180 paired trials** — substantially exceeding the minimum recommended within-subject replication count for biomechanical studies (Bates, 1996; Dufek et al., 1995).
- **Skill-Level Stratification:** The cohort spans a meaningful gradient from competitive black belt (P01) through continental champion (P02) to senior master (P03). This enables a secondary research question: *does Gi-induced degradation scale with execution velocity and explosive power?* P02's championship-speed strikes are expected to induce the most severe tracking failure, while P03's controlled kata-tempo execution provides a lower-velocity reference.
- **Anthropometric Diversity:** Three practitioners of different body compositions ensure that findings are not artefacts of a single body type interacting with a specific Gi cut.
- **Inter-Practitioner Reliability:** Replication across three independent athletes transforms the study from a single-case feasibility pilot into a legitimate small-cohort validation study, strengthening both internal and external validity.

**Ethics:** Full approval from the university ethics board will be secured prior to data collection, adhering strictly to institutional guidelines. Written informed consent will be obtained from all three participants. No vulnerable populations are involved. Participants may withdraw at any time without consequence.

### 6.3 Data Collection Protocol

#### 6.3.1 Hardware & Camera Specifications

| Parameter | Primary Stream | Validation Stream |
| :--- | :--- | :--- |
| Device | iPhone 15 Pro Max | iPhone 15 Pro Max |
| Resolution | 4K (3840×2160) | 1080p |
| Framerate | 60 FPS | 240 FPS (High-Speed) |
| Shutter Speed | 1/500s | 1/500s |
| Color Profile | SDR Rec.709 | SDR Rec.709 |
| Auto-Functions | Locked AE/AF | Locked AE/AF |

#### 6.3.2 Spatial & Geometric Calibration
- **Tripod Configuration:** Height set at 1.0m ± 0.02m (aligned roughly with the subject's iliac crest).
- **Distance & Orientation:** Camera positioned 4.5m orthogonal to the primary plane of execution.
- **Field of View (FOV):** Framed to capture the full body extending through maximum translation, with a 0.5m safety margin on all borders.
- **Angles:** A 90° true sagittal view (primary analysis) and a 45° oblique view (supplementary/qualitative).
- **Scale:** A 1.00m rigid calibration rod placed precisely in the focal plane to establish the pixel-to-metric scaling factor ($S_{px}$).

#### 6.3.3 Lighting & Background
Illumination will be maintained at ≥500 lux using diffuse, balanced indoor lighting, verified via a physical lux meter prior to each session. The backdrop will be matte, neutral, and uniform, strictly avoiding backlighting or high-contrast shadows that could artificially assist or hinder contour detection.

#### 6.3.4 Garment Conditions
- **Set A (Control):** Matte black form-fitting compression clothing. To provide a rigorous proxy ground truth, 25mm high-contrast fluorescent adhesive markers will be affixed to 6 critical anatomical landmarks: lateral malleolus (ankle), lateral femoral condyle (knee), greater trochanter (hip), acromion (shoulder), lateral epicondyle (elbow), and ulnar styloid (wrist).
- **Set B (Experimental):** A pristine 14oz (≈400 g/m²) 100% cotton canvas Karate Gi, secured with a standard WKF-approved knot, representing the extreme rigidity and mass typical of elite kata competition attire.

```text
[Diagrammatic Representation: Control vs Experimental]
     SET A (Control)                 SET B (Experimental)
   Compression + Markers               14oz Canvas Gi
        o (Head)                         O (Head)
       /|\                              /|\ (Hidden contours)
      / | \                            / | \
       / \                             // \\ (Thick folds)
      /   \                           //   \\
```

#### 6.3.5 Standardized Technique Execution
A strict 4-phase audio-cued execution protocol will be enforced using a 60 BPM metronome to standardize temporal pacing:
- **[0s]** Static Calibration Pose (Yoi / T-pose)
- **[3s]** Stance Initiation & Preparation
- **[4s]** Execution resulting in Peak Kime (maximal focal tension)
- **[5s]** Isometric Hold (sustaining Kime)
- **[8s]** Controlled Recovery

**Selected Techniques:**
1. **Oi-zuki (Lunge Punch in Zenkutsu-dachi):** Evaluates rapid anterior translation, ankle/knee velocity, and ballistic elbow extension.
2. **Gedan-barai (Downward Block in Zenkutsu-dachi):** Induces severe canvas folding across the torso and upper arm, coupled with rapid deceleration.
3. **Stance Transition (Zenkutsu-dachi to Kiba-dachi):** Maximizes canvas folding across the pelvic and femoral regions while forcing a distinct multi-planar CoG shift.

*Volume:* 3 techniques × 10 repetitions × 2 conditions × 3 practitioners = **180 primary execution trials**, augmented by subsampled 240 FPS validation takes per practitioner.

#### 6.3.6 Counterbalancing & Fatigue Control
To isolate garment effects from muscular fatigue or practice effects, strict counterbalancing is mandated:
- **Condition Order:** For each practitioner, condition order is counterbalanced across sessions. Practitioner P01 and P03 begin with Set A; P02 begins with Set B. If multiple sessions are conducted per practitioner, the starting condition alternates.
- **Rest Protocol:** A mandatory 5–10 minute rest period is enforced between conditions to mitigate cumulative fatigue.
- **Perceived Exertion:** The Rating of Perceived Exertion (RPE) will be logged after each condition block to ensure physiological equivalence across conditions.
- **Technique Randomisation:** Technique order within each condition block is randomised to prevent systematic ordering effects.
- **Inter-Practitioner Standardisation:** All three practitioners will be briefed using the same verbal cues, metronome cadence, and SOP documentation to maximise execution consistency. Each practitioner will perform a familiarisation set (3–5 warm-up repetitions, not recorded) before data capture begins.

#### 6.3.7 Recording Procedure
1. Execute pre-flight checklist (lens cleaning, tripod leveling, lighting verification, metronome sync).
2. Initiate recording sequence.
3. Perform slate identifier and audio-sync clap.
4. Assume 3-second static calibration posture.
5. Execute 10 consecutive repetitions conforming to metronome cues (3-second resets between reps).
6. Hold final static posture for 3 seconds.
7. Terminate recording and log metadata into `data_collection_sheet.csv`.

#### 6.3.8 File Management
- **Directory Structure:** `/raw_data/calibration/`, `/raw_data/set_a_control/`, `/raw_data/set_b_experimental/`
- **Naming Convention:** `[SET]_[TECHNIQUE]_[TAKE_ID]_[FPS]FPS.mp4`

### 6.4 Data Processing Pipeline

#### 6.4.1 Frame Extraction & Verification
Video streams will be decoded using OpenCV (`VideoCapture`), with rigorous frame-rate constancy checks. The 1.00m calibration rod will be localized to compute the metric scaling factor $S_{px} = \text{pixels/meter}$.

#### 6.4.2 Pose Inference
- **MediaPipe Pose:** Extracts 33 3D landmarks providing $(x, y, z)$ coordinates and a visibility score $v \in [0,1]$.
- **YOLOv8x-Pose:** Extracts 17 COCO keypoints providing $(x, y)$ coordinates and a confidence score $c \in [0,1]$.
Inference is executed strictly on a per-frame basis, explicitly disabling any internal multi-frame temporal smoothing, to capture the raw, unadulterated degradation profile.

#### 6.4.3 Degradation Metrics (Paper 1)
**Spatial Jitter ($J_t$):** Measures the erratic inter-frame displacement of a specific joint $j$.
$$ J_t^{(j)} = \sqrt{\left(x_t^{(j)} - x_{t-1}^{(j)}\right)^2 + \left(y_t^{(j)} - y_{t-1}^{(j)}\right)^2} $$

**Confidence Score Volatility ($V_c$):** Quantifies the instability of the network's predictive certainty.
$$ V_c^{(j)} = \frac{1}{M-1} \sum_{t=2}^{M} \left| c_t^{(j)} - c_{t-1}^{(j)} \right| $$

**Joint Angle Error (New Metric):** Anatomical joint angles (e.g., knee flexion, hip abduction) will be calculated. Angular kinematics are intrinsically more robust to global silhouette translation and provide a biomechanically relevant measure of structural degradation.

#### 6.4.4 Per-Repetition Aggregation
**CRITICAL:** Frame-level data cannot be treated as independent observations due to massive temporal autocorrelation at 60 FPS (16.7ms intervals). Treating frames independently constitutes severe pseudoreplication. Therefore, all metrics will be aggregated per repetition (e.g., median spatial jitter across the rep, AUC of the jitter profile, peak velocity). This yields $N=30$ paired observations per technique/condition combination (10 reps × 3 practitioners), or $N=10$ per practitioner for within-subject analyses.

### 6.5 Statistical Analysis Plan

#### 6.5.1 Pre-Registration Statement
To ensure maximal scientific rigor and prevent p-hacking, all hypotheses, statistical tests, and data exclusion rules outlined in this document are pre-registered a priori.

#### 6.5.2 Unit of Analysis
The fundamental unit of analysis is the individual movement repetition. As articulated by Hurlbert (1984), failing to aggregate highly autocorrelated temporal frames artificially inflates degrees of freedom, guaranteeing spurious statistical significance. Per-repetition aggregation ensures the independence assumption of inferential statistics is maintained. For pooled analyses, $N=30$ paired observations (10 reps × 3 practitioners) are available per technique/condition. For within-subject analyses stratified by practitioner, $N=10$ paired observations per practitioner are available.

#### 6.5.3 Primary Statistical Tests

**Within-Subject Paired Comparison (Primary):**
The **Wilcoxon Signed-Rank Test** (two-tailed, paired, non-parametric) will compare aggregated metrics between Set A and Set B within each practitioner.
- *Justification:* Given the per-practitioner sample size ($N=10$) and the anticipated right-skewed jitter distributions, the non-parametric Wilcoxon test is robust and appropriate.
- *Assumptions:* Paired differences are independent (each repetition is a separate motor execution); derived from a continuous distribution; approximately symmetric (verified via visual inspection).

**Pooled Analysis (Secondary):**
A **Linear Mixed-Effects Model (LMM)** will be fitted with garment condition (Set A vs. Set B) as the fixed effect, and practitioner as a random intercept, to properly account for the nested structure (repetitions within practitioners). This allows estimation of the overall garment effect while modelling inter-practitioner variability:

$$Y_{ijk} = \beta_0 + \beta_1 \cdot \text{Condition}_{j} + u_i + \varepsilon_{ijk}$$

where $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is the random practitioner intercept and $\varepsilon_{ijk}$ is the residual error.

**Skill-Level Stratification (Exploratory):**
To test whether Gi-induced degradation scales with execution velocity and skill level, the **Kruskal-Wallis test** (non-parametric one-way ANOVA) will compare the magnitude of garment-induced jitter increase ($\Delta J = J_{\text{SetB}} - J_{\text{SetA}}$) across the three practitioners, followed by Dunn's post-hoc pairwise comparisons if significant.

#### 6.5.4 Effect Size Reporting
Significance testing will be heavily supplemented by effect size quantification using the **matched-pairs rank-biserial correlation ($r_{rb}$)**:
$$ r_{rb} = \frac{W_+ - W_-}{W_+ + W_-} $$
Thresholds for interpretation: 0.1 (Small), 0.3 (Medium), 0.5 (Large). Additionally, the Hodges-Lehmann median difference estimator will be reported with 95% Confidence Intervals to provide magnitude context in native metric units (pixels/cm).

#### 6.5.5 Multiple Comparisons Correction
To control for Type I error inflation across numerous joint and metric tests, the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure will be applied. Given potentially 24 parallel tests (6 joints × 2 models × 2 metrics), FDR will be controlled at $q = 0.05$. Results will report both nominal and FDR-corrected p-values. To further mitigate testing burden, joints may be statistically pooled into functional regions (Upper Limb, Torso, Lower Limb).

#### 6.5.6 Normality Assessment
The Shapiro-Wilk test will evaluate the normality of paired differences. If normality is supported ($p > 0.05$), standard paired t-tests will be reported alongside the Wilcoxon tests to demonstrate robustness and analytic stability.

#### 6.5.7 Reliability Metrics
To substantiate the experimental design, both intra- and inter-practitioner reliability will be rigorously quantified:

**Intra-Practitioner (Within-Subject):**
- **Coefficient of Variation (CV%)** across the 10 replicates per condition per practitioner.
- **Standard Error of Measurement (SEM)**.
- **Minimal Detectable Change (MDC)**, establishing the noise floor above which algorithm differences are meaningful.

**Inter-Practitioner (Between-Subject):**
- **Intraclass Correlation Coefficient (ICC 3,1)** — assessing consistency of degradation patterns across the three practitioners. A high ICC (>0.75) would indicate that Gi-induced degradation is a systematic, practitioner-independent phenomenon.
- **Execution Velocity Profiling** — peak limb velocities will be extracted per practitioner to verify the expected skill-level velocity gradient (P02 > P01 > P03 for explosive techniques; P03 > P01 ≈ P02 for controlled kata-tempo execution). This validates that inter-practitioner differences in degradation magnitude can be meaningfully attributed to velocity rather than random variation.

#### 6.5.8 Advanced Analysis Frameworks
Where feasible, **Statistical Parametric Mapping (SPM1D)** (Pataky et al., 2012; 2015) will be utilized. SPM allows for the topological comparison of entire continuous 1D kinematic waveforms (e.g., a full punch cycle) rather than relying on arbitrary zero-dimensional point-estimates (like peak values), providing a vastly more comprehensive temporal understanding of occlusion impact. Additionally, Linear Mixed-Effects Models (LMM) with AR(1) temporal correlation structures will be explored to model frame-within-rep nested hierarchies.

#### 6.5.9 Software Stack
Python 3.10+ utilizing: `SciPy` (Wilcoxon, Shapiro), `statsmodels` (LMM), `spm1d` (waveform topology), `pingouin` (effect sizes), `pandas`, and `seaborn`.

### 6.6 Phase 2: Event-Triggered Kinematic Filter (Paper 2)

#### 6.6.1 State-Space Representation
The filter operates on a per-joint basis. The kinematic state vector $x_t \in \mathbb{R}^6$ encompasses position, velocity, and acceleration:
$$ x_t = \begin{bmatrix} x_t & y_t & \dot{x}_t & \dot{y}_t & \ddot{x}_t & \ddot{y}_t \end{bmatrix}^T $$

A constant-acceleration Newtonian transition model ($\Delta t = 1/60$s) drives the system:
$$ x_t = F \cdot x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) $$
$$ F = \begin{bmatrix} 1 & 0 & \Delta t & 0 & 0.5\Delta t^2 & 0 \\ 0 & 1 & 0 & \Delta t & 0 & 0.5\Delta t^2 \\ 0 & 0 & 1 & 0 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 & \Delta t \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix} $$

#### 6.6.2 Confidence-Gated Measurement Update
The observation vector $z_t = [z_x, z_y]^T$ is coupled with the network's confidence score $c_t \in [0,1]$. A defining innovation of this research is the dynamic scaling of the measurement noise covariance matrix $R_t$:
$$ R_t = \sigma_0^2 \cdot \exp(\gamma(1 - c_t)) \cdot I_2 $$
Where $\sigma_0^2$ is the baseline sensor noise and $\gamma$ is a tuning parameter dictating the exponential penalty applied to low-confidence detections.

**Event-Trigger Logic:**
- **High Confidence ($c_t \geq T_{thresh}$):** The system executes a standard Kalman measurement update, incorporating the observation $z_t$ via the Kalman Gain $K_t$.
- **Occlusion Event ($c_t < T_{thresh}$):** The observation is rejected. The filter enters a predict-only phase, utilizing Newtonian momentum projection ($\ddot{x}, \dot{x}$) to estimate position through the occlusion gap.
$T_{thresh}$ will be empirically optimized on a separate held-out validation set.

#### 6.6.3 Anatomical Constraints
To prevent biomechanically impossible projections during extended occlusions, the filter enforces soft constraints based on rigid bone lengths ($||p_{parent} - p_{child}|| \approx L_{segment}$) and physiological joint angle ranges.

#### 6.6.4 Biomechanical CoG Calculation
The global Center of Gravity (CoG) is reconstructed by integrating the filtered 2D coordinates with de Leva's (1996) segmental mass distribution parameters:
- Head/Neck: 0.0681
- Trunk: 0.4346
- Upper Arm (×2): 0.0263
- Forearm (×2): 0.0150
- Hand (×2): 0.0059
- Thigh (×2): 0.1416
- Shank (×2): 0.0433
- Foot (×2): 0.0137

Global coordinates are computed as mass-weighted sums:
$$ X_{CoG} = \sum_{i} (w_i \cdot x_i), \quad Y_{CoG} = \sum_{i} (w_i \cdot y_i) $$
Stance Depth ($D_{stance}$) is defined dynamically:
$$ D_{stance} = |x_{ankle\_lead} - x_{ankle\_rear}| \times S_{px}^{-1} $$

#### 6.6.5 Evaluation Metrics
The filter's efficacy will be validated against the Set A pseudo-ground truth using:
1. **Mean Per Joint Position Error (MPJPE)**
2. **Percentage of Correct Keypoints (PCK@0.2)** normalized by torso diameter.
3. **Phase Lag Analysis:** Ensuring the filter does not temporally delay the critical moment of peak Kime.
4. **Peak Velocity Preservation Ratio:** Ensuring high-frequency ballistic spikes are not smoothed out.

#### 6.6.6 Baselines for Comparison
1. Raw, unadulterated occluded output (Set B).
2. Continuous Standard Kalman Filter (SKF) without confidence gating.
3. Simple Linear Interpolation spanning regions where $c_t < 0.50$.
4. The proposed Event-Triggered EKF.

### 6.7 Ablation Study Plan (Paper 2)
To rigorously isolate the contribution of specific algorithmic innovations, an ablation study will systematically deactivate components:
1. **Confidence Gating Only:** Standard Kalman updates, but $R_t$ dynamically scales (no momentum projection).
2. **Momentum Projection Only:** Fixed $R_t$, but activates projection when $c_t < T_{thresh}$.
3. **No Event Triggering:** Continuous application of dynamic $R_t$ without binary thresholding.
4. **Full System:** The complete integrated architecture.

---

# 7. Hardware & Software Infrastructure

| Category | Specification / Tool |
| :--- | :--- |
| **Data Capture** | iPhone 15 Pro Max (60 FPS Primary, 240 FPS High-Speed Validation) |
| **ML Inference Engine** | ASUS ROG G513 (AMD Ryzen 7 6800H, NVIDIA RTX 3060 6GB, CUDA 12.1, cuDNN 8.9) |
| **Analysis & Drafting** | MacBook Air M4 |
| **Software Stack** | Python 3.10+, OpenCV, MediaPipe, Ultralytics YOLOv8, Pandas, NumPy, SciPy, statsmodels, spm1d, pingouin, Matplotlib, Seaborn, LaTeX |

> [!NOTE] 
> The deliberate selection of lightweight inference models (MediaPipe, YOLOv8) and consumer-grade hardware aligns with the project's foundational objective: developing solutions viable for real-world, decentralized dojo deployment, rather than relying on inaccessible HPC clusters.

---

# 8. Ethical Considerations
The research will adhere strictly to the ethical frameworks mandated by the University of Plymouth and NSBM Green University. Formal ethical approval will be secured prior to data collection. Written informed consent will be obtained from all three participants (P01, P02, P03), ensuring each understands the study objectives, data usage, and their right to withdraw at any time without consequence. No vulnerable populations are involved. All video and metadata will be stored on encrypted local storage with strict adherence to GDPR principles regarding the processing and retention of biometric video data. Participants' visual likenesses in video recordings will be handled in accordance with data protection regulations; anonymisation or pseudonymisation will be applied in all published materials unless explicit consent for identifiable imagery is obtained.

---

# 9. Expected Contributions
1. **Empirical Benchmark:** The first rigorously controlled dataset isolating and quantifying the degradative impact of rigid 14oz canvas occlusion in high-velocity martial arts HPE, validated across three elite practitioners (1st Dan to 4th Dan) including a continental championship winner.
2. **Skill-Level Stratification Analysis:** Novel evidence characterising whether Gi-induced tracking degradation scales with execution velocity and competitive skill level, providing actionable insights for model deployment across practitioner populations.
3. **Methodological Rigor:** Demonstration of robust statistical frameworks (avoiding pseudoreplication, applying FDR correction, using mixed-effects models) for small-cohort biomechanical time-series analysis in computer vision.
4. **Algorithmic Innovation:** The architecture and validation of a novel event-triggered, confidence-gated kinematic EKF capable of preserving ballistic motion while mitigating occlusion noise.
5. **Biomechanical Application:** A validated pipeline synthesising filtered 2D pose coordinates with de Leva's segmental model to achieve robust Center of Gravity tracking under severe occlusion.
6. **Open Science:** Delivery of an open-source Python processing pipeline tailored for martial arts analytics.

---

# 10. Limitations & Threats to Validity
- **Cohort Size:** While the $N=3$ multi-practitioner design with 180 total paired trials substantially exceeds the statistical power of a single-subject pilot, the cohort remains small relative to large-scale population studies. Findings are therefore generalisable to elite-level Shotokan practitioners but should be validated with broader cohorts encompassing different martial arts disciplines, belt levels, and demographics before claiming universal applicability.
- **Ground Truth Constraints:** Set A (compression + markers) serves as a robust proxy ground truth, but lacks the sub-millimeter precision of a 12-camera optoelectronic Vicon MoCap system. The 240 FPS high-speed validation subsample partially mitigates this limitation.
- **Model Scope:** Benchmarking is limited to MediaPipe and YOLOv8-Pose. Evaluation of heavy transformer architectures (e.g., ViTPose) is reserved for future work due to edge deployment constraints.
- **Kinematic Confounding:** The physical weight and drag of a 14oz Gi genuinely alter the wearer's kinematics compared to compression gear. While the filter targets algorithmic tracking error, it cannot perfectly decouple this from true physical kinematic alterations induced by the garment. Inter-practitioner velocity profiling partially addresses this by establishing whether degradation correlates with velocity rather than garment interaction alone.
- **Threshold Sensitivity:** The event-trigger threshold ($T_{thresh} = 0.50$) is empirically derived and data-driven; it may require recalibration across different architectures or martial arts disciplines.
- **Dimensionality:** The analysis is constrained to 2D monocular estimation. Full 3D biomechanical reconstruction requires multi-view stereo triangulation setups outside the scope of this mobile-first objective.

---

# 11. Project Timeline

| Phase | Months | Key Activities | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **1. Foundations** | 1–3 | Comprehensive literature review; finalize Python pipeline setup; secure ethics board approval. | Annotated bibliography; functional data extraction scripts. |
| **2. Acquisition** | 4–6 | Execute counterbalanced video capture protocol; verify data quality; sync streams. | Complete Set A/B video corpus + 240 FPS validation set. |
| **3. Benchmark (Paper 1)** | 7–12 | Run pose inference; compute spatial jitter/confidence metrics; execute statistical analysis; draft manuscript. | Manuscript 1 Drafted (Benchmark & Quantification). |
| **4. Integration** | 13–16 | Implement de Leva CoG tracking on Set A; develop linear interpolation baseline. | Working CoG visualization overlay; baseline filter functioning. |
| **5. Physics Filter (Paper 2)** | 17–22 | Implement event-triggered EKF; conduct ablation studies; validate against baselines; draft manuscript. | Manuscript 2 Drafted (Filter Architecture & Validation). |
| **6. Submission** | 23–24 | Incorporate supervisor feedback; finalize formatting; final proofreading; submission. | Final dissertation submission (Both manuscripts). |

---

# 12. References

1. Bates, B. T. (1996). Single-subject methodology: An alternative approach. *Medicine & Science in Sports & Exercise*, 28(5), 631–638.
2. Dael, F. A., Absadykov, D., Shayea, I., & Sanida, Y. (2026). Comparative Evaluation of MediaPipe and YOLOv8 for Real-Time Pose Estimation. In *Selected Papers from FICAILY 2025*, Springer Nature, pp. 212–227.
3. de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223–1230.
4. Dumas, R., Chèze, L., & Verriest, J.-P. (2007). Adjustments to McConville et al. and Young et al. body segment inertial parameters. *Journal of Biomechanics*, 40(3), 543–553.
5. Dufek, J. S., Bates, B. T., Stergiou, N., & James, C. R. (1995). Interactive effects between group and single-subject response patterns. *Human Movement Science*, 14(3), 301–323.
6. Echeverria, J., & Santos, O. C. (2021). Toward Modeling Psychomotor Performance in Karate Combats Using Computer Vision Pose Estimation. *Sensors*, 21(24), 8378.
7. Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2), 187–211.
8. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 [Software]. https://github.com/ultralytics/ultralytics
9. Lu, F. (2022). AiShifu: AI Karate Pose Trainer Using Human Pose Estimation. *Journal of Student Research*, 11(4).
10. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.
11. Mullineaux, D. R., Bartlett, R. M., & Bennett, S. (2001). Research design and statistics in biomechanics and motor control. *Journal of Sports Sciences*, 19(10), 739–760.
12. Needham, R., Naemi, R., & Chockalingam, N. (2021). Can Markerless Pose Estimation Algorithms Estimate 3D Mass Centre Positions and Velocities during Linear Sprinting Activities? *Sensors*, 21(8), 2889.
13. Pataky, T. C. (2012). One-dimensional statistical parametric mapping in Python. *Computer Methods in Biomechanics and Biomedical Engineering*, 15(3), 295–301.
14. Pataky, T. C., Vanrenterghem, J., & Robinson, M. A. (2015). Zero- vs. one-dimensional, parametric vs. non-parametric, and confidence interval vs. hypothesis testing procedures in one-dimensional biomechanical trajectory analysis. *Journal of Biomechanics*, 48(7), 1277–1285.
15. Welch, G., & Bishop, G. (1995). An introduction to the Kalman filter. *University of North Carolina at Chapel Hill*.
16. Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. *NeurIPS 2022*.
17. Yamaguchi, T., Mikami, D., Matsumura, S., Saijo, N., & Kashino, M. (2022). Pose estimation for human wearing loose-fitting clothes: Obtaining ground truth posture using HFR camera and blinking LEDs. *arXiv:2212.04820*.
18. Zhang, W., Liu, Z., Zhou, L., Leung, H., & Chan, A. B. (2017). Martial arts, dancing and sports dataset: A challenging stereo and multi-view dataset for 3D human pose estimation. *Image and Vision Computing*, 61, 22–39.
19. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.

---

# 13. Appendices

### Appendix A: Data Collection Standard Operating Procedure
*Please refer to the supplementary standard operating procedure document (`data_collection_protocol.md`) for the exact chronological breakdown of the environmental setup, participant briefing, and camera initialization procedures to guarantee reproducible replication of this dataset.*

### Appendix B: Data Collection Sheet Template
The following metadata structure will be strictly maintained for every recorded trial to ensure analytical integrity and traceability during statistical aggregation:

| Date | Time | Condition | Technique | Rep_ID | RPE | Notes/Anomalies | File_Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YYYY-MM-DD | HH:MM | Set A | Oi-zuki | 01 | 3 | Clean execution | `SetA_Oizuki_01_60FPS.mp4` |
| YYYY-MM-DD | HH:MM | Set A | Oi-zuki | 02 | 3 | Minor heel slip | `SetA_Oizuki_02_60FPS.mp4` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| YYYY-MM-DD | HH:MM | Set B | Gedan-barai | 10 | 6 | Heavy Gi drag noted | `SetB_Gedan_10_60FPS.mp4` |

### Appendix C: Self-Guided Study Index
Key domain-specific terminologies actively explored throughout this dissertation process:

- **Python Engineering:** Object-Oriented processing pipelines, asynchronous video decoding, vectorization (NumPy), decorators for telemetry.
- **Computer Vision:** Convolutional Neural Networks (CNN), bounding box regression, keypoint heatmaps, anchor-free detection, bounding box IOU, temporal smoothing.
- **Mathematics, Physics & Statistics:** Newtonian state transition matrices, covariance, non-parametric paired inference (Wilcoxon), False Discovery Rate (FDR), kinematics (velocity, acceleration, jerk), Center of Gravity (CoG) integration, Anthropometry.
