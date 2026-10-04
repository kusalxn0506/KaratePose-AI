# Kinematic Refinement of Coordinate-Based Human Pose Estimation: The Impact of Rigid Garment Occlusion (Paper 1 Draft)

**Author:** Kusal Nirmala
*BSc (Hons) Artificial Intelligence — NSBM Green University / University of Plymouth*

---

## Abstract
The application of monocular Human Pose Estimation (HPE) in sports biomechanics represents a paradigm shift toward democratized, ubiquitous motion capture. However, lightweight edge-deployable architectures such as MediaPipe Pose and YOLOv8-Pose suffer severe degradation when confronted with the compounding challenges of high-velocity ballistic movements and extreme rigid garment occlusion. This research establishes the first rigorously controlled benchmark quantifying the extent of this tracking degradation by comparing performance under a 14oz canvas Gi against a form-fitting control condition. Utilising a multi-practitioner cohort (N=3), the study employs a within-subject, repeated-measures experimental design. Spatial jitter, confidence volatility, and joint angle error are statistically modelled using non-parametric inference with False Discovery Rate correction.

---

## 1. Introduction
The evolution of computer vision, particularly in the domain of Human Pose Estimation (HPE), has catalysed profound advancements in sports analytics and biomechanical modeling. The transition from marker-based optoelectronic systems (e.g., Vicon) to markerless, monocular HPE offers unparalleled ecological validity, allowing performance capture outside the constrained laboratory environment. Contemporary lightweight models, notably MediaPipe Pose (BlazePose) and YOLOv8-Pose, provide real-time, edge-capable inference. However, their reliability precipitously declines when subjected to the dual perturbations of high-velocity kinetics and complex garment occlusion. 

The central problem addressed in this research is the failure of these lightweight models under concurrent high-velocity motion and rigid garment occlusion. The Shotokan karate discipline serves as an optimal testbed due to the unique properties of the traditional attire. The 14oz canvas Karate Gi is a heavyweight, high-stiffness garment. Unlike casual wear or winter clothing which drape and conform dynamically to the body's contours, the rigid canvas forms persistent, geometric folds. These folds create sharp edge gradients and false limb boundaries that deceive convolutional and transformer-based feature extractors. 

This compounding effect—where ballistic velocities exceeding 8 m/s intersect with rigid canvas occlusion—precipitates systemic confidence collapse and extreme spatial jitter in pose estimation outputs. The algorithms struggle to differentiate between the true anatomical joint and the structural vertex of a canvas fold.

## 2. Related Work
### 2.1 Human Pose Estimation Architectures
The landscape of HPE is broadly divided into top-down and bottom-up paradigms. MediaPipe Pose (Lugaresi et al., 2019) represents a state-of-the-art top-down pipeline optimised for real-time mobile inference, outputting 33 3D landmarks. Conversely, YOLOv8-Pose (Jocher et al., 2023) is a single-stage, anchor-free model predicting 17 COCO keypoints simultaneously, offering superior multi-person handling. Both architectures prioritise speed over absolute precision, making them susceptible to occlusion-induced jitter.

### 2.2 Computer Vision in Martial Arts
The application of CV in martial arts has seen growing interest (Echeverria & Santos, 2021; Lu, 2022). Recently, Dael et al. (2026) conducted comparative evaluations of MediaPipe and YOLOv8 across various dynamic activities. However, a critical gap pervades this literature: these studies operate under the implicit assumption of standard clothing, failing to isolate the specific degradation caused by martial arts attire.

### 2.3 Clothing-Induced Occlusion in Pose Estimation
The specific impact of clothing on HPE has only recently garnered rigorous attention. Yamaguchi and colleagues (2022, 2023) quantified degradation under loose-fitting clothes, finding error inflations of 30-150%. A vital distinction, however, separates loose/draping fabrics from rigid canvas. The 14oz Karate Gi behaves distinctly, creating persistent false limb boundaries that remain completely uncharacterized in current benchmarking literature.

## 3. Methodology
### 3.1 Research Philosophy & Design
The research adopts a positivist paradigm, employing a quantitative, experimental methodology. The core structure is a within-subject paired comparison across two controlled states: Set A (Control - Compression + Markers) and Set B (Experimental - 14oz Canvas Gi), replicated across a multi-practitioner cohort (N=3) with skill-level stratification.

### 3.2 Data Collection Protocol
A strict 4-phase audio-cued execution protocol is enforced using a 60 BPM metronome.
- **Selected Techniques:** Oi-zuki (Lunge Punch), Gedan-barai (Downward Block), Stance Transition.
- **Volume:** 3 techniques × 10 repetitions × 2 conditions × 3 practitioners = 180 primary execution trials.
- **Counterbalancing:** Condition order is counterbalanced across sessions, with mandatory 5–10 minute rest periods to mitigate cumulative fatigue.

### 3.3 Data Processing & Degradation Metrics
Inference is executed strictly on a per-frame basis, explicitly disabling internal multi-frame temporal smoothing.
1. **Spatial Jitter ($J_t$):** Measures the erratic inter-frame displacement of a specific joint.
2. **Confidence Score Volatility ($V_c$):** Quantifies the instability of the network's predictive certainty.
3. **Joint Angle Error:** Anatomical joint angles providing a biomechanically relevant measure of structural degradation.

All metrics are aggregated per repetition to prevent pseudoreplication, yielding $N=30$ paired observations per technique/condition combination.

### 3.4 Statistical Analysis
- **Wilcoxon Signed-Rank Test:** Two-tailed, paired, non-parametric comparison of aggregated metrics between Set A and Set B.
- **Linear Mixed-Effects Model (LMM):** Fitted with garment condition as a fixed effect and practitioner as a random intercept.
- **Effect Size:** Matched-pairs rank-biserial correlation ($r_{rb}$).
- **FDR Correction:** Benjamini-Hochberg False Discovery Rate controlled at $q = 0.05$.

---

## 4. Results
*(Awaiting 1-2 Month Data Collection)*
- **Figure 1:** Spatial Jitter Comparison (MediaPipe vs YOLOv8)
- **Figure 2:** Confidence Volatility Boxplots
- **Figure 3:** Joint Angle Time-Series Overlay
- **Table 1:** Wilcoxon Test Results and Effect Sizes

## 5. Discussion
*(To be drafted post data-collection)*

## 6. Conclusion
*(To be drafted post data-collection. Will bridge to Paper 2 - Event-Triggered Kinematic Filter)*
