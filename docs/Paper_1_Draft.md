# Comparative Analysis of AI Pose Estimation Degradation Under Clothing Occlusion in Karate Biomechanics

**Authors:** Kusal Nirmala (1, 2)
*(1) BSc (Hons) Artificial Intelligence, University of Plymouth*
*(2) NSBM Green University*

---

## Abstract
**Background:** AI-based pose estimation models (e.g., MediaPipe, YOLOv8) are increasingly used for sports biomechanics. However, their accuracy is heavily compromised by loose-fitting traditional uniforms (e.g., a Karate Gi) which occlude joint landmarks and introduce independent fabric motion.
**Objective:** This study quantifies the degradation in spatial accuracy, confidence volatility, and joint angle estimation when using standard pose models on subjects wearing a Karate Gi compared to tight-fitting athletic wear.
**Methods:** Three Karate practitioners executed 10 repetitions of the *Oizuki* (lunge punch) under two conditions: Control (Rashguard) and Experimental (Karate Gi). Video data was processed through MediaPipe BlazePose and YOLOv8-Pose. Metrics including Spatial Jitter (Euclidean pixel displacement), Confidence Volatility, and Joint Angle differentials were extracted. Statistical significance was evaluated using the Wilcoxon signed-rank test and Cohen’s *d* for effect size.
**Results:** [PLACEHOLDER - Awaiting data collection. E.g., The Gi condition introduced a statistically significant increase in maximum spatial jitter (p < 0.05) and a mean confidence drop of X% across elbow and knee joints].
**Conclusion:** The findings demonstrate that standard AI pose estimators fail to maintain kinematic stability under severe clothing occlusion, proving the necessity for secondary physics-based filters (e.g., Extended Kalman Filters) to recover true anatomical Center of Gravity (CoG) in traditional martial arts analysis.

---

## 1. Introduction
The democratization of computer vision has fundamentally transformed sports biomechanics. Traditional motion capture systems, which rely on expensive, multi-camera infrared arrays and reflective markers (e.g., VICON), are increasingly being replaced by markerless AI pose estimation frameworks. Tools such as MediaPipe (BlazePose) and YOLOv8-Pose allow coaches and practitioners to extract kinematic data using standard smartphone video. While these models perform with high accuracy in controlled environments and tight-fitting sportswear, their robustness deteriorates significantly in real-world scenarios involving complex apparel.

Traditional martial arts present a unique adversarial challenge for computer vision models. Practitioners wear a *Gi*—a heavy, loose-fitting cotton uniform designed to snap during terminal extension phases (Kime) of strikes. This fabric motion acts independent of the human anatomical structure, creating severe occlusion of major joints (e.g., elbows and knees) and introducing false high-frequency artifacts into the optical tracking data. When applied without correction, these models track the fabric rather than the underlying bone structure, corrupting critical downstream metrics such as Center of Gravity (CoG), strike velocity, and joint angle approximations.

This study hypothesizes that loose-fitting martial arts apparel introduces statistically significant degradation in both spatial tracking accuracy (jitter) and model confidence compared to tight-fitting athletic wear. By quantifying this degradation across MediaPipe and YOLOv8 architectures, this research aims to highlight the limitations of purely vision-based kinematic analysis in martial arts and establish the necessity for secondary physics-based filtering.

## 2. Related Work

**2.1 Markerless Pose Estimation in Sports**
Recent advancements in human pose estimation (HPE) have yielded two primary architectural paradigms: top-down approaches, heavily utilized by the YOLO (You Only Look Once) family, and bottom-up/hybrid approaches like MediaPipe BlazePose. Research indicates that MediaPipe excels in single-subject, on-device deployments by generating 33 highly detailed 3D landmarks (including distal extremities), making it popular for form analysis. Conversely, YOLOv8-Pose (typically yielding 17 COCO keypoints) relies on bounding-box localization before keypoint extraction, providing superior robustness in multi-person or noisy environments. Despite their widespread use in sports like swimming and athletics, neither architecture is explicitly trained to handle the unique physics of martial arts uniforms.

**2.2 Clothing Occlusion and Tracking Degradation**
Clothing occlusion remains one of the most significant unsolved challenges in 2D and 3D pose estimation. Loose garments hide the precise location of underlying joints, forcing models to rely on learned spatial priors that often fail when the fabric itself undergoes dynamic deformation. Recent literature highlights that "motion artifacts" caused by loose fabric lead to severe drops in spatial accuracy. While some studies attempt to mitigate this by generating synthetic datasets with simulated loose clothing or by utilizing Secondary Motion AutoEncoders (SeMo-AE), these solutions are computationally expensive and rarely implemented in edge-based coaching tools.

**2.3 Biomechanics of Martial Arts**
Existing research into martial arts kinematics heavily relies on physical IMU sensors or expensive marker-based systems to analyze techniques like the *Oizuki* (lunge punch). While a few studies have attempted to apply open-source AI pose estimators to martial arts, they generally assume the AI's output is ground-truth, failing to account for the spatial noise introduced by the Gi. This study bridges the gap by directly benchmarking the AI's degradation caused by the uniform itself, establishing a baseline error metric for future algorithmic corrections.

## 3. Methodology
### 3.1 Experimental Design
- **Participants:** N=3 experienced Karate practitioners.
- **Protocol:** Each participant performs 10 repetitions of *Oizuki* (lunge punch) in a tight Rashguard (Set A) and then in a traditional heavy-cotton Gi (Set B). 
- **Camera Setup:** Static tripod, 60fps, 1080p, lateral/frontal angles.

### 3.2 Computational Pipeline
- **Extraction:** Videos decomposed into frames.
- **Inference Engine:** Frames passed simultaneously to YOLOv8-Pose and MediaPipe.
- **Data Aggregation:** Repetitions segmented to prevent pseudoreplication.

### 3.3 Metrics & Mathematical Formulation
Define the three core metrics we built in the pipeline:
1. **Spatial Jitter:** Euclidean distance between joint $j$ in frame $t$ and frame $t-1$.
   $$ J_{j, t} = \sqrt{(x_{j, t} - x_{j, t-1})^2 + (y_{j, t} - y_{j, t-1})^2} $$
2. **Confidence Volatility:** The variance in the model's self-reported confidence score across the repetition window.
3. **Joint Angle Tracking:** Angle $\theta$ at the elbow/knee computed via dot product of adjacent vectors.

### 3.4 Statistical Analysis
- Shapiro-Wilk test to check for normality (justifying non-parametric tests).
- Wilcoxon signed-rank test for paired comparisons (Rashguard vs. Gi per participant).
- Cohen's *d* to measure the effect size of the degradation.

---

## 4. Results
*(This section will be populated once the 1-2 month data collection is complete)*

### 4.1 Impact on Spatial Jitter
- **[Insert Figure 1: Spatial Jitter Comparison (MediaPipe vs YOLOv8)]**
- *Draft text:* The Gi condition resulted in massive spikes in spatial jitter, particularly during the "snap" phase (Kime) of the punch. MediaPipe showed a jitter increase of X%, while YOLOv8 showed Y%.

### 4.2 Impact on Model Confidence
- **[Insert Figure 2: Confidence Volatility Boxplots]**
- *Draft text:* YOLOv8 confidence scores plummeted by X% when the elbow joint was occluded by the Gi sleeve. 

### 4.3 Joint Angle Distortion
- **[Insert Figure 3: Joint Angle Time-Series Overlay]**
- *Draft text:* The true biomechanical angle of the elbow was distorted by the fabric fold, causing the AI to report a false hyperextension of X degrees.

### 4.4 Statistical Significance
- **[Insert Table 1: Wilcoxon Test Results and Cohen's d values]**

---

## 5. Discussion
- **Interpretation of Findings:** Why did the models fail? (Lack of temporal awareness, training data bias).
- **YOLOv8 vs MediaPipe:** Which model handled the occlusion better and why? (MediaPipe's 33 points might offer better kinematic priors, or YOLO's top-down approach might be more robust to missing limbs).
- **The "Kime" Problem:** Discuss how the snapping of the Gi specifically tricks the optical flow/keypoint trackers into tracking the fabric instead of the wrist.

## 6. Conclusion & Future Work (Bridge to Paper 2)
- Summarize that off-the-shelf AI models are insufficient for accurate Karate biomechanics due to clothing occlusion.
- **The Bridge:** Propose the solution for Paper 2 — implementing an **Extended Kalman Filter (EKF)** with physical kinematic constraints (a "physics police") to smooth the jitter, ignore the fabric snap, and accurately recover the true Center of Gravity (CoG).

---

## References
*(To be populated with citations for MediaPipe, YOLOv8, and relevant biomechanics papers)*
