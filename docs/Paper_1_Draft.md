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
- **Context:** The rise of computer vision in sports analytics and the democratization of biomechanical analysis via smartphones.
- **The Problem:** Models like YOLO and MediaPipe are trained primarily on standard datasets (COCO) featuring everyday clothing or sportswear. Traditional martial arts uniforms (Gi) present a unique adversarial challenge: the fabric snaps independently of the limb, masking true joint locations (occlusion) and introducing false high-frequency motion.
- **The Impact:** Without accurate joint tracking, downstream metrics like Center of Gravity (CoG) and velocity profiles become corrupted, rendering the AI useless for serious coaching or injury prevention.
- **Hypothesis:** We hypothesize that the Gi condition will exhibit significantly higher spatial jitter and confidence volatility compared to the Rashguard condition, specifically around terminal extension phases of strikes.

## 2. Related Work
- **Pose Estimation Models:** Briefly compare the architecture of YOLOv8-Pose (top-down, 17 keypoints) vs MediaPipe BlazePose (bottom-up/hybrid, 33 landmarks). 
- **Clothing Occlusion in CV:** Discuss existing literature on how loose clothing degrades bounding box and keypoint detection.
- **Sports Biomechanics AI:** Review previous applications of AI in martial arts and where they fall short (usually ignoring the clothing variable).

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
