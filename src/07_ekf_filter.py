# =============================================================================
# KaratePose-AI — Module 07: Event-Triggered Kinematic Extended Kalman Filter
# =============================================================================
#
# PURPOSE:
#   This module implements the core algorithmic contribution of Paper 2.
#   It applies a physics-informed Kalman filter to the raw landmark
#   coordinates produced by Module 02 (pose_inference), producing smoothed
#   trajectories that survive the confidence collapses caused by the 14oz
#   Karate Gi while PRESERVING the genuine ballistic peaks (Kime).
#
# WHY THIS EXISTS:
#   Standard smoothing (Butterworth, Savitzky-Golay) treats ALL high-frequency
#   content as noise. In karate, the fastest part of a punch IS the signal.
#   A standard low-pass filter destroys Kime — the explosive peak that defines
#   elite technique. Our event-triggered approach only activates filtering
#   when the pose estimator's confidence drops (i.e., when the Gi occludes a
#   joint), and otherwise trusts the raw measurement.
#
# RESEARCH NOTE:
#   This module implements §6.6 of the research proposal:
#     §6.6.1 — State-space representation (6-state constant-acceleration model)
#     §6.6.2 — Confidence-gated measurement update (dynamic R_t)
#     §6.6.3 — Anatomical constraints (future extension)
#     §6.6.6 — Four filter modes for baseline comparison
#     §6.7   — Ablation study toggles
#
# HOW TO USE:
#   from src.config import EKF_CONFIG
#   import importlib
#   ekf = importlib.import_module("src.07_ekf_filter")
#
#   filtered_df, diagnostics_df = ekf.run_filter(
#       landmarks_df, model_name="mediapipe", filter_mode="ekf"
#   )
#
# FILTER MODES (§6.6.6):
#   "raw"           — No filtering (passthrough). Baseline for comparison.
#   "skf"           — Standard Kalman Filter. Constant R, no confidence gating.
#   "linear_interp" — Linear interpolation across low-confidence gaps.
#   "ekf"           — Full event-triggered confidence-gated EKF (our method).
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from src.config import PATHS, EKF_CONFIG, JOINTS_MP, JOINTS_YOLO


# =============================================================================
# 1. KALMAN FILTER MATH — Building Blocks
# =============================================================================
# These functions construct the matrices that define the Kalman filter's
# mathematical model of how a joint moves through space.
#
# If you're new to Kalman filters, here's the intuition:
#   - We PREDICT where a joint will be next frame using physics (Newton's laws)
#   - We MEASURE where the pose estimator says it is
#   - We BLEND prediction and measurement, trusting whichever is more reliable
#
# The "Extended" in EKF usually means non-linear dynamics. Our dynamics are
# actually linear (constant acceleration), but the MEASUREMENT NOISE is
# non-linear (it depends on confidence), which motivates the EKF framing
# in the proposal.
# =============================================================================

def build_transition_matrix(dt: float) -> np.ndarray:
    """
    Constructs the 6×6 state transition matrix F for a constant-acceleration
    Newtonian motion model.

    CONCEPT: This matrix encodes Newton's equations of motion:
        position_new = position_old + velocity × Δt + 0.5 × acceleration × Δt²
        velocity_new = velocity_old + acceleration × Δt
        acceleration_new = acceleration_old  (constant acceleration assumption)

    The state vector is: [x, y, ẋ, ẏ, ẍ, ÿ]ᵀ
    where x,y = position, ẋ,ẏ = velocity, ẍ,ÿ = acceleration.

    MATH (from Proposal §6.6.1):
        F = | 1  0  Δt  0   0.5Δt²  0      |
            | 0  1  0   Δt  0       0.5Δt² |
            | 0  0  1   0   Δt      0      |
            | 0  0  0   1   0       Δt     |
            | 0  0  0   0   1       0      |
            | 0  0  0   0   0       1      |

    Args:
        dt: Time step between frames in seconds (1/60 for 60 FPS).

    Returns:
        np.ndarray: 6×6 state transition matrix.
    """
    dt2 = 0.5 * dt * dt  # Half dt-squared, for the kinematic equation
    F = np.array([
        [1, 0, dt, 0,  dt2, 0  ],
        [0, 1, 0,  dt, 0,   dt2],
        [0, 0, 1,  0,  dt,  0  ],
        [0, 0, 0,  1,  0,   dt ],
        [0, 0, 0,  0,  1,   0  ],
        [0, 0, 0,  0,  0,   1  ],
    ], dtype=np.float64)
    return F


def build_observation_matrix() -> np.ndarray:
    """
    Constructs the 2×6 observation matrix H.

    CONCEPT: The pose estimator only gives us (x, y) positions — it doesn't
    directly measure velocity or acceleration. The H matrix extracts just the
    position components from our 6-state vector.

    H = | 1  0  0  0  0  0 |
        | 0  1  0  0  0  0 |

    Returns:
        np.ndarray: 2×6 observation matrix.
    """
    H = np.zeros((2, 6), dtype=np.float64)
    H[0, 0] = 1.0  # We observe x (state index 0)
    H[1, 1] = 1.0  # We observe y (state index 1)
    return H


def build_process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """
    Constructs the 6×6 process noise covariance matrix Q using a
    continuous white-noise jerk model.

    CONCEPT: Our model assumes constant acceleration, but real karate strikes
    involve CHANGING acceleration (jerk). The Q matrix accounts for this
    modelling error by adding uncertainty proportional to how much the
    acceleration might change between frames.

    The white-noise jerk model assumes jerk (rate of change of acceleration)
    is a zero-mean Gaussian process. This is the standard approach in
    target tracking (Bar-Shalom, Li & Kirubarajan, 2001).

    For a single axis with state [pos, vel, acc], the jerk-driven Q block is:
        Q_block = σ_a² × | Δt⁵/20   Δt⁴/8   Δt³/6 |
                          | Δt⁴/8    Δt³/3   Δt²/2 |
                          | Δt³/6    Δt²/2   Δt    |

    We construct the full 6×6 Q by placing two of these blocks diagonally
    (one for x-axis, one for y-axis).

    Args:
        dt: Time step between frames in seconds.
        sigma_a: Standard deviation of the acceleration process noise.
                 Controls how much we expect acceleration to change between frames.
                 Higher = more agile motion allowed. Typical: 0.5–5.0 for human motion.

    Returns:
        np.ndarray: 6×6 process noise covariance matrix.
    """
    # Pre-compute powers of dt
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt

    # Single-axis Q block (3×3) from continuous white-noise jerk model
    q_block = np.array([
        [dt5 / 20, dt4 / 8, dt3 / 6],
        [dt4 / 8,  dt3 / 3, dt2 / 2],
        [dt3 / 6,  dt2 / 2, dt     ],
    ], dtype=np.float64)

    # Full 6×6 Q: two blocks on the diagonal (x-axis and y-axis are independent)
    Q = np.zeros((6, 6), dtype=np.float64)
    Q[0:3, 0:3] = q_block  # x, ẋ, ẍ
    Q[3:6, 3:6] = q_block  # y, ẏ, ÿ

    # Note: The indexing maps state [x, y, ẋ, ẏ, ẍ, ÿ] but we need to
    # rearrange because our state groups by axis: [x, ẋ, ẍ] and [y, ẏ, ÿ].
    # HOWEVER, our state vector is [x, y, ẋ, ẏ, ẍ, ÿ] (interleaved).
    # So we need to map the block to the correct indices.
    #
    # x-axis states are at indices [0, 2, 4]
    # y-axis states are at indices [1, 3, 5]
    Q_full = np.zeros((6, 6), dtype=np.float64)
    x_idx = [0, 2, 4]  # x, ẋ, ẍ
    y_idx = [1, 3, 5]  # y, ẏ, ÿ

    for i in range(3):
        for j in range(3):
            Q_full[x_idx[i], x_idx[j]] = q_block[i, j]
            Q_full[y_idx[i], y_idx[j]] = q_block[i, j]

    return Q_full * (sigma_a ** 2)


def compute_dynamic_R(confidence: float, sigma_0: float, gamma: float) -> np.ndarray:
    """
    Computes the 2×2 measurement noise covariance R_t, dynamically scaled
    by the pose estimator's confidence score.

    CONCEPT: This is the KEY INNOVATION of our filter. Instead of assuming
    a fixed measurement noise (like a standard Kalman filter), we say:

      "If the pose estimator is confident (high c_t), trust its measurement
       more (small R). If it's uncertain (low c_t, probably because the Gi
       is hiding the joint), trust the measurement less (large R)."

    This means during occlusion events, the filter automatically shifts
    towards relying on its physics-based prediction rather than the noisy
    measurement.

    MATH (from Proposal §6.6.2):
        R_t = σ₀² × exp(γ × (1 - c_t)) × I₂

    When c_t = 1.0 (perfect confidence): R_t = σ₀² × exp(0) = σ₀²  (small)
    When c_t = 0.5 (T_thresh):           R_t = σ₀² × exp(2.5)      (≈12× larger)
    When c_t = 0.0 (no confidence):      R_t = σ₀² × exp(5.0)      (≈148× larger)

    Args:
        confidence: The pose estimator's confidence score c_t ∈ [0, 1].
        sigma_0: Base measurement noise standard deviation (σ₀).
        gamma: Exponential scaling factor (γ). Higher = harsher penalty
               for low confidence. Default from config: 5.0.

    Returns:
        np.ndarray: 2×2 measurement noise covariance matrix.
    """
    # Clip confidence to [0, 1] for numerical safety
    c = np.clip(confidence, 0.0, 1.0)

    # Exponential scaling: low confidence → exponentially larger noise
    scale = sigma_0 ** 2 * np.exp(gamma * (1.0 - c))

    # I₂ = 2×2 identity (x and y noise are independent and equal)
    R = scale * np.eye(2, dtype=np.float64)
    return R


# =============================================================================
# 2. SINGLE-JOINT EKF — The Core Algorithm
# =============================================================================
# This is the heart of the module. It processes one joint's time series
# through the full predict-update Kalman cycle, with event triggering.
# =============================================================================

def run_ekf_single_joint(
    xs: np.ndarray,
    ys: np.ndarray,
    confs: np.ndarray,
    dt: float,
    sigma_0: float,
    gamma: float,
    threshold: float,
    sigma_a: float,
    use_confidence_gating: bool = True,
    use_momentum_projection: bool = True,
    use_event_trigger: bool = True,
) -> dict:
    """
    Runs the full event-triggered confidence-gated EKF for a single joint
    across all frames.

    ALGORITHM (per frame):
        1. PREDICT: Use Newton's laws to project where the joint should be.
        2. CHECK CONFIDENCE:
           a) If c_t >= threshold (joint is visible):
              → Compute dynamic R_t based on confidence
              → Execute standard Kalman measurement update
           b) If c_t < threshold (joint is occluded by Gi):
              → SKIP the measurement update
              → Trust the physics prediction (momentum projection)
        3. STORE: Save filtered position and diagnostic info.

    ABLATION TOGGLES (§6.7):
        - use_confidence_gating: If False, R_t is fixed (standard Kalman).
        - use_momentum_projection: If False, always do measurement update
          even during occlusion (ignoring event trigger).
        - use_event_trigger: If False, never skip updates — just scale R_t
          continuously via confidence gating.

    Args:
        xs: Array of raw x-coordinates (normalised 0–1), shape (N,).
        ys: Array of raw y-coordinates (normalised 0–1), shape (N,).
        confs: Array of confidence scores ∈ [0,1], shape (N,).
        dt: Time step between frames (seconds).
        sigma_0: Base measurement noise std.
        gamma: Confidence scaling factor for R_t.
        threshold: Confidence threshold T_thresh for event triggering.
        sigma_a: Process noise acceleration std for Q matrix.
        use_confidence_gating: Enable/disable dynamic R_t scaling.
        use_momentum_projection: Enable/disable predict-only during occlusion.
        use_event_trigger: Enable/disable binary confidence thresholding.

    Returns:
        dict with keys:
            'filtered_x':       np.ndarray of filtered x positions
            'filtered_y':       np.ndarray of filtered y positions
            'event_triggered':  np.ndarray of booleans (True = predict-only frame)
            'kalman_gain_norm': np.ndarray of Kalman gain magnitudes
            'velocity_x':      np.ndarray of estimated x-velocities
            'velocity_y':      np.ndarray of estimated y-velocities
    """
    N = len(xs)

    # ── Build constant matrices ──
    F = build_transition_matrix(dt)
    H = build_observation_matrix()
    Q = build_process_noise(dt, sigma_a)

    # ── Initialise state and covariance ──
    # CONCEPT: We initialise the state at the first measurement.
    # Velocity and acceleration start at zero (we don't know them yet).
    # The initial covariance P must be scaled to match the coordinate system.
    #
    # IMPORTANT: Our coordinates are NORMALISED (0.0–1.0), so the expected
    # range of motion per frame is tiny (~0.001–0.01). Setting P too large
    # (e.g., P=I) causes the velocity/acceleration estimates to wildly
    # overshoot before converging, which is catastrophic if an occlusion
    # event occurs during this transient. We use physically reasonable
    # initial uncertainties instead.
    x_state = np.array([
        xs[0], ys[0],   # Initial position from first measurement
        0.0, 0.0,       # Initial velocity = 0 (unknown)
        0.0, 0.0,       # Initial acceleration = 0 (unknown)
    ], dtype=np.float64)

    # Initial covariance — scaled for normalised coordinates
    # Position uncertainty: ~0.01 (1% of frame, reasonable for pose estimators)
    # Velocity uncertainty: ~0.1 (we don't know initial velocity)
    # Acceleration uncertainty: ~1.0 (we have no idea about acceleration)
    P = np.diag([
        0.01**2, 0.01**2,   # Position variance (x, y)
        0.1**2,  0.1**2,    # Velocity variance (ẋ, ẏ)
        1.0**2,  1.0**2,    # Acceleration variance (ẍ, ÿ)
    ]).astype(np.float64)

    # ── Pre-allocate output arrays ──
    filtered_x = np.zeros(N, dtype=np.float64)
    filtered_y = np.zeros(N, dtype=np.float64)
    event_triggered = np.zeros(N, dtype=bool)
    kalman_gain_norm = np.zeros(N, dtype=np.float64)
    velocity_x = np.zeros(N, dtype=np.float64)
    velocity_y = np.zeros(N, dtype=np.float64)

    # ── First frame: just use the measurement directly ──
    filtered_x[0] = xs[0]
    filtered_y[0] = ys[0]

    # Track consecutive occlusion frames for progressive damping
    coast_frames = 0
    last_good_x = xs[0]
    last_good_y = ys[0]

    # ── Main Kalman loop ──
    for t in range(1, N):
        # ─── STEP 1: PREDICT ─────────────────────────────────────────
        # Project state forward using Newton's laws
        x_pred = F @ x_state
        P_pred = F @ P @ F.T + Q

        # ─── STEP 2: DECIDE — Update or Project? ─────────────────────
        c_t = confs[t]

        # Determine if this is an occlusion event
        is_occluded = (use_event_trigger and c_t < threshold)

        if is_occluded and use_momentum_projection:
            # ── OCCLUSION: Predict-only (momentum projection) ─────────
            # The Gi is hiding this joint. We DON'T trust the measurement.
            # Instead, we coast on our physics prediction.
            #
            # PROGRESSIVE DAMPING STRATEGY:
            # Real data shows occlusion gaps up to 86 frames (1.4s).
            # A pure constant-acceleration coast diverges badly over
            # such long gaps. We use a "fading memory" approach from
            # target tracking literature (Li & Jilkov, 2003):
            #
            #   - Velocity decays gently: preserves short ballistic
            #     projections but prevents runaway over long gaps.
            #   - Acceleration decays faster: ballistic impulse fades
            #     rapidly as the strike completes.
            #   - After many frames, the filter holds roughly the last
            #     known position (graceful degradation to "hold").
            #
            # Decay rates (per frame):
            #   velocity:     0.98 → halves every ~34 frames (~0.57s)
            #   acceleration: 0.90 → halves every ~7 frames (~0.12s)
            #
            # This means for short gaps (5-10 frames): mostly momentum.
            # For long gaps (50+ frames): effectively holding position.
            coast_frames += 1

            x_state = x_pred.copy()
            P = P_pred

            # ADAPTIVE DAMPING: damping increases with coast duration.
            # Short occlusions (1-10 frames): light damping, momentum preserved.
            # Medium occlusions (10-30 frames): moderate damping.
            # Long occlusions (30+ frames): heavy damping, approaching hold.
            #
            # We use a sigmoid-shaped decay that transitions from gentle
            # to aggressive around the 15-frame mark (~0.25s at 60 FPS).
            # This preserves the ballistic projection for brief Gi flutter
            # while preventing runaway drift during extended occlusions.
            #
            # Physics motivation: a karate strike's ballistic phase lasts
            # ~0.1-0.2s. After that, the limb decelerates. So trusting
            # momentum beyond ~15 frames is increasingly unsound.
            vel_decay = max(0.80, 0.98 - 0.01 * coast_frames)   # 0.98→0.80 over 18 frames
            acc_decay = max(0.50, 0.90 - 0.02 * coast_frames)   # 0.90→0.50 over 20 frames

            x_state[2] *= vel_decay   # ẋ (velocity x)
            x_state[3] *= vel_decay   # ẏ (velocity y)
            x_state[4] *= acc_decay   # ẍ (acceleration x)
            x_state[5] *= acc_decay   # ÿ (acceleration y)

            event_triggered[t] = True
            kalman_gain_norm[t] = 0.0  # No measurement incorporated

        else:
            # ── VISIBLE: Kalman measurement update ────────────────────
            # Build the measurement noise matrix R_t
            if use_confidence_gating:
                R = compute_dynamic_R(c_t, sigma_0, gamma)
            else:
                # Standard Kalman: fixed R (as if confidence is always 1.0)
                R = (sigma_0 ** 2) * np.eye(2, dtype=np.float64)

            # The actual measurement from the pose estimator
            z = np.array([xs[t], ys[t]], dtype=np.float64)

            # ── Innovation (measurement residual) ──
            # CONCEPT: "How far off was our prediction from what we actually saw?"
            y_innov = z - H @ x_pred

            # ── Innovation covariance ──
            S = H @ P_pred @ H.T + R

            # ── Kalman Gain ──
            # CONCEPT: K determines how much we trust the measurement vs prediction.
            # If R is large (low confidence), K will be small → trust prediction more.
            # If R is small (high confidence), K will be large → trust measurement more.
            K = P_pred @ H.T @ np.linalg.inv(S)

            # ── State update ──
            x_state = x_pred + K @ y_innov

            # ── Covariance update (Joseph form for numerical stability) ──
            # CONCEPT: The Joseph form is mathematically equivalent to the
            # standard P = (I - KH)P but is guaranteed to stay symmetric
            # and positive-definite, which prevents numerical blow-up.
            I_KH = np.eye(6, dtype=np.float64) - K @ H
            P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

            event_triggered[t] = False
            kalman_gain_norm[t] = np.linalg.norm(K)
            coast_frames = 0  # Reset coast counter on good measurement
            last_good_x = x_state[0]
            last_good_y = x_state[1]

        # ─── STEP 3: STORE results ───────────────────────────────────
        filtered_x[t] = x_state[0]
        filtered_y[t] = x_state[1]
        velocity_x[t] = x_state[2]
        velocity_y[t] = x_state[3]

    return {
        'filtered_x':       filtered_x,
        'filtered_y':       filtered_y,
        'event_triggered':  event_triggered,
        'kalman_gain_norm': kalman_gain_norm,
        'velocity_x':       velocity_x,
        'velocity_y':       velocity_y,
    }


# =============================================================================
# 3. BASELINE FILTERS — For Comparison (§6.6.6)
# =============================================================================
# These implement the three baseline methods against which we compare
# our event-triggered EKF in the benchmark module (10_filter_benchmark.py).
# =============================================================================

def run_standard_kalman(
    xs: np.ndarray,
    ys: np.ndarray,
    confs: np.ndarray,
    dt: float,
    sigma_0: float,
    sigma_a: float,
) -> dict:
    """
    Standard Kalman Filter — constant R, no confidence gating, no event trigger.

    CONCEPT: This is the "textbook" Kalman filter. It treats every measurement
    equally regardless of confidence. This means it will happily incorporate
    wildly wrong measurements during Gi occlusion, pulling the estimate off-track.
    It also tends to over-smooth genuine Kime peaks because it can't distinguish
    fast motion from noise.

    This serves as Baseline 2 in §6.6.6.

    Args:
        xs, ys, confs: Raw coordinate and confidence arrays.
        dt: Time step.
        sigma_0: Fixed measurement noise std.
        sigma_a: Process noise acceleration std.

    Returns:
        dict matching the output format of run_ekf_single_joint.
    """
    return run_ekf_single_joint(
        xs, ys, confs, dt,
        sigma_0=sigma_0,
        gamma=0.0,        # gamma=0 makes R constant: σ₀²·exp(0)=σ₀²
        threshold=0.0,    # threshold=0 means nothing triggers predict-only
        sigma_a=sigma_a,
        use_confidence_gating=False,
        use_momentum_projection=False,
        use_event_trigger=False,
    )


def run_linear_interpolation(
    xs: np.ndarray,
    ys: np.ndarray,
    confs: np.ndarray,
    threshold: float,
) -> dict:
    """
    Linear interpolation across low-confidence gaps.

    CONCEPT: When the pose estimator loses confidence (c_t < threshold),
    we throw away those measurements and draw a straight line from the
    last good point to the next good point. This is simple but has a
    critical flaw: it cannot follow curved trajectories or preserve
    acceleration during occlusion — every gap becomes a straight line,
    destroying the biomechanics of arcs and rotations.

    This serves as Baseline 3 in §6.6.6.

    Args:
        xs, ys: Raw coordinate arrays.
        confs: Confidence score arrays.
        threshold: Below this confidence, measurements are considered unreliable.

    Returns:
        dict with 'filtered_x', 'filtered_y', 'event_triggered', 'kalman_gain_norm',
        'velocity_x', 'velocity_y'.
    """
    N = len(xs)
    filtered_x = xs.copy().astype(np.float64)
    filtered_y = ys.copy().astype(np.float64)
    event_triggered = confs < threshold

    # Find indices where confidence is above threshold (reliable points)
    good_mask = ~event_triggered
    good_indices = np.where(good_mask)[0]

    if len(good_indices) >= 2:
        # Interpolate x and y separately across the bad regions
        # np.interp does linear interpolation: given known (x,y) points,
        # it fills in values for all other x positions.
        all_indices = np.arange(N)
        filtered_x = np.interp(all_indices, good_indices, xs[good_indices])
        filtered_y = np.interp(all_indices, good_indices, ys[good_indices])

    # Compute crude velocity via finite differences for output compatibility
    velocity_x = np.zeros(N, dtype=np.float64)
    velocity_y = np.zeros(N, dtype=np.float64)
    if N > 1:
        velocity_x[1:] = np.diff(filtered_x)
        velocity_y[1:] = np.diff(filtered_y)

    return {
        'filtered_x':       filtered_x,
        'filtered_y':       filtered_y,
        'event_triggered':  event_triggered,
        'kalman_gain_norm': np.where(event_triggered, 0.0, 1.0),
        'velocity_x':       velocity_x,
        'velocity_y':       velocity_y,
    }


# =============================================================================
# 4. ORCHESTRATOR — Process All Joints
# =============================================================================
# This is the main entry point. It takes a full landmarks DataFrame
# (with all joints and all frames), selects a filter mode, and processes
# each joint independently through the chosen filter.
# =============================================================================

def run_filter(
    landmarks_df: pd.DataFrame,
    model_name: str = "mediapipe",
    filter_mode: str = "ekf",
    config: dict = None,
    use_confidence_gating: bool = True,
    use_momentum_projection: bool = True,
    use_event_trigger: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters all joints in a landmarks DataFrame using the specified filter mode.

    This is the PRIMARY entry point for Phase 2 filtering. It processes
    each joint independently through the selected filter, preserving the
    original DataFrame schema while adding filtered coordinates and
    diagnostic columns.

    FILTER MODES (§6.6.6):
        "raw"           — Passthrough. No filtering applied. Returns original data.
        "skf"           — Standard Kalman Filter. Constant R, no gating.
        "linear_interp" — Linear interpolation across low-confidence gaps.
        "ekf"           — Full event-triggered confidence-gated EKF (our method).

    Args:
        landmarks_df: Raw landmarks from 02_pose_inference.
            Columns: [frame, joint_idx, joint_name, x, y, z, confidence, model]
        model_name: Which model to filter ('mediapipe' or 'yolov8').
        filter_mode: One of 'raw', 'skf', 'linear_interp', 'ekf'.
        config: EKF parameter dict. Defaults to EKF_CONFIG from config.py.
        use_confidence_gating: Ablation toggle (§6.7) — dynamic R_t.
        use_momentum_projection: Ablation toggle (§6.7) — predict-only on occlusion.
        use_event_trigger: Ablation toggle (§6.7) — binary confidence threshold.

    Returns:
        Tuple of (filtered_df, diagnostics_df):
            - filtered_df: Same schema as input + filtered_x, filtered_y,
              filter_mode, event_triggered, kalman_gain_norm columns.
            - diagnostics_df: Per-frame per-joint diagnostic information.
    """
    if config is None:
        config = EKF_CONFIG

    # Extract parameters from config
    dt        = config["dt"]
    sigma_0   = config["sigma_0"]
    gamma     = config["gamma"]
    threshold = config["confidence_threshold"]
    sigma_a   = config.get("sigma_a", 1.0)  # Default if not in config yet

    # ── Filter to the requested model ──
    model_df = landmarks_df[landmarks_df['model'] == model_name].copy()
    if model_df.empty:
        print(f"  ⚠️  No data found for model '{model_name}'")
        empty_filtered = pd.DataFrame()
        empty_diag = pd.DataFrame()
        return empty_filtered, empty_diag

    # ── Get the list of joints to process ──
    joint_names = model_df['joint_name'].unique()
    n_frames = model_df['frame'].nunique()
    print(f"  🔧 Filter mode: {filter_mode}")
    print(f"  📐 Processing {len(joint_names)} joints × {n_frames} frames "
          f"for model '{model_name}'")

    # ── Process each joint independently ──
    all_results = []
    all_diagnostics = []

    for joint_name in sorted(joint_names):
        # Extract this joint's time series, sorted by frame
        joint_data = model_df[model_df['joint_name'] == joint_name].sort_values('frame')

        xs = joint_data['x'].values.astype(np.float64)
        ys = joint_data['y'].values.astype(np.float64)
        confs = joint_data['confidence'].values.astype(np.float64)
        frames = joint_data['frame'].values

        # ── Select and run filter ──
        if filter_mode == "raw":
            # Passthrough: no filtering at all
            result = {
                'filtered_x':       xs.copy(),
                'filtered_y':       ys.copy(),
                'event_triggered':  np.zeros(len(xs), dtype=bool),
                'kalman_gain_norm': np.ones(len(xs), dtype=np.float64),
                'velocity_x':       np.zeros(len(xs), dtype=np.float64),
                'velocity_y':       np.zeros(len(xs), dtype=np.float64),
            }

        elif filter_mode == "skf":
            result = run_standard_kalman(xs, ys, confs, dt, sigma_0, sigma_a)

        elif filter_mode == "linear_interp":
            result = run_linear_interpolation(xs, ys, confs, threshold)

        elif filter_mode == "ekf":
            result = run_ekf_single_joint(
                xs, ys, confs, dt, sigma_0, gamma, threshold, sigma_a,
                use_confidence_gating=use_confidence_gating,
                use_momentum_projection=use_momentum_projection,
                use_event_trigger=use_event_trigger,
            )

        else:
            raise ValueError(
                f"Unknown filter_mode '{filter_mode}'. "
                f"Choose from: 'raw', 'skf', 'linear_interp', 'ekf'"
            )

        # ── Build filtered rows for this joint ──
        joint_filtered = joint_data.copy()
        joint_filtered['filtered_x'] = result['filtered_x']
        joint_filtered['filtered_y'] = result['filtered_y']
        joint_filtered['filter_mode'] = filter_mode
        joint_filtered['event_triggered'] = result['event_triggered']
        joint_filtered['kalman_gain_norm'] = result['kalman_gain_norm']
        all_results.append(joint_filtered)

        # ── Build diagnostics for this joint ──
        diag = pd.DataFrame({
            'frame':            frames,
            'joint_name':       joint_name,
            'model':            model_name,
            'filter_mode':      filter_mode,
            'raw_x':            xs,
            'raw_y':            ys,
            'filtered_x':       result['filtered_x'],
            'filtered_y':       result['filtered_y'],
            'confidence':       confs,
            'event_triggered':  result['event_triggered'],
            'kalman_gain_norm': result['kalman_gain_norm'],
            'velocity_x':      result['velocity_x'],
            'velocity_y':      result['velocity_y'],
            'displacement':    np.sqrt(
                (result['filtered_x'] - xs) ** 2 +
                (result['filtered_y'] - ys) ** 2
            ),
        })
        all_diagnostics.append(diag)

    # ── Combine all joints ──
    filtered_df = pd.concat(all_results, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)

    # ── Summary statistics ──
    n_events = diagnostics_df['event_triggered'].sum()
    total_frames = len(diagnostics_df)
    pct_events = (n_events / total_frames * 100) if total_frames > 0 else 0
    avg_displacement = diagnostics_df['displacement'].mean()

    print(f"  ✅ Filtering complete:")
    print(f"     Event-triggered frames: {n_events}/{total_frames} ({pct_events:.1f}%)")
    print(f"     Mean displacement (raw→filtered): {avg_displacement:.6f}")

    return filtered_df, diagnostics_df


# =============================================================================
# 5. FILE I/O — Save Results
# =============================================================================

def save_filtered_landmarks(
    filtered_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    filename: str = "filtered_landmarks.csv",
) -> Path:
    """
    Saves filtered landmarks to CSV.

    Args:
        filtered_df: The filtered DataFrame from run_filter().
        output_dir: Directory to save to. Defaults to PATHS['filtered'].
        filename: Output filename.

    Returns:
        Path to the saved CSV file.
    """
    if output_dir is None:
        output_dir = PATHS["filtered"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    filtered_df.to_csv(output_path, index=False)
    print(f"  💾 Filtered landmarks saved: {output_path}")
    return output_path


def save_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    filename: str = "ekf_diagnostics.csv",
) -> Path:
    """
    Saves filter diagnostics to CSV.

    The diagnostics CSV contains per-frame, per-joint information about
    what the filter did: raw vs filtered coordinates, whether the event
    trigger activated, the Kalman gain magnitude, and estimated velocities.
    This is essential for the ablation study and for generating Paper 2 figures.

    Args:
        diagnostics_df: The diagnostics DataFrame from run_filter().
        output_dir: Directory to save to. Defaults to PATHS['filtered'].
        filename: Output filename.

    Returns:
        Path to the saved CSV file.
    """
    if output_dir is None:
        output_dir = PATHS["filtered"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    diagnostics_df.to_csv(output_path, index=False)
    print(f"  💾 Diagnostics saved: {output_path}")
    return output_path


def load_filtered_landmarks(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Loads previously saved filtered landmarks CSV.

    Args:
        filepath: Path to the CSV. Defaults to PATHS['filtered'] / 'filtered_landmarks.csv'.

    Returns:
        pd.DataFrame with filtered landmark data.
    """
    if filepath is None:
        filepath = PATHS["filtered"] / "filtered_landmarks.csv"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Filtered landmarks not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"  📂 Loaded filtered landmarks: {len(df)} rows from {filepath.name}")
    return df


# =============================================================================
# 6. STANDALONE EXECUTION
# =============================================================================
# When you run this file directly (python -m src.07_ekf_filter), it processes
# any available landmark CSV as a quick test.
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  🥋  KaratePose-AI — Module 07: EKF Filter (Standalone Test)")
    print("=" * 70)

    # ── Find a landmark CSV to test on ──
    landmark_dir = PATHS["landmarks"]
    csv_files = list(landmark_dir.glob("*.csv")) if landmark_dir.exists() else []

    if not csv_files:
        print(f"\n  ⚠️  No landmark CSVs found in {landmark_dir}")
        print(f"     Run Module 02 first to generate landmarks.")
        print(f"     Or place test CSVs in: {landmark_dir}")
        exit(1)

    # Use the first available CSV
    test_csv = csv_files[0]
    print(f"\n  📂 Test input: {test_csv.name}")

    # Load landmarks
    landmarks_df = pd.read_csv(test_csv)
    print(f"  📊 Loaded {len(landmarks_df)} rows, "
          f"{landmarks_df['frame'].nunique()} frames")

    # ── Run all four filter modes ──
    print("\n" + "─" * 70)
    print("  Running all filter modes for comparison...")
    print("─" * 70)

    for mode in ["raw", "skf", "linear_interp", "ekf"]:
        print(f"\n  {'━' * 60}")
        print(f"  Filter mode: {mode.upper()}")
        print(f"  {'━' * 60}")

        filtered_df, diag_df = run_filter(
            landmarks_df,
            model_name="mediapipe",
            filter_mode=mode,
        )

        if not filtered_df.empty:
            # Save with mode-specific filename
            save_filtered_landmarks(
                filtered_df,
                filename=f"filtered_{mode}_{test_csv.stem}.csv",
            )
            save_diagnostics(
                diag_df,
                filename=f"diagnostics_{mode}_{test_csv.stem}.csv",
            )

    print("\n" + "=" * 70)
    print("  ✅ Module 07 standalone test complete!")
    print(f"  📂 Output directory: {PATHS['filtered']}")
    print("=" * 70)
