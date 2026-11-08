"""
10_filter_benchmark.py — Comparative Filter Evaluation
Compares all 4 filter modes (raw, skf, linear_interp, ekf) using:
  - MPJPE: Mean Per-Joint Position Error (confidence-masked)
  - PCK@0.2: Percentage of Correct Keypoints (normalised by torso diameter)
  - Phase Lag: Cross-correlation peak offset in frames
  - Peak Velocity Preservation Ratio (confidence-masked)

NOTE ON GROUND TRUTH:
  When no separate --reference CSV is provided, the reference defaults to
  the raw landmarks CSV. This means the "raw" filter mode will produce
  MPJPE ≈ 0 by definition (it IS the reference). This is expected and
  noted in the benchmark output. The informative comparisons are between
  the non-raw filters and the reference.

  Confidence masking: frames where the raw detector confidence falls below
  CONF_MASK_THRESHOLD are excluded from MPJPE and PVR calculations. This
  prevents low-confidence glitch frames from corrupting the metrics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from src.config import PATHS, EKF_CONFIG

FILTER_MODES = ["raw", "skf", "linear_interp", "ekf"]

# Frames with raw confidence below this threshold are excluded from
# MPJPE and PVR calculations, so low-confidence glitches do not
# inflate artificial velocity spikes or drag reference accuracy.
CONF_MASK_THRESHOLD = EKF_CONFIG.get("confidence_threshold", 0.50)


# ── Metric Functions ────────────────────────────────────────────────────────

def compute_mpjpe(filtered_df: pd.DataFrame, reference_df: pd.DataFrame,
                  model: str, conf_mask_threshold: float = None) -> pd.DataFrame:
    """Mean Per-Joint Position Error between filtered and reference coordinates.

    When conf_mask_threshold is set, frames where the reference confidence
    is below the threshold are excluded from the error calculation.
    """
    if conf_mask_threshold is None:
        conf_mask_threshold = CONF_MASK_THRESHOLD

    filt = filtered_df[filtered_df["model"] == model].copy()
    ref = reference_df[reference_df["model"] == model].copy()

    # Build merge keys — include confidence from reference if available
    ref_cols = ["frame", "joint_name", "x", "y"]
    if "confidence" in ref.columns:
        ref_cols.append("confidence")

    merged = pd.merge(
        filt[["frame", "joint_name", "filtered_x", "filtered_y"]],
        ref[ref_cols],
        on=["frame", "joint_name"], suffixes=("_filt", "_ref")
    )
    if merged.empty:
        return pd.DataFrame()

    # Confidence masking: exclude low-confidence reference frames
    if "confidence" in merged.columns and conf_mask_threshold > 0:
        merged = merged[merged["confidence"] >= conf_mask_threshold]

    if merged.empty:
        return pd.DataFrame()

    merged["error"] = np.sqrt(
        (merged["filtered_x"] - merged["x"]) ** 2 +
        (merged["filtered_y"] - merged["y"]) ** 2
    )

    per_joint = merged.groupby("joint_name")["error"].agg(
        mpjpe="mean", mpjpe_std="std", mpjpe_max="max"
    ).reset_index()
    per_joint["model"] = model
    return per_joint


def compute_pck(filtered_df: pd.DataFrame, reference_df: pd.DataFrame,
                model: str, threshold: float = 0.2) -> pd.DataFrame:
    """Percentage of Correct Keypoints within threshold * torso_diameter."""
    filt = filtered_df[filtered_df["model"] == model].copy()
    ref = reference_df[reference_df["model"] == model].copy()

    merged = pd.merge(
        filt[["frame", "joint_name", "filtered_x", "filtered_y"]],
        ref[["frame", "joint_name", "x", "y"]],
        on=["frame", "joint_name"]
    )
    if merged.empty:
        return pd.DataFrame()

    # Torso diameter per frame: distance between shoulder midpoint and hip midpoint
    torso = _compute_torso_diameter(ref)
    merged = pd.merge(merged, torso, on="frame", how="left")
    merged["torso_diam"] = merged["torso_diam"].fillna(merged["torso_diam"].median())

    merged["error"] = np.sqrt(
        (merged["filtered_x"] - merged["x"]) ** 2 +
        (merged["filtered_y"] - merged["y"]) ** 2
    )
    merged["correct"] = merged["error"] < (threshold * merged["torso_diam"])

    per_joint = merged.groupby("joint_name")["correct"].mean().reset_index()
    per_joint.columns = ["joint_name", f"pck_at_{threshold}"]
    per_joint["model"] = model
    return per_joint


def _compute_torso_diameter(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Torso diameter = dist(shoulder_mid, hip_mid) per frame."""
    pivoted = ref_df.pivot_table(index="frame", columns="joint_name", values=["x", "y"])

    results = []
    for frame in pivoted.index:
        try:
            ls_x = pivoted.loc[frame, ("x", "left_shoulder")]
            rs_x = pivoted.loc[frame, ("x", "right_shoulder")]
            ls_y = pivoted.loc[frame, ("y", "left_shoulder")]
            rs_y = pivoted.loc[frame, ("y", "right_shoulder")]
            lh_x = pivoted.loc[frame, ("x", "left_hip")]
            rh_x = pivoted.loc[frame, ("x", "right_hip")]
            lh_y = pivoted.loc[frame, ("y", "left_hip")]
            rh_y = pivoted.loc[frame, ("y", "right_hip")]

            sh_mid = np.array([(ls_x + rs_x) / 2, (ls_y + rs_y) / 2])
            hp_mid = np.array([(lh_x + rh_x) / 2, (lh_y + rh_y) / 2])
            diam = np.linalg.norm(sh_mid - hp_mid)
            results.append({"frame": frame, "torso_diam": diam})
        except (KeyError, TypeError):
            continue

    return pd.DataFrame(results)


def compute_phase_lag(filtered_series: np.ndarray,
                      reference_series: np.ndarray) -> float:
    """Phase lag via cross-correlation peak offset (frames)."""
    if len(filtered_series) < 10 or len(reference_series) < 10:
        return np.nan

    f = filtered_series - np.mean(filtered_series)
    r = reference_series - np.mean(reference_series)

    corr = signal.correlate(f, r, mode="full")
    lags = signal.correlation_lags(len(f), len(r), mode="full")

    peak_idx = np.argmax(np.abs(corr))
    return float(lags[peak_idx])


def compute_peak_velocity_ratio(filtered_df: pd.DataFrame,
                                reference_df: pd.DataFrame,
                                joint_name: str, model: str,
                                dt: float = 1/60,
                                conf_mask_threshold: float = None) -> float:
    """Peak velocity of filtered / peak velocity of reference.

    Both filtered and reference peak velocities are confidence-masked:
    velocity is only computed between consecutive frames where the
    REFERENCE confidence is above the threshold. This ensures that:
      - Reference peak velocity reflects true physical motion (not glitches)
      - Filtered peak velocity is evaluated only on the same reliable frames,
        so PVR ≈ 1.0 means "the filter faithfully reproduces visible motion"
        and PVR > 1.0 means "the filter captures motion peaks that raw data
        loses to glitches" (the desired Kime-preservation effect).
    """
    if conf_mask_threshold is None:
        conf_mask_threshold = CONF_MASK_THRESHOLD

    # Get reference confidence mask for this joint
    ref_jdf = reference_df[
        (reference_df["joint_name"] == joint_name) &
        (reference_df["model"] == model)
    ].sort_values("frame")

    if len(ref_jdf) < 2:
        return np.nan

    ref_frames = ref_jdf["frame"].values
    ref_confs = ref_jdf["confidence"].values if "confidence" in ref_jdf.columns else np.ones(len(ref_jdf))
    conf_mask = ref_confs >= conf_mask_threshold

    if conf_mask.sum() < 2:
        return np.nan

    def _masked_peak_vel(df, x_col, y_col):
        """Compute peak velocity only between consecutive confident frames."""
        jdf = df[
            (df["joint_name"] == joint_name) & (df["model"] == model)
        ].sort_values("frame")
        if len(jdf) < 2:
            return np.nan

        x_vals = jdf[x_col].values
        y_vals = jdf[y_col].values
        frames = jdf["frame"].values

        # Use the same confidence mask (from reference) for both sides
        # Build a frame→mask lookup
        frame_mask = dict(zip(ref_frames, conf_mask))

        speeds = []
        prev_idx = None
        for i in range(len(jdf)):
            if frame_mask.get(frames[i], False):
                if prev_idx is not None and (frames[i] - frames[prev_idx]) == 1:
                    vx = (x_vals[i] - x_vals[prev_idx]) / dt
                    vy = (y_vals[i] - y_vals[prev_idx]) / dt
                    speeds.append(np.sqrt(vx**2 + vy**2))
                prev_idx = i
        return np.max(speeds) if speeds else np.nan

    v_filt = _masked_peak_vel(filtered_df, "filtered_x", "filtered_y")
    v_ref = _masked_peak_vel(reference_df, "x", "y")

    if np.isnan(v_ref) or v_ref < 1e-9:
        return np.nan
    return float(v_filt / v_ref)


# ── Per-Mode Evaluation ─────────────────────────────────────────────────────

def evaluate_mode(filtered_path: Path, reference_df: pd.DataFrame,
                  filter_mode: str, dt: float = 1/60) -> pd.DataFrame:
    """Run all metrics for a single filter mode."""
    filt_df = pd.read_csv(filtered_path)
    models = filt_df["model"].unique()

    rows = []
    for model in models:
        # MPJPE
        mpjpe_df = compute_mpjpe(filt_df, reference_df, model)
        # PCK
        pck_df = compute_pck(filt_df, reference_df, model)

        if mpjpe_df.empty:
            continue

        merged = pd.merge(mpjpe_df, pck_df, on=["joint_name", "model"], how="outer")

        # Phase lag & velocity ratio per joint
        for _, row in merged.iterrows():
            jname = row["joint_name"]

            # Phase lag on x-coordinate
            fj = filt_df[(filt_df["joint_name"] == jname) & (filt_df["model"] == model)].sort_values("frame")
            rj = reference_df[(reference_df["joint_name"] == jname) & (reference_df["model"] == model)].sort_values("frame")

            if len(fj) > 10 and len(rj) > 10:
                lag = compute_phase_lag(fj["filtered_x"].values, rj["x"].values)
                pvr = compute_peak_velocity_ratio(filt_df, reference_df, jname, model, dt)
            else:
                lag = np.nan
                pvr = np.nan

            rows.append({
                "model": model,
                "filter_mode": filter_mode,
                "joint_name": jname,
                "mpjpe": row.get("mpjpe", np.nan),
                "mpjpe_std": row.get("mpjpe_std", np.nan),
                "pck_at_0.2": row.get("pck_at_0.2", np.nan),
                "phase_lag_frames": lag,
                "peak_velocity_preservation_ratio": pvr,
            })

    return pd.DataFrame(rows)


# ── Batch Benchmark ──────────────────────────────────────────────────────────

def run_benchmark(filtered_dir: Path, reference_df: pd.DataFrame,
                  output_dir: Path, modes: list[str] = None,
                  dt: float = 1/60) -> pd.DataFrame:
    """Run benchmark across all filter modes."""
    if modes is None:
        modes = FILTER_MODES

    all_results = []
    for mode in modes:
        files = sorted(filtered_dir.glob(f"filtered_{mode}_*.csv"))
        if not files:
            print(f"  [SKIP] No files for mode: {mode}")
            continue

        for fpath in files:
            print(f"  ▸ Evaluating: {fpath.name}")
            result = evaluate_mode(fpath, reference_df, mode, dt)
            all_results.append(result)
            if not result.empty:
                print(f"    {mode}: {len(result)} joint-model rows evaluated")

    if not all_results:
        print("  ⚠️  No results generated.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "benchmark_detailed.csv"
    combined.to_csv(detail_path, index=False)
    print(f"\n  💾 Detailed results → {detail_path}")

    # Summary table: mean across joints
    summary = combined.groupby(["model", "filter_mode"]).agg(
        mpjpe_mean=("mpjpe", "mean"),
        mpjpe_max=("mpjpe", "max"),
        pck_mean=("pck_at_0.2", "mean"),
        phase_lag_mean=("phase_lag_frames", "mean"),
        pvr_mean=("peak_velocity_preservation_ratio", "mean"),
    ).reset_index()

    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  💾 Summary results  → {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"\n  Model: {row['model']}  |  Filter: {row['filter_mode']}")
        print(f"    MPJPE (mean):  {row['mpjpe_mean']:.6f}  (max: {row['mpjpe_max']:.6f})")
        print(f"    PCK@0.2:       {row['pck_mean']:.4f}")
        print(f"    Phase Lag:     {row['phase_lag_mean']:.1f} frames")
        print(f"    Peak Vel Ratio:{row['pvr_mean']:.4f}")
    print("=" * 80)

    return combined


def main():
    parser = argparse.ArgumentParser(description="10: Filter Benchmark")
    parser.add_argument("--filtered-dir", type=str, default=str(PATHS["filtered"]))
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference landmark CSV (proxy ground truth). "
                             "Defaults to raw landmarks in data/landmarks/")
    parser.add_argument("--output-dir", type=str, default=str(PATHS["benchmark"]))
    parser.add_argument("--modes", type=str, nargs="+", default=FILTER_MODES)
    args = parser.parse_args()

    print("=" * 60)
    print("10_filter_benchmark — Comparative Filter Evaluation")
    print("=" * 60)

    # Load reference (ground truth proxy)
    if args.reference:
        ref_path = Path(args.reference)
    else:
        # Auto-discover from landmarks dir
        landmark_dir = PATHS["landmarks"]
        ref_files = sorted(landmark_dir.glob("*_landmarks.csv")) if landmark_dir.exists() else []
        if not ref_files:
            print(f"  ⚠️  No landmark CSVs in {landmark_dir}")
            sys.exit(1)
        ref_path = ref_files[0]

    print(f"  📂 Reference: {ref_path.name}")
    reference_df = pd.read_csv(ref_path)
    print(f"     {len(reference_df)} rows, {reference_df['frame'].nunique()} frames")

    filtered_dir = Path(args.filtered_dir)
    output_dir = Path(args.output_dir)

    results = run_benchmark(filtered_dir, reference_df, output_dir, args.modes)

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
