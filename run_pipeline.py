# =============================================================================
# KaratePose-AI — Main Pipeline Orchestrator
# =============================================================================
# PURPOSE:
#   This is the MAIN ENTRY POINT for the entire pipeline.
#   Run this file to execute the full analysis from video → results.
#
#   Usage:
#     python run_pipeline.py                     # Run full Phase 1 pipeline
#     python run_pipeline.py --step extract      # Run only frame extraction
#     python run_pipeline.py --step inference     # Run only pose inference
#     python run_pipeline.py --step metrics       # Run only metrics computation
#     python run_pipeline.py --step aggregate     # Run only per-rep aggregation
#     python run_pipeline.py --step statistics    # Run only statistical tests
#     python run_pipeline.py --step visualise     # Run only plot generation
#     python run_pipeline.py --sample             # Run on sample/test data
#
#     python run_pipeline.py --phase 2            # Run full Phase 2 pipeline
#     python run_pipeline.py --phase 2 --step filter    # EKF filtering only
#     python run_pipeline.py --phase 2 --step cog       # CoG tracking only
#     python run_pipeline.py --phase 2 --step benchmark # Filter comparison only
#
# HOW IT CONNECTS:
#   Phase 1 (Paper 1 — The Degradation Problem):
#     01_frame_extractor  → extracts frames from video files
#     02_pose_inference   → runs MediaPipe + YOLOv8 on frames
#     03_metrics          → computes jitter, confidence volatility, joint angles
#     04_aggregator       → aggregates per-repetition (anti-pseudoreplication)
#     05_statistics       → runs Wilcoxon, LMM, FDR correction
#     08_visualiser       → generates publication-quality plots
#
#   Phase 2 (Paper 2 — The EKF & CoG Solution):
#     07_ekf_filter       → Event-Triggered Kinematic EKF + baselines
#     09_cog_tracker      → De Leva (1996) Center of Gravity engine
#     10_filter_benchmark → MPJPE / PCK / phase-lag / velocity evaluation
# =============================================================================

import argparse       # CONCEPT: argparse lets you pass command-line arguments
import sys            #   like --step or --sample when running the script
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Import our modules ──────────────────────────────────────────────────────
# CONCEPT: These are the modules WE wrote. Python finds them because we have
# an __init__.py in the src/ folder, making it a "package".
#
# CONCEPT: Python module names starting with numbers (like 01_frame_extractor)
# can't be imported with the normal "from src.01_module import ..." syntax.
# Instead, we use importlib, which lets us import modules by string name.
# This is a common workaround when files have numbered prefixes for ordering.
import importlib

# Load each numbered module using importlib
_frame_extractor = importlib.import_module("src.01_frame_extractor")
_pose_inference  = importlib.import_module("src.02_pose_inference")
_metrics         = importlib.import_module("src.03_metrics")
_aggregator      = importlib.import_module("src.04_aggregator")
_statistics      = importlib.import_module("src.05_statistics")
_visualiser      = importlib.import_module("src.08_visualiser")

# Phase 2 modules
_ekf_filter      = importlib.import_module("src.07_ekf_filter")
_cog_tracker     = importlib.import_module("src.09_cog_tracker")

# Now pull out the specific functions we need (same as "from module import func")
from src.config import PATHS, VIDEO_CONFIG, PRACTITIONERS, TECHNIQUES

extract_frames      = _frame_extractor.extract_frames
get_video_metadata  = _frame_extractor.get_video_metadata
run_inference       = _pose_inference.run_inference
load_landmarks_csv  = _pose_inference.load_landmarks_csv
compute_all_metrics = _metrics.compute_all_metrics
detect_repetitions  = _aggregator.detect_repetitions
aggregate_per_repetition = _aggregator.aggregate_per_repetition
add_metadata        = _aggregator.add_metadata
run_full_analysis   = _statistics.run_full_analysis
generate_all_plots  = _visualiser.generate_all_plots

# Phase 2 functions
run_filter              = _ekf_filter.run_filter
save_filtered_landmarks = _ekf_filter.save_filtered_landmarks
save_diagnostics        = _ekf_filter.save_diagnostics
cog_batch_process       = _cog_tracker.batch_process


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(phase: int = 1):
    """Print a nice header when the pipeline starts."""
    label = "Phase 1 — Degradation Analysis" if phase == 1 else "Phase 2 — EKF & CoG Solution"
    print("=" * 70)
    print(f"  🥋  KaratePose-AI — {label}")
    print("=" * 70)
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project:    {PATHS['sample_data'].parent.parent}")
    print("=" * 70)
    print()


def print_step(step_number: int, step_name: str):
    """Print a section header for each pipeline step."""
    print()
    print(f"  ┌{'─' * 58}┐")
    print(f"  │  Step {step_number}: {step_name:<50}│")
    print(f"  └{'─' * 58}┘")
    print()


def find_videos(directory: Path) -> list[Path]:
    """
    Find all video files in a directory.

    CONCEPT: Path.glob() searches for files matching a pattern.
    We check multiple extensions (.mov, .mp4, etc.) because different
    cameras save in different formats.
    """
    videos = []
    for ext in VIDEO_CONFIG["video_extensions"]:
        # glob() returns all files matching the pattern in this directory
        videos.extend(directory.glob(f"*{ext}"))
    # Sort alphabetically for consistent processing order
    return sorted(videos)


def discover_data(data_dir: Path) -> dict:
    """
    Scan the data directory and report what's available.

    Returns a dict mapping condition labels to lists of video paths.
    """
    discovered = {}

    # Check for sample data (for testing)
    sample_videos = find_videos(PATHS["sample_data"])
    if sample_videos:
        discovered["sample"] = sample_videos

    # Check for real Set A data
    set_a_videos = find_videos(PATHS["set_a"])
    if set_a_videos:
        discovered["set_a"] = set_a_videos

    # Check for real Set B data
    set_b_videos = find_videos(PATHS["set_b"])
    if set_b_videos:
        discovered["set_b"] = set_b_videos

    return discovered


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_1_extract(video_paths: list[Path]) -> dict:
    """
    Step 1: Extract frames from all videos.

    RESEARCH NOTE: §6.4.1 — "Video streams will be decoded using OpenCV
    (VideoCapture), with rigorous frame-rate constancy checks."
    """
    print_step(1, "Frame Extraction")

    all_metadata = {}
    for video_path in video_paths:
        print(f"  📹 Processing: {video_path.name}")

        # Get video metadata (FPS, resolution, duration)
        metadata = get_video_metadata(str(video_path))
        print(f"     Resolution: {metadata['width']}x{metadata['height']}")
        print(f"     FPS: {metadata['fps']}, Frames: {metadata['total_frames']}")
        print(f"     Duration: {metadata['duration_s']:.1f} seconds")

        all_metadata[video_path.name] = metadata

    print(f"\n  ✅ Discovered {len(video_paths)} video(s)")
    return all_metadata


def step_2_inference(video_paths: list[Path]) -> dict:
    """
    Step 2: Run MediaPipe + YOLOv8 pose inference on all videos.

    RESEARCH NOTE: §6.4.2 — "Inference is executed strictly on a per-frame
    basis, explicitly disabling any internal multi-frame temporal smoothing."
    """
    print_step(2, "Pose Inference (MediaPipe + YOLOv8)")

    all_landmarks = {}
    for video_path in video_paths:
        print(f"  🤖 Running inference on: {video_path.name}")

        # Create output CSV path: data/landmarks/video_name_landmarks.csv
        csv_name = video_path.stem + "_landmarks.csv"
        csv_path = PATHS["landmarks"] / csv_name

        # Check if we already processed this video (skip if CSV exists)
        if csv_path.exists():
            print(f"     ⏭️  Already processed — loading from {csv_name}")
            landmarks_df = load_landmarks_csv(str(csv_path))
        else:
            # Run both MediaPipe and YOLOv8
            landmarks_df = run_inference(
                video_path=str(video_path),
                output_csv_path=str(csv_path)
            )

        all_landmarks[video_path.name] = landmarks_df
        n_frames = landmarks_df["frame"].nunique()
        print(f"     ✅ Extracted landmarks for {n_frames} frames")

    return all_landmarks


def step_3_metrics(all_landmarks: dict) -> dict:
    """
    Step 3: Compute degradation metrics (jitter, confidence, angles).

    RESEARCH NOTE: §6.4.3 — "Spatial Jitter measures erratic inter-frame
    displacement. Confidence Volatility quantifies detection instability."
    """
    print_step(3, "Degradation Metrics")

    all_metrics = {}
    for video_name, landmarks_df in all_landmarks.items():
        print(f"  📊 Computing metrics for: {video_name}")
        metrics = compute_all_metrics(landmarks_df)
        all_metrics[video_name] = metrics
        print(f"     ✅ Jitter: {len(metrics['jitter'])} rows")
        print(f"     ✅ Confidence: {len(metrics['confidence'])} rows")
        print(f"     ✅ Angles: {len(metrics['angles'])} rows")

    return all_metrics


def step_4_aggregate(all_metrics: dict, n_reps: int = 10) -> dict:
    """
    Step 4: Aggregate frame-level data into per-repetition summaries.

    RESEARCH NOTE: §6.4.4 — "All metrics will be aggregated per repetition.
    This yields N=30 paired observations per technique/condition combination."

    THIS IS THE CRITICAL PSEUDOREPLICATION FIX.
    """
    print_step(4, "Per-Repetition Aggregation")

    all_aggregated = {}
    for video_name, metrics in all_metrics.items():
        print(f"  📦 Aggregating: {video_name}")

        # Detect rep boundaries (simple equal-division for now)
        jitter_df = metrics["jitter"]
        if len(jitter_df) == 0:
            print(f"     ⚠️  No jitter data — skipping")
            continue

        # Get repetition boundaries using the jitter data
        # detect_repetitions uses len(landmarks_df) to determine total frames
        rep_boundaries = detect_repetitions(
            landmarks_df=jitter_df,
            technique="unknown",
            n_reps=n_reps,
        )

        # Aggregate jitter per repetition
        jitter_agg = aggregate_per_repetition(
            metrics_df=jitter_df,
            rep_boundaries=rep_boundaries,
            metric_column="jitter"
        )

        # Aggregate confidence per repetition
        conf_agg = aggregate_per_repetition(
            metrics_df=metrics["confidence"],
            rep_boundaries=rep_boundaries,
            metric_column="conf_volatility"
        )

        all_aggregated[video_name] = {
            "jitter_agg": jitter_agg,
            "confidence_agg": conf_agg,
            "rep_boundaries": rep_boundaries
        }
        print(f"     ✅ {len(rep_boundaries)} repetitions aggregated")

    return all_aggregated


def step_5_statistics(agg_set_a: dict, agg_set_b: dict) -> dict:
    """
    Step 5: Run the full statistical analysis.

    RESEARCH NOTE: §6.5 — "Wilcoxon Signed-Rank Test, Benjamini-Hochberg
    FDR correction, Linear Mixed-Effects Model."
    """
    print_step(5, "Statistical Analysis")

    # Run the full analysis suite
    results = run_full_analysis(agg_set_a, agg_set_b)

    # Save results
    output_path = PATHS["statistics"] / "statistical_results.csv"
    print(f"  💾 Results saved to: {output_path}")

    return results


def step_6_visualise(agg_a, agg_b, landmarks_a, landmarks_b, stats_results):
    """
    Step 6: Generate all publication-quality plots.

    RESEARCH NOTE: These figures will go directly into Paper 1.
    """
    print_step(6, "Visualisation")

    output_dir = PATHS["plots"]
    generate_all_plots(agg_a, agg_b, landmarks_a, landmarks_b,
                       stats_results, output_dir)

    print(f"  📈 All plots saved to: {output_dir}")


# =============================================================================
# PHASE 2 PIPELINE STEPS
# =============================================================================

def phase2_step_filter(landmark_csvs: list[Path]) -> dict:
    """Phase 2 Step 1: Run all 4 filter modes on landmark CSVs."""
    print_step(1, "EKF Filtering (4 modes)")

    all_filtered = {}
    for csv_path in landmark_csvs:
        print(f"  📂 Input: {csv_path.name}")
        landmarks_df = pd.read_csv(csv_path)
        n_frames = landmarks_df["frame"].nunique()
        models = landmarks_df["model"].unique() if "model" in landmarks_df.columns else ["mediapipe"]
        print(f"     {len(landmarks_df)} rows, {n_frames} frames, models: {list(models)}")

        for mode in ["raw", "skf", "linear_interp", "ekf"]:
            print(f"\n  ▸ Filter mode: {mode.upper()}")
            for model in models:
                filtered_df, diag_df = run_filter(
                    landmarks_df, model_name=model, filter_mode=mode
                )
                if not filtered_df.empty:
                    fname = f"filtered_{mode}_{csv_path.stem}.csv"
                    dname = f"diagnostics_{mode}_{csv_path.stem}.csv"
                    save_filtered_landmarks(filtered_df, filename=fname)
                    save_diagnostics(diag_df, filename=dname)
                    all_filtered[f"{mode}_{model}_{csv_path.stem}"] = filtered_df
                    print(f"    {model}: ✅ {len(filtered_df)} rows → {fname}")

    return all_filtered


def phase2_step_cog(scale_px_per_m: float = None) -> None:
    """Phase 2 Step 2: Compute CoG from filtered landmarks."""
    print_step(2, "De Leva (1996) Center of Gravity")

    filtered_dir = PATHS["filtered"]
    output_dir = PATHS["cog"]

    if not filtered_dir.exists() or not list(filtered_dir.glob("filtered_*.csv")):
        print("  ⚠️  No filtered CSVs found. Run --step filter first.")
        return

    cog_batch_process(
        input_dir=filtered_dir,
        output_dir=output_dir,
        filter_modes=["raw", "skf", "linear_interp", "ekf"],
        scale_px_per_m=scale_px_per_m,
    )


def phase2_step_benchmark() -> None:
    """Phase 2 Step 3: Run filter benchmark comparison."""
    print_step(3, "Filter Benchmark (MPJPE / PCK / Phase Lag)")

    try:
        _benchmark = importlib.import_module("src.10_filter_benchmark")
        _benchmark.main()
    except ModuleNotFoundError:
        print("  ⚠️  Module 10 (filter_benchmark) not yet built.")
        print("     This step will be available after 10_filter_benchmark.py is created.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function — orchestrates the entire Phase A pipeline.

    CONCEPT: argparse creates a command-line interface. When you run
    `python run_pipeline.py --step extract`, argparse reads '--step extract'
    and stores it so we can decide which step to run.
    """
    # ── Parse command-line arguments ────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="KaratePose-AI Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which research phase to run (1=degradation, 2=EKF+CoG)"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "extract", "inference", "metrics",
                 "aggregate", "statistics", "visualise",
                 "filter", "cog", "benchmark"],
        help="Which pipeline step to run (default: all)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample/test data instead of real data"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom input directory containing videos"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Phase 2: pixels per metre for stance depth conversion"
    )

    args = parser.parse_args()

    # Validate step belongs to the chosen phase
    phase1_steps = {"all", "extract", "inference", "metrics", "aggregate", "statistics", "visualise"}
    phase2_steps = {"all", "filter", "cog", "benchmark"}

    if args.phase == 1 and args.step not in phase1_steps:
        print(f"  ⚠️  Step '{args.step}' is a Phase 2 step. Use --phase 2 --step {args.step}")
        sys.exit(1)
    if args.phase == 2 and args.step not in phase2_steps:
        print(f"  ⚠️  Step '{args.step}' is a Phase 1 step. Use --phase 1 --step {args.step}")
        sys.exit(1)

    # ── Print welcome banner ────────────────────────────────────────────
    print_banner(args.phase)

    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — EKF Filtering → CoG Tracking → Benchmark
    # ════════════════════════════════════════════════════════════════════
    if args.phase == 2:
        # Discover landmark CSVs (input for Phase 2)
        landmark_dir = PATHS["landmarks"]
        if args.input_dir:
            landmark_dir = Path(args.input_dir)

        landmark_csvs = sorted(landmark_dir.glob("*_landmarks.csv")) if landmark_dir.exists() else []

        if not landmark_csvs:
            print(f"  ⚠️  No landmark CSVs found in {landmark_dir}")
            print(f"     Run Phase 1 inference first: python run_pipeline.py --step inference")
            sys.exit(1)

        print(f"  📂 Found {len(landmark_csvs)} landmark file(s)")
        for lc in landmark_csvs:
            print(f"     • {lc.name}")
        print()

        # Step 1: EKF Filtering
        if args.step in ("all", "filter"):
            phase2_step_filter(landmark_csvs)

        # Step 2: CoG Tracking
        if args.step in ("all", "cog"):
            phase2_step_cog(scale_px_per_m=args.scale)

        # Step 3: Benchmark
        if args.step in ("all", "benchmark"):
            phase2_step_benchmark()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — Video → Frames → Pose → Metrics → Stats → Plots
    # ════════════════════════════════════════════════════════════════════
    else:
        # ── Discover available data ─────────────────────────────────────
        if args.input_dir:
            data_dir = Path(args.input_dir)
            video_paths = find_videos(data_dir)
            print(f"  Using custom input: {data_dir}")
        elif args.sample:
            video_paths = find_videos(PATHS["sample_data"])
            print(f"  🧪 Running on SAMPLE data")
        else:
            discovered = discover_data(PATHS["raw_data"].parent)
            if not discovered:
                print("  ⚠️  No video files found!")
                print(f"     Place test videos in: {PATHS['sample_data']}")
                print(f"     Or real data in: {PATHS['set_a']} and {PATHS['set_b']}")
                sys.exit(1)

            video_paths = []
            for category, paths in discovered.items():
                print(f"  Found {len(paths)} video(s) in {category}/")
                video_paths.extend(paths)

        if not video_paths:
            print("  ⚠️  No video files found! Add videos and try again.")
            print(f"     Sample data folder: {PATHS['sample_data']}")
            sys.exit(1)

        print(f"\n  Total videos to process: {len(video_paths)}")

        # ── Run Phase 1 pipeline steps ──────────────────────────────────
        if args.step in ("all", "extract"):
            metadata = step_1_extract(video_paths)

        if args.step in ("all", "inference"):
            all_landmarks = step_2_inference(video_paths)

        if args.step in ("all", "metrics"):
            if "all_landmarks" not in locals():
                print("  Loading previously extracted landmarks...")
                all_landmarks = {}
                for vp in video_paths:
                    csv_path = PATHS["landmarks"] / (vp.stem + "_landmarks.csv")
                    if csv_path.exists():
                        all_landmarks[vp.name] = load_landmarks_csv(str(csv_path))
            all_metrics = step_3_metrics(all_landmarks)

        if args.step in ("all", "aggregate"):
            if "all_metrics" not in locals():
                print("  ⚠️  Run metrics step first: python run_pipeline.py --step metrics")
                sys.exit(1)
            all_aggregated = step_4_aggregate(all_metrics)

        if args.step in ("all", "statistics"):
            if args.sample:
                print("\n  📝 Note: Skipping full statistics since we only have a sample video.")
                print("     (Full stats require both Set A and Set B data)")
            else:
                print("\n  📝 Note: Full statistical comparison requires both Set A and Set B data.")

        if args.step in ("all", "visualise"):
            if args.sample:
                print("\n  🎨 Generating visualisations from sample data...")
                import subprocess
                subprocess.run([sys.executable, "-m", "src.08_visualiser"])
            else:
                print("\n  📝 Note: Full visualisation requires completed statistical analysis.")

    # ── Print summary ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  ✅ Pipeline completed in {elapsed:.1f} seconds")
    print(f"  📂 Output directory: {PATHS['metrics'].parent}")
    print("=" * 70)


# CONCEPT: This is a Python idiom. It means "only run main() if this file
# is executed directly (python run_pipeline.py), NOT if it's imported by
# another file." This lets other scripts import our functions without
# accidentally running the whole pipeline.
if __name__ == "__main__":
    main()
