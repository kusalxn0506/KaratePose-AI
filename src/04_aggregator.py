"""
=============================================================================
Module 04: Frame-to-Repetition Aggregator
=============================================================================

WHY THIS MODULE EXISTS (READ THIS CAREFULLY):
In biomechanics and AI pose estimation, videos are typically recorded at 60 FPS
(Frames Per Second). This means consecutive frames are only 16.7ms apart.
Because the human body barely moves in 16.7ms, the data in frame `t` and
frame `t+1` are practically identical. They are HIGHLY correlated.

If a student records a 10-second video of a karate punch at 60 FPS, they get
600 frames of data. If they then run a statistical test (like a t-test) treating
those 600 frames as 600 independent observations, the stats software will spit
out a tiny, "highly significant" p-value (e.g., p < 0.001) even if there is NO
real effect.

This critical error is called PSEUDOREPLICATION. It violates the core assumption
of independence in almost all standard statistical tests.

THE FIX:
Instead of analyzing 600 highly correlated frames, we must AGGREGATE the frames
that belong to a single "event" (a single repetition of a punch). By condensing
all the frames of one punch into a SINGLE summary statistic (like the median jitter),
we collapse those 600 fake data points into 1 honest, independent data point.
If a practitioner does 10 punches (reps), we get N=10 valid data points per
condition.

RESEARCH NOTE:
See §6.4.4 and §6.5.2 of your BSc Honours research proposal for the formal
justification of this aggregation step.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from src.config import TECHNIQUES

def detect_repetitions(landmarks_df: pd.DataFrame, technique: str, model_name: str = 'mediapipe', n_reps: int = 10) -> List[Tuple[int, int]]:
    """
    Detects individual repetition boundaries within a continuous video timeline.
    
    CONCEPT:
    A video contains a continuous recording of multiple repetitions (e.g., 10 punches).
    To avoid pseudoreplication, we need to know exactly which frames belong to Rep 1, 
    Rep 2, etc. This function figures out the start and end frame for each repetition.
    
    Args:
        landmarks_df: DataFrame containing the raw or processed tracking data over time.
        technique: Name of the technique (e.g., 'OIZUKI') to fetch expected reps from config.
        model_name: The AI model used ('mediapipe' or 'yolov8').
        n_reps: The expected number of repetitions. Defaults to 10.
        
    Returns:
        A list of tuples, where each tuple contains (start_frame, end_frame) for a rep.
    """
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: Equal Division (Implemented First)
    # -------------------------------------------------------------------------
    # The simplest approach. If the video was perfectly trimmed to only contain 
    # the active exercise period, and the practitioner paced themselves perfectly, 
    # we can divide the total frames evenly by the number of expected repetitions.
    # -------------------------------------------------------------------------
    
    total_frames = len(landmarks_df)
    
    # Check if config overrides the default n_reps
    if technique in TECHNIQUES:
        n_reps = TECHNIQUES[technique]["reps"]
        
    segment_length = total_frames // n_reps
    rep_boundaries = []
    
    for i in range(n_reps):
        start_frame = i * segment_length
        # Ensure the last repetition goes all the way to the end to catch rounding gaps
        end_frame = (i + 1) * segment_length - 1 if i < (n_reps - 1) else total_frames - 1
        rep_boundaries.append((start_frame, end_frame))
        
    # -------------------------------------------------------------------------
    # TODO: STRATEGY 2: Valley Detection (More robust, to be implemented later)
    # -------------------------------------------------------------------------
    # In reality, practitioners take a 3-second static reset between reps. 
    # During these static resets, the physical movement speed (and associated jitter) 
    # drops to near-zero. 
    # 
    # Advanced logic:
    # 1. Calculate a rolling average of overall body velocity.
    # 2. Find the "valleys" (local minima) in this velocity curve.
    # 3. These valleys represent the static resets, marking the boundaries between reps.
    # -------------------------------------------------------------------------
    
    return rep_boundaries


def aggregate_per_repetition(metrics_df: pd.DataFrame, rep_boundaries: List[Tuple[int, int]], metric_column: str) -> pd.DataFrame:
    """
    Aggregates frame-by-frame metrics into a single summary statistic per repetition.
    
    CONCEPT:
    This is where we actually cure the pseudoreplication. For a given repetition
    (e.g., frames 100 to 250), we take all 150 values of the metric and condense
    them into summary numbers.
    
    Args:
        metrics_df: The frame-by-frame data (e.g., jitter for each joint).
        rep_boundaries: The list of (start_frame, end_frame) tuples.
        metric_column: The specific column we are aggregating (e.g., 'jitter_magnitude').
        
    Returns:
        A DataFrame with the summarized data per repetition.
    """
    aggregated_records = []
    
    # We assume metrics_df has a 'joint_name' column or we are grouping by it.
    # If the df is in long format: [frame, joint_name, metric_value]
    # We'll group by joint_name and then apply the logic per repetition.
    
    joints = metrics_df['joint_name'].unique() if 'joint_name' in metrics_df.columns else ['unknown_joint']
    
    for rep_id, (start_frame, end_frame) in enumerate(rep_boundaries, start=1):
        
        # Slice the dataframe to only include frames within this specific repetition
        # Ensure we have a frame index or column to filter by. Assuming index is the frame number.
        rep_data = metrics_df.loc[start_frame:end_frame]
        
        for joint in joints:
            if 'joint_name' in rep_data.columns:
                joint_data = rep_data[rep_data['joint_name'] == joint][metric_column]
            else:
                joint_data = rep_data[metric_column]
                
            if joint_data.empty:
                continue
                
            # -----------------------------------------------------------------
            # SUMMARY STATISTICS EXPLAINED
            # -----------------------------------------------------------------
            
            # 1. Median: The middle value. 
            # WHY IT'S PREFERRED: AI pose estimation occasionally produces wild outliers 
            # (e.g., the elbow suddenly glitches to coordinates 0,0 for two frames). 
            # The 'mean' gets heavily skewed by these outliers. The 'median' ignores 
            # extreme outliers, giving a much more robust measure of typical behavior.
            val_median = joint_data.median()
            
            # 2. Mean: The standard average. Included for completeness, but median is safer.
            val_mean = joint_data.mean()
            
            # 3. Standard Deviation (std): How spread out the values are.
            val_std = joint_data.std()
            
            # 4. Max: The absolute peak value during the repetition (worst-case jitter).
            val_max = joint_data.max()
            
            # 5. IQR (Interquartile Range): The difference between the 75th and 25th percentiles.
            # Measures the spread of the "middle 50%" of the data, ignoring outliers.
            val_iqr = joint_data.quantile(0.75) - joint_data.quantile(0.25)
            
            # 6. AUC (Area Under the Curve): The sum of values across the rep.
            # INTUITIVE EXPLANATION: If we are measuring jitter, AUC represents the 
            # "total accumulated jitter" across the entire motion. If one model has a 
            # lower average jitter but takes twice as long to complete the motion, 
            # its AUC might actually be higher.
            val_auc = joint_data.sum()
            
            # Determine model from data if available, else 'unknown'
            model = metrics_df['model'].iloc[0] if 'model' in metrics_df.columns else 'unknown'
            
            record = {
                'rep_id': rep_id,
                'joint_name': joint,
                'median': val_median,
                'mean': val_mean,
                'std': val_std,
                'max': val_max,
                'iqr': val_iqr,
                'auc': val_auc,
                'model': model
            }
            aggregated_records.append(record)
            
    return pd.DataFrame(aggregated_records)


def aggregate_all(jitter_df: pd.DataFrame, confidence_df: pd.DataFrame, rep_boundaries: List[Tuple[int, int]]) -> Dict[str, pd.DataFrame]:
    """
    Runs the aggregation pipeline for both jitter and confidence datasets.
    
    Args:
        jitter_df: DataFrame containing frame-by-frame jitter data.
        confidence_df: DataFrame containing frame-by-frame confidence data.
        rep_boundaries: List of (start_frame, end_frame) tuples.
        
    Returns:
        A dictionary containing the aggregated DataFrames for both metrics.
    """
    
    print("Aggregating jitter data...")
    # Assuming the value column in jitter_df is 'jitter'
    metric_col_jitter = 'jitter' if 'jitter' in jitter_df.columns else jitter_df.columns[-1]
    agg_jitter = aggregate_per_repetition(jitter_df, rep_boundaries, metric_column=metric_col_jitter)
    
    print("Aggregating confidence data...")
    # Assuming the value column in confidence_df is 'confidence'
    metric_col_conf = 'confidence' if 'confidence' in confidence_df.columns else confidence_df.columns[-1]
    agg_confidence = aggregate_per_repetition(confidence_df, rep_boundaries, metric_column=metric_col_conf)
    
    return {
        "jitter": agg_jitter,
        "confidence": agg_confidence
    }


def add_metadata(agg_df: pd.DataFrame, practitioner_id: str, technique: str, condition: str) -> pd.DataFrame:
    """
    Adds contextual metadata to the aggregated DataFrame.
    
    CONCEPT:
    Once we aggregate data for one video (e.g., P01 doing Oizuki in the Gi), we 
    need to label it before we combine it with all the other videos. This allows 
    us to do "pooled analysis" (grouping all participants together) later on.
    
    Args:
        agg_df: The aggregated DataFrame (one row per repetition).
        practitioner_id: e.g., 'P01', 'P02', 'P03'.
        technique: e.g., 'OIZUKI'.
        condition: e.g., 'Set_A' (Control) or 'Set_B' (Experimental/Gi).
        
    Returns:
        The DataFrame with new metadata columns inserted at the beginning.
    """
    df = agg_df.copy()
    
    # Insert metadata columns at the front for readability
    df.insert(0, 'practitioner_id', practitioner_id)
    df.insert(1, 'technique', technique)
    df.insert(2, 'condition', condition)
    
    return df


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Example usage / Test block
    # -------------------------------------------------------------------------
    print("--- Testing Module 04: Aggregator ---")
    
    # Create some dummy frame-by-frame data for testing
    # Let's pretend we have 600 frames of video (10 seconds at 60 FPS)
    dummy_frames = 600
    
    # Create dummy jitter dataframe
    df_jitter = pd.DataFrame({
        'joint_name': ['right_wrist'] * dummy_frames,
        'jitter': np.random.normal(loc=0.02, scale=0.005, size=dummy_frames),
        'model': ['mediapipe'] * dummy_frames
    })
    
    # Create dummy confidence dataframe
    df_conf = pd.DataFrame({
        'joint_name': ['right_wrist'] * dummy_frames,
        'confidence': np.random.uniform(low=0.7, high=1.0, size=dummy_frames),
        'model': ['mediapipe'] * dummy_frames
    })
    
    # 1. Detect boundaries
    # Using 'OIZUKI' from config which specifies 10 reps
    print("1. Detecting repetition boundaries...")
    boundaries = detect_repetitions(df_jitter, technique="OIZUKI", model_name='mediapipe')
    print(f"Found {len(boundaries)} repetitions.")
    
    # 2. Aggregate
    print("\n2. Aggregating frame-level data into rep-level summaries...")
    results = aggregate_all(df_jitter, df_conf, boundaries)
    
    # 3. Add Metadata
    print("\n3. Adding experiment metadata for pooled analysis...")
    final_jitter = add_metadata(results["jitter"], practitioner_id="P01", technique="OIZUKI", condition="Set_A")
    final_conf = add_metadata(results["confidence"], practitioner_id="P01", technique="OIZUKI", condition="Set_A")
    
    print("\nSUCCESS! Preview of the final aggregated Jitter DataFrame:")
    print(final_jitter.head())
    
    print("\nRemember: Each row now represents ONE honest, independent repetition. Pseudoreplication averted!")
