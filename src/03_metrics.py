"""
03_metrics.py - KaratePose-AI Degradation Metrics

This module calculates the primary evaluation metrics for Paper 1 of the study.
It takes the raw coordinate data extracted from our pose estimation models
(MediaPipe and YOLOv8) and computes how much the pose predictions "jump around"
(jitter), how stable their confidence scores are, and the actual joint angles.

RESEARCH NOTE:
These metrics correspond directly to the degradation analysis described in
Section 6.4.3 of the research proposal. They quantify the difference in pose
estimation quality between the Control Set (Set A - rashguard) and the
Experimental Set (Set B - Karate Gi).
"""

import numpy as np
import pandas as pd
from typing import Dict
from src.config import PATHS, JOINT_ANGLES_MP, JOINT_ANGLES_YOLO

def compute_spatial_jitter(landmarks_df: pd.DataFrame, model_name: str = 'mediapipe') -> pd.DataFrame:
    """
    Computes the frame-to-frame spatial displacement (jitter) for each joint.
    
    CONCEPT: Spatial jitter measures how much a predicted joint location "twitches"
    or moves between consecutive video frames. In a perfectly smooth karate strike,
    the joints should follow a smooth curve. If the model is confused by the baggy
    Karate Gi, it might guess the elbow is at (X, Y) in frame 1, and (X+10, Y+10)
    in frame 2, creating a high jitter value.
    
    MATH: Euclidean Distance
    We use the Pythagorean theorem to find the straight-line distance between the
    joint's position in the current frame (t) and the previous frame (t-1).
    J_t = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    
    Args:
        landmarks_df (pd.DataFrame): The raw landmarks dataframe.
            Expected columns: [frame, joint_idx, joint_name, x, y, z, confidence, model]
        model_name (str): 'mediapipe' or 'yolov8'. Default is 'mediapipe'.
            
    Returns:
        pd.DataFrame: A dataframe containing [frame, joint_name, jitter, model].
    """
    # 1. Filter the data to only include rows for the requested model.
    df = landmarks_df[landmarks_df['model'] == model_name].copy()
    
    # Check edge case: If the dataframe is empty after filtering, return empty.
    if df.empty:
        return pd.DataFrame(columns=['frame', 'joint_name', 'jitter', 'model'])
    
    # 2. Sort the data by joint_name and then by frame.
    # CONCEPT: Sorting is crucial because we want to compare frame 2 with frame 1
    # FOR THE SAME JOINT. If we don't sort, we might accidentally calculate the distance
    # between the 'nose' in frame 1 and the 'left_ankle' in frame 2!
    df = df.sort_values(by=['joint_name', 'frame']).reset_index(drop=True)
    
    # 3. Calculate the differences (deltas) between consecutive rows.
    # We group by 'joint_name' so that the difference is only calculated within the same joint's data.
    # The .diff() function subtracts the previous row's value from the current row's value.
    # For the first frame of each joint, there is no previous row, so it becomes NaN (Not a Number).
    # CONCEPT: Vectorisation
    # Instead of writing a 'for' loop to go through every single row one by one, we use
    # pandas/numpy built-in functions. These functions process entire columns at once
    # using highly optimised C code in the background. This is called 'vectorisation'
    # and it makes our Python code run incredibly fast.
    delta_x = df.groupby('joint_name')['x'].diff()
    delta_y = df.groupby('joint_name')['y'].diff()
    
    # 4. Apply the Euclidean distance formula.
    # MATH: np.sqrt calculates the square root. ** 2 raises the value to the power of 2 (squares it).
    df['jitter'] = np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    # 5. Handle edge cases where confidence might be 0 (meaning the model completely lost the joint).
    # If confidence is 0, the coordinates might be (0,0) or garbage. We should set jitter to NaN
    # because a jump to (0,0) isn't real movement, it's a tracking failure.
    df.loc[df['confidence'] == 0, 'jitter'] = np.nan
    
    # 6. Select only the columns we want to return.
    result_df = df[['frame', 'joint_name', 'jitter', 'model']].copy()
    
    return result_df

def compute_confidence_volatility(landmarks_df: pd.DataFrame, model_name: str = 'mediapipe') -> pd.DataFrame:
    """
    Computes the frame-to-frame absolute change in confidence score for each joint.
    
    CONCEPT: Confidence volatility measures how "sure" the model is over time.
    If the model gives a confidence of 0.99 in one frame, and 0.40 in the next, it
    means the model is struggling (likely due to clothing occlusion from the Gi).
    We want to quantify these unstable confidence scores.
    
    MATH: Absolute Difference
    We subtract the previous frame's confidence (c_{t-1}) from the current frame's
    confidence (c_t), and take the absolute value (meaning we remove any negative signs).
    V_t = |c_t - c_{t-1}|
    
    Args:
        landmarks_df (pd.DataFrame): The raw landmarks dataframe.
        model_name (str): 'mediapipe' or 'yolov8'. Default is 'mediapipe'.
            
    Returns:
        pd.DataFrame: A dataframe containing [frame, joint_name, conf_volatility, confidence, model].
    """
    # 1. Filter for the requested model.
    df = landmarks_df[landmarks_df['model'] == model_name].copy()
    if df.empty:
        return pd.DataFrame(columns=['frame', 'joint_name', 'conf_volatility', 'confidence', 'model'])
    
    # 2. Sort by joint, then by frame to ensure correct chronological order.
    df = df.sort_values(by=['joint_name', 'frame']).reset_index(drop=True)
    
    # 3. Calculate the difference in confidence.
    # Again, we use .diff() grouped by joint_name.
    conf_diff = df.groupby('joint_name')['confidence'].diff()
    
    # 4. Calculate the absolute value.
    # MATH: np.abs() turns negative numbers into positive numbers (e.g., -0.5 becomes 0.5).
    # We don't care IF confidence went up or down, we just care about the MAGNITUDE of the change.
    df['conf_volatility'] = np.abs(conf_diff)
    
    # 5. Select the final columns to return. We include the raw confidence for reference.
    result_df = df[['frame', 'joint_name', 'conf_volatility', 'confidence', 'model']].copy()
    
    return result_df

def compute_joint_angles(landmarks_df: pd.DataFrame, model_name: str = 'mediapipe') -> pd.DataFrame:
    """
    Computes the anatomical joint angles defined in our configuration file.
    
    CONCEPT: In biomechanics, we rarely care about raw X,Y coordinates. We care about
    joint angles (e.g., "is the knee fully extended?"). To calculate an angle at a joint
    (like the knee), we need three points: the hip, the knee, and the ankle. We then
    measure the angle formed between the hip-knee line and the knee-ankle line.
    
    MATH: Dot Product and Arccos
    Imagine two arrows (vectors) starting from the knee.
    Arrow 1 (Vector BA) points from Knee to Hip.
    Arrow 2 (Vector BC) points from Knee to Ankle.
    
    The "Dot Product" is a mathematical operation that tells us how much these two
    arrows point in the same direction. 
    The formula is: Dot Product = (Length of BA) * (Length of BC) * cos(theta)
    Where theta (θ) is the angle between them.
    
    We can rearrange this formula to solve for cos(θ):
    cos(θ) = (BA · BC) / (|BA| × |BC|)
    
    Once we have cos(θ), we use the "arccos" (inverse cosine) function to find the actual
    angle θ. Finally, we convert it from radians (which math functions use) to degrees
    (which humans understand).
    
    Args:
        landmarks_df (pd.DataFrame): The raw landmarks dataframe.
        model_name (str): 'mediapipe' or 'yolov8'. Default is 'mediapipe'.
            
    Returns:
        pd.DataFrame: A dataframe containing [frame, angle_name, angle_degrees, model].
    """
    # Filter for the specified model
    df = landmarks_df[landmarks_df['model'] == model_name].copy()
    if df.empty:
        return pd.DataFrame(columns=['frame', 'angle_name', 'angle_degrees', 'model'])
    
    # Select the correct angle definitions from config based on the model
    # RESEARCH NOTE: MediaPipe and YOLOv8 use different numbering systems for joints,
    # so we must use the correct dictionary from our central config!
    if model_name.lower() == 'mediapipe':
        angle_defs = JOINT_ANGLES_MP
    elif model_name.lower() == 'yolov8':
        angle_defs = JOINT_ANGLES_YOLO
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # We will collect our calculated angles in a list, then build a DataFrame at the end.
    results = []
    
    # Get all unique frames in the data
    frames = df['frame'].unique()
    
    # We create a fast lookup dictionary for coordinates in this specific frame
    # A pivot table helps us rearrange the data so we can look up coordinates by joint index quickly
    # Index = frame, Columns = joint_idx, Values = x or y
    pivot_x = df.pivot(index='frame', columns='joint_idx', values='x')
    pivot_y = df.pivot(index='frame', columns='joint_idx', values='y')
    
    # Iterate over every angle defined in our config
    for angle_name, joints in angle_defs.items():
        idx_A = joints['A']
        idx_B = joints['B']  # This is the vertex (the joint where the angle is measured)
        idx_C = joints['C']
        
        # Check if all required joints actually exist in our data for this model
        if idx_A not in pivot_x.columns or idx_B not in pivot_x.columns or idx_C not in pivot_x.columns:
            continue # Skip this angle if the model doesn't track these joints
            
        # Extract the X and Y coordinates for all frames for points A, B, and C
        # This is another example of vectorisation: we extract entire columns (all frames) at once.
        A_x = pivot_x[idx_A].values
        A_y = pivot_y[idx_A].values
        
        B_x = pivot_x[idx_B].values
        B_y = pivot_y[idx_B].values
        
        C_x = pivot_x[idx_C].values
        C_y = pivot_y[idx_C].values
        
        # Calculate Vector BA (from B to A)
        BA_x = A_x - B_x
        BA_y = A_y - B_y
        
        # Calculate Vector BC (from B to C)
        BC_x = C_x - B_x
        BC_y = C_y - B_y
        
        # Calculate the Dot Product (BA · BC)
        dot_product = (BA_x * BC_x) + (BA_y * BC_y)
        
        # Calculate the magnitudes (lengths) of vectors BA and BC
        # Using Pythagorean theorem again
        mag_BA = np.sqrt(BA_x**2 + BA_y**2)
        mag_BC = np.sqrt(BC_x**2 + BC_y**2)
        
        # Calculate cos(theta). 
        # We add a tiny number (1e-7) to the denominator to prevent "Division by Zero" errors
        # just in case a vector has 0 length (e.g., if point B and A are in exactly the same spot).
        cos_theta = dot_product / ((mag_BA * mag_BC) + 1e-7)
        
        # Floating point math can sometimes result in numbers slightly outside [-1, 1],
        # like 1.0000000002. The arccos function will crash if you give it a number > 1.
        # So we "clip" the values to strictly stay within -1.0 and 1.0.
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calculate the angle in radians, then convert to degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        # Append the results for all frames
        for i, frame in enumerate(pivot_x.index):
            results.append({
                'frame': frame,
                'angle_name': angle_name,
                'angle_degrees': angle_deg[i],
                'model': model_name
            })
            
    # Convert our list of results back into a clean pandas DataFrame
    angles_df = pd.DataFrame(results)
    return angles_df

def compute_all_metrics(landmarks_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Master function to run all three metric calculations for BOTH models.
    
    CONCEPT: This is a "wrapper" or "orchestrator" function. Instead of calling
    the three individual functions multiple times in our main script, we just call
    this one function, which handles all the heavy lifting and returns a tidy dictionary.
    
    Args:
        landmarks_df (pd.DataFrame): The complete raw landmarks dataframe containing
                                     data from both MediaPipe and YOLOv8.
                                     
    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing three DataFrames:
                                 {'jitter': jitter_df, 'confidence': conf_df, 'angles': angles_df}
    """
    models = ['mediapipe', 'yolov8']
    
    all_jitter = []
    all_conf = []
    all_angles = []
    
    # Loop through both models and compute their metrics
    for model in models:
        print(f"Computing metrics for {model.upper()}...")
        
        jitter = compute_spatial_jitter(landmarks_df, model_name=model)
        conf = compute_confidence_volatility(landmarks_df, model_name=model)
        angles = compute_joint_angles(landmarks_df, model_name=model)
        
        all_jitter.append(jitter)
        all_conf.append(conf)
        all_angles.append(angles)
        
    # Combine the lists of DataFrames into single large DataFrames using pd.concat
    # ignore_index=True ensures we get clean 0, 1, 2, 3... row numbers.
    final_jitter = pd.concat(all_jitter, ignore_index=True)
    final_conf = pd.concat(all_conf, ignore_index=True)
    final_angles = pd.concat(all_angles, ignore_index=True)
    
    # --- SAVING TO CSV ---
    # RESEARCH NOTE: We save these intermediate metric files so we don't have to 
    # recalculate them every time we want to run statistical tests or plot graphs.
    
    # Ensure the output directory exists
    PATHS["metrics"].mkdir(parents=True, exist_ok=True)
    
    # Save the dataframes
    jitter_path = PATHS["metrics"] / "spatial_jitter.csv"
    conf_path = PATHS["metrics"] / "confidence_volatility.csv"
    angles_path = PATHS["metrics"] / "joint_angles.csv"
    
    final_jitter.to_csv(jitter_path, index=False)
    final_conf.to_csv(conf_path, index=False)
    final_angles.to_csv(angles_path, index=False)
    
    print(f"Metrics saved to {PATHS['metrics']}")
    
    return {
        'jitter': final_jitter,
        'confidence': final_conf,
        'angles': final_angles
    }

if __name__ == '__main__':
    # =========================================================================
    # DEMONSTRATION BLOCK
    # =========================================================================
    # CONCEPT: The code inside this 'if __name__ == "__main__":' block only runs
    # when you execute THIS specific file (e.g., `python 03_metrics.py`). 
    # It will NOT run if another file imports functions from here.
    # This is standard Python practice for testing and demonstration.
    
    print("--- KaratePose-AI Metrics Demonstration ---")
    
    # 1. Create a tiny dummy dataset to show how the functions work
    # We pretend we have 3 frames of data for a left knee and left hip using MediaPipe.
    dummy_data = {
        'frame': [1, 2, 3, 1, 2, 3],
        'joint_idx': [25, 25, 25, 23, 23, 23],  # 25=left_knee, 23=left_hip
        'joint_name': ['left_knee', 'left_knee', 'left_knee', 'left_hip', 'left_hip', 'left_hip'],
        'x': [0.5, 0.52, 0.55, 0.5, 0.5, 0.5],     # Knee is moving right, hip is still
        'y': [0.8, 0.81, 0.83, 0.5, 0.5, 0.5],     # Knee is moving down, hip is still
        'z': [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        'confidence': [0.99, 0.80, 0.00, 0.95, 0.95, 0.95], # Knee confidence drops to 0!
        'model': ['mediapipe', 'mediapipe', 'mediapipe', 'mediapipe', 'mediapipe', 'mediapipe']
    }
    
    dummy_df = pd.DataFrame(dummy_data)
    
    print("\n[Input] Raw Dummy Data:")
    print(dummy_df)
    
    # 2. Test Spatial Jitter
    print("\n[Output] Spatial Jitter:")
    # Note how the first frame is NaN, and the third frame for the knee is NaN because confidence dropped to 0.
    print(compute_spatial_jitter(dummy_df, model_name='mediapipe'))
    
    # 3. Test Confidence Volatility
    print("\n[Output] Confidence Volatility:")
    # Note the jump from 0.99 to 0.80 is a volatility of 0.19.
    print(compute_confidence_volatility(dummy_df, model_name='mediapipe'))
    
    # Note: We can't easily test joint angles with just hip and knee (we need the ankle too!)
    # But the master function would process it all.
    print("\nDemonstration complete. In production, pass the full landmarks_df to compute_all_metrics().")
