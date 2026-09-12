"""
Module: 01_frame_extractor.py
Purpose: Extracts individual frames from karate video files for pose estimation analysis.

RESEARCH NOTE: This corresponds to §6.4.1 Frame Extraction in the proposal. 
Before we can run AI pose estimation, we need to break the video down into individual static images (frames).

CONCEPT: A video is essentially a fast slideshow of still images. 
This module uses a library called OpenCV (cv2) to read a video file and extract these still images 
so we can analyze them one by one.
"""

import cv2 # OpenCV library for computer vision tasks
import numpy as np # NumPy for numerical operations, like handling image matrices
from pathlib import Path # Pathlib for easy and safe file path manipulation
from typing import Tuple, List, Dict, Optional # Type hints to make code readable and help catch errors
from tqdm import tqdm # Tqdm for showing progress bars in the terminal

# CONCEPT: We try to import configuration settings.
# Using a try-except block here so the script doesn't completely crash if the config file is missing while testing.
try:
    from src.config import PATHS, VIDEO_CONFIG
except ImportError:
    print("Warning: Could not import config. Make sure you are running this from the project root.")
    # Fallback default values if config is missing
    PATHS = {"RAW_DATA": Path("data/raw"), "PROCESSED_DATA": Path("data/processed/frames")}
    VIDEO_CONFIG = {"TARGET_FPS": 30, "IMAGE_EXT": ".png"}


def get_video_metadata(video_path: Path) -> Dict[str, float]:
    """
    Reads basic information (metadata) about a video file without loading the entire video into memory.
    
    CONCEPT: 'Metadata' means 'data about data'. Here, it's information like width, height, and fps, 
    rather than the actual pixel colors of the video.
    
    Args:
        video_path (Path): The file path to the video.
        
    Returns:
        Dict[str, float]: A dictionary containing fps, total_frames, width, height, and duration_s.
    """
    # CONCEPT: Ensure video_path is a Path object for consistent handling
    video_path = Path(video_path)
    
    # Check if the file actually exists before trying to open it
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # CONCEPT: cv2.VideoCapture is OpenCV's way of reading video files frame by frame.
    # It creates an object that acts like a connection to the video file.
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file (might be corrupted or unsupported format): {video_path}")

    # CONCEPT: cap.get() retrieves specific properties of the video using predefined OpenCV constants (cv2.CAP_PROP_...).
    fps = cap.get(cv2.CAP_PROP_FPS) # Frames per second
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Total number of frames in the video
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Width of the video in pixels
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Height of the video in pixels
    
    # Calculate duration in seconds (total frames / frames per second)
    duration_s = total_frames / fps if fps > 0 else 0.0

    # CONCEPT: Always release the VideoCapture object when done to free up system resources!
    cap.release()

    # Return a dictionary containing the extracted metadata
    return {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
        "duration_s": float(duration_s)
    }


def extract_frames(video_path: Path, output_dir: Optional[Path] = None) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Opens a video file, reads every frame, and returns them. 
    Optionally saves the frames to a directory.
    
    RESEARCH NOTE: High frame rate extraction is vital for capturing explosive karate movements (e.g., gyaku-tsuki).
    
    Args:
        video_path (Path): Path to the video file.
        output_dir (Optional[Path]): If provided, frames will be saved as images in this directory.
        
    Returns:
        Tuple[List[np.ndarray], Dict[str, float]]: A tuple containing a list of frames (as numpy arrays) and the metadata dictionary.
    """
    video_path = Path(video_path)
    print(f"Extracting frames from: {video_path.name}")
    
    # First, get the metadata so we know how many frames to expect
    metadata = get_video_metadata(video_path)
    total_frames = metadata["total_frames"]
    
    # If an output directory is provided, create it if it doesn't exist
    if output_dir:
        output_dir = Path(output_dir)
        # parents=True creates parent directories if needed, exist_ok=True prevents errors if it already exists
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Frames will be saved to: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    frames = [] # List to store the extracted frames
    
    # CONCEPT: tqdm creates a progress bar. We tell it the total number of frames so it can calculate the percentage.
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        frame_idx = 0
        while True:
            # cap.read() returns two values:
            # - success: A boolean indicating if a frame was successfully read
            # - frame: The frame itself as a NumPy array (if success is True)
            success, frame = cap.read()
            
            # If success is False, we've reached the end of the video (or there's an error reading the frame)
            if not success:
                break
                
            # Add the frame to our list
            frames.append(frame)
            
            # If an output directory was provided, save the frame as an image file
            if output_dir:
                # Format the filename with leading zeros (e.g., frame_0001.png) so they sort correctly alphabetically
                filename = f"frame_{frame_idx:04d}.png"
                out_path = output_dir / filename
                # cv2.imwrite saves the NumPy array as an image file
                cv2.imwrite(str(out_path), frame)
                
            frame_idx += 1
            # Update the progress bar
            pbar.update(1)
            
    # Release the video object
    cap.release()
    print(f"Extraction complete. Successfully extracted {len(frames)} frames.")
    
    return frames, metadata


def detect_calibration_rod(frame: np.ndarray, rod_length_m: float = 1.0) -> Optional[float]:
    """
    Attempts to detect a calibration rod in the frame and calculate the pixel-to-meter scaling factor.
    
    RESEARCH NOTE: This addresses §6.3.2 2D to 3D Scaling Strategy. We need a known reference 
    in the video to translate abstract pixel distances into real-world meters for kinematic analysis.
    
    Args:
        frame (np.ndarray): The video frame image as a NumPy array.
        rod_length_m (float): The known length of the calibration rod in real-world meters.
        
    Returns:
        Optional[float]: The scaling factor (pixels per meter), or None if detection fails.
    """
    print("\n--- Calibration Rod Detection ---")
    print(f"Looking for a {rod_length_m}m calibration rod...")
    
    # CONCEPT: Automated object detection (like finding a specific rod) can be very complex 
    # depending on lighting, background clutter, and rod color.
    # For this beginner module, we will provide a simplified placeholder that warns the user.
    
    print("Warning: Automatic rod detection is highly dependent on scene conditions.")
    print("Currently using a fallback/dummy scaling factor for demonstration.")
    
    # TODO: Implement manual clicking interface using cv2.setMouseCallback
    # This would pop up a window and let the user click the two ends of the rod.
    # print("TODO: Implement manual point-and-click calibration.")
    
    # Dummy calculation for demonstration: assume the rod is 500 pixels long in the image
    dummy_rod_length_pixels = 500.0 
    
    # Calculate the scaling factor: how many pixels represent one meter?
    scaling_factor = dummy_rod_length_pixels / rod_length_m
    
    print(f"Result: Calculated scaling factor = {scaling_factor} pixels/meter (DUMMY VALUE)")
    return scaling_factor


if __name__ == '__main__':
    # CONCEPT: The code inside this block only runs if you execute this script directly 
    # (e.g., `python 01_frame_extractor.py`). It will NOT run if you import this module into another script.
    
    print("=== Karate Pose AI: Frame Extractor Module Test ===")
    
    # Setup dummy paths for demonstration
    # Replace these with actual video paths when running with real data
    sample_video = Path("sample_karate_kick.mp4")
    output_directory = Path("output_frames")
    
    print(f"\n1. Testing Metadata Extraction...")
    # We use a try-except block to gracefully handle the case where the sample video doesn't exist
    try:
        # Create a dummy file just to show the error handling or allow tests to pass if mocked
        # with open(sample_video, 'w') as f: f.write('dummy') # Uncomment to create a fake file
        
        # In a real scenario, make sure 'sample_video.mp4' exists in the directory!
        if not sample_video.exists():
            print(f"  -> Skipping. (Please place a video named '{sample_video.name}' in the current directory to test this properly.)")
        else:
            meta = get_video_metadata(sample_video)
            print(f"  -> Metadata found: {meta}")
            
            print(f"\n2. Testing Frame Extraction...")
            frames, meta = extract_frames(sample_video, output_dir=output_directory)
            
            if len(frames) > 0:
                print(f"\n3. Testing Calibration Detection on the first frame...")
                first_frame = frames[0]
                scale_factor = detect_calibration_rod(first_frame, rod_length_m=1.0)
                
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        
    print("\nTest complete.")
