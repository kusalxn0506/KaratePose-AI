import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Import central configuration
from src.config import PATHS

# Define how the joints connect to form a skeleton
CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    
    # Right Arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    
    # Left Arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    
    # Right Leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    
    # Left Leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    
    # Feet (MediaPipe only)
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
]

def extract_points(model_data, width, height):
    """Extracts valid points (confidence > 0.3) into a dictionary."""
    points = {}
    for _, row in model_data.iterrows():
        if row['confidence'] > 0.3:
            x = int(row['x'] * width)
            y = int(row['y'] * height)
            points[row['joint_name']] = (x, y)
    return points

def draw_points(canvas, points_dict, color):
    """Draws ONLY the dots (for the real video side)."""
    for point in points_dict.values():
        cv2.circle(canvas, point, 8, color, -1)

def draw_skeleton(canvas, points_dict, color):
    """Draws the bones AND the dots (for the dark side)."""
    # 1. Draw the bones (lines)
    for joint1, joint2 in CONNECTIONS:
        if joint1 in points_dict and joint2 in points_dict:
            cv2.line(canvas, points_dict[joint1], points_dict[joint2], color, 3)
            
    # 2. Draw the joints (dots) on top of the lines
    for point in points_dict.values():
        cv2.circle(canvas, point, 8, color, -1)

def play_overlay():
    video_path = PATHS["sample_data"] / "test_oizuki.mp4"
    csv_path = PATHS["landmarks"] / "test_oizuki_landmarks.csv"

    if not csv_path.exists():
        print(f"❌ Cannot find CSV file at: {csv_path}")
        return
    if not video_path.exists():
        print(f"❌ Cannot find Video file at: {video_path}")
        return

    print("Loading landmark data...")
    df = pd.read_csv(csv_path)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    fps = str(fps)

    print("\n" + "="*60)
    print("🎬 PLAYING SIDE-BY-SIDE OVERLAY")
    print("  • LEFT:  Real video with tracking dots")
    print("  • RIGHT: Extracted skeleton on black background")
    print("  • Blue = MediaPipe | Red = YOLOv8")
    print("  • Press 'q' to close the video window early.")
    print("="*60 + "\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_data = df[df['frame'] == frame_idx]
        
        # Create a pure black canvas for the skeleton side
        skeleton_canvas = np.zeros_like(frame)
        
        # Extract points for both models
        mp_data = frame_data[frame_data['model'] == 'mediapipe']
        mp_points = extract_points(mp_data, width, height)
        
        yolo_data = frame_data[frame_data['model'] == 'yolov8']
        yolo_points = extract_points(yolo_data, width, height)

        # --- LEFT SIDE (Real Video) ---
        # Only draw dots on the real person
        draw_points(frame, mp_points, (255, 0, 0))    # Blue
        draw_points(frame, yolo_points, (0, 0, 255))  # Red
        
        # --- RIGHT SIDE (Black Canvas) ---
        # Draw full skeleton on the blank canvas
        draw_skeleton(skeleton_canvas, mp_points, (255, 0, 0))   # Blue
        draw_skeleton(skeleton_canvas, yolo_points, (0, 0, 255)) # Red

        # Resize both frames to 50% so they fit side-by-side on your screen
        # (1920x1080 becomes 960x540 each)
        left_side = cv2.resize(frame, (width // 2, height // 2))
        right_side = cv2.resize(skeleton_canvas, (width // 2, height // 2))
        
        # Stick them together horizontally
        combined_display = np.hstack((left_side, right_side))
        
        # Add labels to the display
        cv2.putText(combined_display,"Real Video (Points Only)" + "\n\n\n\n\n\n\n\n Frame Rate:" +fps, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_display, "Extracted Skeleton", (width // 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Legend (Bottom Left)
        cv2.putText(combined_display, "MediaPipe (Blue)", (20, height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined_display, "YOLOv8 (Red)", (20, height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Karate Pose AI - Side-by-Side Analysis", combined_display)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_overlay()
