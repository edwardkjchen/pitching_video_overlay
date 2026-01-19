"""
This script implements a video processing pipeline to first stabilize two videos
individually and then overlay them with spatial alignment based on pose estimation.

The pipeline is as follows:
1.  **Stabilize Videos**: Each input video is stabilized independently using
    a function `func_stablize_video`. This function estimates camera motion,
    smooths the motion trajectory, and applies corrections to remove jitter.
    The stabilized videos are saved as temporary files.
2.  **Estimate Initial Alignment**: It uses MediaPipe Pose to detect the
    back foot in the initial frames of the original videos. The difference
    in foot positions is used to calculate the spatial displacement needed
    to align the two subjects.
3.  **Render Overlay**: The two stabilized videos are then overlaid using
    `func_render_overlay`. This function takes the stabilized video paths
    and the calculated displacement to create the final composite video,
    saving it to a file.
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import subprocess
import sys

# --- Core Helper Functions ---

def get_back_foot_position(frame, pose_detector):
    """
    Analyzes a single frame to find the pixel coordinates of the right heel.

    Args:
        frame (np.array): The video frame (in BGR format) to analyze.
        pose_detector: An initialized MediaPipe Pose detector instance.

    Returns:
        tuple: The (x, y) pixel coordinates of the right heel, or None if not found.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    if not results.pose_landmarks:
        return None
    h, w, _ = frame.shape
    landmarks = results.pose_landmarks.landmark
    right_heel = mp.solutions.pose.PoseLandmark.RIGHT_HEEL
    x = int(landmarks[right_heel].x * w)
    y = int(landmarks[right_heel].y * h)
    return (x, y)

def estimate_stable_foot_position(video_path, pose_detector):
    """
    Estimates a stable position of the back foot from the first few frames of a video.

    Args:
        video_path (str): Path to the video file.
        pose_detector: An initialized MediaPipe Pose detector instance.

    Returns:
        tuple: The median (x, y) coordinates of the foot, or None.
    """
    print(f"Estimating stable foot position for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    positions = []
    for _ in range(10): # Analyze the first 10 frames
        ret, frame = cap.read()
        if not ret: break
        pos = get_back_foot_position(frame, pose_detector)
        if pos: positions.append(pos)
    cap.release()
    if not positions:
        print("  -> No foot detected.")
        return None
    median_x = int(np.median([p[0] for p in positions]))
    median_y = int(np.median([p[1] for p in positions]))
    print(f"  -> Stable position found: ({median_x}, {median_y})")
    return (median_x, median_y)


# --- Main Functional Units (Wrappers for scripts) ---

def func_stabilize_video(input_path, output_path):
    """
    Wrapper function that calls the func_stabilize_video.py script.

    Args:
        input_path (str): Path to the original shaky video.
        output_path (str): Path to save the stabilized video.
    """
    print(f"Calling stabilization script for {input_path}...")
    subprocess.run([sys.executable, "func_stabilize_video.py", input_path, output_path], check=True)
    print(f"  -> Stabilization script finished for {input_path}.")

def func_render_overlay(video1_path, video2_path, displacement, output_path, alpha=0.5):
    """
    Wrapper function that calls the func_render_overlay.py script.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.
        displacement (tuple): The (dx, dy) offset to apply to the second video.
        output_path (str): Path to save the final overlaid video.
        alpha (float): Transparency of the first video.
    """
    print(f"Calling overlay rendering script...")
    dx, dy = str(displacement[0]), str(displacement[1])
    subprocess.run([
        sys.executable,
        "func_render_overlay.py",
        video1_path,
        video2_path,
        dx,
        dy,
        output_path,
        "--alpha",
        str(alpha)
    ], check=True)
    print(f"  -> Overlay rendering script finished.")

# --- Main Execution ---

def main():
    """Main function to run the full video stabilization and overlay pipeline."""
    # --- Configuration ---
    video1_path = "Input_Video\cutsIMG_2725.mp4"
    video2_path = "Input_Video\cutsIMG_2726.mp4"
    stabilized1_path = "video1_stabilized_v01.mp4"
    stabilized2_path = "video2_stabilized_v01.mp4"
    final_overlay_path = "system_overlay_v01.mp4"

    # --- Pipeline ---
    
    # Step 1: Stabilize both videos individually
    func_stabilize_video(video1_path, stabilized1_path)
    func_stabilize_video(video2_path, stabilized2_path)

    # Step 2: Estimate initial spatial alignment based on back foot position
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    stable_pos1 = estimate_stable_foot_position(video1_path, pose_detector)
    stable_pos2 = estimate_stable_foot_position(video2_path, pose_detector)
    pose_detector.close()

    if stable_pos1 and stable_pos2:
        displacement = (stable_pos1[0] - stable_pos2[0], stable_pos1[1] - stable_pos2[1])
        print(f"Calculated initial displacement: {displacement}")
    else:
        print("Could not determine foot positions. Using zero displacement.")
        displacement = (0, 0)

    # Step 3: Render the final overlay using the stabilized videos
    func_render_overlay(stabilized1_path, stabilized2_path, displacement, final_overlay_path)
    
    # Optional: Clean up intermediate stabilized files
    # os.remove(stabilized1_path)
    # os.remove(stabilized2_path)
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
