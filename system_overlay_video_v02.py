"""
This script implements a video processing pipeline to synchronize and overlay two videos.

The pipeline is as follows:
1.  **Stabilize Videos**: Each input video is stabilized independently.
2.  **Temporal Alignment**: The script calls `func_temporal_alignment` to find the
    frame offset between the two original videos.
3.  **Trim Videos**: The video that starts earlier is trimmed by the calculated
    frame offset to ensure both videos start at the same moment.
4.  **Estimate Initial Alignment**: It uses MediaPipe Pose to find the spatial
    displacement needed to align the subjects in the videos.
5.  **Render Overlay**: The stabilized, temporally aligned, and spatially aligned
    videos are overlaid to create the final composite video.
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import subprocess
import sys

# --- Import functions from other scripts ---
from func_temporal_alignment import temporal_align_videos
from func_render_overlay import render_overlay

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

def trim_video(input_path, output_path, frames_to_skip):
    """
    Trims a specified number of frames from the beginning of a video.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the trimmed video.
        frames_to_skip (int): Number of frames to skip from the beginning.
    """
    print(f"Trimming {frames_to_skip} frames from {input_path} -> {output_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path} for trimming.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Skip the initial frames
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            print("Warning: Video ended before all frames could be skipped.")
            break

    # Write the remaining frames to the new video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"  -> Trimming complete.")


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


# --- Main Execution ---

def main():
    """Main function to run the full video processing pipeline."""
    # --- Configuration ---
    video1_path = "Input_Video/cutsIMG_2725.mp4"
    video2_path = "Input_Video/cutsIMG_2726.mp4"
    
    stabilized1_path = "video1_stabilized_v02.mp4"
    stabilized2_path = "video2_stabilized_v02.mp4"
    
    trimmed1_path = "video1_trimmed_v02.mp4"
    trimmed2_path = "video2_trimmed_v02.mp4"
    
    final_overlay_path = "system_overlay_v02.mp4"

    # --- Pipeline ---

    # Step 1: Stabilize both videos individually
    func_stabilize_video(video1_path, stabilized1_path)
    func_stabilize_video(video2_path, stabilized2_path)

    # Step 2: Temporal Alignment
    print("Starting temporal alignment...")
    # It's better to use original videos for alignment to avoid stabilization artifacts
    frame_shift = temporal_align_videos(video1_path, video2_path)
    frame_shift = int(round(frame_shift))
    print(f"Temporal alignment complete. Frame shift: {frame_shift}")

    # Step 3: Trim videos based on the calculated shift
    video1_to_render = stabilized1_path
    video2_to_render = stabilized2_path

    if frame_shift < 0:
        # Video 2 starts later, so we trim the beginning of Video 1
        print(f"Video 2 is shifted by {frame_shift} frames. Trimming Video 1.")
        trim_video(stabilized1_path, trimmed1_path, frame_shift)
        video1_to_render = trimmed1_path
        # The second video does not need trimming, but we copy it for consistent naming
        os.system(f"copy {stabilized2_path} {trimmed2_path}")
        video2_to_render = trimmed2_path
    elif frame_shift > 0:
        # Video 1 starts later, so we trim the beginning of Video 2
        shift = abs(frame_shift)
        print(f"Video 1 is shifted by {shift} frames. Trimming Video 2.")
        trim_video(stabilized2_path, trimmed2_path, shift)
        video2_to_render = trimmed2_path
        # The first video does not need trimming, but we copy it for consistent naming
        os.system(f"copy {stabilized1_path} {trimmed1_path}")
        video1_to_render = trimmed1_path
    else:
        print("Videos are already synchronized. No trimming needed.")
        os.system(f"copy {stabilized1_path} {trimmed1_path}")
        os.system(f"copy {stabilized2_path} {trimmed2_path}")
        video1_to_render = trimmed1_path
        video2_to_render = trimmed2_path


    # Step 4: Estimate initial spatial alignment based on back foot position
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    # Use original videos for foot position to avoid stabilization affecting coordinates
    stable_pos1 = estimate_stable_foot_position(video1_path, pose_detector)
    stable_pos2 = estimate_stable_foot_position(video2_path, pose_detector)
    pose_detector.close()

    if stable_pos1 and stable_pos2:
        displacement = (stable_pos1[0] - stable_pos2[0], stable_pos1[1] - stable_pos2[1])
        print(f"Calculated initial displacement: {displacement}")
    else:
        print("Could not determine foot positions. Using zero displacement.")
        displacement = (0, 0)

    # Step 5: Render the final overlay using the trimmed and stabilized videos
    render_overlay(video1_to_render, video2_to_render, displacement, final_overlay_path)
    
    # Optional: Clean up intermediate files
    # os.remove(stabilized1_path)
    # os.remove(stabilized2_path)
    # os.remove(trimmed1_path)
    # os.remove(trimmed2_path)
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
