"""
This module estimates the spatial displacement required to align two subjects
in different videos.

It uses MediaPipe Pose to detect the back foot in the initial frames of the
videos and calculates the difference in their median positions.
"""
import cv2
import numpy as np
import mediapipe as mp

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

def get_spatial_displacement(video1_path, video2_path):
    """
    Calculates the spatial displacement between two videos based on foot position.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.

    Returns:
        tuple: The (dx, dy) displacement to align video2 with video1.
    """
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
    
    return displacement

if __name__ == '__main__':
    # Example of how to use the function
    video1 = "Input_Video/cutsIMG_2725.mp4"
    video2 = "Input_Video/cutsIMG_2726.mp4"
    try:
        displacement = get_spatial_displacement(video1, video2)
        print(f"\nDisplacement (dx, dy): {displacement}")
    except FileNotFoundError:
        print("Error: Make sure the video files exist at the specified paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
