"""
This module estimates the spatial displacement required to align two subjects
in different videos.

It uses MediaPipe Pose to detect the back foot in the initial frames of the
videos and calculates the difference in their median positions.
"""
import cv2
import numpy as np
import mediapipe as mp
import os

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

def estimate_stable_foot_position(video_path, pose_detector, debug_output_path=None):
    """
    Estimates a stable position of the back foot from the first few frames of a video.
    Optionally saves a debug image with the detected foot position.

    Args:
        video_path (str): Path to the video file.
        pose_detector: An initialized MediaPipe Pose detector instance.
        debug_output_path (str, optional): Path to save a debug image with the
                                           detected foot position. Defaults to None.

    Returns:
        tuple: The median (x, y) coordinates of the foot, or None.
    """
    print(f"Estimating stable foot position for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    positions = []
    # Store the first frame where a foot was detected for debug visualization
    first_frame_with_foot = None
    first_frame_read_success = False

    for i in range(10): # Analyze the first 10 frames
        ret, frame = cap.read()
        if not ret: break

        if not first_frame_read_success:
            first_frame_with_foot = frame.copy() # Make a copy before potential modification
            first_frame_read_success = True

        pos = get_back_foot_position(frame, pose_detector)
        if pos: 
            positions.append(pos)
            if first_frame_with_foot is None: # Only store the first frame where a foot is found
                first_frame_with_foot = frame.copy()
    cap.release()

    if not positions:
        print("  -> No foot detected.")
        return None
    
    median_x = int(np.median([p[0] for p in positions]))
    median_y = int(np.median([p[1] for p in positions]))
    print(f"  -> Stable position found: ({median_x}, {median_y})")

    # Draw debug dot if path is provided
    if debug_output_path and first_frame_with_foot is not None:
        if median_x is not None and median_y is not None:
            cv2.circle(first_frame_with_foot, (median_x, median_y), 10, (0, 0, 255), -1) # Red dot
            cv2.imwrite(debug_output_path, first_frame_with_foot)
            print(f"  -> Debug image saved to: {debug_output_path}")

    return (median_x, median_y)

def get_spatial_displacement(video1_path, video2_path, debug_output_dir=None):
    """
    Calculates the spatial displacement between two videos based on foot position.
    Optionally saves debug images of detected foot positions.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.
        debug_output_dir (str, optional): Directory to save debug images. Defaults to None.

    Returns:
        tuple: The (dx, dy) displacement to align video2 with video1.
    """
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

    debug_path1 = None
    debug_path2 = None
    if debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        base1 = os.path.splitext(os.path.basename(video1_path))[0]
        base2 = os.path.splitext(os.path.basename(video2_path))[0]
        debug_path1 = os.path.join(debug_output_dir, f"{base1}_foot_pos.png")
        debug_path2 = os.path.join(debug_output_dir, f"{base2}_foot_pos.png")

    stable_pos1 = estimate_stable_foot_position(video1_path, pose_detector, debug_output_path=debug_path1)
    stable_pos2 = estimate_stable_foot_position(video2_path, pose_detector, debug_output_path=debug_path2)
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
