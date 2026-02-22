import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import shutil
import math

# Import necessary functions from func_scale_alignment
from func_scale_alignment import analyze_pitching_motion, find_motion_start_frame

def get_stable_landmark_position(
    coords_list: list, 
    start_frame: int, 
    end_frame: int
) -> np.ndarray:
    """
    Calculates the median position of a landmark over a given frame window.

    Args:
        coords_list (list): A list of (x, y) coordinates for a landmark over time.
        start_frame (int): The starting frame of the window.
        end_frame (int): The ending frame of the window.

    Returns:
        np.ndarray: The median (x, y) coordinate, or None if not found.
    """
    valid_coords = [
        c for c in coords_list[start_frame:end_frame] if c is not None
    ]
    if not valid_coords:
        return None
    
    # Apply median filter element-wise (x, y)
    x_coords = [c[0] for c in valid_coords]
    y_coords = [c[1] for c in valid_coords]
    
    median_coord = np.array([np.median(x_coords), np.median(y_coords)])
    return median_coord

def calculate_tilt_angle_for_video(video_path: str) -> float:
    """
    Calculates the camera tilt angle for a single video.

    The tilt is determined by the angle of the line connecting two key points:
    1. The right ankle's position during a static phase before motion.
    2. The left ankle's position at the moment of the right wrist's maximum speed.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        float: The calculated tilt angle in degrees, or None if it cannot be determined.
    """
    print(f" Analyzing video for tilt: {video_path}")
    all_landmark_coords, all_landmark_speeds, fps = analyze_pitching_motion(video_path)
    
    if not any(val is not None for val_list in all_landmark_coords.values() for val in val_list):
        print("  -> Could not detect any pose landmarks in the video.")
        return None

    # 1. Find the location of the right ankle during the static phase
    motion_start_frame = find_motion_start_frame(all_landmark_speeds, fps)
    print(f"  -> Detected major motion starting around frame {motion_start_frame}.")
    
    # Define a static window of 0.5s before motion starts
    static_window_size = int(fps * 0.5)
    static_period_start = max(0, motion_start_frame - static_window_size)
    static_period_end = motion_start_frame
    
    print(f"  -> Using static analysis window from frame {static_period_start} to {static_period_end}.")
    
    right_ankle_coords = all_landmark_coords.get('RIGHT_ANKLE', [])
    static_right_ankle_pos = get_stable_landmark_position(
        right_ankle_coords, static_period_start, static_period_end
    )
    
    if static_right_ankle_pos is None:
        print("  -> Could not determine stable right ankle position in static phase.")
        return None
    print(f"  -> Stable right ankle position: ({static_right_ankle_pos[0]:.2f}, {static_right_ankle_pos[1]:.2f})")

    # 2. Find the location of the left ankle at the moment of max right wrist speed
    right_wrist_speeds = all_landmark_speeds.get('RIGHT_WRIST', [])
    if not right_wrist_speeds or pd.Series(right_wrist_speeds).isnull().all():
        print("  -> Could not find right wrist speed data.")
        return None
        
    # Find the frame with the maximum speed for the right wrist
    max_speed_frame = pd.Series(right_wrist_speeds).idxmax()
    print(f"  -> Right wrist reached maximum speed at frame {max_speed_frame}.")
    
    # Define a 3-frame window around the max speed frame
    motion_window_start = max(0, max_speed_frame - 1)
    motion_window_end = min(len(right_wrist_speeds), max_speed_frame + 2)

    left_ankle_coords = all_landmark_coords.get('LEFT_ANKLE', [])
    motion_left_ankle_pos = get_stable_landmark_position(
        left_ankle_coords, motion_window_start, motion_window_end
    )

    if motion_left_ankle_pos is None:
        print("  -> Could not determine left ankle position at peak motion.")
        return None
    print(f"  -> Left ankle position at peak motion: ({motion_left_ankle_pos[0]:.2f}, {motion_left_ankle_pos[1]:.2f})")

    # 3. Use these two positions to find the tilt of the video camera
    p1 = static_right_ankle_pos
    p2 = motion_left_ankle_pos
    
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    
    # Calculate angle and convert to degrees
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    
    print(f"  -> Calculated tilt angle: {angle_deg:.2f} degrees.")
    return angle_deg

def compare_video_tilts(video_path1: str, video_path2: str):
    """
    Compares the camera tilt angles between two videos and reports the difference.

    Args:
        video_path1 (str): Path to the first video.
        video_path2 (str): Path to the second video.
    """
    tilt1 = calculate_tilt_angle_for_video(video_path1)
    tilt2 = calculate_tilt_angle_for_video(video_path2)
    
    print("\n  --- Tilt Alignment Report ---")
    if tilt1 is not None and tilt2 is not None:
        difference = tilt1 - tilt2
        print(f"  Tilt Angle 1: {tilt1:.2f}°")
        print(f"  Tilt Angle 2: {tilt2:.2f}°")
        print(f"  Angle Difference (Video 1 - Video 2): {difference:.2f}°")
        
        if abs(difference) < 1.0:
            print("  -> Conclusion: The camera tilts are well-aligned.")
        else:
            print(f"  -> Conclusion: The camera in Video 1 is tilted {'more counter-clockwise' if difference > 0 else 'more clockwise'} by {abs(difference):.2f}° relative to Video 2.")
    else:
        print("  Could not calculate the tilt difference due to errors in analyzing one or both videos.")

def create_tilted_video(input_video_path: str, output_video_path: str, angle_degrees: float):
    """
    Creates a new video by rotating each frame of the input video by the given angle.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input video file {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot open video writer for {output_video_path}")

    center = (frame_width / 2, frame_height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
        out.write(rotated_frame)

    cap.release()
    out.release()
    print(f"Rotated video saved to: {output_video_path} with {angle_degrees}° rotation.")

def tilt_align_videos(video1_path: str, video2_path: str, output_video1_path: str, output_video2_path: str):
    """
    Performs tilt alignment between two videos. The second video is rotated
    to match the tilt angle of the first video. The first video is copied
    without rotation.

    Args:
        video1_path (str): Path to the first input video.
        video2_path (str): Path to the second input video.
        output_video1_path (str): Path to save the processed first video (copied).
        output_video2_path (str): Path to save the tilt-aligned second video (rotated).
    """
    print(f"\nPerforming tilt alignment for {video1_path} and {video2_path}...")

    tilt1 = calculate_tilt_angle_for_video(video1_path)
    tilt2 = calculate_tilt_angle_for_video(video2_path)

    if tilt1 is not None and tilt2 is not None:
        angle_to_rotate_v2 = tilt2 - tilt1
        print(f"  Video 1 Tilt: {tilt1:.2f}°")
        print(f"  Video 2 Tilt: {tilt2:.2f}°")
        print(f"  Rotating Video 2 by {angle_to_rotate_v2:.2f}° to match Video 1's tilt.")

        # Rotate video 2
        create_tilted_video(video2_path, output_video2_path, angle_to_rotate_v2)
        
        # Copy video 1 as is
        print(f"  Copying Video 1 to {output_video1_path} without rotation.")
        shutil.copy(video1_path, output_video1_path)
        print("  -> Tilt alignment process complete.")
    else:
        print("  Could not perform tilt alignment: failed to determine tilt angle for one or both videos.")

if __name__ == '__main__':
    # --- Example Usage ---
    # Replace with the actual paths to your video files.
    # These videos should be trimmed to the relevant motion (e.g., using util_auto_pitch_cut.py).
    video1_path = "Input_Video/cutsIMG_1889.mp4"
    video2_path = "Test_Video/cutsIMG_1889_n6.mp4"
    
    if not (os.path.exists(video1_path) and os.path.exists(video2_path)):
        print("Error: Please update the video paths in the __main__ block of func_tilt_alignment.py.")
    else:
        compare_video_tilts(video1_path, video2_path)
        # To perform tilt alignment and save the output videos, uncomment the following lines:
        output_video1_path = "Output_Overlay/tilt_aligned_video1.mp4"
        output_video2_path = "Output_Overlay/tilt_aligned_video2.mp4"
        tilt_align_videos(video1_path, video2_path, output_video1_path, output_video2_path)   
