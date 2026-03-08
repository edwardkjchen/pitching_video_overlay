import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import bisect

from func_render_overlay import render_overlay

def _median_mean(data):
    """Calculates the mean of the central 50% of the sorted data."""
    sorted_data = sorted(data)
    mid_index = len(sorted_data) // 4
    mid_data = sorted_data[mid_index:-mid_index]
    return np.mean(mid_data) if mid_data else 0

def _max_by_overlapping_histogram(data, error_bound):
    """Finds the value with the highest density in a 1D dataset."""
    if not data:
        return 0
    data_sorted = sorted(data)
    min_val, max_val = data_sorted[0], data_sorted[-1]
    centers = range(int(min_val), int(max_val) + 1)
    counts = [bisect.bisect_right(data_sorted, center + error_bound) - bisect.bisect_left(data_sorted, center - error_bound) for center in centers]
    max_count = max(counts)
    max_centers = [center for center, count in zip(centers, counts) if count == max_count]
    return np.mean(max_centers) if max_centers else 0

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import bisect
import matplotlib.pyplot as plt
import os

def _median_mean(data):
    """Calculates the mean of the central 50% of the sorted data."""
    sorted_data = sorted(data)
    mid_index = len(sorted_data) // 4
    mid_data = sorted_data[mid_index:-mid_index]
    return np.mean(mid_data) if mid_data else 0

def _max_by_overlapping_histogram(data, error_bound):
    """Finds the value with the highest density in a 1D dataset."""
    if not data:
        return 0
    data_sorted = sorted(data)
    min_val, max_val = data_sorted[0], data_sorted[-1]
    centers = range(int(min_val), int(max_val) + 1)
    counts = [bisect.bisect_right(data_sorted, center + error_bound) - bisect.bisect_left(data_sorted, center - error_bound) for center in centers]
    max_count = max(counts)
    max_centers = [center for center, count in zip(centers, counts) if count == max_count]
    return np.mean(max_centers) if max_centers else 0

def extract_pose_features(video_path: str, model_complexity: int = 2, denoise_window: int = 5):
    """
    Processes a video to extract pose landmarks and their speeds.
    Assumes videos are already scaled to the same size.

    Args:
        video_path (str): Path to the input video file.
        model_complexity (int): Complexity of the pose model (0, 1, or 2).
        denoise_window (int): Size of the sliding window for landmark smoothing.

    Returns:
        pd.DataFrame: DataFrame with joint speeds for each frame.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=model_complexity,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmark_window = deque(maxlen=denoise_window)
    prev_landmarks = None
    
    all_features = []

    # Subset of landmarks to use for alignment
    landmark_subset = {
        'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
        'RIGHT_HEEL': mp_pose.PoseLandmark.RIGHT_HEEL,
        'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
        'LEFT_HEEL': mp_pose.PoseLandmark.LEFT_HEEL
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            landmark_window.append(landmarks)
            smoothed_landmarks = np.median(landmark_window, axis=0)
            
            frame_features = {}
            if prev_landmarks is not None:
                # Calculate speeds for selected joints in the subset
                for joint_name, landmark_enum in landmark_subset.items():
                    idx = landmark_enum.value
                    horizontal_speed = (smoothed_landmarks[idx, 0] - prev_landmarks[idx, 0]) * frame_width
                    vertical_speed = (smoothed_landmarks[idx, 1] - prev_landmarks[idx, 1]) * frame_height
                    frame_features[f'{joint_name}_h_speed'] = horizontal_speed
                    frame_features[f'{joint_name}_v_speed'] = vertical_speed
                
                all_features.append(frame_features)

            prev_landmarks = smoothed_landmarks
        else:
            # Append empty data if no landmarks are found
            if all_features: # ensure we have feature names
                all_features.append({key: 0 for key in all_features[0].keys()})
            prev_landmarks = None
            
    cap.release()
    pose.close()

    if not all_features:
        return pd.DataFrame()

    df = pd.DataFrame(all_features).fillna(0)
    return df


def align_features_dtw(features1: pd.DataFrame, features2: pd.DataFrame,
                       framerate1: float, framerate2: float,
                       video_name1: str = "Video 1", video_name2: str = "Video 2"):
    """
    Aligns two feature series using Dynamic Time Warping (DTW) and computes the time shift.
    Also plots joint speeds and DTW shifts.

    Args:
        features1 (pd.DataFrame): Feature data for the first video.
        features2 (pd.DataFrame): Feature data for the second video.
        framerate1 (float): Framerate of the first video.
        framerate2 (float): Framerate of the second video.
        video_name1 (str): Label for the first video in plots.
        video_name2 (str): Label for the second video in plots.

    Returns:
        float: The estimated time shift in frames.
    """
    if features1.empty or features2.empty:
        return 0

    # Ensure columns match before processing
    common_columns = list(features1.columns.intersection(features2.columns))
    data1 = features1[common_columns].to_numpy()
    data2 = features2[common_columns].to_numpy()

    # Assuming videos are already scaled, we don't apply median_length scaling.
    # We still handle framerate differences if necessary.
    framerate_ratio = framerate1 / framerate2
    data2_ready = data2 * framerate_ratio
    data1_ready = data1

    # Simple interpolation for different framerates
    if framerate1 == framerate2 * 2:
        new_shape = (data2_ready.shape[0] * 2 - 1, data2_ready.shape[1])
        data2_interp = np.empty(new_shape)
        data2_interp[::2] = data2_ready
        data2_interp[1::2] = (data2_ready[:-1] + data2_ready[1:]) / 2
        data2_ready = data2_interp
    elif framerate2 == framerate1 * 2:
        new_shape = (data1.shape[0] * 2 - 1, data1.shape[1])
        data1_interp = np.empty(new_shape)
        data1_interp[::2] = data1
        data1_interp[1::2] = (data1[:-1] + data1[1:]) / 2
        data1_ready = data1_interp
        
    distance, best_path = fastdtw(data1_ready, data2_ready, dist=euclidean)
    
    shifts_per_frame = [p2 - p1 for p1, p2 in best_path]
    
    # --- Plotting ---
    output_dir = "Alignment_Plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Choose a robust shift estimation method
    time_shift = _max_by_overlapping_histogram(shifts_per_frame, error_bound=1)

    # Plotting Helper for Individual Joint Speeds
    def plot_video_speeds(features_df, video_name):
        plt.figure(figsize=(12, 6))
        # Filter to specific requested components for visual clarity
        desired_columns = ['RIGHT_WRIST_h_speed', 'LEFT_KNEE_v_speed', 'LEFT_HEEL_v_speed', 'LEFT_HEEL_h_speed']
        for col in desired_columns:
            if col in features_df.columns:
                plt.plot(features_df[col], label=col, alpha=0.8)
        plt.title(f'Target Joint Speeds: {video_name}')
        plt.xlabel('Frame Number')
        plt.ylabel('Speed (px/frame)')
        plt.ylim(-200, 200) # Fixed y-axis for consistent scale
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"joint_speeds_{video_name}.png".replace("/", "_").replace("\\", "_"))
        plt.savefig(plot_path)
        print(f"Joint speed plot saved to {plot_path}")
        plt.close()

    # Output individual plots per video
    plot_video_speeds(features1, video_name1)
    plot_video_speeds(features2, video_name2)

    # Plot 2: DTW Shifts per frame with the final time_shift as a reference line
    plt.figure(figsize=(12, 6))
    plt.plot(shifts_per_frame, label='Warping Path Shift (idx2 - idx1)', color='tab:blue', alpha=0.6)
    plt.axhline(y=time_shift, color='red', linestyle='--', linewidth=2, label=f'Calculated Time Shift: {time_shift:.2f}')
    
    plt.title(f'DTW Alignment Path: {video_name1} vs {video_name2}')
    plt.xlabel('Step in Alignment Path')
    plt.ylabel('Shift (Frame2_idx - Frame1_idx)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    shift_plot_path = os.path.join(output_dir, f"dtw_shifts_{video_name1}_{video_name2}.png".replace("/", "_").replace("\\", "_"))
    plt.savefig(shift_plot_path)
    print(f"DTW shift plot saved to {shift_plot_path}")
    plt.close()
    
    return time_shift

def temporal_align_videos(video_path1: str, video_path2: str, **kwargs):
    """
    Temporally aligns two videos by finding the optimal time shift between them.
    """
    cap1 = cv2.VideoCapture(video_path1)
    framerate1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.release()

    cap2 = cv2.VideoCapture(video_path2)
    framerate2 = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    
    name1 = os.path.basename(video_path1)
    name2 = os.path.basename(video_path2)

    print(f"Extracting features from {name1}...")
    features1 = extract_pose_features(video_path1, **kwargs)
    
    print(f"Extracting features from {name2}...")
    features2 = extract_pose_features(video_path2, **kwargs)
    
    print("Aligning features using DTW...")
    time_shift = align_features_dtw(features1, features2, framerate1, framerate2, name1, name2)
    
    return time_shift

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You would need to provide actual video file paths.
    # For instance:
    # video1 = 'Input_Video/seq7p_8_0.mp4'
    # video2 = 'Input_Video/seqmlb60r_bh_1_0.mp4'
    # video1_path = "Input_Video/cutsIMG_1922.mp4"
    # video2_path = "Input_Video/cutsIMG_1923.mp4"

    input_dir = "Input_Video"

    # Get all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_files = ['cutsIMG_1899.mp4', 'cutsIMG_1902.mp4']
    video_files.sort()
    
    # Process all pairs of videos
    for i in range(len(video_files) - 1):
        video1_name = video_files[i]
        video2_name = video_files[i + 1]
        
        video1_path = os.path.join(input_dir, video1_name)
        video2_path = os.path.join(input_dir, video2_name)

        try:
            print(f"Attempting to align '{video1_path}' and '{video2_path}'...")
            shift = temporal_align_videos(video1_path, video2_path)
            print(f"\nEstimated time shift: {shift:.2f} frames.")

        except (IOError, FileNotFoundError) as e:
            print(f"Error: {e}. Please ensure video paths are correct and the videos exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")