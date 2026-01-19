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

def extract_pose_features(video_path: str, model_complexity: int = 2, denoise_window: int = 5):
    """
    Processes a video to extract pose landmarks, their speeds, and leg length measurements.

    Args:
        video_path (str): Path to the input video file.
        model_complexity (int): Complexity of the pose model (0, 1, or 2).
        denoise_window (int): Size of the sliding window for landmark smoothing.

    Returns:
        A tuple containing:
        - pd.DataFrame: DataFrame with joint speeds and knee-to-heel length for each frame.
        - float: The median of the knee-to-heel lengths, used for scaling.
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
                # Calculate speeds for selected joints
                for joint_name, landmark_enum in mp_pose.PoseLandmark.__members__.items():
                    if 'HIP' in joint_name or 'KNEE' in joint_name or 'WRIST' in joint_name or 'TOE' in joint_name or 'FOOT_INDEX' in joint_name:
                        idx = landmark_enum.value
                        horizontal_speed = (smoothed_landmarks[idx, 0] - prev_landmarks[idx, 0]) * frame_width
                        vertical_speed = (smoothed_landmarks[idx, 1] - prev_landmarks[idx, 1]) * frame_height
                        frame_features[f'{joint_name}_h_speed'] = horizontal_speed
                        frame_features[f'{joint_name}_v_speed'] = vertical_speed
                
                # Calculate knee-to-heel distance
                left_knee = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_heel = smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
                right_knee = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_heel = smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                
                left_dist = np.linalg.norm(left_knee - left_heel) * frame_width
                right_dist = np.linalg.norm(right_knee - right_heel) * frame_width
                avg_leg_length = (left_dist + right_dist) / 2
                frame_features['knee_to_heel_length'] = avg_leg_length
                
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
        return pd.DataFrame(), 0

    df = pd.DataFrame(all_features).fillna(0)
    median_length = df["knee_to_heel_length"].median()
    
    return df, median_length


def align_features_dtw(features1: pd.DataFrame, median_length1: float,
                       features2: pd.DataFrame, median_length2: float,
                       framerate1: float, framerate2: float):
    """
    Aligns two feature series using Dynamic Time Warping (DTW) and computes the time shift.

    Args:
        features1 (pd.DataFrame): Feature data for the first video.
        median_length1 (float): Median leg length for scaling reference.
        features2 (pd.DataFrame): Feature data for the second video.
        median_length2 (float): Median leg length for scaling.
        framerate1 (float): Framerate of the first video.
        framerate2 (float): Framerate of the second video.

    Returns:
        float: The estimated time shift in frames.
    """
    if features1.empty or features2.empty:
        return 0

    # Ensure columns match before processing
    common_columns = list(features1.columns.intersection(features2.columns))
    data1 = features1[common_columns].to_numpy()
    data2 = features2[common_columns].to_numpy()

    # Rescale data2 to match the scale of data1
    scale_factor = (median_length1 / median_length2) if median_length2 > 0 else 1
    framerate_ratio = framerate1 / framerate2
    
    data2_rescaled = data2 * scale_factor * framerate_ratio

    # Handle framerate differences by simple interpolation (doubling rows)
    if framerate1 == framerate2 * 2:
        new_shape = (data2_rescaled.shape[0] * 2 - 1, data2_rescaled.shape[1])
        data2_ready = np.empty(new_shape)
        data2_ready[::2] = data2_rescaled
        data2_ready[1::2] = (data2_rescaled[:-1] + data2_rescaled[1:]) / 2
        data1_ready = data1
    elif framerate2 == framerate1 * 2:
        new_shape = (data1.shape[0] * 2 - 1, data1.shape[1])
        data1_ready = np.empty(new_shape)
        data1_ready[::2] = data1
        data1_ready[1::2] = (data1[:-1] + data1[1:]) / 2
        data2_ready = data2_rescaled
    else: # Assuming same framerate
        data1_ready, data2_ready = data1, data2_rescaled
        
    distance, best_path = fastdtw(data1_ready, data2_ready, dist=euclidean)
    
    shifts_per_frame = [p2 - p1 for p1, p2 in best_path]
    
    # Choose a robust shift estimation method
    time_shift = _max_by_overlapping_histogram(shifts_per_frame, error_bound=1)
    
    return time_shift

def temporal_align_videos(video_path1: str, video_path2: str, **kwargs):
    """
    Temporally aligns two videos by finding the optimal time shift between them.

    This function processes both videos to extract pose motion features, then uses
    Dynamic Time Warping (DTW) to find the alignment path and calculate the shift.

    Args:
        video_path1 (str): Path to the first video.
        video_path2 (str): Path to the second video.
        **kwargs: Optional arguments for `extract_pose_features`.

    Returns:
        float: The time shift in frames (video2 is shifted by this amount relative to video1).
    """
    cap1 = cv2.VideoCapture(video_path1)
    framerate1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.release()

    cap2 = cv2.VideoCapture(video_path2)
    framerate2 = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    
    print("Extracting features from the first video...")
    features1, median_length1 = extract_pose_features(video_path1, **kwargs)
    
    print("Extracting features from the second video...")
    features2, median_length2 = extract_pose_features(video_path2, **kwargs)
    
    print("Aligning features using DTW...")
    time_shift = align_features_dtw(features1, median_length1, features2, median_length2, framerate1, framerate2)
    
    return time_shift

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You would need to provide actual video file paths.
    # For instance:
    # video1 = 'Input_Video/seq7p_8_0.mp4'
    # video2 = 'Input_Video/seqmlb60r_bh_1_0.mp4'
    video1_path = "Input_Video/cutsIMG_2725.mp4"
    video2_path = "Input_Video/cutsIMG_2726.mp4"

    # Since I cannot assume files exist, this part is commented out.
    # try:
    #     shift = temporal_align_videos(video1, video2)
    #     print(f"\nEstimated time shift: {shift:.2f} frames.")
    #     print(f"This means video2 should be shifted by {shift / 30:.2f} seconds if at 30 FPS.")
    # except (IOError, FileNotFoundError) as e:
    #     print(f"Error: {e}. Please ensure video paths are correct.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    try:
        print(f"Attempting to align '{video1_path}' and '{video2_path}'...")
        shift = temporal_align_videos(video1_path, video2_path)
        print(f"\nEstimated time shift: {shift:.2f} frames.")

    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}. Please ensure video paths are correct and the videos exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")