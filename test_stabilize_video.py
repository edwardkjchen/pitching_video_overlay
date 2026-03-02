import os
import glob
import sys
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from existing scripts
from func_stabilize_video import stabilize_video

def generate_motion_data(frames, width, height):
    """
    Generates a series of transformations to simulate camera motion.

    This function creates a simple, predictable motion pattern (sinusoidal)
    to test the video stabilization algorithm.

    Args:
        frames (int): The number of frames in the video.
        width (int): The width of the video frames.
        height (int): The height of the video frames.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'dx' and 'dy' translation for a corresponding frame.
    """
    motion_data = []
    # Generate smooth, looping motion using sine function
    x_translation = 20 * np.sin(np.linspace(0, 2 * np.pi, frames))
    y_translation = 20 * np.sin(np.linspace(0, 4 * np.pi, frames)) # Different frequency for y to create more complex motion

    for i in range(frames):
        dx = x_translation[i]
        dy = y_translation[i]
        motion_data.append({'dx': dx, 'dy': dy})
    return motion_data

def add_camera_motion(input_video_path, output_video_path, motion_csv_path):
    """
    Reads a stable video, applies simulated camera motion to it, and saves
    the resulting shaky video. This is used to create test data for stabilization.

    Args:
        input_video_path (str): Path to the stable input video.
        output_video_path (str): Path to save the shaky output video.
        motion_csv_path (str): Path to save the CSV file containing the applied motion data.
    """
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties (width, height, frames per second, etc.)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Generate the synthetic motion data for the duration of the video
    motion_data = generate_motion_data(frame_count, width, height)

    # Loop through each frame of the input video
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the simulated motion (dx, dy) for the current frame
        motion = motion_data[i]
        dx = motion['dx']
        dy = motion['dy']

        # Create a 2x3 transformation matrix for affine transformation (translation)
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the translation to the frame using warpAffine
        transformed_frame = cv2.warpAffine(frame, M, (width, height))

        # Write the transformed (shaky) frame to the output video
        out.write(transformed_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Save the generated motion data to a CSV file for later analysis or comparison
    with open(motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(motion_data)

    print(f"Successfully created {output_video_path} and {motion_csv_path}")

def compare_motion_data(ground_truth_csv, estimated_raw_csv, estimated_smoothed_csv, output_image_path):
    """
    Reads, plots, and compares ground truth motion data with the raw and
    smoothed motion data estimated by the stabilization algorithm.
    """
    try:
        gt_df = pd.read_csv(ground_truth_csv)
        est_raw_df = pd.read_csv(estimated_raw_csv)
        est_smoothed_df = pd.read_csv(estimated_smoothed_csv)
    except Exception as e:
        print(f"      Error reading CSVs for plot: {e}")
        return

    # Compute Trajectories (Accumulative Motion)
    est_raw_traj = est_raw_df.cumsum()
    est_smoothed_traj = est_smoothed_df.cumsum()
    gt_traj = gt_df.copy()

    # Align lengths
    min_len = min(len(gt_traj), len(est_raw_traj), len(est_smoothed_traj))
    gt_traj = gt_traj.iloc[:min_len]
    est_raw_traj = est_raw_traj.iloc[:min_len]
    est_smoothed_traj = est_smoothed_traj.iloc[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Accumulative Motion (Trajectory) Comparison', fontsize=16)

    # Plot DX
    ax1.plot(gt_traj.index, gt_traj['dx'], label='Ground Truth', color='blue', linewidth=2)
    ax1.plot(est_raw_traj.index, est_raw_traj['dx'], label='Estimated (Raw)', color='red', linestyle='--', alpha=0.6)
    ax1.plot(est_smoothed_traj.index, est_smoothed_traj['dx'], label='Estimated (Smoothed)', color='purple', linewidth=2.5)
    ax1.set_ylabel('Horizontal Offset (dx)')
    ax1.legend(); ax1.grid(True, linestyle='--')

    # Plot DY
    ax2.plot(gt_traj.index, gt_traj['dy'], label='Ground Truth', color='green', linewidth=2)
    ax2.plot(est_raw_traj.index, est_raw_traj['dy'], label='Estimated (Raw)', color='orange', linestyle='--', alpha=0.6)
    ax2.plot(est_smoothed_traj.index, est_smoothed_traj['dy'], label='Estimated (Smoothed)', color='cyan', linewidth=2.5)
    ax2.set_xlabel('Frame Number'); ax2.set_ylabel('Vertical Offset (dy)')
    ax2.legend(); ax2.grid(True, linestyle='--')

    plt.savefig(output_image_path)
    plt.close()

    mae_raw = (np.abs(gt_traj['dx'] - est_raw_traj['dx']).mean() + np.abs(gt_traj['dy'] - est_raw_traj['dy']).mean()) 
    mae_smoothed = (np.abs(gt_traj['dx'] - est_smoothed_traj['dx']).mean() + np.abs(gt_traj['dy'] - est_smoothed_traj['dy']).mean()) 

    return {'avg_abs_diff_raw': mae_raw, 'avg_abs_diff_smoothed': mae_smoothed}

def main():
    input_dir = "Input_Stable_Video"
    output_dir = "Test_Results"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Find all videos in the Input_Stable_Video folder
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"No video files found in {input_dir}. Please run pre_stabilization.py first.")
        return

    results_data = []
    print(f"Found {len(video_files)} stable videos for testing.\n")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        print(f"Processing Test: {filename}...")

        # Setup paths
        shaky_video = os.path.join(output_dir, f"{base_name}_shaky.mp4")
        gt_motion_csv = os.path.join(output_dir, f"{base_name}_gt_motion.csv")
        stabilized_video = os.path.join(output_dir, f"{base_name}_stabilized.mp4")
        raw_motion_csv = os.path.join(output_dir, f"{base_name}_est_raw.csv")
        smoothed_motion_csv = os.path.join(output_dir, f"{base_name}_est_smoothed.csv")
        plot_path = os.path.join(output_dir, f"{base_name}_comparison.png")

        # Run Test Steps
        add_camera_motion(video_path, shaky_video, gt_motion_csv)
        stabilize_video(shaky_video, stabilized_video, raw_motion_csv, smoothed_motion_csv, debug=False)
        metrics = compare_motion_data(gt_motion_csv, raw_motion_csv, smoothed_motion_csv, plot_path)

        # Metrics
        if metrics:
            results_data.append({
                'video_name': filename,
                'avg_abs_diff_raw': metrics['avg_abs_diff_raw'],
                'avg_abs_diff_smoothed': metrics['avg_abs_diff_smoothed']
            })
            print(f"  Result: Raw Error={metrics['avg_abs_diff_raw']:.4f}, Smoothed Error={metrics['avg_abs_diff_smoothed']:.4f}")

    # Summary Report
    df_results = pd.DataFrame(results_data)
    print("\n--- Final Stabilization Performance Report ---")
    print(df_results.to_string(index=False))
    
    report_csv = "stabilization_performance_summary.csv"
    df_results.to_csv(report_csv, index=False)
    print(f"\nSummary saved to {report_csv}")

if __name__ == "__main__":
    main()
