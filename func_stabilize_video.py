import cv2
import numpy as np
import csv
from scipy.signal import savgol_filter
import argparse
import sys

def stabilize_video(input_path, output_path, raw_motion_csv_path, smoothed_motion_csv_path):
    """
    Stabilizes a video by estimating and correcting camera motion.
    This version uses Gaussian blur, RANSAC, and a Savitzky-Golay filter
    for robust and accurate stabilization.
    """
    # === 1. Setup ===
    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the stabilized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # === 2. Process Frames: Estimate Motion ===
    # This list will store the estimated motion (dx, dy) between each frame
    frame_motions = []

    # Reset capture to the first frame for motion estimation
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame_for_motion = cap.read()
    if not ret: return # Exit if the first frame cannot be read
    # Convert the first frame to grayscale for feature detection
    prev_gray_for_motion = cv2.cvtColor(prev_frame_for_motion, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and improve feature detection
    prev_gray_for_motion = cv2.GaussianBlur(prev_gray_for_motion, (5, 5), 0)


    # Loop through all frames to estimate inter-frame motion
    for i in range(n_frames - 1):
        ret, curr_frame_for_motion = cap.read()
        if not ret:
            break # Exit if a frame cannot be read
        # Convert the current frame to grayscale
        curr_gray_for_motion = cv2.cvtColor(curr_frame_for_motion, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        curr_gray_for_motion = cv2.GaussianBlur(curr_gray_for_motion, (5, 5), 0)

        # Create a mask to focus feature detection on the central region of the image,
        # avoiding noisy edges.
        mask = np.zeros_like(prev_gray_for_motion)
        h, w = mask.shape
        h_margin = int(h * 0.05) # 5% margin from top and bottom
        w_margin = int(w * 0.05) # 5% margin from left and right
        mask[h_margin:h - h_margin, w_margin:w - w_margin] = 255

        # Detect good features to track in the previous frame within the masked area
        prev_pts = cv2.goodFeaturesToTrack(prev_gray_for_motion, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3, mask=mask)
        # To-Do: consider grey out the pitcher in the middle of the frame if present

        dx, dy = 0, 0
        if prev_pts is not None:
            # Calculate the optical flow (motion) of the detected features
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray_for_motion, curr_gray_for_motion, prev_pts, None)

            # Filter out points that were not successfully tracked
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Estimate the affine transformation (translation, rotation, scale) using RANSAC.
            # RANSAC is robust to outliers (mismatched feature points).
            if len(prev_pts) > 5:
                transform_matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

                if transform_matrix is not None:
                    # Extract the translation components (dx, dy) from the transformation matrix
                    dx = transform_matrix[0, 2]
                    dy = transform_matrix[1, 2]
                else:
                    # If the transformation could not be estimated, assume no motion
                    dx, dy = 0, 0
        
        # Store the estimated motion for the current frame
        frame_motions.append([dx, dy])
        # Set the current frame's grayscale image as the previous one for the next iteration
        prev_gray_for_motion = curr_gray_for_motion

    # === 3. Calculate Smoothed Trajectory ===
    # Convert the list of motions to a NumPy array for numerical operations
    frame_motions_np = np.array(frame_motions)
    
    # Apply a Savitzky-Golay filter to smooth the raw motion data.
    # This filter fits a polynomial to a window of data points, which helps to
    # remove jitter while preserving the overall motion trend.
    window_length = 31 # The length of the filter window (must be an odd number)
    polyorder = 3      # The order of the polynomial used to fit the samples
    smoothed_motions_np = np.copy(frame_motions_np)
    smoothed_motions_np[:, 0] = savgol_filter(frame_motions_np[:, 0], window_length, polyorder)
    smoothed_motions_np[:, 1] = savgol_filter(frame_motions_np[:, 1], window_length, polyorder)

    # === 4. Calculate Accumulated Motion ===
    # Calculate the cumulative sum of the smoothed motions to get the overall trajectory
    # of the camera. This represents the total displacement from the starting position.
    trajectory = np.cumsum(smoothed_motions_np, axis=0)

    # === 5. Apply Accumulated Transformations ===
    # Reset the video stream to the first frame to apply the stabilization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame_stabilized = cap.read()
    if not ret: return # Exit if the first frame cannot be read
    # The first frame is the reference, so write it to the output without transformation
    out.write(prev_frame_stabilized) 

    # Loop through the trajectory data to apply the transformation to each frame
    for i in range(len(trajectory)):
        ret, curr_frame_stabilized = cap.read()
        if not ret:
            break # Exit if a frame cannot be read

        # Get the total accumulated motion up to this frame
        dx_total = trajectory[i, 0]
        dy_total = trajectory[i, 1]

        # Create an inverse transformation matrix to counteract the camera's accumulated motion
        m = np.float32([[1, 0, -dx_total], [0, 1, -dy_total]])

        # Apply the affine transformation (warping) to the current frame to stabilize it
        frame_stabilized = cv2.warpAffine(curr_frame_stabilized, m, (width, height))

        # Write the stabilized frame to the output video file
        out.write(frame_stabilized)

    # === 6. Release Resources and Save Data ===
    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Save the raw, unfiltered motion data to a CSV file for analysis
    raw_estimated_motion_dict = [{'dx': row[0], 'dy': row[1]} for row in frame_motions_np]
    with open(raw_motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_estimated_motion_dict)
    print(f"Successfully created {raw_motion_csv_path}")

    # Save the smoothed motion data to another CSV file for comparison and analysis
    smoothed_motion_dict = [{'dx': row[0], 'dy': row[1]} for row in smoothed_motions_np]
    with open(smoothed_motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(smoothed_motion_dict)
    print(f"Successfully created {smoothed_motion_csv_path}")


# This block executes when the script is run directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Stabilize a video file.')
        parser.add_argument('input_path', type=str, help='Path to the input video file.')
        parser.add_argument('output_path', type=str, help='Path to save the stabilized video.')
        args = parser.parse_args()

        stabilize_video(args.input_path, args.output_path, 'calculated_motion_raw.csv', 'calculated_motion_smoothed.csv')
    else:
        print("No command line arguments provided. Running a test stabilization...")
        stabilize_video("Input_Video\cutsIMG_2725.mp4", "stabilized_test_video_2725.mp4", 'calculated_motion_raw_2725.csv', 'calculated_motion_smoothed_2725.csv')
        stabilize_video("Input_Video\cutsIMG_2726.mp4", "stabilized_test_video_2726.mp4", 'calculated_motion_raw_2726.csv', 'calculated_motion_smoothed_2726.csv')