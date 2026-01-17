import cv2
import numpy as np
import csv
from scipy.signal import savgol_filter

def stabilize_video(input_path, output_path, raw_motion_csv_path, smoothed_motion_csv_path):
    """
    Stabilizes a video by estimating and correcting camera motion.
    This version uses Gaussian blur, RANSAC, and a Savitzky-Golay filter
    for robust and accurate stabilization.
    """
    # === 1. Setup ===
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # === 2. Process Frames: Estimate Motion ===
    frame_motions = []

    # Reset capture to re-read frames for motion estimation
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame_for_motion = cap.read()
    if not ret: return
    prev_gray_for_motion = cv2.cvtColor(prev_frame_for_motion, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the first frame
    prev_gray_for_motion = cv2.GaussianBlur(prev_gray_for_motion, (5, 5), 0)


    for i in range(n_frames - 1):
        ret, curr_frame_for_motion = cap.read()
        if not ret:
            break
        curr_gray_for_motion = cv2.cvtColor(curr_frame_for_motion, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to the current frame
        curr_gray_for_motion = cv2.GaussianBlur(curr_gray_for_motion, (5, 5), 0)

        # Create a mask to focus feature detection on the center of the image
        mask = np.zeros_like(prev_gray_for_motion)
        h, w = mask.shape
        h_margin = int(h * 0.1)
        w_margin = int(w * 0.1)
        mask[h_margin:h - h_margin, w_margin:w - w_margin] = 255

        # Detect good features to track, using the mask
        prev_pts = cv2.goodFeaturesToTrack(prev_gray_for_motion, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3, mask=mask)

        dx, dy = 0, 0
        if prev_pts is not None:
            # Calculate optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray_for_motion, curr_gray_for_motion, prev_pts, None)

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find transformation using RANSAC
            if len(prev_pts) > 5:
                transform_matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

                if transform_matrix is not None:
                    dx = transform_matrix[0, 2]
                    dy = transform_matrix[1, 2]
                else:
                    dx, dy = 0, 0
        
        frame_motions.append([dx, dy])
        prev_gray_for_motion = curr_gray_for_motion

    # === 3. Calculate Smoothed Trajectory ===
    frame_motions_np = np.array(frame_motions)
    
    # Apply Savitzky-Golay filter to the raw frame-to-frame motions
    window_length = 31 # Must be odd
    polyorder = 3
    smoothed_motions_np = np.copy(frame_motions_np)
    smoothed_motions_np[:, 0] = savgol_filter(frame_motions_np[:, 0], window_length, polyorder)
    smoothed_motions_np[:, 1] = savgol_filter(frame_motions_np[:, 1], window_length, polyorder)

    # --- NEW: ACCUMULATE MOTION ---
    # Calculate the cumulative sum of motions to get the trajectory
    trajectory = np.cumsum(smoothed_motions_np, axis=0)


    # === 4. Apply Accumulated Transformations ===
    # Reset stream to first frame for actual stabilization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame_stabilized = cap.read()
    if not ret: return
    out.write(prev_frame_stabilized) # Write first frame as-is

    for i in range(len(trajectory)):
        ret, curr_frame_stabilized = cap.read()
        if not ret:
            break

        # Get the total accumulated motion for this frame
        dx_total = trajectory[i, 0]
        dy_total = trajectory[i, 1]

        # Build the inverse transformation matrix to counteract the accumulated motion
        m = np.float32([[1, 0, -dx_total], [0, 1, -dy_total]])

        # Apply the transformation to the current frame
        frame_stabilized = cv2.warpAffine(curr_frame_stabilized, m, (width, height))

        # Write the stabilized frame
        out.write(frame_stabilized)

    # === 5. Release Resources and Save Data ===
    cap.release()
    out.release()

    # Save raw motion data
    raw_estimated_motion_dict = [{'dx': row[0], 'dy': row[1]} for row in frame_motions_np]
    with open(raw_motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_estimated_motion_dict)
    print(f"Successfully created {raw_motion_csv_path}")

    # Save smoothed motion data
    smoothed_motion_dict = [{'dx': row[0], 'dy': row[1]} for row in smoothed_motions_np]
    with open(smoothed_motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(smoothed_motion_dict)
    print(f"Successfully created {smoothed_motion_csv_path}")


if __name__ == '__main__':
    stabilize_video(
        'test_video.mp4',
        'results_video.mp4',
        'calculated_motion_raw.csv',
        'calculated_motion_smoothed.csv'
    )
