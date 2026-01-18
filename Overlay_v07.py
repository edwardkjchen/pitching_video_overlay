import cv2
import numpy as np
import mediapipe as mp

# --- Core Helper & Motion Estimation Logic ---

def get_back_foot_position(frame, pose_detector):
    """
    Returns the (x, y) of the right back foot (right heel) in the frame.
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

def get_motion_trajectory(video_path):
    print(f"Step 3: Getting motion trajectory for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    ret, prev_frame = cap.read()
    if not ret: return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    frame_to_frame_transforms = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_pts is None or len(prev_pts) < 5:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is None: 
            frame_to_frame_transforms.append([0,0,0]); continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
        good_new = next_pts[status==1]
        good_old = prev_pts[status==1]
        
        T, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        if T is None: 
            frame_to_frame_transforms.append([0,0,0]); continue

        dx, dy = T[0, 2], T[1, 2]
        da = np.arctan2(T[1, 0], T[0, 0])
        frame_to_frame_transforms.append([dx, dy, da])
        
        prev_gray = frame_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    trajectory = np.cumsum(np.array(frame_to_frame_transforms), axis=0)
    cap.release()
    print(f"  -> Trajectory calculated for {len(trajectory)} frames.")
    return trajectory

def smooth_trajectory(trajectory, smoothing_radius=30):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(len(trajectory)):
        start = max(0, i - smoothing_radius)
        end = min(len(trajectory) - 1, i + smoothing_radius)
        smoothed_trajectory[i] = np.mean(trajectory[start:end+1], axis=0)
    return smoothed_trajectory

def stabilize_video_and_save(input_path, output_path, trajectory, smoothed_trajectory):
    print(f"Step 4: Stabilizing {input_path} -> {output_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    difference = trajectory - smoothed_trajectory
    T = np.zeros((2, 3), np.float32)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx < len(difference):
            dx, dy, da = difference[frame_idx]
            T[0, 0], T[0, 1] = np.cos(da), -np.sin(da)
            T[1, 0], T[1, 1] = np.sin(da), np.cos(da)
            T[0, 2], T[1, 2] = -dx, -dy
            stabilized_frame = cv2.warpAffine(frame, T, (w, h))
            out.write(stabilized_frame)
        else:
            out.write(frame)
        frame_idx += 1

    cap.release(); out.release()
    print(f"  -> Saved stabilized video to '{output_path}'.")

# --- NEW Pipeline Step 1 & 2: Foot-based Initial Displacement ---

def estimate_stable_foot_position(video_path, pose_detector):
    print(f"Step 1: Estimating stable foot position for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    positions = []
    for _ in range(10):
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

def estimate_initial_displacement_by_foot(stable_pos1, stable_pos2):
    print(f"Step 2: Estimating initial displacement based on foot coordinates...")
    if not stable_pos1 or not stable_pos2:
        print("  -> Missing foot coordinates. Returning zero displacement.")
        return (0, 0)
    
    dx = stable_pos1[0] - stable_pos2[0]
    dy = stable_pos1[1] - stable_pos2[1]
    print(f"  -> Initial displacement estimated as: dx={dx}, dy={dy}")
    return (dx, dy)

# --- Pipeline Step 5: Overlay Videos ---

def overlay_videos(stabilized_path1, stabilized_path2, initial_displacement, output_path, alpha=0.5):
    print(f"Step 5: Overlaying videos -> {output_path}...")
    cap1 = cv2.VideoCapture(stabilized_path1)
    cap2 = cv2.VideoCapture(stabilized_path2)
    if not cap1.isOpened() or not cap2.isOpened(): return
    fps, h, w = cap1.get(cv2.CAP_PROP_FPS), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    dx, dy = initial_displacement
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2: break
        frame1, frame2 = cv2.resize(frame1, (w, h)), cv2.resize(frame2, (w, h))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_frame2 = cv2.warpAffine(frame2, M, (w, h))
        blended = cv2.addWeighted(frame1, alpha, aligned_frame2, 1 - alpha, 0)
        out.write(blended)
    cap1.release(); cap2.release(); out.release()
    print(f"  -> Final overlay saved to '{output_path}'.")

# --- Main Execution ---

def main():
    video1_path = "cutsIMG_1839.mp4"
    video2_path = "cutsIMG_1840.mp4"
    stabilized1_path = "video1_stabilized_v7.mp4"
    stabilized2_path = "video2_stabilized_v7.mp4"
    final_overlay_path = "final_overlay_v7.mp4"

    # Step 1: Estimate stable foot position for each video
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    stable_pos1 = estimate_stable_foot_position(video1_path, pose_detector)
    stable_pos2 = estimate_stable_foot_position(video2_path, pose_detector)
    pose_detector.close()

    # Step 2: Estimate initial displacement from foot positions
    initial_displacement = estimate_initial_displacement_by_foot(stable_pos1, stable_pos2)
    
    # Step 3: Get shaky motion trajectory for each video
    shaky_trajectory1 = get_motion_trajectory(video1_path)
    shaky_trajectory2 = get_motion_trajectory(video2_path)

    # Step 4: Smooth trajectory and stabilize each video
    if shaky_trajectory1 is not None:
        smooth_trajectory1 = smooth_trajectory(shaky_trajectory1)
        stabilize_video_and_save(video1_path, stabilized1_path, shaky_trajectory1, smooth_trajectory1)
    else:
        print(f"Could not generate trajectory for {video1_path}. Stabilization skipped.")

    if shaky_trajectory2 is not None:
        smooth_trajectory2 = smooth_trajectory(shaky_trajectory2)
        stabilize_video_and_save(video2_path, stabilized2_path, shaky_trajectory2, smooth_trajectory2)
    else:
        print(f"Could not generate trajectory for {video2_path}. Stabilization skipped.")

    # Step 5: Overlay the two stabilized videos
    overlay_videos(stabilized1_path, stabilized2_path, initial_displacement, final_overlay_path, alpha=0.5)
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
