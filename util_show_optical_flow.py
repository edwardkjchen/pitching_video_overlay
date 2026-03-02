import cv2
import numpy as np
import os
import mediapipe as mp

def show_optical_flow_between_frames(video_path, frame_idx1, frame_idx2, output_image_path="optical_flow_12_to_48_no_human.png"):
    """
    Reads a video, extracts two specific frames, and calculates/visualizes 
    the optical flow (motion vectors) from frame_idx1 to frame_idx2.
    It uses MediaPipe to exclude feature points that are on or near the human body.
    """
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Step 1: Extract frames ---
    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {idx}")
            return None
        return frame

    frame1 = get_frame(frame_idx1)
    frame2 = get_frame(frame_idx2)
    cap.release()

    if frame1 is None or frame2 is None:
        return

    # --- Step 2: Prepare for feature detection ---
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

    # Define a central mask (5% margin)
    mask = np.zeros_like(gray1)
    h, w = mask.shape
    h_margin, w_margin = int(h * 0.05), int(w * 0.05)
    mask[h_margin:h - h_margin, w_margin:w - w_margin] = 255

    # Detect initial features in frame 1
    prev_pts = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=3, mask=mask)

    if prev_pts is None:
        print("Error: No features found in frame 1")
        return

    # --- Step 3: MediaPipe Body Tracking & Filtering ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
    
    # Convert frame1 to RGB for MediaPipe
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame1)

    filtered_prev_pts = []
    
    if results.pose_landmarks:
        # Calculate bounding box of the human body with a buffer
        landmarks = results.pose_landmarks.landmark
        x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]

        if x_coords and y_coords:
            # Create a bounding box with 10% padding
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            pad_w = (max_x - min_x) * 0.15
            pad_h = (max_y - min_y) * 0.15
            
            body_rect = (min_x - pad_w, min_y - pad_h, max_x + pad_w, max_y + pad_h)
            
            # Filter points: only keep those OUTSIDE the body rectangle
            for pt in prev_pts:
                px, py = pt.ravel()
                if not (body_rect[0] <= px <= body_rect[2] and body_rect[1] <= py <= body_rect[3]):
                    filtered_prev_pts.append(pt)
            
            print(f"Human detected. Filtered {len(prev_pts) - len(filtered_prev_pts)} points near the body.")
            
            # Draw exclusion zone for visualization
            cv2.rectangle(frame1, (int(body_rect[0]), int(body_rect[1])), 
                          (int(body_rect[2]), int(body_rect[3])), (255, 0, 0), 2)
            cv2.putText(frame1, "Exclusion Zone", (int(body_rect[0]), int(body_rect[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            filtered_prev_pts = list(prev_pts)
    else:
        print("No human body detected. Keeping all features.")
        filtered_prev_pts = list(prev_pts)

    pose.close()
    
    if not filtered_prev_pts:
        print("Error: No points left after filtering")
        return

    filtered_prev_pts = np.array(filtered_prev_pts, dtype=np.float32)

    # --- Step 4: Calculate Optical Flow (to frame 2) ---
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, filtered_prev_pts, None)

    # Filter successful points
    idx = np.where(status == 1)[0]
    good_new = curr_pts[idx]
    good_old = filtered_prev_pts[idx]

    # --- Step 5: Filter Outliers by Vector Length ---
    # Calculate lengths of all motion vectors
    vectors = good_new - good_old
    lengths = np.sqrt(np.sum(vectors**2, axis=2)).flatten()
    
    if len(lengths) > 0:
        median_length = np.median(lengths)
        # Eliminate vectors whose length is more than 2x the median length
        valid_idx = np.where(lengths <= 2 * median_length)[0]
        
        print(f"Median vector length: {median_length:.2f} pixels.")
        print(f"Filtered {len(good_new) - len(valid_idx)} outliers based on length (> {2*median_length:.2f} px).")
        
        good_new = good_new[valid_idx]
        good_old = good_old[valid_idx]
    else:
        print("No valid motion vectors found to filter.")

    # --- Step 6: Final Visualization ---
    display_frame = frame1.copy()
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(display_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(display_frame, (int(a), int(b)), 4, (0, 0, 255), -1)

    cv2.putText(display_frame, f"Filtered & Outlier-Free Flow: Frame {frame_idx1} -> {frame_idx2}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save and show
    cv2.imwrite(output_image_path, display_frame)
    print(f"Motion visualization saved to {output_image_path}")
    
    cv2.imshow('Filtered Optical Flow', display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = os.path.join("Test_Results", "cutsIMG_2727_shaky.mp4")
    show_optical_flow_between_frames(video_file, 12, 48)
