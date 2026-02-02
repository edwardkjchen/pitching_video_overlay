"""
This script uses MediaPipe to perform pose estimation on a video, draws markers
on a specific set of joints, and saves the result to a new video file.

It is intended for debugging and visualizing the accuracy of the motion tracking.

Usage:
    python debug_motion_tracking.py <input_video_path> <output_video_path>
"""
import cv2
import mediapipe as mp
import argparse
import sys
import os

def track_and_draw_landmarks(input_path, output_path):
    """
    Reads a video, tracks specified body joints using MediaPipe Pose,
    draws markers on them, and saves the output video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file with drawn landmarks.
    """
    # --- 1. Setup ---
    print(f"Processing video: {input_path}")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Define the landmarks to track as per user request
    tracked_landmarks_enums = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    # Video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- 2. Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find landmarks
        results = pose.process(rgb_frame)

        # Draw the landmarks on the frame
        if results.pose_landmarks:
            for landmark_enum in tracked_landmarks_enums:
                landmark = results.pose_landmarks.landmark[landmark_enum]
                
                # Check if the landmark is visible
                if landmark.visibility > 0.5:
                    # Get coordinates
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    
                    # Draw a circle on the landmark
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) # Green dot
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1) # White outline

        # Write the frame with landmarks to the output file
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    # --- 3. Cleanup ---
    cap.release()
    out.release()
    pose.close()
    
    print(f"Processing complete. Output saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Track specific body joints in a video using MediaPipe and save the result."
    )
    parser.add_argument(
        'input_path', 
        type=str, 
        help='Path to the input video file.'
    )
    parser.add_argument(
        'output_path', 
        type=str, 
        help='Path to save the output video file.'
    )

    if len(sys.argv) < 3:
        parser.print_help()
        # Provide a test case if no arguments are given
        print("\nRunning a test case with example files...")
        # Assume an input video exists for testing purposes
        test_input = "Output_Overlay\\cutsIMG_2723_trimmed_v04.mp4"
        test_output = "debug_motion_tracking_2723.mp4"
        if os.path.exists(test_input):
            track_and_draw_landmarks(test_input, test_output)
        else:
            print(f"Test input video '{test_input}' not found. Please provide arguments.")
    else:
        args = parser.parse_args()
        track_and_draw_landmarks(args.input_path, args.output_path)
