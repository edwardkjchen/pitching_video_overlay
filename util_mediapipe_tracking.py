import cv2
import mediapipe as mp
import numpy as np
import os

def track_and_overlay_video(input_file, output_file):
    """
    Tracks body joints in an input video using MediaPipe Pose and overlays the landmarks
    and connections onto a new output video.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to save the output video with overlaid landmarks.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,  # Use model_complexity=1 for faster processing, 2 for higher accuracy
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can use 'XVID' or 'MJPG' if 'mp4v' causes issues
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_file}")
        cap.release()
        return

    print(f"Processing {input_file}...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find pose landmarks
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Write the frame with landmarks to the output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    pose.close()
    print(f"Finished processing. Output saved to {output_file}. Processed {frame_count} frames.")

if __name__ == '__main__':
    # Example usage:
    # This part will be executed only when the script is run directly
    # and not when imported as a module.
    
    # Create a dummy raw directory and a dummy video for testing if they don't exist
    input_dir = "raw"
    output_dir = "tracked"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dummy_input_video = os.path.join(input_dir, "dummy_video.mp4")
    dummy_output_video = os.path.join(output_dir, "dummy_video_tracked.mp4")

    if not os.path.exists(dummy_input_video):
        print(f"Creating a dummy video at {dummy_input_video} for testing...")
        # Create a simple black video for testing
        width, height = 640, 480
        fps = 30
        duration_seconds = 3
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(dummy_input_video, fourcc, fps, (width, height))
        
        if writer.isOpened():
            for _ in range(fps * duration_seconds):
                frame = np.zeros((height, width, 3), dtype=np.uint8) # Black frame
                writer.write(frame)
            writer.release()
            print(f"Dummy video created at {dummy_input_video}")
            track_and_overlay_video(dummy_input_video, dummy_output_video)
        else:
            print(f"Error: Could not create dummy video writer for {dummy_input_video}")
    else:
        track_and_overlay_video(dummy_input_video, dummy_output_video)
