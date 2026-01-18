import cv2
import numpy as np
import csv

def generate_motion_data(frames, width, height):
    """Generates a series of transformations to simulate camera motion."""
    motion_data = []
    # Simple sinusoidal motion for translation
    x_translation = 20 * np.sin(np.linspace(0, 2 * np.pi, frames))
    y_translation = 10 * np.cos(np.linspace(0, 2 * np.pi, frames))

    for i in range(frames):
        dx = x_translation[i]
        dy = y_translation[i]
        motion_data.append({'dx': dx, 'dy': dy})
    return motion_data

def add_camera_motion(input_video_path, output_video_path, motion_csv_path):
    """
    Reads a video, adds simulated camera motion, and saves the output.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Generate motion data
    motion_data = generate_motion_data(frame_count, width, height)

    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the transformation for the current frame
        motion = motion_data[i]
        dx = motion['dx']
        dy = motion['dy']

        # Create transformation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Apply translation
        transformed_frame = cv2.warpAffine(frame, M, (width, height))

        # Write the transformed frame
        out.write(transformed_frame)

    # Release resources
    cap.release()
    out.release()

    # Save motion data to CSV
    with open(motion_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dx', 'dy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(motion_data)

    print(f"Successfully created {output_video_path} and {motion_csv_path}")

if __name__ == '__main__':
    add_camera_motion(
        'stable_video.mp4',
        'test_video.mp4',
        'motion.csv'
    )
