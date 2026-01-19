import cv2
import numpy as np
import csv

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
    # Generate smooth, looping motion using sine and cosine functions
    x_translation = 20 * np.sin(np.linspace(0, 2 * np.pi, frames))
    y_translation = 10 * np.cos(np.linspace(0, 2 * np.pi, frames))

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

# This block executes when the script is run directly
if __name__ == '__main__':
    # Creates a shaky video ('test_video.mp4') from a stable one ('stable_video.mp4')
    # and saves the motion data used to 'motion.csv'.
    add_camera_motion(
        'stable_video.mp4',
        'test_video.mp4',
        'motion.csv'
    )
