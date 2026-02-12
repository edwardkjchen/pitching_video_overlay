import cv2
import numpy as np
import os
import math
import glob
from func_tilt_alignment import calculate_tilt_angle_for_video
import pandas as pd

# Define a reasonable tolerance for angle comparison
ANGLE_TOLERANCE_DEGREES = 5.0

def get_all_video_paths():
    """
    Finds all .mp4 and .mov video files in the Input_Video directory.
    """
    video_dir = "Input_Video"
    if not os.path.isdir(video_dir):
        print(f"Error: Input directory not found: {video_dir}")
        return []
    
    paths = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.mov"))
    if not paths:
        print(f"Error: No .mp4 or .mov videos found in {video_dir}")
    return paths

def create_tilted_video(input_video_path: str, output_video_path: str, angle_degrees: float):
    """
    Creates a new video by rotating each frame of the input video by the given angle.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input video file {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot open video writer for {output_video_path}")

    center = (frame_width / 2, frame_height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
        out.write(rotated_frame)

    cap.release()
    out.release()
    print(f"Created tilted video at: {output_video_path} with {angle_degrees}° tilt.")

def run_tilt_test_for_video(input_video_path: str, results_list: list):
    """
    Runs the tilt alignment accuracy test for a single video and records the results.
    """
    print(f"\n\n--- Processing Video: {os.path.basename(input_video_path)} ---")
    angles_to_test = range(-6, 10, 4)  # -6, -2, +2, 6

    # First, analyze the original input video to determine its "base" tilt.
    base_tilt = calculate_tilt_angle_for_video(input_video_path)
    if base_tilt is None:
        print(f"Could not determine base tilt for {input_video_path}. Skipping this video.")
        return
    print(f"Base tilt of original video: {base_tilt:.2f}°")

    for ground_truth_angle in angles_to_test:
        print(f"\n--- Testing with ground truth angle: {ground_truth_angle}° for {os.path.basename(input_video_path)} ---")
        
        angle_str = f"n{-ground_truth_angle}" if ground_truth_angle < 0 else f"p{ground_truth_angle}"
        base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = "Test_Video"
        tilted_video_path = os.path.join(output_dir, f"{base_filename}_{angle_str}.mp4")
        
        create_tilted_video(input_video_path, tilted_video_path, ground_truth_angle)
        
        calculated_tilt = calculate_tilt_angle_for_video(tilted_video_path)
        
        result_data = {
            "video_filename": os.path.basename(input_video_path),
            "ground_truth_angle": ground_truth_angle,
            "base_tilt": base_tilt,
        }

        if calculated_tilt is None:
            print(f"Failed to calculate tilt for video {tilted_video_path}.")
            result_data.update({"expected_tilt": "N/A", "calculated_tilt": "Failed"})
        else:
            expected_tilt = base_tilt - ground_truth_angle
            print(f"  Expected Tilt: {expected_tilt:.2f}°")
            print(f"  Calculated Tilt: {calculated_tilt:.2f}°")
            result_data.update({"expected_tilt": expected_tilt, "calculated_tilt": calculated_tilt})

            if not math.isclose(calculated_tilt, expected_tilt, abs_tol=ANGLE_TOLERANCE_DEGREES):
                print(f"  [WARNING] Tilt mismatch for {os.path.basename(input_video_path)} with {ground_truth_angle}° rotation. "
                      f"Expected: {expected_tilt:.2f}°, Got: {calculated_tilt:.2f}°")
        
        results_list.append(result_data)

def main():
    """
    Main function to run the tilt alignment test suite.
    """
    all_results = []
    video_paths = get_all_video_paths()

    for video_path in video_paths:
        run_tilt_test_for_video(video_path, all_results)

    print("\n--- Saving all tilt test results to CSV ---")
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = "tilt_test_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()
