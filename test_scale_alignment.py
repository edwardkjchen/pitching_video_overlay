"""
This script tests the accuracy of the `func_scale_alignment.py` module.

It works by:
1. Taking a source video.
2. Programmatically scaling this video by a range of known factors (e.g., 0.5x, 1.2x).
3. Running the scale alignment function on the original vs. each scaled video.
4. Comparing the estimated scale ratio returned by the function to the ground truth.
5. Displaying the results in a summary table.
"""
import cv2
import os
import shutil
import pandas as pd
import glob
from func_scale_alignment import calculate_scale_ratios, analyze_pitching_motion, find_motion_start_frame

def test_motion_start_detection():
    """
    Scans all videos in the Input_Video directory and prints the detected
    motion_start_frame for each one.
    """
    print("\n\n--- Testing Motion Start Detection ---")
    video_dir = "Input_Video/"
    # Use glob to find all common video types
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                  glob.glob(os.path.join(video_dir, "*.mov")) + \
                  glob.glob(os.path.join(video_dir, "*.avi"))

    if not video_paths:
        print(f"No video files (.mp4, .mov, .avi) found in '{video_dir}'")
        return

    for video_path in sorted(video_paths):
        print(f"\n----- Analyzing '{os.path.basename(video_path)}' -----")
        try:
            # We only need the speeds and fps for this test
            _, all_landmark_speeds, fps = analyze_pitching_motion(video_path)

            if not all_landmark_speeds or not any(s is not None for speeds in all_landmark_speeds.values() for s in speeds):
                print("  -> Could not analyze motion (no speeds detected).")
                continue
            
            motion_start_frame = find_motion_start_frame(all_landmark_speeds, fps)
            # The find_motion_start_frame function already prints debug info.
            # We'll just print a summary line here.
            print(f"  --> Final Detected Motion Start Frame: {motion_start_frame}")

        except Exception as e:
            print(f"  -> An error occurred during analysis: {e}")

def scale_video(input_path: str, output_path: str, scale_factor: float):
    """
    Reads a video, scales each frame, and writes to a new file.

    Args:
        input_path (str): Path to the source video.
        output_path (str): Path to save the scaled video.
        scale_factor (float): The factor by which to scale the video dimensions.
    """
    print(f"Scaling video '{os.path.basename(input_path)}' by {scale_factor}x -> '{os.path.basename(output_path)}'...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions
    new_w = int(original_w * scale_factor)
    new_h = int(original_h * scale_factor)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()
    print("  -> Scaling complete.")

def main():
    """Main function to run the scale alignment test suite."""
    # --- Configuration ---
    scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    output_dir = "temp_scaled_videos"
    video_dir = "Input_Video/"
    all_results = []

    # Get all video files
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                  glob.glob(os.path.join(video_dir, "*.mov")) + \
                  glob.glob(os.path.join(video_dir, "*.avi"))

    if not video_paths:
        print(f"No video files found in '{video_dir}'")
        return

    # --- Test Execution for each video ---
    for original_video_path in sorted(video_paths):
        video_name = os.path.splitext(os.path.basename(original_video_path))[0]
        print(f"\n\n=== Testing '{video_name}' ===")
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for factor in scale_factors:
            ground_truth_ratio = 1.0 / factor
            
            scaled_video_filename = f"{video_name}_scaled_{factor:.1f}.mp4"
            scaled_video_path = os.path.join(output_dir, scaled_video_filename)

            try:
                # 1. Create the scaled video
                scale_video(original_video_path, scaled_video_path, factor)
                
                # 2. Run the alignment function
                print(f"Running alignment for scale factor: {factor}")
                estimated_ratios = calculate_scale_ratios(original_video_path, scaled_video_path)
                
                # 3. Store the results
                result_row = {"Video": video_name, "Scale Factor": factor, "Ground Truth Ratio": ground_truth_ratio}
                for seg_name, ratio in estimated_ratios.items():
                    col_name = f"Estimated: {seg_name.replace('_', ' ').title()}"
                    result_row[col_name] = ratio
                all_results.append(result_row)
            except Exception as e:
                print(f"  Error processing {video_name} at scale {factor}: {e}")

    # --- Display and Save Results ---
    if not all_results:
        print("No results were generated.")
        return
        
    df = pd.DataFrame(all_results)
    pd.options.display.float_format = '{:,.4f}'.format
    
    print("\n\n--- Test Summary: Scale Alignment Accuracy ---")
    print(df.to_string())
    
    # Save to CSV
    csv_output = "scale_alignment_results.csv"
    df.to_csv(csv_output, index=False)
    print(f"\nResults saved to '{csv_output}'")

    # --- Cleanup ---
    print(f"\nCleaning up temporary directory: {output_dir}")
    shutil.rmtree(output_dir)
    print("Test complete.")

if __name__ == "__main__":
    # Per the user's request, run the new diagnostic function.
    # The full test suite can be run by uncommenting main()
    # test_motion_start_detection()
    main()
