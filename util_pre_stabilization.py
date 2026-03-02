import os
import glob
from func_stabilize_video import stabilize_video

""" 
This file is responsible for pre-stabilizing all videos in the "Input_Video" directory and saving the stabilized versions to "Input_Stable_Video".
It uses the stabilize_video function from func_stabilize_video.py to perform the stabilization process.
We will use the stablized videos are the ground truth for testing, so we want to ensure they are as stable as possible.
"""

def pre_stabilize_all_videos(input_dir="Input_Video", output_dir="Input_Stable_Video"):
    """
    Reads all video files from the input directory, stabilizes them,
    and saves the results to the output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Define supported video extensions
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    # Find all video files in the input directory
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to stabilize.")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, filename)
        
        # Define paths for temporary motion CSVs (using filename to avoid collisions)
        raw_csv = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_raw.csv")
        smoothed_csv = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_smoothed.csv")

        print(f"Stabilizing: {filename}...")
        try:
            stabilize_video(
                video_path, 
                output_path, 
                raw_csv, 
                smoothed_csv, 
                debug=False
            )
            print(f"  Done -> {output_path}")
            
            # Optional: Clean up the temporary CSV files if they aren't needed
            if os.path.exists(raw_csv): os.remove(raw_csv)
            if os.path.exists(smoothed_csv): os.remove(smoothed_csv)
            
        except Exception as e:
            print(f"  Error stabilizing {filename}: {e}")

    print("All stabilization tasks complete.")

if __name__ == "__main__":
    pre_stabilize_all_videos()
