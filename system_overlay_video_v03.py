"""
This script implements a video processing pipeline to synchronize and overlay two videos.
This version modularizes the stabilization and alignment steps.

The pipeline is as follows:
1.  **Stabilize Videos**: Each input video is stabilized by calling the
    `stabilize_video` function directly.
2.  **Temporal Alignment**: The script calls `func_temporal_alignment` to find the
    frame offset between the two original videos.
3.  **Trim Videos**: The video that starts earlier is trimmed by the calculated
    frame offset to ensure both videos start at the same moment.
4.  **Spatial Alignment**: It calls `func_spatial_alignment` to find the spatial
    displacement needed to align the subjects in the videos.
5.  **Render Overlay**: The stabilized, temporally aligned, and spatially aligned
    videos are overlaid to create the final composite video.
"""
import cv2
import os
import shutil
import sys

# --- Import functions from other scripts ---
from func_temporal_alignment import temporal_align_videos
from func_render_overlay import render_overlay
from func_stabilize_video import stabilize_video
from func_spatial_alignment import get_spatial_displacement

# --- Core Helper Functions ---

def trim_video(input_path, output_path, frames_to_skip):
    """
    Trims a specified number of frames from the beginning of a video.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the trimmed video.
        frames_to_skip (int): Number of frames to skip from the beginning.
    """
    print(f"Trimming {frames_to_skip} frames from {input_path} -> {output_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path} for trimming.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Skip the initial frames
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            print("Warning: Video ended before all frames could be skipped.")
            break

    # Write the remaining frames to the new video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"  -> Trimming complete.")


# --- Main Execution ---

def main():
    """Main function to run the full video processing pipeline."""
    # --- Configuration ---
    video1_path = "Input_Video/cutsIMG_2725.mp4"
    video2_path = "Input_Video/cutsIMG_2726.mp4"

    stabilized1_path = "video1_stabilized_v03.mp4"
    stabilized2_path = "video2_stabilized_v03.mp4"
    
    trimmed1_path = "video1_trimmed_v03.mp4"
    trimmed2_path = "video2_trimmed_v03.mp4"

    final_overlay_path = "system_overlay_v03.mp4"

    # --- Pipeline ---

    # Step 1: Stabilize both videos individually by calling the function directly
    print("Stabilizing videos...")
    stabilize_video(video1_path, stabilized1_path, 'motion_raw1_v03.csv', 'motion_smoothed1_v03.csv')
    stabilize_video(video2_path, stabilized2_path, 'motion_raw2_v03.csv', 'motion_smoothed2_v03.csv')
    print("Stabilization complete.")

    # Step 2: Temporal Alignment
    print("\nStarting temporal alignment...")
    # It's better to use original videos for alignment to avoid stabilization artifacts
    frame_shift = temporal_align_videos(video1_path, video2_path)
    frame_shift = int(round(frame_shift))
    print(f"Temporal alignment complete. Frame shift: {frame_shift}")

    # Step 3: Trim videos based on the calculated shift
    print("\nTrimming videos for synchronization...")
    if frame_shift < 0:
        # Video 2 starts 'frame_shift' frames after Video 1, so trim Video 1.
        print(f"Video 2 is delayed by {frame_shift} frames. Trimming Video 1.")
        trim_video(stabilized1_path, trimmed1_path, frame_shift)
        shutil.copy(stabilized2_path, trimmed2_path)
    elif frame_shift > 0:
        # Video 1 starts 'frame_shift' frames after Video 2, so trim Video 2.
        shift = abs(frame_shift)
        print(f"Video 1 is delayed by {shift} frames. Trimming Video 2.")
        trim_video(stabilized2_path, trimmed2_path, shift)
        shutil.copy(stabilized1_path, trimmed1_path)
    else:
        # Videos are already synchronized.
        print("Videos are already synchronized. No trimming needed.")
        shutil.copy(stabilized1_path, trimmed1_path)
        shutil.copy(stabilized2_path, trimmed2_path)
    print("Trimming complete.")

    # Step 4: Spatial Alignment
    print("\nStarting spatial alignment...")
    # Use original videos for foot position to avoid stabilization affecting coordinates
    displacement = get_spatial_displacement(video1_path, video2_path)
    print("Spatial alignment complete.")

    # Step 5: Render the final overlay using the trimmed and stabilized videos
    print("\nRendering final overlay...")
    render_overlay(trimmed1_path, trimmed2_path, displacement, final_overlay_path)
    
    # Optional: Clean up intermediate files
    # files_to_remove = [
    #     stabilized1_path, stabilized2_path,
    #     trimmed1_path, trimmed2_path,
    #     'motion_raw1_v03.csv', 'motion_smoothed1_v03.csv',
    #     'motion_raw2_v03.csv', 'motion_smoothed2_v03.csv'
    # ]
    # for f in files_to_remove:
    #     try:
    #         os.remove(f)
    #         print(f"Removed intermediate file: {f}")
    #     except OSError as e:
    #         print(f"Error removing file {f}: {e}")
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
