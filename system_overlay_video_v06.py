"""
This script implements a video processing pipeline to synchronize and overlay two videos.
This version (v05) adds a tilt alignment step to the pipeline.

The pipeline is as follows:
1.  **Stabilize Videos**: Both input videos are stabilized.
2.  **Scale Alignment**: The script calculates the scale difference between the
    subjects in the two videos and scales the larger video down to match the
    smaller one. This is done on the stabilized videos.
3.  **Tilt Alignment**: This new step aligns the tilt (rotation) of the subjects.
4.  **Spatial Alignment**: It calculates the spatial displacement required to
    align the subjects' initial positions using the original videos.
4.  **Temporal Alignment**: It finds the frame offset between the two original
    videos to synchronize their start times.
5.  **Trim Videos**: The video that starts earlier is trimmed (from the
    stabilized and scaled versions) to match the start time of the other.
6.  **Render Overlay**: The fully processed videos (stabilized, scaled, aligned,
    and trimmed) are overlaid to create the final composite video.
"""
import cv2
import os
import shutil
import sys
import numpy as np

# --- Import functions from other scripts ---
from func_temporal_alignment import temporal_align_videos
from func_render_overlay import render_overlay
from func_stabilize_video import stabilize_video
from func_spatial_alignment import get_spatial_displacement
from func_scale_alignment import calculate_scale_ratios
from func_tilt_alignment import tilt_align_videos

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


def scale_video(input_path, output_path, scale_factor):
    """
    Scales a video by a given factor.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the scaled video.
        scale_factor (float): The factor by which to scale the video.
    """
    print(f"Scaling {input_path} by {scale_factor} -> {output_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path} for scaling.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # We need to create a black canvas of the original size and paste the scaled video in the center
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        scaled_frame = cv2.resize(frame, (new_w, new_h))

        # Create a black canvas of original size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Calculate top-left corner to paste the scaled frame
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        # Paste the scaled frame onto the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled_frame
        
        out.write(canvas)

    cap.release()
    out.release()
    print(f"  -> Scaling complete.")


# --- Main Execution ---

def main():
    """Main function to run the full video processing pipeline."""
    # --- Configuration ---
    input_dir = "Input_Video"
    output_dir = "Output_Overlay"
    debug_dir = os.path.join(output_dir, "debug_spatial_alignment")
    
    # Alignment Step Flags
    ENABLE_STABILIZATION = True
    ENABLE_SCALE_ALIGNMENT = True
    ENABLE_TILT_ALIGNMENT = True
    ENABLE_SPATIAL_ALIGNMENT = True
    ENABLE_TEMPORAL_ALIGNMENT = True
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create debug directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    
    # Get all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files.sort()
    
    # Process all pairs of videos
    for i in range(len(video_files) - 1):
        video1_name = video_files[i]
        video2_name = video_files[i + 1]
        
        video1_path = os.path.join(input_dir, video1_name)
        video2_path = os.path.join(input_dir, video2_name)
        
        # Generate output filenames based on input names
        base1 = os.path.splitext(video1_name)[0]
        base2 = os.path.splitext(video2_name)[0]
        
        stabilized1_path = os.path.join(output_dir, f"{base1}_stabilized_v06.mp4")
        stabilized2_path = os.path.join(output_dir, f"{base2}_stabilized_v06.mp4")

        scaled1_path = os.path.join(output_dir, f"{base1}_scaled_v06.mp4")
        scaled2_path = os.path.join(output_dir, f"{base2}_scaled_v06.mp4")
        
        tilt_aligned1_path = os.path.join(output_dir, f"{base1}_tilt_aligned_v06.mp4")
        tilt_aligned2_path = os.path.join(output_dir, f"{base2}_tilt_aligned_v06.mp4")
        
        trimmed1_path = os.path.join(output_dir, f"{base1}_trimmed_v06.mp4")
        trimmed2_path = os.path.join(output_dir, f"{base2}_trimmed_v06.mp4")
        
        final_overlay_path = os.path.join(output_dir, f"{base1}_overlay_{base2}_v06.mp4")
        
        print(f"{'='*60}")
        print(f"Processing pair: {video1_name} + {video2_name}")
        print(f"{'='*60}")

        # --- Pipeline ---

        # Step 1: Stabilize both videos individually
        if ENABLE_STABILIZATION:
            print("Step 1: Stabilizing videos...")
            stabilize_video(video1_path, stabilized1_path, 'motion_raw1_v06.csv', 'motion_smoothed1_v06.csv')
            stabilize_video(video2_path, stabilized2_path, 'motion_raw2_v06.csv', 'motion_smoothed2_v06.csv')
            print("Stabilization complete.")
        else:
            print("Step 1: Stabilization disabled. Skipping.")
            shutil.copy(video1_path, stabilized1_path)
            shutil.copy(video2_path, stabilized2_path)

        # Step 2: Scale Alignment
        if ENABLE_SCALE_ALIGNMENT:
            print("\nStep 2: Performing scale alignment...")
            # Use original videos for scale calculation
            scale_ratios = calculate_scale_ratios(video1_path, video2_path)
            # Using right_lower_leg as the reference for scaling
            scale_ratio = scale_ratios.get('right_lower_leg', 1.0)
            
            if scale_ratio > 1.0 and scale_ratio != 0:
                # Video 1's subject is larger, so scale it down
                print(f"Video 1 is larger by a factor of {scale_ratio:.2f}. Scaling it down.")
                scale_factor = 1.0 / scale_ratio
                scale_video(stabilized1_path, scaled1_path, scale_factor)
                shutil.copy(stabilized2_path, scaled2_path)
            elif scale_ratio < 1.0 and scale_ratio != 0:
                # Video 2's subject is larger, so scale it down
                print(f"Video 2 is larger by a factor of {1/scale_ratio:.2f}. Scaling it down.")
                scale_factor = scale_ratio
                scale_video(stabilized2_path, scaled2_path, scale_factor)
                shutil.copy(stabilized1_path, scaled1_path)
            else:
                print("Videos are already at a similar scale. No scaling needed.")
                shutil.copy(stabilized1_path, scaled1_path)
                shutil.copy(stabilized2_path, scaled2_path)
            print("Scale alignment complete.")
        else:
            print("\nStep 2: Scale alignment disabled. Skipping.")
            shutil.copy(stabilized1_path, scaled1_path)
            shutil.copy(stabilized2_path, scaled2_path)

        # Step 3: Tilt Alignment
        if ENABLE_TILT_ALIGNMENT:
            print("\nStep 3: Performing tilt alignment...")
            tilt_align_videos(scaled1_path, scaled2_path, tilt_aligned1_path, tilt_aligned2_path)
            print("Tilt alignment complete.")
        else:
            print("\nStep 3: Tilt alignment disabled. Skipping.")
            shutil.copy(scaled1_path, tilt_aligned1_path)
            shutil.copy(scaled2_path, tilt_aligned2_path)

        # Step 4: Spatial Alignment
        displacement = (0, 0)
        if ENABLE_SPATIAL_ALIGNMENT:
            print("\nStep 4: Starting spatial alignment...")
            # Use the scaled videos to get the correct displacement
            displacement = get_spatial_displacement(tilt_aligned1_path, tilt_aligned2_path, debug_output_dir=debug_dir)
            print("Spatial alignment complete.")
        else:
            print("\nStep 4: Spatial alignment disabled. Skipping.")

        # Step 5: Temporal Alignment
        frame_shift = 0
        if ENABLE_TEMPORAL_ALIGNMENT:
            print("\nStep 5: Starting temporal alignment...")
            # Use tilt-aligned videos for temporal alignment to avoid stabilization/scaling/tilting artifacts
            frame_shift = temporal_align_videos(tilt_aligned1_path, tilt_aligned2_path)
            frame_shift = int(round(frame_shift))
            print(f"Temporal alignment complete. Frame shift: {frame_shift}")
        else:
            print("\nStep 5: Temporal alignment disabled. Skipping.")

        # Step 6: Trim videos based on the calculated shift
        print("\nStep 6: Trimming videos for synchronization...")
        if frame_shift > 0:
            # Video 1 starts 'frame_shift' frames after Video 2, so trim Video 2.
            print(f"Video 1 is delayed by {abs(frame_shift)} frames. Trimming Video 2.")
            trim_video(tilt_aligned2_path, trimmed2_path, abs(frame_shift))
            shutil.copy(tilt_aligned1_path, trimmed1_path)
        elif frame_shift < 0:
            # Video 2 starts 'frame_shift' frames after Video 1, so trim Video 1.
            print(f"Video 2 is delayed by {abs(frame_shift)} frames. Trimming Video 1.")
            trim_video(tilt_aligned1_path, trimmed1_path, abs(frame_shift))
            shutil.copy(tilt_aligned2_path, trimmed2_path)
        else:
            # Videos are already synchronized or temporal alignment disabled.
            print("Videos are already synchronized or alignment disabled. No trimming needed.")
            shutil.copy(tilt_aligned1_path, trimmed1_path)
            shutil.copy(tilt_aligned2_path, trimmed2_path)
        print("Trimming complete.")

        # Step 7: Render the final overlay
        print("\nStep 7: Rendering final overlay...")
        render_overlay(trimmed1_path, trimmed2_path, displacement, final_overlay_path)
        
        print("Pipeline complete.")

        # break

if __name__ == "__main__":
    main()
