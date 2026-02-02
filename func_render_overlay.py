"""
This script overlays two video files with a specified spatial displacement.

It is designed to be called from the command line, taking the paths to the
two input videos, the displacement values (dx, dy), and the output path
as arguments.
"""
import os
import cv2
import numpy as np
import argparse
import sys

def render_overlay(video1_path, video2_path, displacement, output_path, alpha=0.5):
    """
    Overlays two videos with a given spatial displacement and saves the result.

    This function is designed to handle videos of potentially different resolutions.
    It determines the smallest resolution between the two and sets that as the
    output size. It then applies the displacement to the second video by creating
    a temporary canvas, ensuring that shifts in any direction (positive or negative)
    are handled correctly by clipping off-screen content and padding with black where
    necessary. Finally, it crops both source frames to the output resolution before
    blending.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.
        displacement (tuple): The (dx, dy) offset to apply to the second video.
        output_path (str): Path to save the final overlaid video.
        alpha (float): Transparency of the first video.
    """
    print(f"Rendering overlay of {video1_path} and {video2_path} -> {output_path}...")
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files for rendering.")
        return

    # Get original dimensions and FPS from both video captures
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Determine the final output resolution by using the smaller of the two videos.
    # This ensures the smaller video is never scaled up, preventing distortion.
    out_w = min(w1, w2)
    out_h = min(h1, h2)

    # Define the codec and create VideoWriter object to save the final video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Unpack displacement values into integer coordinates.
    dx, dy = int(displacement[0]), int(displacement[1])

    while True:
        # Read one frame from each video stream.
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # If either video has ended, break the loop.
        if not ret1 or not ret2:
            break

        # --- Frame Alignment and Cropping Logic ---

        # 1. DISPLACE FRAME 2: Create a black canvas the size of frame1. We will place
        #    frame2 onto this canvas at the specified (dx, dy) offset. This method correctly
        #    handles both positive (down/right) and negative (up/left) displacements.
        frame2_displaced_canvas = np.zeros_like(frame1)

        # 2. CALCULATE DESTINATION REGION on the canvas. This is where frame2 will be pasted.
        #    - `max(0, ...)` prevents using negative coordinates if frame2 is shifted up/left.
        #    - `min(h1, ...)` prevents writing past the canvas boundaries.
        y_start_dest = max(0, dy)
        x_start_dest = max(0, dx)
        y_end_dest = min(h1, dy + h2)
        x_end_dest = min(w1, dx + w2)

        # 3. CALCULATE SOURCE REGION from frame2. This is the part of frame2 to be copied.
        #    - `max(0, -dy)` handles up/left shifts by selecting a starting point
        #      within frame2, effectively clipping the part that moved off-screen.
        y_start_src = max(0, -dy)
        x_start_src = max(0, -dx)
        #    - The end of the source region is calculated from the size of the destination region.
        y_end_src = y_start_src + (y_end_dest - y_start_dest)
        x_end_src = x_start_src + (x_end_dest - x_start_dest)

        # 4. PASTE: Copy the source region from frame2 onto the destination region of the canvas.
        #    This completes the displacement operation.
        frame2_displaced_canvas[y_start_dest:y_end_dest, x_start_dest:x_end_dest] = frame2[y_start_src:y_end_src, x_start_src:x_end_src]

        # 5. CROP to final output size. Now that frame2 is aligned relative to frame1,
        #    we crop both frame1 and the displaced frame2 canvas to the final output size.
        #    This ensures the larger of the two original videos is center-cropped.
        y_crop_offset1 = (h1 - out_h) // 2
        x_crop_offset1 = (w1 - out_w) // 2
        final_frame1 = frame1[y_crop_offset1:y_crop_offset1 + out_h, x_crop_offset1:x_crop_offset1 + out_w]

        # The displaced canvas is also cropped from the same origin to maintain alignment.
        final_frame2 = frame2_displaced_canvas[y_crop_offset1:y_crop_offset1 + out_h, x_crop_offset1:x_crop_offset1 + out_w]
        
        # 6. BLEND the two frames, which are now correctly aligned and identically sized.
        blended = cv2.addWeighted(final_frame1, alpha, final_frame2, 1 - alpha, 0)
        
        # 7. WRITE the final blended frame to the output video file.
        out.write(blended)

    # --- Cleanup ---
    # Release the video capture and writer objects to free up resources.
    cap1.release()
    cap2.release()
    out.release()
    print(f"  -> Final overlay saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay two video files, cropping the larger one to match the smaller one.')
    parser.add_argument('video1_path', type=str, help='Path to the first input video file.')
    parser.add_argument('video2_path', type=str, help='Path to the second input video file.')
    parser.add_argument('dx', type=float, help='Horizontal displacement for the second video.')
    parser.add_argument('dy', type=float, help='Vertical displacement for the second video.')
    parser.add_argument('output_path', type=str, help='Path to save the overlaid video.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha transparency for the first video.')
    
    if len(sys.argv) > 1:
        args = parser.parse_args()
        displacement = (args.dx, args.dy)
        render_overlay(args.video1_path, args.video2_path, displacement, args.output_path, args.alpha)
    else:
        print("No command line arguments provided. Running a test overlay with example files...")
        # Example usage for testing purposes
        video1_path = "Output_Overlay/cutsIMG_2734_trimmed_v04.mp4" # Replace with a valid video path
        video2_path = "Output_Overlay/cutsIMG_2747_trimmed_v04.mp4" # Replace with a valid video path
        displacement = (-43, 28)
        #video1_path = "Output_Overlay/cutsIMG_1844_trimmed_v04.mp4" # Replace with a valid video path
        #video2_path = "Output_Overlay/cutsIMG_2723_trimmed_v04.mp4" # Replace with a valid video path
        #displacement = (920, 730)
        output_path = "test_overlay_refined.mp4"
        
        if os.path.exists(video1_path) and os.path.exists(video2_path):
             render_overlay(video1_path, video2_path, displacement, output_path, alpha=0.5)
        else:
            print(f"Warning: Test video files not found. Please provide arguments or place videos at:\n{video1_path}\n{video2_path}")
