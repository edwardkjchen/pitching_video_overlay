"""
This script overlays two video files with a specified spatial displacement.

It is designed to be called from the command line, taking the paths to the
two input videos, the displacement values (dx, dy), and the output path
as arguments.
"""
import cv2
import numpy as np
import argparse
import sys

def render_overlay(video1_path, video2_path, displacement, output_path, alpha=0.5):
    """
    Overlays two videos with a given spatial displacement and saves the result.

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

    fps = cap1.get(cv2.CAP_PROP_FPS)
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    dx, dy = displacement

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # Ensure frames are the same size
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))

        # Apply the spatial alignment to the second video's frame
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_frame2 = cv2.warpAffine(frame2, M, (w, h))

        # Blend the two frames
        blended = cv2.addWeighted(frame1, alpha, aligned_frame2, 1 - alpha, 0)
        out.write(blended)

    cap1.release()
    cap2.release()
    out.release()
    print(f"  -> Final overlay saved to '{output_path}'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Overlay two video files.')
        parser.add_argument('video1_path', type=str, help='Path to the first input video file.')
        parser.add_argument('video2_path', type=str, help='Path to the second input video file.')
        parser.add_argument('dx', type=float, help='Horizontal displacement for the second video.')
        parser.add_argument('dy', type=float, help='Vertical displacement for the second video.')
        parser.add_argument('output_path', type=str, help='Path to save the overlaid video.')
        parser.add_argument('--alpha', type=float, default=0.5, help='Alpha transparency for the first video.')
        args = parser.parse_args()

        displacement = (args.dx, args.dy)
        render_overlay(args.video1_path, args.video2_path, displacement, args.output_path, args.alpha)
    else:
        print("No command line arguments provided. Running a test overlay...")
        render_overlay("Input_Video\cutsIMG_2725.mp4", "Input_Video\cutsIMG_2726.mp4", (0, 0), "test_overlay.mp4", alpha=0.5)