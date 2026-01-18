import cv2

def overlay_pitch_videos(video1_path, video2_path, alpha=0.5):
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: One or both videos could not be opened.")
        return

    print("Press 'q' to quit, 'p' to pause/play, '+' or '-' to adjust transparency.")

    paused = False

    while True:
        if not paused:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Resize to match the same frame size
            height = min(frame1.shape[0], frame2.shape[0])
            width = min(frame1.shape[1], frame2.shape[1])
            frame1 = cv2.resize(frame1, (width, height))
            frame2 = cv2.resize(frame2, (width, height))

            # Overlay the two frames
            blended = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)

            # Show the overlay
            cv2.imshow('Pitch Motion Overlay', blended)

        # Key controls
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('+'):
            alpha = min(1.0, alpha + 0.05)
            print(f"Transparency: {alpha:.2f}")
        elif key == ord('-'):
            alpha = max(0.0, alpha - 0.05)
            print(f"Transparency: {alpha:.2f}")

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Example usage:
overlay_pitch_videos("cutsIMG_2725.mp4", "cutsIMG_2726.mp4", alpha=0.5)
