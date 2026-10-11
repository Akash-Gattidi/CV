import os
import cv2
import json

def inspect_video(video_path="input/input_video.mp4"):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    print(f"Video File: {video_path}")
    print(f"File Size: {size_mb:.2f} MB")
    print(f"Resolution: {width}x{height}")
    print(f"Frame Rate: {fps:.2f} fps")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    cap.release()

if __name__ == "__main__":
    inspect_video()
