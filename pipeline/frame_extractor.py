import os
import cv2
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

class FrameExtractor:
    """
    Extracts, filters (for sharpness/blur), and downsamples video frames
    to prepare clean keyframes for DUSt3R/MASt3R and Gaussian Splatting,
    optimized to fit strictly within 4GB VRAM.
    """
    def __init__(
        self,
        target_fps: float = 2.0,
        max_dim: int = 512,
        min_laplacian_var: float = 20.0,
        output_dir: str = "output/frames"
    ):
        self.target_fps = target_fps
        self.max_dim = max_dim
        self.min_laplacian_var = min_laplacian_var
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def calculate_sharpness(image_bgr: np.ndarray) -> float:
        """Computes variance of the Laplacian as a metric for image sharpness."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def resize_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Resizes image so that its longest side is at most `max_dim`.
        Preserves aspect ratio and ensures dimensions are multiples of 16 (friendly for ViTs).
        """
        h, w = image_bgr.shape[:2]
        scale = self.max_dim / max(h, w)
        if scale < 1.0:
            new_w = int(round(w * scale / 16.0) * 16)
            new_h = int(round(h * scale / 16.0) * 16)
            new_w = max(16, new_w)
            new_h = max(16, new_h)
            return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Snap to multiples of 16
            new_w = int(round(w / 16.0) * 16)
            new_h = int(round(h / 16.0) * 16)
            if new_w != w or new_h != h:
                return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return image_bgr

    def extract(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        max_frames: Optional[int] = 100
    ) -> List[Dict]:
        """
        Extracts keyframes from video_path between start_sec and end_sec.
        Filters out excessively blurry frames.
        Returns a list of frame metadata dictionaries.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        if end_sec is None or end_sec > duration:
            end_sec = duration

        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        frame_interval = max(1, int(round(video_fps / self.target_fps)))

        print(f"[FrameExtractor] Video: {video_path}")
        print(f"[FrameExtractor] FPS: {video_fps:.2f}, Total Duration: {duration:.2f}s")
        print(f"[FrameExtractor] Clipping: {start_sec:.2f}s to {end_sec:.2f}s (Frames {start_frame} -> {end_frame})")
        print(f"[FrameExtractor] Sampling every {frame_interval} frames (target ~{self.target_fps} fps)")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        extracted_metadata = []
        current_frame_idx = start_frame

        candidate_window = []
        window_size = max(1, frame_interval // 2)

        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            sharpness = self.calculate_sharpness(frame)
            candidate_window.append((current_frame_idx, frame, sharpness))

            # When window has enough samples or at target interval, pick best frame
            if len(candidate_window) >= window_size or current_frame_idx == end_frame:
                # Sort candidate window by sharpness descending
                candidate_window.sort(key=lambda x: x[2], reverse=True)
                best_idx, best_frame, best_sharpness = candidate_window[0]

                # Resize to fit 4GB VRAM constraint
                resized_frame = self.resize_frame(best_frame)
                timestamp = best_idx / video_fps
                out_filename = f"frame_{len(extracted_metadata):04d}.jpg"
                out_path = os.path.join(self.output_dir, out_filename)

                cv2.imwrite(out_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                meta = {
                    "frame_id": len(extracted_metadata),
                    "video_frame_index": best_idx,
                    "timestamp": round(timestamp, 3),
                    "sharpness": round(best_sharpness, 2),
                    "image_path": out_path,
                    "width": resized_frame.shape[1],
                    "height": resized_frame.shape[0]
                }
                extracted_metadata.append(meta)

                # Reset window and jump forward
                candidate_window = []
                current_frame_idx += frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

                if max_frames and len(extracted_metadata) >= max_frames:
                    print(f"[FrameExtractor] Reached max_frames limit ({max_frames}). Stopping extraction.")
                    break
            else:
                current_frame_idx += 1

        cap.release()

        # Save metadata manifest
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(extracted_metadata, f, indent=2)

        print(f"[FrameExtractor] Successfully extracted {len(extracted_metadata)} keyframes to {self.output_dir}")
        return extracted_metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract keyframes from video")
    parser.add_argument("--video", type=str, default="single room.mp4", help="Path to input video")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS")
    parser.add_argument("--max_dim", type=int, default=512, help="Max image dimension for 4GB VRAM")
    parser.add_argument("--out_dir", type=str, default="output/frames", help="Output directory")
    args = parser.parse_args()

    extractor = FrameExtractor(target_fps=args.fps, max_dim=args.max_dim, output_dir=args.out_dir)
    extractor.extract(video_path=args.video, start_sec=args.start, end_sec=args.end)
