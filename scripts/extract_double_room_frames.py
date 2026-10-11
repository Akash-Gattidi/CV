import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.frame_extractor import FrameExtractor

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract keyframes for multi-room reconstruction")
    parser.add_argument("--video", type=str, default="input/input_video.mp4", help="Path to input video")
    parser.add_argument("--out_dir", type=str, default="output/frames_double", help="Output directory")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    # 52s video -> at 1.8-2.0 fps with blur filtering gives ~90-100 high-quality keyframes
    extractor = FrameExtractor(
        target_fps=1.8,
        max_dim=512,
        min_laplacian_var=15.0,
        output_dir=output_dir
    )

    metadata = extractor.extract(
        video_path=video_path,
        start_sec=0.0,
        end_sec=None,
        max_frames=120
    )

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Extraction Summary]")
    print(f"Extracted {len(metadata)} clean keyframes to {output_dir}")
    if metadata:
        sample = metadata[0]
        print(f"Frame resolution: {sample['width']}x{sample['height']}")
        sharpness_vals = [m['sharpness'] for m in metadata]
        print(f"Mean sharpness (Laplacian var): {sum(sharpness_vals)/len(sharpness_vals):.1f}")
        print(f"Min sharpness: {min(sharpness_vals):.1f}, Max: {max(sharpness_vals):.1f}")

if __name__ == "__main__":
    main()
