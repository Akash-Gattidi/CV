import os
import sys
import json
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.frame_extractor import FrameExtractor
from pipeline.dust3r_runner import DUSt3RGeometryEstimator
from pipeline.splat_init import SplatInitializer
from pipeline.splat_trainer import GaussianModel, SplatTrainer

def run_milestone_1(
    video_path: str = "single room.mp4",
    start_sec: float = 0.0,
    end_sec: float = None,
    target_fps: float = 2.0,
    max_dim: int = 512,
    iterations: int = 500,
    voxel_size: float = 0.02
):
    print("================================================================")
    print("      MILESTONE 1: SINGLE-ROOM 3D RECONSTRUCTION PIPELINE       ")
    print("================================================================")
    t_start = time.time()

    # Step 1: Keyframe Extraction & Filtering
    print("\n--- STEP 1: KEYFRAME EXTRACTION & SHARPNESS FILTERING ---")
    frames_dir = os.path.join("output", "frames")
    extractor = FrameExtractor(target_fps=target_fps, max_dim=max_dim, output_dir=frames_dir)
    manifest = extractor.extract(video_path=video_path, start_sec=start_sec, end_sec=end_sec)
    
    if len(manifest) < 4:
        raise ValueError(f"Extracted only {len(manifest)} frames. Need at least 4 frames for reconstruction.")

    image_paths = [m["image_path"] for m in manifest]
    print(f"Step 1 Complete: {len(image_paths)} clean keyframes ready.")

    # Step 2: DUSt3R Geometry & Pose Estimation (Handles Blank Walls)
    print("\n--- STEP 2: DUST3R DENSE GEOMETRY & CAMERA POSE ESTIMATION ---")
    dust3r_dir = os.path.join("output", "dust3r")
    cameras_json_path = os.path.join(dust3r_dir, "cameras.json")
    ply_path = os.path.join(dust3r_dir, "dust3r_points.ply")

    if os.path.exists(cameras_json_path) and os.path.exists(ply_path):
        print(f"[DUSt3R] Found existing reconstruction at {dust3r_dir}. Loading cached geometry and camera poses...")
        with open(cameras_json_path, "r") as f:
            camera_data = json.load(f)
        import plyfile
        ply_data = plyfile.PlyData.read(ply_path)
        pts = np.stack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']], axis=-1).astype(np.float32)
        colors = (np.stack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)
        dust3r_results = {
            "cameras_json": cameras_json_path,
            "ply_path": ply_path,
            "points": pts,
            "colors": colors,
            "cameras": camera_data
        }
        print(f"[DUSt3R] Loaded {len(pts)} 3D points and {len(camera_data)} camera poses from cache.")
    else:
        weights_path = os.path.join("checkpoints", "dust3r_512")
        estimator = DUSt3RGeometryEstimator(model_name=weights_path)
        dust3r_results = estimator.run_reconstruction(
            image_paths=image_paths,
            output_dir=dust3r_dir,
            n_iters=200,
            voxel_downsample=voxel_size
        )
        print("Step 2 Complete: DUSt3R camera poses and 3D pointmaps generated.")


    # Step 3: Splat Initialization (Direct from DUSt3R, No COLMAP)
    print("\n--- STEP 3: INITIALIZING 3D GAUSSIAN SPLATS ---")
    init_helper = SplatInitializer()
    init_params = init_helper.prepare_initial_gaussians(
        points=dust3r_results["points"],
        colors=dust3r_results["colors"],
        voxel_size=voxel_size,
        max_initial_points=120000
    )
    gaussians = GaussianModel(init_params=init_params, device="cuda")
    print(f"Step 3 Complete: {gaussians.get_xyz.shape[0]} Gaussians initialized.")

    # Step 4: Gaussian Splat Training per Room Segment (< 4GB VRAM)
    print(f"\n--- STEP 4: 3D GAUSSIAN SPLAT TRAINING ({iterations} ITERATIONS) ---")
    splat_dir = os.path.join("output", "splats")
    trainer = SplatTrainer(gaussians=gaussians, cameras=dust3r_results["cameras"])
    final_ply = trainer.train(total_iterations=iterations, output_dir=splat_dir)
    print("Step 4 Complete: 3D Gaussian Splatting model trained and saved.")

    total_time = time.time() - t_start
    print("\n================================================================")
    print("             MILESTONE 1 RUN COMPLETED SUCCESSFULLY             ")
    print(f" Total Pipeline Time: {total_time:.1f} seconds ({total_time/60.0:.2f} mins)")
    print(f" Output Point Cloud:  {dust3r_results['ply_path']}")
    print(f" Output Splat Scene:  {final_ply}")
    print(f" Camera Poses:        {dust3r_results['cameras_json']}")
    print("================================================================")
    return {
        "final_ply": final_ply,
        "cameras_json": dust3r_results["cameras_json"],
        "dust3r_ply": dust3r_results["ply_path"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 1 Single-Room Pipeline")
    parser.add_argument("--video", type=str, default="single room.mp4", help="Path to input video")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS")
    parser.add_argument("--max_dim", type=int, default=512, help="Max image dimension for 4GB VRAM")
    parser.add_argument("--iters", type=int, default=500, help="Splat training iterations")
    parser.add_argument("--voxel", type=float, default=0.02, help="Voxel size in meters")
    args = parser.parse_args()

    run_milestone_1(
        video_path=args.video,
        start_sec=args.start,
        end_sec=args.end,
        target_fps=args.fps,
        max_dim=args.max_dim,
        iterations=args.iters,
        voxel_size=args.voxel
    )
