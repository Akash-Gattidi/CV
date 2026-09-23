import os
import sys
import json
import time
import torch
import numpy as np
import plyfile
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DUST3R_PATH = os.path.join(PROJECT_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.inference import inference
from pipeline.splat_init import SplatInitializer
from pipeline.splat_trainer import GaussianModel, SplatTrainer
from scripts.render_novel_views import render_novel_views_and_orbit

def train_high_density(
    target_gaussians: int = 80000,
    voxel_size: float = 0.006,
    iterations: int = 2000,
    output_dir: str = "output/splats"
):
    print("================================================================")
    print("     HIGH-DENSITY PHOTOREALISTIC GAUSSIAN SPLATTING PIPELINE    ")
    print(f" Target: ~{target_gaussians} Gaussians | Voxel: {voxel_size*1000:.1f}mm | Iters: {iterations}")
    print("================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load calibrated cameras
    cams_path = "output/dust3r/cameras.json"
    with open(cams_path, "r") as f:
        cams = json.load(f)
    print(f"Loaded {len(cams)} calibrated camera poses.")

    # Check if dense seed points already exist
    dense_ply_path = "output/dust3r/dust3r_points_dense.ply"
    if os.path.exists(dense_ply_path):
        print(f"Loading cached dense points from {dense_ply_path}...")
        ply_data = plyfile.PlyData.read(dense_ply_path)
        pts = np.stack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']], axis=-1).astype(np.float32)
        colors = (np.stack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)
        print(f"Loaded {len(pts)} dense seed points from cache.")
    else:
        # 2. Extract dense points from DUSt3R
        print("\n--- STEP 1: EXTRACTING DENSE POINTMAPS FROM DUST3R (~45 SECONDS) ---")
        ckpt = "checkpoints/dust3r_512"
        model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(device)
        model.eval()

        img_paths = [c["img_path"] for c in cams]
        loaded_imgs = load_images(img_paths, size=512)

        pairs = []
        for i in range(len(loaded_imgs) - 1):
            pairs.append((loaded_imgs[i], loaded_imgs[i+1]))

        print(f"Running forward inference on {len(pairs)} consecutive image pairs...")
        all_pts_world = []
        all_colors = []

        batch_size = 2
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), batch_size), desc="Dense Pointmaps"):
                batch_pairs = pairs[i:i+batch_size]
                out = inference(batch_pairs, model, device, batch_size=batch_size, verbose=False)

                pred1_pts = out["pred1"]["pts3d"].detach().cpu().numpy()  # (B, H, W, 3)
                pred1_conf = out["pred1"]["conf"].detach().cpu().numpy()   # (B, H, W)

                for b in range(len(batch_pairs)):
                    pair_idx = i + b
                    cam = cams[pair_idx]
                    c2w = np.array(cam["c2w"], dtype=np.float32)
                    R_c2w = c2w[:3, :3]
                    t_c2w = c2w[:3, 3]

                    p3d_cam = pred1_pts[b]  # (H, W, 3)
                    conf = pred1_conf[b]    # (H, W)
                    rgb = (batch_pairs[b][0]["img"][0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0

                    mask = (conf > 1.5) & (p3d_cam[..., 2] > 0.1) & (p3d_cam[..., 2] < 15.0)
                    valid_cam_pts = p3d_cam[mask]  # (M, 3)
                    valid_rgb = rgb[mask]

                    if len(valid_cam_pts) > 0:
                        # Transform to world coordinate frame
                        pts_world = valid_cam_pts @ R_c2w.T + t_c2w
                        all_pts_world.append(pts_world)
                        all_colors.append(valid_rgb)

        pts = np.concatenate(all_pts_world, axis=0).astype(np.float32)
        colors = np.concatenate(all_colors, axis=0).astype(np.float32)
        print(f"Extracted {len(pts)} total raw confident 3D points.")

        # Voxel downsampling to target ~80,000 points
        init_helper = SplatInitializer()
        pts, colors = init_helper.voxel_downsample(pts, colors, voxel_size=voxel_size)
        print(f"Points after {voxel_size*1000:.1f}mm voxel filter: {len(pts)}")

        if len(pts) > target_gaussians:
            idx = np.random.choice(len(pts), target_gaussians, replace=False)
            pts = pts[idx]
            colors = colors[idx]
            print(f"Subsampled to target cap: {len(pts)} points")

        # Save dense PLY
        vertex_data = np.empty(
            len(pts),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        vertex_data['x'] = pts[:, 0]
        vertex_data['y'] = pts[:, 1]
        vertex_data['z'] = pts[:, 2]
        vertex_data['red'] = (colors[:, 0] * 255).astype(np.uint8)
        vertex_data['green'] = (colors[:, 1] * 255).astype(np.uint8)
        vertex_data['blue'] = (colors[:, 2] * 255).astype(np.uint8)
        el = plyfile.PlyElement.describe(vertex_data, 'vertex')
        plyfile.PlyData([el]).write(dense_ply_path)
        print(f"Saved dense seed points to: {dense_ply_path}")

        # Free DUSt3R model memory before splat training
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Splat Initialization with high density
    print("\n--- STEP 2: INITIALIZING HIGH-DENSITY GAUSSIAN SCENE ---")
    init_helper = SplatInitializer()
    init_params = init_helper.prepare_initial_gaussians(
        points=pts,
        colors=colors,
        voxel_size=0.0, # already downsampled
        max_initial_points=target_gaussians
    )
    gaussians = GaussianModel(init_params=init_params, device=device)
    print(f"Active 3D Gaussians: {gaussians.get_xyz.shape[0]}")

    # 4. Train with vectorized GPU rasterizer
    print(f"\n--- STEP 3: HIGH-DENSITY SPLAT TRAINING ({iterations} ITERATIONS) ---")
    trainer = SplatTrainer(
        gaussians=gaussians,
        cameras=cams,
        lr_xyz=8e-4,
        lr_features=2.5e-3,
        lr_opacity=0.05,
        lr_scaling=4e-3
    )
    out_ply = trainer.train(total_iterations=iterations, output_dir=output_dir)

    # 5. Render updated novel views and orbit animation
    print("\n--- STEP 4: RENDERING UPDATED NOVEL VIEWS & ORBIT WALKTHROUGH ---")
    render_novel_views_and_orbit(
        ply_path=out_ply,
        cams_path=cams_path,
        out_dir="output/renders",
        n_views=6,
        orbit_frames=60
    )

    # Copy to artifact folder
    import shutil, glob
    art_dir = r"C:\Users\akash\.gemini\antigravity-ide\brain\98e73429-616d-4959-84c9-efc1ac00bb95"
    for f in glob.glob("output/renders/*.*"):
        shutil.copy(f, os.path.join(art_dir, os.path.basename(f)))
    print("Copied updated renders to artifact directory for inspection.")

    print("\n================================================================")
    print("      HIGH-DENSITY RECONSTRUCTION COMPLETED SUCCESSFULLY!       ")
    print(f" Trained Model: {out_ply}")
    print(f" Total Gaussians: {gaussians.get_xyz.shape[0]}")
    print("================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts", type=int, default=75000, help="Target Gaussians")
    parser.add_argument("--iters", type=int, default=2000, help="Training iterations")
    parser.add_argument("--voxel", type=float, default=0.006, help="Voxel size in meters")
    args = parser.parse_args()

    train_high_density(
        target_gaussians=args.pts,
        voxel_size=args.voxel,
        iterations=args.iters
    )
