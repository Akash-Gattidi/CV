import os
import sys
import json
import math
import cv2
import torch
import numpy as np
import plyfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.render_novel_views import load_splats, create_lookat_w2c
from pipeline.splat_trainer import DifferentiableSplatRenderer

def inspect_and_render_raw(
    ply_path: str = "output/splats/room_splat.ply",
    cams_path: str = "output/dust3r/cameras.json",
    out_dir: str = "output/renders",
    n_views: int = 6
):
    # 1. Exact Gaussian Count from PLY
    ply_data = plyfile.PlyData.read(ply_path)
    n_gaussians = len(ply_data['vertex'])
    file_size_kb = os.path.getsize(ply_path) / 1024.0

    print("================================================================")
    print(f" EXACT GAUSSIAN COUNT IN '{ply_path}': {n_gaussians:,}")
    print(f" FILE SIZE: {file_size_kb:.2f} KB")
    print("================================================================")

    # 2. Render Raw Novel Views (sharpness=1.0, opacity=1.0, NO BOOSTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians = load_splats(ply_path, device=device)
    renderer = DifferentiableSplatRenderer(bg_color=(0.95, 0.95, 0.95))

    with open(cams_path, "r") as f:
        cams = json.load(f)

    base_cam = cams[0]
    fx, fy = base_cam["fx"], base_cam["fy"]
    cx, cy = base_cam["cx"], base_cam["cy"]
    w, h = base_cam["width"], base_cam["height"]

    all_cam_pos = np.array([[c["c2w"][0][3], c["c2w"][1][3], c["c2w"][2][3]] for c in cams], dtype=np.float32)
    xyz_points = gaussians.get_xyz.detach().cpu().numpy()
    center = np.median(xyz_points, axis=0)

    cam_center = np.mean(all_cam_pos, axis=0)
    radius = np.mean(np.linalg.norm(all_cam_pos[:, :2] - center[:2], axis=1))
    if radius < 0.2:
        radius = 0.5
    cam_z = np.median(all_cam_pos[:, 2])

    print("\nRendering RAW unboosted novel views (sharpness=1.0, opacity=1.0)...")
    angles = np.linspace(0, 2 * math.pi, n_views, endpoint=False)
    
    raw_paths = []
    for i, angle in enumerate(angles):
        pos = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            cam_z + 0.1 * math.sin(angle * 2)
        ])
        w2c = create_lookat_w2c(pos, center)
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)

        with torch.no_grad():
            projected = renderer.project_gaussians(gaussians, w2c_t, fx, fy, cx, cy, w, h)
            if projected is not None:
                # RAW: sharpness=1.0, opacity=1.0 (no multiplier, no boost)
                rendered_rgb = renderer.render(
                    projected, w, h,
                    sharpness_multiplier=1.0,
                    opacity_boost=1.0
                )
                img_np = (rendered_rgb.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_np = np.full((h, w, 3), 240, dtype=np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, f"novel_view_{i+1:02d}_raw.jpg")
        cv2.imwrite(out_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw_paths.append(out_path)
        print(f"Rendered RAW novel view {i+1}/{n_views}: {out_path}")

    # Copy raw renders to artifact directory for side-by-side inspection
    art_dir = r"C:\Users\akash\.gemini\antigravity-ide\brain\98e73429-616d-4959-84c9-efc1ac00bb95"
    if os.path.exists(art_dir):
        import shutil
        for p in raw_paths:
            shutil.copy(p, os.path.join(art_dir, os.path.basename(p)))
        print("Raw unboosted renders copied to artifact directory.")

if __name__ == "__main__":
    inspect_and_render_raw()
