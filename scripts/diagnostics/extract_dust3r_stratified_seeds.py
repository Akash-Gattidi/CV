import os
import sys
import json
import glob
import time
import torch
import numpy as np
import plyfile

REPO_ROOT = os.path.abspath('.')
DUST3R_PATH = os.path.join(REPO_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images

def extract_stratified_seeds(
    frames_dir: str = "output/frames_double",
    output_dir: str = "output/dust3r_double",
    grid_rows: int = 53,
    grid_cols: int = 30,
    conf_floor: float = 0.8,
    n_iters: int = 60
):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    n_frames = len(frame_paths)
    print(f"[DUSt3R Stratified] Loading {n_frames} keyframes...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_images(frame_paths, size=512)

    cache_path = os.path.join(output_dir, "pairwise_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Pairwise cache not found at {cache_path}. Run extract_dust3r_100k_seeds.py first.")

    print(f"[DUSt3R Stratified] Loading cached pairwise inferences from {cache_path}...")
    output = torch.load(cache_path, map_location="cpu")

    print(f"[DUSt3R Stratified] Running global alignment ({n_iters} iterations)...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=True)
    loss = scene.compute_global_alignment(init="mst", niter=n_iters, schedule="cosine", lr=0.01)
    print(f"[DUSt3R Stratified] Global alignment completed with loss: {loss:.4f}")

    # Extract camera poses & intrinsics
    poses = scene.get_im_poses()
    if torch.is_tensor(poses): poses = poses.detach().cpu().numpy()
    focals = scene.get_focals()
    if torch.is_tensor(focals): focals = focals.detach().cpu().numpy()
    pps = scene.get_principal_points()
    if torch.is_tensor(pps): pps = pps.detach().cpu().numpy()

    pts3d = scene.get_pts3d()
    confs = scene.get_conf()

    all_points = []
    all_colors = []
    camera_data = []

    # Regional tracking: Left, Center, Right thirds for every frame
    per_frame_region_counts = [] # list of (left, center, right)
    per_frame_total_counts = []

    print(f"[DUSt3R Stratified] Sampling points via {grid_rows}x{grid_cols} grid (conf floor: {conf_floor})...")

    for idx, img_path in enumerate(frame_paths):
        h, w = images[idx]["true_shape"][0].tolist()
        f_val = focals[idx]
        f = float(f_val[0]) if hasattr(f_val, "__len__") else float(f_val)
        cx, cy = float(pps[idx][0]), float(pps[idx][1])
        c2w_mat = np.array(poses[idx], dtype=np.float64)
        w2c_mat = np.linalg.inv(c2w_mat)

        camera_data.append({
            "id": idx,
            "img_name": os.path.basename(img_path),
            "img_path": img_path,
            "width": int(w), "height": int(h),
            "fx": f, "fy": f, "cx": cx, "cy": cy,
            "c2w": c2w_mat.tolist(),
            "w2c": w2c_mat.tolist()
        })

        p3d_tensor = pts3d[idx]
        p3d = p3d_tensor.detach().cpu().numpy() if torch.is_tensor(p3d_tensor) else np.array(p3d_tensor)
        conf_tensor = confs[idx]
        conf = conf_tensor.detach().cpu().numpy() if torch.is_tensor(conf_tensor) else np.array(conf_tensor)
        rgb = (images[idx]["img"][0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0

        H_img, W_img = conf.shape
        row_bins = np.linspace(0, H_img, grid_rows + 1).astype(int)
        col_bins = np.linspace(0, W_img, grid_cols + 1).astype(int)

        sampled_p3d = []
        sampled_rgb = []
        sampled_x_coords = []

        for r in range(grid_rows):
            r0, r1 = row_bins[r], row_bins[r + 1]
            for c in range(grid_cols):
                c0, c1 = col_bins[c], col_bins[c + 1]
                cell_conf = conf[r0:r1, c0:c1]
                
                # Check valid points
                above_floor = cell_conf >= conf_floor
                if np.any(above_floor):
                    # Pick argmax within cell
                    local_idx = np.argmax(cell_conf)
                    local_r, local_c = np.unravel_index(local_idx, cell_conf.shape)
                    best_y = r0 + local_r
                    best_x = c0 + local_c
                else:
                    # Fallback to lower floor (conf > 0.2) to ensure coverage in dark/mirror areas
                    above_low = cell_conf > 0.2
                    if np.any(above_low):
                        local_idx = np.argmax(cell_conf)
                        local_r, local_c = np.unravel_index(local_idx, cell_conf.shape)
                        best_y = r0 + local_r
                        best_x = c0 + local_c
                    else:
                        continue # truly empty / invalid
                
                pt = p3d[best_y, best_x]
                # Check finite
                if np.all(np.isfinite(pt)) and np.linalg.norm(pt) < 50.0:
                    sampled_p3d.append(pt)
                    sampled_rgb.append(rgb[best_y, best_x])
                    sampled_x_coords.append(best_x)

        sampled_p3d = np.array(sampled_p3d, dtype=np.float32)
        sampled_rgb = np.array(sampled_rgb, dtype=np.float32)
        sampled_x_coords = np.array(sampled_x_coords)

        all_points.append(sampled_p3d)
        all_colors.append(sampled_rgb)
        per_frame_total_counts.append(len(sampled_p3d))

        # Split into Left, Center, Right thirds based on W_img
        w_third = W_img / 3.0
        n_left = np.sum(sampled_x_coords < w_third)
        n_center = np.sum((sampled_x_coords >= w_third) & (sampled_x_coords < 2 * w_third))
        n_right = np.sum(sampled_x_coords >= 2 * w_third)
        per_frame_region_counts.append((int(n_left), int(n_center), int(n_right)))

    # Save updated cameras.json
    cams_out = os.path.join(output_dir, "cameras.json")
    with open(cams_out, "w") as f:
        json.dump(camera_data, f, indent=2)

    merged_pts = np.concatenate(all_points, axis=0).astype(np.float32)
    merged_colors = np.concatenate(all_colors, axis=0).astype(np.float32)

    # Save stratified ply
    out_ply = os.path.join(output_dir, "dust3r_points_100k_stratified.ply")
    vertex_data = np.empty(
        len(merged_pts),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    vertex_data['x'] = merged_pts[:, 0]
    vertex_data['y'] = merged_pts[:, 1]
    vertex_data['z'] = merged_pts[:, 2]
    vertex_data['red'] = (merged_colors[:, 0] * 255).astype(np.uint8)
    vertex_data['green'] = (merged_colors[:, 1] * 255).astype(np.uint8)
    vertex_data['blue'] = (merged_colors[:, 2] * 255).astype(np.uint8)
    plyfile.PlyData([plyfile.PlyElement.describe(vertex_data, 'vertex')]).write(out_ply)

    # Also overwrite dust3r_points_100k.ply so all downstream scripts use the stratified seeds
    out_ply_main = os.path.join(output_dir, "dust3r_points_100k.ply")
    plyfile.PlyData([plyfile.PlyElement.describe(vertex_data, 'vertex')]).write(out_ply_main)

    # Calculate statistics
    all_region_counts = []
    for l, c, r in per_frame_region_counts:
        all_region_counts.extend([l, c, r])
    
    lowest_region_count = min(all_region_counts)
    lowest_region_idx = np.argmin(all_region_counts)
    frame_with_lowest = lowest_region_idx // 3
    region_names = ["LEFT", "CENTER", "RIGHT"]
    region_type = region_names[lowest_region_idx % 3]

    f33_left, f33_center, f33_right = per_frame_region_counts[33]

    stats = {
        "total_seeds": len(merged_pts),
        "min_per_cam": int(min(per_frame_total_counts)),
        "max_per_cam": int(max(per_frame_total_counts)),
        "median_per_cam": int(np.median(per_frame_total_counts)),
        "frame_33_left_third": int(f33_left),
        "frame_33_center_third": int(f33_center),
        "frame_33_right_third": int(f33_right),
        "frame_33_total": int(per_frame_total_counts[33]),
        "lowest_per_region_count": int(lowest_region_count),
        "lowest_per_region_frame": int(frame_with_lowest),
        "lowest_per_region_name": region_type,
        "per_frame_region_counts": per_frame_region_counts
    }

    with open(os.path.join(output_dir, "stratified_seed_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n============================================================")
    print("      STRATIFIED DUSt3R SEED SELECTION RESULTS               ")
    print("============================================================")
    print(f"Total Stratified Seeds: {len(merged_pts):,}")
    print(f"Per-Camera Range: {min(per_frame_total_counts)} to {max(per_frame_total_counts)} (Median: {int(np.median(per_frame_total_counts))})")
    print(f"Frame 33 Breakdown:")
    print(f"  - LEFT third:   {f33_left} seeds")
    print(f"  - CENTER third: {f33_center} seeds")
    print(f"  - RIGHT third:  {f33_right} seeds")
    print(f"  - Total Frame 33: {per_frame_total_counts[33]} seeds")
    print(f"Lowest Region Count across ALL {n_frames} frames (195 regions total):")
    print(f"  - Count: {lowest_region_count} seeds in Frame {frame_with_lowest} ({region_type} third)")
    print("============================================================\n")

if __name__ == "__main__":
    extract_stratified_seeds()
