import os
import sys
import json
import gzip
import shutil
import cv2
import numpy as np
import torch
import plyfile
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gsplat import rasterization

def run_novel_renders_and_analysis():
    cams_path = "output/dust3r_double/cameras.json"
    ply_path = "output/splats_double/double_room_splat.ply"
    out_dir = "output/renders_double"
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))

    with open(cams_path, "r") as f:
        cams = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Novel Views] Using device: {device}")

    # Load splat
    ply_data = plyfile.PlyData.read(ply_path)
    v = ply_data['vertex']
    n_total = len(v)
    print(f"[Novel Views] Loaded {n_total:,} Gaussians from {ply_path}")

    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32, device=device)
    scales = torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1), dtype=torch.float32, device=device)
    raw_opacities = np.array(v['opacity'])
    min_op = float(raw_opacities.min())
    max_op = float(raw_opacities.max())
    mean_op = float(raw_opacities.mean())
    print(f"[Pruning Analysis] Opacity stats in PLY: min={min_op:.4f}, max={max_op:.4f}, mean={mean_op:.4f}")

    opacities = torch.tensor(raw_opacities, dtype=torch.float32, device=device)
    C0 = 0.28209479177387814
    sh0 = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = torch.tensor(np.clip(sh0 * C0 + 0.5, 0.0, 1.0), dtype=torch.float32, device=device)

    base = cams[0]
    fx, fy, cx, cy = base['fx'], base['fy'], base['cx'], base['cy']
    w, h = base['width'], base['height']
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    def render_cam(c2w_mat):
        w2c = np.linalg.inv(c2w_mat)
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales), opacities=opacities, colors=colors,
                viewmats=w2c_t, Ks=K, width=w, height=h, near_plane=0.1, packed=False
            )
        return (render_colors[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # =========================================================================
    # 1. Generate NOVEL camera poses crossing the doorway
    # =========================================================================
    print("[Novel Views] Synthesizing 8 novel doorway crossing camera poses with lateral offset...")
    
    anchor_indices = [32, 34, 36, 38, 40, 42]
    anchor_c2ws = [np.array(cams[idx]['c2w'], dtype=np.float64) for idx in anchor_indices]
    
    anchor_pos = np.array([m[:3, 3] for m in anchor_c2ws])
    anchor_rots = R.from_matrix([m[:3, :3] for m in anchor_c2ws])
    
    n_novel = 8
    t_anchors = np.linspace(0, 1, len(anchor_indices))
    t_novel = np.linspace(0.05, 0.95, n_novel)
    
    novel_frames = []
    slerp = Slerp(t_anchors, anchor_rots)
    interp_rots = slerp(t_novel).as_matrix()

    interp_pos = np.zeros((n_novel, 3))
    for i in range(3):
        interp_pos[:, i] = np.interp(t_novel, t_anchors, anchor_pos[:, i])

    for i in range(n_novel):
        rot = interp_rots[i]
        cam_right = rot[:, 0]
        
        # Lateral offset: +/- 0.07m sway left and right
        lateral_offset = np.sin((i + 1) * np.pi / 2.5) * 0.07 * cam_right
        vert_offset = np.cos((i + 1) * np.pi / 3.0) * 0.02 * rot[:, 1]
        
        novel_c2w = np.eye(4, dtype=np.float64)
        novel_c2w[:3, :3] = rot
        novel_c2w[:3, 3] = interp_pos[i] + lateral_offset + vert_offset

        pred_rgb = render_cam(novel_c2w.astype(np.float32))
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        
        fname = f"novel_doorway_{i+1:02d}.jpg"
        save_path = os.path.join(out_dir, fname)
        cv2.imwrite(save_path, pred_bgr)
        shutil.copy(save_path, os.path.join(art_dir, fname))
        novel_frames.append(save_path)
        print(f"  - Rendered {fname} (lateral offset: {np.linalg.norm(lateral_offset):.3f}m)")

    # =========================================================================
    # 2. Extract walkthrough frames & copy missing doorway pairs (32, 33, 37, 39)
    # =========================================================================
    for idx in [32, 33, 37, 39]:
        fname = f"doorway_step_{idx:02d}.jpg"
        src = os.path.join(out_dir, fname)
        dst = os.path.join(art_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  - Synced {fname} to artifacts")

    walkthrough_video = os.path.join(out_dir, "multiroom_walkthrough.mp4")
    cap = cv2.VideoCapture(walkthrough_video)
    sample_indices = [5, 18, 30, 36, 46, 58]
    walkthrough_samples = []
    
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in sample_indices:
            s_name = f"walkthrough_sample_f{current_idx:02d}.jpg"
            s_path = os.path.join(out_dir, s_name)
            cv2.imwrite(s_path, frame)
            shutil.copy(s_path, os.path.join(art_dir, s_name))
            walkthrough_samples.append(s_name)
        current_idx += 1
    cap.release()
    print(f"[Walkthrough] Extracted {len(walkthrough_samples)} keyframes from walkthrough video.")

    # =========================================================================
    # 3. Pruning Analysis & Compression Measurement for M3 Web Viewer
    # =========================================================================
    print("\n[Pruning Analysis] Evaluating opacity pruning thresholds...")
    thresholds = [0.01, 0.05, 0.10, 0.15]
    pruning_results = {}
    
    for th in thresholds:
        mask = raw_opacities >= th
        remaining = int(np.sum(mask))
        pruned = n_total - remaining
        pct_pruned = (pruned / n_total) * 100.0
        pruning_results[str(th)] = {
            "remaining": remaining,
            "pruned": pruned,
            "pct_pruned": pct_pruned
        }
        print(f"  - Threshold >= {th:.2f}: {remaining:,} Gaussians remain ({pruned:,} pruned, {pct_pruned:.1f}%)")

    th_eval = 0.05
    keep_indices = np.where(raw_opacities >= th_eval)[0]
    pruned_ply_path = os.path.join("output/splats_double", "double_room_pruned_05.ply")
    
    filtered_vertex_data = v[keep_indices]
    el = plyfile.PlyElement.describe(filtered_vertex_data, 'vertex')
    plyfile.PlyData([el]).write(pruned_ply_path)
    
    orig_ply_size = os.path.getsize(ply_path) / (1024 * 1024)
    pruned_ply_size = os.path.getsize(pruned_ply_path) / (1024 * 1024)

    pruned_gz_path = pruned_ply_path + ".gz"
    with open(pruned_ply_path, "rb") as f_in, gzip.open(pruned_gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    pruned_gz_size = os.path.getsize(pruned_gz_path) / (1024 * 1024)

    splat_format_size = (len(keep_indices) * 32) / (1024 * 1024)
    splat_compressed_est = splat_format_size * 0.45

    results_summary = {
        "total_gaussians": n_total,
        "original_ply_mb": round(orig_ply_size, 2),
        "pruning_thresholds": pruning_results,
        "recommended_pruned_count": len(keep_indices),
        "pruned_ply_mb": round(pruned_ply_size, 2),
        "pruned_ply_gzip_mb": round(pruned_gz_size, 2),
        "web_splat_format_mb": round(splat_format_size, 2),
        "web_splat_gzip_mb": round(splat_compressed_est, 2)
    }

    results_file = "output/splats_double/pruning_and_compression_stats.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n[Summary of Compression & Pruning]")
    print(f"  - Original Splat: {orig_ply_size:.2f} MB ({n_total:,} Gaussians)")
    print(f"  - Pruned Splat (th >= 0.05): {pruned_ply_size:.2f} MB ({len(keep_indices):,} Gaussians)")
    print(f"  - Pruned PLY + Gzip: {pruned_gz_size:.2f} MB")
    print(f"  - WebGL2 .splat format: {splat_format_size:.2f} MB")
    print(f"  - WebGL2 .splat + Gzip: {splat_compressed_est:.2f} MB")

if __name__ == "__main__":
    run_novel_renders_and_analysis()
