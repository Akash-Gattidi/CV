import os
import sys
import json
import glob
import time
import torch
import numpy as np

REPO_ROOT = os.path.abspath('.')
DUST3R_PATH = os.path.join(REPO_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images

def extract_100k_seeds(
    frames_dir: str = "output/frames_double",
    output_dir: str = "output/dust3r_double",
    target_points_per_cam: int = 1600,
    n_iters: int = 60
):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    print(f"[DUSt3R 100k] Loading {len(frame_paths)} keyframes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_images(frame_paths, size=512)
    pairs = make_pairs(images, scene_graph="swin-3-noncyclic", symmetrize=True)
    print(f"[DUSt3R 100k] Scene graph has {len(pairs)} edges.")
    
    cache_path = os.path.join(output_dir, "pairwise_cache.pt")
    if os.path.exists(cache_path):
        print(f"[DUSt3R 100k] Loading pairwise cache from {cache_path}...")
        output = torch.load(cache_path, map_location="cpu")
    else:
        print("[DUSt3R 100k] Initializing DUSt3R ViTLarge model (fp32 weights, fp16 autocast)...")
        model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            output = inference(pairs, model, device, batch_size=1, verbose=True)
        print(f"[DUSt3R 100k] Pairwise inference completed in {time.time()-t0:.1f}s")
        torch.save(output, cache_path)
        print(f"[DUSt3R 100k] Cached pairwise output to {cache_path}")
        del model
        torch.cuda.empty_cache()

    print(f"[DUSt3R 100k] Running global alignment ({n_iters} iterations)...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=True)
    loss = scene.compute_global_alignment(init="mst", niter=n_iters, schedule="cosine", lr=0.01)
    print(f"[DUSt3R 100k] Global alignment finished with loss: {loss:.4f}")

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
    per_cam_counts = []
    camera_data = []

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

        # Filter by confidence > 1.5
        mask = conf > 1.5
        valid_p = p3d[mask]
        valid_c = rgb[mask]
        valid_conf = conf[mask]

        # Uniformly distribute points across cameras by selecting top confident points up to target
        if len(valid_p) > target_points_per_cam:
            top_k_idx = np.argsort(valid_conf)[-target_points_per_cam:]
            sel_p = valid_p[top_k_idx]
            sel_c = valid_c[top_k_idx]
        else:
            sel_p = valid_p
            sel_c = valid_c

        per_cam_counts.append(len(sel_p))
        all_points.append(sel_p)
        all_colors.append(sel_c)

    # Save cameras.json
    with open(os.path.join(output_dir, "cameras.json"), "w") as f:
        json.dump(camera_data, f, indent=2)

    merged_pts = np.concatenate(all_points, axis=0).astype(np.float32)
    merged_colors = np.concatenate(all_colors, axis=0).astype(np.float32)

    # Save to dust3r_points_100k.ply
    out_ply = os.path.join(output_dir, "dust3r_points_100k.ply")
    import plyfile
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

    print(f"\n[DUSt3R 100k] Successfully generated {len(merged_pts):,} distributed seed points in {out_ply}")
    print(f"[DUSt3R 100k] Per-camera seed stats:")
    print(f"  - Minimum seeds per camera: {min(per_cam_counts)}")
    print(f"  - Maximum seeds per camera: {max(per_cam_counts)}")
    print(f"  - Median seeds per camera: {int(np.median(per_cam_counts))}")
    print(f"  - Frame 33 seeds: {per_cam_counts[33]}")

if __name__ == "__main__":
    extract_100k_seeds()
