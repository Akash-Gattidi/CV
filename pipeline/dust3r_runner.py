import os
import sys
import json
import math
import hashlib
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add dust3r repo and croco submodule to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


class DUSt3RGeometryEstimator:
    """
    Runs DUSt3R dense geometric reconstruction and camera pose estimation.
    Engineered to handle low-texture/blank walls where COLMAP fails,
    with strict memory optimizations designed for consumer low-VRAM GPUs.
    """
    def __init__(
        self,
        model_name: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        confidence_threshold: float = 1.5
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        print(f"[DUSt3R] Initializing model '{model_name}' on {self.device} (dtype: {self.dtype})...")
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        self.model.eval()
        print("[DUSt3R] Model initialized successfully.")

    def run_reconstruction(
        self,
        image_paths: List[str],
        output_dir: str = "output/dust3r",
        scene_graph_type: str = "consecutive",
        step: int = 1,
        n_iters: int = 200,
        voxel_downsample: Optional[float] = None,
        voxel_downsample_ratio: float = 0.012,
        stratified_sampling: bool = True,
        heldout_eval: bool = False,
        heldout_step: int = 5,
        dedup_voxel_size: Optional[float] = None,
        dedup_voxel_ratio: float = 0.004,
        use_cache: bool = True
    ) -> Dict:
        """
        Runs DUSt3R pairwise inference and global alignment across image_paths.
        Saves estimated camera poses, intrinsics, and aligned 3D point cloud.
        """
        os.makedirs(output_dir, exist_ok=True)
        n_images = len(image_paths)
        print(f"[DUSt3R] Running geometry estimation for {n_images} frames...")

        # 1. Load images with dust3r utilities (preserves dimensions)
        images = load_images(image_paths, size=512)

        # 2. Build memory-efficient scene graph (sliding window 3 along video sequence)
        pairs = make_pairs(images, scene_graph="swin-3-noncyclic", symmetrize=True)
        print(f"[DUSt3R] Formed {len(pairs)} pairwise graph edges for inference.")

        # 3. Content-addressed pairwise cache (keyed on model, graph, resolution, and image paths in pair order)
        cache_str = f"model={self.model_name}|graph={scene_graph_type}|res=512|" + "|".join(
            f"{os.path.basename(p)}:{os.path.getmtime(p) if os.path.exists(p) else 0}" for p in image_paths
        )
        cache_hash = hashlib.sha256(cache_str.encode("utf-8")).hexdigest()[:16]
        cache_path = os.path.join(output_dir, f"pairwise_cache_{cache_hash}.pt")
        legacy_cache_path = os.path.join(output_dir, "pairwise_cache.pt")

        if use_cache and os.path.exists(cache_path):
            print(f"[DUSt3R] Loading content-addressed pairwise cache ({cache_hash}) from {cache_path}...")
            output = torch.load(cache_path, map_location="cpu")
        elif use_cache and os.path.exists(legacy_cache_path) and len(image_paths) == 65:
            print(f"[DUSt3R] Loading legacy pairwise cache from {legacy_cache_path}...")
            output = torch.load(legacy_cache_path, map_location="cpu")
        else:
            with torch.no_grad():
                output = inference(
                    pairs,
                    self.model,
                    self.device,
                    batch_size=1,
                    verbose=True
                )
            if use_cache:
                try:
                    torch.save(output, cache_path)
                    print(f"[DUSt3R] Cached pairwise output to {cache_path}")
                except Exception as e:
                    print(f"[DUSt3R] Warning: could not cache pairwise output: {e}")

        # Clear PyTorch caching to free activation memory before global alignment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[DUSt3R] Pairwise inference complete. Starting global alignment ({n_iters} iterations)...")

        # 4. Global alignment: solves for globally consistent camera poses and point scale
        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=True
        )

        # Optimize camera poses and 3D points
        loss = scene.compute_global_alignment(
            init="mst",
            niter=n_iters,
            schedule="cosine",
            lr=0.01
        )
        print(f"[DUSt3R] Global alignment finished. Final alignment loss: {loss:.4f}")

        # 5. Extract camera poses & intrinsics
        poses = scene.get_im_poses()
        if torch.is_tensor(poses):
            poses = poses.detach().cpu().numpy()
        elif isinstance(poses, (list, tuple)):
            poses = np.array([p.detach().cpu().numpy() if torch.is_tensor(p) else np.array(p) for p in poses])

        focals = scene.get_focals()
        if torch.is_tensor(focals):
            focals = focals.detach().cpu().numpy()
        elif isinstance(focals, (list, tuple)):
            focals = [f.detach().cpu().numpy() if torch.is_tensor(f) else np.array(f) for f in focals]

        pps = scene.get_principal_points()
        if torch.is_tensor(pps):
            pps = pps.detach().cpu().numpy()
        elif isinstance(pps, (list, tuple)):
            pps = [p.detach().cpu().numpy() if torch.is_tensor(p) else np.array(p) for p in pps]

        pts3d = scene.get_pts3d()  # list of (H, W, 3)
        confs = scene.get_conf()   # list of (H, W)

        # Compute scene extent from estimated camera path length
        cam_positions = np.array([p[:3, 3] for p in poses]) if len(poses) > 0 else np.zeros((0, 3))
        if len(cam_positions) > 1:
            path_len = float(np.sum(np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)))
        else:
            path_len = 0.0

        if path_len > 1e-4:
            scene_extent = path_len
        else:
            # Fallback to bounding box diagonal of confident points
            all_valid_pts = []
            for p_t, c_t in zip(pts3d, confs):
                p_np = p_t.detach().cpu().numpy() if torch.is_tensor(p_t) else np.array(p_t)
                c_np = c_t.detach().cpu().numpy() if torch.is_tensor(c_t) else np.array(c_t)
                valid = p_np[c_np > self.confidence_threshold]
                if len(valid) > 0:
                    all_valid_pts.append(valid)
            if len(all_valid_pts) > 0:
                merged_v = np.concatenate(all_valid_pts, axis=0)
                scene_extent = float(np.linalg.norm(np.max(merged_v, axis=0) - np.min(merged_v, axis=0)))
            else:
                scene_extent = 1.0

        # Derive scene-relative spatial thresholds
        norm_cutoff = max(scene_extent * 5.0, 10.0)
        eff_dedup_voxel = dedup_voxel_size if dedup_voxel_size is not None else scene_extent * dedup_voxel_ratio
        eff_voxel_downsample = voxel_downsample if voxel_downsample is not None else scene_extent * voxel_downsample_ratio
        print(f"[DUSt3R] Scene extent: {scene_extent:.4f} units (path length: {path_len:.4f}). Relative thresholds: norm_cutoff={norm_cutoff:.1f}, dedup_voxel={eff_dedup_voxel:.5f}, downsample_voxel={eff_voxel_downsample:.5f}")

        camera_data = []
        all_points = []
        all_colors = []

        for idx, img_path in enumerate(image_paths):
            h, w = images[idx]["true_shape"][0].tolist()
            f_val = focals[idx]
            f = float(f_val[0]) if hasattr(f_val, "__len__") else float(f_val)
            cx, cy = float(pps[idx][0]), float(pps[idx][1])
            c2w = poses[idx].tolist() if hasattr(poses[idx], "tolist") else poses[idx]

            # World to Camera: inv(c2w)
            c2w_mat = np.array(c2w, dtype=np.float64)
            w2c_mat = np.linalg.inv(c2w_mat)

            cam_info = {
                "id": idx,
                "img_name": os.path.basename(img_path),
                "img_path": img_path,
                "width": int(w),
                "height": int(h),
                "fx": f,
                "fy": f,
                "cx": cx,
                "cy": cy,
                "c2w": c2w_mat.tolist(),
                "w2c": w2c_mat.tolist()
            }
            camera_data.append(cam_info)

            # If heldout_eval is enabled, withhold test frames from contributing 3D seeds
            if heldout_eval and (idx % heldout_step == (heldout_step - 1)):
                continue

            # Extract dense 3D points
            p3d_tensor = pts3d[idx]
            p3d = p3d_tensor.detach().cpu().numpy() if torch.is_tensor(p3d_tensor) else np.array(p3d_tensor)
            conf_tensor = confs[idx]
            conf = conf_tensor.detach().cpu().numpy() if torch.is_tensor(conf_tensor) else np.array(conf_tensor)
            rgb = (images[idx]["img"][0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0  # (H, W, 3) in [0, 1]

            if stratified_sampling:
                H_img, W_img = p3d.shape[:2]
                target_cells = 1600
                aspect = W_img / max(H_img, 1)
                grid_cols = max(1, int(round(math.sqrt(target_cells * aspect))))
                grid_rows = max(1, int(round(math.sqrt(target_cells / aspect))))
                row_bins = np.linspace(0, H_img, grid_rows + 1, dtype=int)
                col_bins = np.linspace(0, W_img, grid_cols + 1, dtype=int)
                for r in range(grid_rows):
                    r0, r1 = row_bins[r], row_bins[r + 1]
                    for c in range(grid_cols):
                        c0, c1 = col_bins[c], col_bins[c + 1]
                        cell_conf = conf[r0:r1, c0:c1]
                        if np.any(cell_conf >= 0.8):
                            lr, lc = np.unravel_index(np.argmax(cell_conf), cell_conf.shape)
                            best_y, best_x = r0 + lr, c0 + lc
                        elif np.any(cell_conf > 0.2):
                            lr, lc = np.unravel_index(np.argmax(cell_conf), cell_conf.shape)
                            best_y, best_x = r0 + lr, c0 + lc
                        else:
                            continue
                        pt = p3d[best_y, best_x]
                        if np.all(np.isfinite(pt)) and np.linalg.norm(pt) < norm_cutoff:
                            all_points.append(pt)
                            all_colors.append(rgb[best_y, best_x])
            else:
                mask = conf > self.confidence_threshold
                valid_p3d = p3d[mask]
                valid_rgb = rgb[mask]
                if len(valid_p3d) > 0:
                    for pt, col in zip(valid_p3d, valid_rgb):
                        if np.all(np.isfinite(pt)) and np.linalg.norm(pt) < norm_cutoff:
                            all_points.append(pt)
                            all_colors.append(col)

        cameras_json_path = os.path.join(output_dir, "cameras.json")
        with open(cameras_json_path, "w") as f:
            json.dump(camera_data, f, indent=2)

        merged_pts = np.array(all_points, dtype=np.float32)
        merged_colors = np.array(all_colors, dtype=np.float32)
        print(f"[DUSt3R] Total extracted 3D points before deduplication: {len(merged_pts):,}")

        if stratified_sampling and eff_dedup_voxel is not None and eff_dedup_voxel > 0 and len(merged_pts) > 0:
            coords = np.floor(merged_pts / eff_dedup_voxel).astype(np.int64)
            _, unique_idx = np.unique(coords, axis=0, return_index=True)
            merged_pts = merged_pts[unique_idx]
            merged_colors = merged_colors[unique_idx]
            print(f"[DUSt3R] Points after spatial deduplication ({eff_dedup_voxel:.4f}m grid): {len(merged_pts):,}")
        elif eff_voxel_downsample is not None and eff_voxel_downsample > 0 and len(merged_pts) > 0:
            from pipeline.splat_init import SplatInitializer
            init_helper = SplatInitializer()
            merged_pts, merged_colors = init_helper.voxel_downsample(merged_pts, merged_colors, eff_voxel_downsample)
            print(f"[DUSt3R] Confident points after voxel downsampling ({eff_voxel_downsample:.4f}m): {len(merged_pts):,}")

        # Save point cloud as PLY
        ply_path = os.path.join(output_dir, "dust3r_points.ply")
        self.save_ply(ply_path, merged_pts, (merged_colors * 255).astype(np.uint8))
        print(f"[DUSt3R] Saved point cloud to: {ply_path}")
        print(f"[DUSt3R] Saved camera poses to: {cameras_json_path}")

        return {
            "cameras_json": cameras_json_path,
            "ply_path": ply_path,
            "points": merged_pts,
            "colors": merged_colors,
            "cameras": camera_data
        }

    @staticmethod
    def save_ply(path: str, points: np.ndarray, colors: np.ndarray):
        """Saves XYZRGB point cloud to PLY file."""
        import plyfile
        vertex_data = np.empty(
            len(points),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]

        el = plyfile.PlyElement.describe(vertex_data, 'vertex')
        plyfile.PlyData([el]).write(path)
