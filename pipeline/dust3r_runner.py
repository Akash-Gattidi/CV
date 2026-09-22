import os
import sys
import json
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
    with strict memory optimizations to remain within 4GB VRAM.
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
        voxel_downsample: float = 0.02
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

        # 3. Run pairwise inference in fp16, batch_size=1 to ensure peak VRAM < 3.5GB
        with torch.no_grad():
            output = inference(
                pairs,
                self.model,
                self.device,
                batch_size=1,
                verbose=True
            )


        # Clear PyTorch caching to free activation memory before global alignment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[DUSt3R] Pairwise inference complete. Starting global alignment ({n_iters} iterations)...")

        # 4. Global alignment: solves for globally consistent camera poses and point scale
        # Use GlobalAlignerMode.PointCloudOptimizer or pairwise optimizer
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

            # Filter dense 3D points by confidence
            p3d_tensor = pts3d[idx]
            p3d = p3d_tensor.detach().cpu().numpy() if torch.is_tensor(p3d_tensor) else np.array(p3d_tensor)
            conf_tensor = confs[idx]
            conf = conf_tensor.detach().cpu().numpy() if torch.is_tensor(conf_tensor) else np.array(conf_tensor)
            rgb = (images[idx]["img"][0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0  # (H, W, 3) in [0, 1]


            mask = conf > self.confidence_threshold
            valid_p3d = p3d[mask]
            valid_rgb = rgb[mask]

            if len(valid_p3d) > 0:
                all_points.append(valid_p3d)
                all_colors.append(valid_rgb)

        cameras_json_path = os.path.join(output_dir, "cameras.json")
        with open(cameras_json_path, "w") as f:
            json.dump(camera_data, f, indent=2)

        merged_pts = np.concatenate(all_points, axis=0).astype(np.float32)
        merged_colors = np.concatenate(all_colors, axis=0).astype(np.float32)
        print(f"[DUSt3R] Total confident 3D points before voxel downsampling: {len(merged_pts)}")

        # Voxel downsample point cloud
        if voxel_downsample > 0:
            from pipeline.splat_init import SplatInitializer
            init_helper = SplatInitializer()
            merged_pts, merged_colors = init_helper.voxel_downsample(merged_pts, merged_colors, voxel_downsample)
            print(f"[DUSt3R] Confident points after voxel downsampling ({voxel_downsample}m): {len(merged_pts)}")

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
