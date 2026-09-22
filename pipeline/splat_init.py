import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

class SplatInitializer:
    """
    Initializes 3D Gaussian Splatting parameters directly from DUSt3R / MASt3R
    dense pointmaps and estimated camera poses, bypassing COLMAP entirely.
    """
    @staticmethod
    def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
        """
        Converts RGB [0, 1] to 0th-order Spherical Harmonics coefficient.
        C0 = 0.28209479177387814
        """
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    @staticmethod
    def compute_knn_scales(points: np.ndarray, k: int = 3, min_scale: float = 1e-4) -> np.ndarray:
        """
        Computes initial Gaussian log-scales using k-nearest neighbor mean distance.
        Uses a KD-Tree or chunked distance computation.
        """
        from scipy.spatial import KDTree
        tree = KDTree(points)
        # Query k+1 neighbors because the closest is the point itself (dist 0)
        dists, _ = tree.query(points, k=min(k + 1, len(points)))
        if dists.ndim == 1:
            mean_dist = np.full(len(points), min_scale)
        else:
            mean_dist = np.mean(dists[:, 1:], axis=1)
            mean_dist = np.clip(mean_dist, min_scale, 10.0)

        log_scales = np.log(mean_dist)[:, None]
        # Repeat for 3 axes (isotropic initialization)
        return np.repeat(log_scales, 3, axis=1)

    @staticmethod
    def voxel_downsample(
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsamples point cloud using a regular 3D voxel grid.
        Keeps Gaussian count well-conditioned for <= 4GB VRAM training.
        """
        if len(points) == 0:
            return points, colors

        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        # Use structured array or dict for fast hashing
        voxel_dict = {}
        for i, idx in enumerate(map(tuple, voxel_indices)):
            if idx not in voxel_dict:
                voxel_dict[idx] = []
            voxel_dict[idx].append(i)

        downsampled_pts = []
        downsampled_colors = []
        for indices in voxel_dict.values():
            downsampled_pts.append(np.mean(points[indices], axis=0))
            downsampled_colors.append(np.mean(colors[indices], axis=0))

        return np.array(downsampled_pts, dtype=np.float32), np.array(downsampled_colors, dtype=np.float32)

    def prepare_initial_gaussians(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.02,
        initial_opacity: float = 0.1,
        max_initial_points: int = 150000
    ) -> Dict[str, np.ndarray]:
        """
        Produces initial Gaussian parameters ready for training.
        """
        print(f"[SplatInit] Raw input points from DUSt3R: {len(points)}")

        # Step 1: Voxel downsample to control density and memory
        if voxel_size > 0.0:
            points, colors = self.voxel_downsample(points, colors, voxel_size)
            print(f"[SplatInit] Points after voxel downsampling ({voxel_size}m): {len(points)}")

        # Step 2: Cap max initial points if needed for 4GB VRAM safety
        if len(points) > max_initial_points:
            subsample_idx = np.random.choice(len(points), max_initial_points, replace=False)
            points = points[subsample_idx]
            colors = colors[subsample_idx]
            print(f"[SplatInit] Subsampled to safety cap: {len(points)} points")

        # Step 3: Compute k-NN scales
        log_scales = self.compute_knn_scales(points, k=3)

        # Step 4: Unit quaternions [w, x, y, z] = [1, 0, 0, 0]
        rotations = np.zeros((len(points), 4), dtype=np.float32)
        rotations[:, 0] = 1.0

        # Step 5: Initial opacities (inverse sigmoid)
        clamped_op = np.clip(initial_opacity, 1e-4, 1.0 - 1e-4)
        logit_opacity = np.log(clamped_op / (1.0 - clamped_op))
        opacities = np.full((len(points), 1), logit_opacity, dtype=np.float32)

        # Step 6: Spherical Harmonics DC
        sh_dc = self.rgb_to_sh0(colors)[:, :, None]  # (N, 3, 1)

        return {
            "xyz": points.astype(np.float32),
            "sh_dc": sh_dc.astype(np.float32),
            "log_scales": log_scales.astype(np.float32),
            "rotations": rotations.astype(np.float32),
            "opacities": opacities.astype(np.float32)
        }
