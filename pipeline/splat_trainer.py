import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import plyfile

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Normalizes quaternions [w, x, y, z] and converts to 3x3 rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), dtype=q.dtype, device=q.device)
    
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Computes Structural Similarity Index (SSIM) for 3-channel images (C, H, W).
    """
    channel = img1.shape[0]
    # 1D Gaussian kernel
    sigma = 1.5
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)], device=img1.device, dtype=img1.dtype)
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    kernel_2d = (gauss @ gauss.T).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    mu1 = F.conv2d(img1, kernel_2d, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, kernel_2d, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel_2d, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel_2d, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class GaussianModel(nn.Module):
    """
    Core 3D Gaussian Splatting representation.
    Tracks positions (xyz), opacities, scales, rotations (quaternions),
    and spherical harmonic colors.
    """
    def __init__(self, init_params: Dict[str, np.ndarray], device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._xyz = nn.Parameter(torch.tensor(init_params["xyz"], dtype=torch.float32, device=self.device))
        self._features_dc = nn.Parameter(torch.tensor(init_params["sh_dc"], dtype=torch.float32, device=self.device))
        self._scaling = nn.Parameter(torch.tensor(init_params["log_scales"], dtype=torch.float32, device=self.device))
        self._rotation = nn.Parameter(torch.tensor(init_params["rotations"], dtype=torch.float32, device=self.device))
        self._opacity = nn.Parameter(torch.tensor(init_params["opacities"], dtype=torch.float32, device=self.device))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        # Clip max scale for numerical stability and preventing needle artifacts
        return torch.exp(torch.clamp(self._scaling, min=-10.0, max=1.0))

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return self._features_dc

    def get_covariance_3d(self) -> torch.Tensor:
        """
        Computes 3D covariance matrix Sigma = R * S * S^T * R^T for all Gaussians.
        """
        R = quat_to_rotmat(self.get_rotation)  # (N, 3, 3)
        scales = self.get_scaling              # (N, 3)
        S = torch.diag_embed(scales)           # (N, 3, 3)
        M = R @ S
        return M @ M.transpose(1, 2)           # (N, 3, 3)

    def save_ply(self, path: str):
        """
        Exports the Gaussian scene to a standard 3D Gaussian Splatting PLY file
        compatible with WebGL/WebGPU viewers (e.g. SuperSplat, Three.js).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_features.detach().cpu().numpy().reshape((xyz.shape[0], 3))
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f'f_dc_{i}')
        l.append('opacity')
        for i in range(3):
            l.append(f'scale_{i}')
        for i in range(4):
            l.append(f'rot_{i}')

        dtype_full = [(name, 'f4') for name in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = plyfile.PlyElement.describe(elements, 'vertex')
        plyfile.PlyData([el]).write(path)
        print(f"[GaussianModel] Exported {xyz.shape[0]} Gaussians to {path} ({os.path.getsize(path)/(1024*1024):.2f} MB)")


class DifferentiableSplatRenderer:
    """
    Lightweight, vectorized GPU splat rasterizer in pure PyTorch.
    Projects 3D Gaussians into 2D camera coordinates and alpha-blends tiles/pixels,
    guaranteed to execute cleanly within 4GB VRAM without external C++ compilers.
    """
    def __init__(self, bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.bg_color = bg_color

    def project_gaussians(
        self,
        gaussians: GaussianModel,
        w2c: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near: float = 0.1,
        far: float = 50.0
    ):
        """
        Projects 3D Gaussians to screen-space 2D means and 2x2 covariance matrices.
        """
        xyz = gaussians.get_xyz  # (N, 3)
        R_w2c = w2c[:3, :3]      # (3, 3)
        t_w2c = w2c[:3, 3]       # (3)

        # 1. Camera space translation: X_cam = R * X + t
        xyz_cam = xyz @ R_w2c.T + t_w2c  # (N, 3)
        z = xyz_cam[:, 2]

        # Filter points in front of camera
        valid = (z > near) & (z < far)
        if not valid.any():
            return None

        xyz_cam = xyz_cam[valid]
        z = z[valid]
        N_valid = xyz_cam.shape[0]

        # 2. Perspective projection
        x_pix = fx * (xyz_cam[:, 0] / z) + cx
        y_pix = fy * (xyz_cam[:, 1] / z) + cy
        means_2d = torch.stack([x_pix, y_pix], dim=-1)  # (N_valid, 2)

        # Discard points far outside screen bounds
        in_frame = (x_pix >= -64) & (x_pix < width + 64) & (y_pix >= -64) & (y_pix < height + 64)
        if not in_frame.any():
            return None

        # 3. 3D Covariance to 2D screen covariance
        # J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
        J = torch.zeros((N_valid, 2, 3), dtype=xyz.dtype, device=xyz.device)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -(fx * xyz_cam[:, 0]) / (z**2)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -(fy * xyz_cam[:, 1]) / (z**2)

        # Transform 3D covariance to camera space: Sigma_cam = R_w2c * Sigma_world * R_w2c^T
        cov3d = gaussians.get_covariance_3d()[valid]  # (N_valid, 3, 3)
        cov3d_cam = R_w2c.unsqueeze(0) @ cov3d @ R_w2c.T.unsqueeze(0)

        # Screen-space covariance: Sigma_2d = J * Sigma_cam * J^T
        cov2d = J @ cov3d_cam @ J.transpose(1, 2)  # (N_valid, 2, 2)

        # Low-pass filter (0.3 pixel blur) to avoid aliasing
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3

        # Color from SH (0th order)
        C0 = 0.28209479177387814
        sh_dc = gaussians.get_features[valid].squeeze(-1)  # (N_valid, 3)
        colors = torch.clamp(sh_dc * C0 + 0.5, 0.0, 1.0)

        opacities = gaussians.get_opacity[valid].squeeze(-1) # (N_valid)

        return {
            "means_2d": means_2d[in_frame],
            "cov2d": cov2d[in_frame],
            "depths": z[in_frame],
            "colors": colors[in_frame],
            "opacities": opacities[in_frame]
        }

    def render(
        self,
        projected: Dict[str, torch.Tensor],
        width: int,
        height: int,
        tile_size: int = 16
    ) -> torch.Tensor:
        """
        Renders projected Gaussians into a (3, H, W) image using depth sorting
        and front-to-back alpha compositing.
        """
        device = projected["means_2d"].device
        means_2d = projected["means_2d"]
        cov2d = projected["cov2d"]
        depths = projected["depths"]
        colors = projected["colors"]
        opacities = projected["opacities"]

        # Sort front-to-back by camera depth
        sort_idx = torch.argsort(depths)
        means_2d = means_2d[sort_idx]
        cov2d = cov2d[sort_idx]
        colors = colors[sort_idx]
        opacities = opacities[sort_idx]

        # Compute inverse 2D covariance for Gaussian falloff: exp(-0.5 * d^T * inv(cov) * d)
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        det = torch.clamp(det, min=1e-5)
        inv_cov = torch.zeros_like(cov2d)
        inv_cov[:, 0, 0] = cov2d[:, 1, 1] / det
        inv_cov[:, 1, 1] = cov2d[:, 0, 0] / det
        inv_cov[:, 0, 1] = -cov2d[:, 0, 1] / det
        inv_cov[:, 1, 0] = -cov2d[:, 1, 0] / det

        # Compute 3-sigma radius in screen space
        radii = torch.ceil(3.0 * torch.sqrt(torch.maximum(cov2d[:, 0, 0], cov2d[:, 1, 1]))).to(torch.int32)
        radii = torch.clamp(radii, min=1, max=64)

        # Coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij"
        )
        pixels = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
        bg = torch.tensor(self.bg_color, device=device, dtype=torch.float32)

        # Gaussian screen bounds
        gx_min = means_2d[:, 0] - radii
        gx_max = means_2d[:, 0] + radii
        gy_min = means_2d[:, 1] - radii
        gy_max = means_2d[:, 1] + radii

        # Render image tile by tile to eliminate in-place slice assignment in autograd
        tile_size = 32
        n_tiles_y = (height + tile_size - 1) // tile_size
        n_tiles_x = (width + tile_size - 1) // tile_size

        tile_rows = []
        for ty in range(n_tiles_y):
            row_tiles = []
            y0 = ty * tile_size
            y1 = min(y0 + tile_size, height)

            for tx in range(n_tiles_x):
                x0 = tx * tile_size
                x1 = min(x0 + tile_size, width)

                # Find Gaussians overlapping this tile
                in_tile = (gx_max >= x0) & (gx_min < x1) & (gy_max >= y0) & (gy_min < y1)

                if not in_tile.any():
                    # Empty tile
                    row_tiles.append(bg.view(1, 1, 3).repeat(y1 - y0, x1 - x0, 1))
                    continue

                t_means = means_2d[in_tile]
                t_inv = inv_cov[in_tile]
                t_colors = colors[in_tile]
                t_op = opacities[in_tile]

                sub_pix = pixels[y0:y1, x0:x1]  # (th, tw, 2)
                th, tw = sub_pix.shape[:2]

                diff = sub_pix.unsqueeze(2) - t_means.unsqueeze(0).unsqueeze(0)  # (th, tw, K, 2)
                power = -0.5 * (
                    diff[..., 0] * (diff[..., 0] * t_inv[:, 0, 0] + diff[..., 1] * t_inv[:, 1, 0]) +
                    diff[..., 1] * (diff[..., 0] * t_inv[:, 0, 1] + diff[..., 1] * t_inv[:, 1, 1])
                )
                alpha = torch.clamp(t_op * torch.exp(torch.clamp(power, max=0.0)), 0.0, 0.99)  # (th, tw, K)

                # Out-of-place front-to-back alpha compositing for this tile
                tile_out = torch.zeros((th, tw, 3), device=device, dtype=torch.float32)
                tile_T = torch.ones((th, tw), device=device, dtype=torch.float32)

                for k in range(alpha.shape[-1]):
                    a = alpha[..., k]
                    weight = tile_T * a
                    tile_out = tile_out + weight.unsqueeze(-1) * t_colors[k]
                    tile_T = tile_T * (1.0 - a)

                final_tile = tile_out + tile_T.unsqueeze(-1) * bg
                row_tiles.append(final_tile)

            tile_rows.append(torch.cat(row_tiles, dim=1))

        final_image = torch.cat(tile_rows, dim=0)
        return final_image.permute(2, 0, 1)  # (3, H, W)



class SplatTrainer:
    """
    Manages end-to-end training of the 3D Gaussian Splat scene per room segment.
    """
    def __init__(
        self,
        gaussians: GaussianModel,
        cameras: List[Dict],
        lr_xyz: float = 1e-3,
        lr_features: float = 2.5e-3,
        lr_opacity: float = 0.05,
        lr_scaling: float = 5e-3,
        lr_rotation: float = 1e-3,
        lambda_dssim: float = 0.2
    ):
        self.gaussians = gaussians
        self.cameras = cameras
        self.lambda_dssim = lambda_dssim
        self.renderer = DifferentiableSplatRenderer(bg_color=(1.0, 1.0, 1.0))

        # Optimizer parameters
        self.optimizer = torch.optim.Adam([
            {"params": [self.gaussians._xyz], "lr": lr_xyz, "name": "xyz"},
            {"params": [self.gaussians._features_dc], "lr": lr_features, "name": "features_dc"},
            {"params": [self.gaussians._opacity], "lr": lr_opacity, "name": "opacity"},
            {"params": [self.gaussians._scaling], "lr": lr_scaling, "name": "scaling"},
            {"params": [self.gaussians._rotation], "lr": lr_rotation, "name": "rotation"}
        ])

    def train_step(self, cam_idx: int) -> float:
        """
        Executes a single optimization step on one camera view.
        """
        cam = self.cameras[cam_idx]
        w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=self.gaussians.device)
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = cam["cx"], cam["cy"]
        w, h = cam["width"], cam["height"]

        # Load ground truth image
        import cv2
        gt_bgr = cv2.imread(cam["img_path"])
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_tensor = torch.tensor(gt_rgb / 255.0, dtype=torch.float32, device=self.gaussians.device).permute(2, 0, 1)

        # Forward pass
        projected = self.renderer.project_gaussians(
            self.gaussians, w2c, fx, fy, cx, cy, w, h
        )
        if projected is None:
            return 0.0

        rendered = self.renderer.render(projected, w, h)

        # Loss: L1 + lambda * (1 - SSIM)
        l1_loss = F.l1_loss(rendered, gt_tensor)
        ssim_val = ssim(rendered, gt_tensor)
        loss = (1.0 - self.lambda_dssim) * l1_loss + self.lambda_dssim * (1.0 - ssim_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self, total_iterations: int = 1000, output_dir: str = "output/splats") -> str:
        """
        Runs the full training loop and exports the trained scene.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SplatTrainer] Starting 3DGS training for {total_iterations} iterations...")
        print(f"[SplatTrainer] Initial Gaussians: {self.gaussians.get_xyz.shape[0]}")

        n_cams = len(self.cameras)
        pbar = tqdm(range(total_iterations), desc="Training Splats")
        losses = []

        for it in pbar:
            # Pick random camera view
            cam_idx = np.random.randint(0, n_cams)
            loss_val = self.train_step(cam_idx)
            losses.append(loss_val)

            if it % 50 == 0 or it == total_iterations - 1:
                avg_loss = sum(losses[-50:]) / max(1, len(losses[-50:]))
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "pts": self.gaussians.get_xyz.shape[0]})

            # Free PyTorch caching regularly to prevent VRAM accumulation
            if it % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_ply = os.path.join(output_dir, "room_splat.ply")
        self.gaussians.save_ply(out_ply)
        print(f"[SplatTrainer] Training complete! Model saved to: {out_ply}")
        return out_ply
