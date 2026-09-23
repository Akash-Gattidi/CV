import os
import sys
import json
import math
import time
import cv2
import numpy as np
import plyfile
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat import rasterization, DefaultStrategy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Computes SSIM between two (1, H, W, C) images in [0, 1].
    """
    img1 = img1.permute(0, 3, 1, 2)  # (1, C, H, W)
    img2 = img2.permute(0, 3, 1, 2)
    C = img1.shape[1]

    # Gaussian kernel
    sigma = 1.5
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)],
        dtype=torch.float32, device=img1.device
    )
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    kernel_2d = (gauss @ gauss.T).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    pad = window_size // 2
    mu1 = F.conv2d(img1, kernel_2d, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel_2d, padding=pad, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel_2d, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel_2d, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class GSplatTrainer:
    """
    Production-grade 3D Gaussian Splatting trainer utilizing gsplat's
    CUDA-accelerated rasterization and DefaultStrategy adaptive densification.
    """
    def __init__(
        self,
        cams_path: str = "output/dust3r/cameras.json",
        points_path: str = "output/dust3r/dust3r_points_dense.ply",
        fallback_points_path: str = "output/dust3r/dust3r_points.ply",
        device: str = "cuda",
        max_init_points: int = 100000
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[GSplatTrainer] Initializing on device: {self.device}")

        # 1. Load calibrated camera poses
        with open(cams_path, "r") as f:
            self.cams_meta = json.load(f)
        print(f"[GSplatTrainer] Loaded {len(self.cams_meta)} camera poses from {cams_path}")

        # Load training images into memory
        self.train_data = []
        for meta in self.cams_meta:
            img_bgr = cv2.imread(meta["img_path"])
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {meta['img_path']}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=self.device)  # (H, W, 3)

            c2w = np.array(meta["c2w"], dtype=np.float32)
            w2c = np.linalg.inv(c2w)
            w2c_t = torch.tensor(w2c, dtype=torch.float32, device=self.device)[None]  # (1, 4, 4)

            fx, fy = meta["fx"], meta["fy"]
            cx, cy = meta["cx"], meta["cy"]
            K_t = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=self.device)

            self.train_data.append({
                "img": img_t,
                "w2c": w2c_t,
                "K": K_t,
                "width": meta["width"],
                "height": meta["height"],
                "cam_pos": c2w[:3, 3]
            })

        # 2. Load seed points from DUSt3R
        ply_file = points_path if os.path.exists(points_path) else fallback_points_path
        print(f"[GSplatTrainer] Loading seed points from {ply_file}...")
        ply_data = plyfile.PlyData.read(ply_file)
        pts = np.stack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']], axis=-1).astype(np.float32)
        colors = (np.stack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)

        if len(pts) > max_init_points:
            idx = np.random.choice(len(pts), max_init_points, replace=False)
            pts = pts[idx]
            colors = colors[idx]
            print(f"[GSplatTrainer] Subsampled initial seeds to: {len(pts)}")
        else:
            print(f"[GSplatTrainer] Loaded {len(pts)} initial seed points.")

        # Compute initial scales via k-NN distance (k=3)
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=min(4, len(pts)))
        mean_dists = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.1)[:, None]
        init_log_scales = np.log(mean_dists).repeat(3, axis=-1).astype(np.float32)

        # Pre-activation opacity: logit(0.8) = 1.386
        init_opacities = np.full((len(pts),), 1.386, dtype=np.float32)

        # Quaternions: identity [1, 0, 0, 0]
        init_quats = np.zeros((len(pts), 4), dtype=np.float32)
        init_quats[:, 0] = 1.0

        # 3. Create Model Parameters
        self.params = nn.ParameterDict({
            "means": nn.Parameter(torch.tensor(pts, dtype=torch.float32, device=self.device)),
            "scales": nn.Parameter(torch.tensor(init_log_scales, dtype=torch.float32, device=self.device)),
            "quats": nn.Parameter(torch.tensor(init_quats, dtype=torch.float32, device=self.device)),
            "opacities": nn.Parameter(torch.tensor(init_opacities, dtype=torch.float32, device=self.device)),
            "colors": nn.Parameter(torch.tensor(colors, dtype=torch.float32, device=self.device)),
        })

        # Calculate scene scale for densification
        self.scene_center = np.median(pts, axis=0)
        self.scene_scale = float(np.percentile(np.linalg.norm(pts - self.scene_center, axis=1), 90))
        print(f"[GSplatTrainer] Scene Center: {self.scene_center}, Scene Scale: {self.scene_scale:.2f}m")

        # 4. Optimizers
        self.optimizers = {
            "means": torch.optim.Adam([self.params["means"]], lr=1.6e-4, eps=1e-15),
            "scales": torch.optim.Adam([self.params["scales"]], lr=5e-3, eps=1e-15),
            "quats": torch.optim.Adam([self.params["quats"]], lr=1e-3, eps=1e-15),
            "opacities": torch.optim.Adam([self.params["opacities"]], lr=0.05, eps=1e-15),
            "colors": torch.optim.Adam([self.params["colors"]], lr=2.5e-3, eps=1e-15),
        }

    def train(self, total_iterations: int = 15000, output_ply: str = "output/splats/room_splat.ply") -> str:
        os.makedirs(os.path.dirname(output_ply), exist_ok=True)
        print(f"\n[GSplatTrainer] Starting training for {total_iterations} iterations...")
        print(f"[GSplatTrainer] Initial Gaussians: {len(self.params['means']):,}")

        # 5. Adaptive Densification Strategy (DefaultStrategy) scaled to total_iterations
        refine_stop = int(total_iterations * 0.85)
        reset_iter = max(1000, total_iterations // 5)
        self.strategy = DefaultStrategy(
            prune_opa=0.005,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            refine_start_iter=500,
            refine_stop_iter=refine_stop,
            refine_every=100,
            reset_every=reset_iter,
            absgrad=True,
            verbose=False
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        print(f"[GSplatTrainer] Densification strategy: refine 500 -> {refine_stop}, reset every {reset_iter}, grow_grad2d=0.0002")

        start_time = time.time()
        pbar = tqdm(range(total_iterations), desc="Training gsplat")
        n_cams = len(self.train_data)

        for step in pbar:
            # Learning rate decay for means
            lr_factor = math.exp(-2.0 * step / total_iterations)
            for param_group in self.optimizers["means"].param_groups:
                param_group["lr"] = 1.6e-4 * lr_factor

            # Pick camera view
            cam_idx = step % n_cams
            cam = self.train_data[cam_idx]
            gt_img = cam["img"]

            # Forward rasterization with 0.1m near plane
            render_colors, render_alphas, info = rasterization(
                means=self.params["means"],
                quats=F.normalize(self.params["quats"], dim=-1),
                scales=torch.exp(self.params["scales"]),
                opacities=torch.sigmoid(self.params["opacities"]),
                colors=self.params["colors"],
                viewmats=cam["w2c"],
                Ks=cam["K"],
                width=cam["width"],
                height=cam["height"],
                near_plane=0.1,
                packed=True,
                absgrad=True
            )

            pred_img = render_colors[0]  # (H, W, 3)

            # Combined loss: 0.8 L1 + 0.2 (1 - SSIM)
            l1_loss = F.l1_loss(pred_img, gt_img)
            ssim_val = compute_ssim(pred_img.unsqueeze(0), gt_img.unsqueeze(0))
            loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

            # Densification pre-backward
            self.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state, step, info)

            # Backward
            loss.backward()

            # Densification post-backward (cloning, splitting, pruning)
            self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step, info, packed=True)

            # Optimizer step
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % 200 == 0 or step == total_iterations - 1:
                cur_n = len(self.params["means"])
                vram_mb = torch.cuda.memory_allocated() / (1024**2)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "SSIM": f"{ssim_val.item():.3f}",
                    "GS": f"{cur_n:,}",
                    "VRAM": f"{vram_mb:.0f}MB"
                })

        elapsed = time.time() - start_time
        speed = total_iterations / max(elapsed, 1e-4)
        print(f"\n[GSplatTrainer] Training completed in {elapsed:.1f}s ({speed:.1f} it/s)!")
        print(f"[GSplatTrainer] Final Gaussians after densification: {len(self.params['means']):,}")

        # Save to PLY
        self.save_ply(output_ply)
        return output_ply

    def save_ply(self, path: str):
        means = self.params["means"].detach().cpu().numpy()
        scales = self.params["scales"].detach().cpu().numpy()
        quats = F.normalize(self.params["quats"], dim=-1).detach().cpu().numpy()
        opacities = torch.sigmoid(self.params["opacities"]).detach().cpu().numpy()
        colors = self.params["colors"].detach().cpu().numpy()

        C0 = 0.28209479177387814
        sh0 = (colors - 0.5) / C0

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        vertex = np.empty(len(means), dtype=dtype)
        vertex['x'] = means[:, 0]
        vertex['y'] = means[:, 1]
        vertex['z'] = means[:, 2]
        vertex['f_dc_0'] = sh0[:, 0]
        vertex['f_dc_1'] = sh0[:, 1]
        vertex['f_dc_2'] = sh0[:, 2]
        vertex['opacity'] = opacities
        vertex['scale_0'] = scales[:, 0]
        vertex['scale_1'] = scales[:, 1]
        vertex['scale_2'] = scales[:, 2]
        vertex['rot_0'] = quats[:, 0]
        vertex['rot_1'] = quats[:, 1]
        vertex['rot_2'] = quats[:, 2]
        vertex['rot_3'] = quats[:, 3]

        el = plyfile.PlyElement.describe(vertex, 'vertex')
        plyfile.PlyData([el]).write(path)
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"[GSplatTrainer] Exported {len(means):,} Gaussians to {path} ({size_mb:.2f} MB)")


def create_lookat_w2c(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, -1, 0])) -> np.ndarray:
    forward = target - cam_pos
    norm_f = np.linalg.norm(forward)
    forward = forward / max(norm_f, 1e-6)
    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    right = right / max(norm_r, 1e-6)
    actual_up = np.cross(right, forward)
    R_c2w = np.stack([right, -actual_up, forward], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = cam_pos
    return np.linalg.inv(c2w)


def interp_pose(c2w1: np.ndarray, c2w2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    pos = (1 - alpha) * c2w1[:3, 3] + alpha * c2w2[:3, 3]
    R = (1 - alpha) * c2w1[:3, :3] + alpha * c2w2[:3, :3]
    u, _, vt = np.linalg.svd(R)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = u @ vt
    c2w[:3, 3] = pos
    return c2w


def render_gsplat_views_and_orbit(
    ply_path: str = "output/splats/room_splat.ply",
    cams_path: str = "output/dust3r/cameras.json",
    out_dir: str = "output/renders",
    orbit_frames: int = 120,
    device: str = "cuda"
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[GSplatRenderer] Loading trained splats from {ply_path}...")
    ply_data = plyfile.PlyData.read(ply_path)
    v = ply_data['vertex']

    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32, device=device)
    scales = torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1), dtype=torch.float32, device=device)
    opacities = torch.tensor(v['opacity'], dtype=torch.float32, device=device)

    C0 = 0.28209479177387814
    sh0 = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = torch.tensor(np.clip(sh0 * C0 + 0.5, 0.0, 1.0), dtype=torch.float32, device=device)

    with open(cams_path, "r") as f:
        cams = json.load(f)

    base = cams[0]
    fx, fy = base["fx"], base["fy"]
    cx, cy = base["cx"], base["cy"]
    w, h = base["width"], base["height"]
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    # 1. Aligned Ground Truth vs Splat Comparison Montage (8 views)
    print("[GSplatRenderer] Rendering 8 aligned Ground Truth vs Splat comparison frames...")
    test_indices = [5, 15, 25, 30, 40, 45, 50, 60]
    rendered_list = []
    gt_list = []

    for idx in test_indices:
        c = cams[idx]
        c2w = np.array(c['c2w'], dtype=np.float32)
        w2c_t = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None]

        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales), opacities=opacities, colors=colors,
                viewmats=w2c_t, Ks=K, width=w, height=h, near_plane=0.1, packed=False
            )
        pred = (render_colors[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rendered_list.append(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        gt = cv2.imread(c['img_path'])
        gt_list.append(gt)

    gt_grid = np.vstack([np.hstack(gt_list[:4]), np.hstack(gt_list[4:])])
    render_grid = np.vstack([np.hstack(rendered_list[:4]), np.hstack(rendered_list[4:])])
    full_comparison = np.vstack([gt_grid, render_grid])
    comp_path = os.path.join(out_dir, "full_recon_comparison.jpg")
    cv2.imwrite(comp_path, full_comparison)
    print(f"Saved aligned comparison montage to {comp_path}")

    # 2. Render 6 Clean Interior Novel Views
    print("[GSplatRenderer] Rendering 6 interior novel viewpoints...")
    novel_configs = [
        (49, 51, 0.5, "novel_view_01.jpg", "Bed Vantage (headboard, pillows, duvet)"),
        (28, 30, 0.5, "novel_view_02.jpg", "Wardrobe Arches (teal doors, handles, top cabinets)"),
        (42, 44, 0.5, "novel_view_03.jpg", "Study Desk & Overhead Glass Cabinet"),
        (37, 39, 0.5, "novel_view_04.jpg", "Window Curtains & Wall AC"),
        (60, 62, 0.5, "novel_view_05.jpg", "Closet Drawers, Niche, and Side Frame"),
        (8, 12, 0.5, "novel_view_06.jpg", "Diagonal Corridor Room Overview")
    ]
    novel_paths = []
    for idx_a, idx_b, alpha, fname, desc in novel_configs:
        c2w_a = np.array(cams[idx_a]['c2w'], dtype=np.float32)
        c2w_b = np.array(cams[idx_b]['c2w'], dtype=np.float32)
        c2w_nov = interp_pose(c2w_a, c2w_b, alpha)
        w2c_t = torch.tensor(np.linalg.inv(c2w_nov), dtype=torch.float32, device=device)[None]

        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales), opacities=opacities, colors=colors,
                viewmats=w2c_t, Ks=K, width=w, height=h, near_plane=0.1, packed=False
            )
        pred = (render_colors[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        novel_paths.append(out_path)
        print(f"Rendered novel view: {fname} - {desc}")

    # 3. Render 120-frame Walkthrough Animation
    print(f"\n[GSplatRenderer] Rendering {orbit_frames}-frame smooth walkthrough animation...")
    orbit_gif = os.path.join(out_dir, "room_orbit.gif")
    orbit_webp = os.path.join(out_dir, "room_orbit.webp")
    orbit_mp4 = os.path.join(out_dir, "room_orbit.mp4")

    all_c2ws = [np.array(c['c2w'], dtype=np.float32) for c in cams]
    walkthrough_frames = []
    video_writer = cv2.VideoWriter(orbit_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))

    for f_i in range(orbit_frames):
        t_global = (f_i / float(orbit_frames)) * (len(all_c2ws) - 1)
        i0 = int(t_global)
        i1 = min(i0 + 1, len(all_c2ws) - 1)
        frac = t_global - i0
        c2w_interp = interp_pose(all_c2ws[i0], all_c2ws[i1], frac)
        w2c_t = torch.tensor(np.linalg.inv(c2w_interp), dtype=torch.float32, device=device)[None]

        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales), opacities=opacities, colors=colors,
                viewmats=w2c_t, Ks=K, width=w, height=h, near_plane=0.1, packed=False
            )
        frame_np = (render_colors[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        walkthrough_frames.append(Image.fromarray(frame_np))
        video_writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

    video_writer.release()
    if walkthrough_frames:
        walkthrough_frames[0].save(orbit_gif, save_all=True, append_images=walkthrough_frames[1:], duration=42, loop=0)
        walkthrough_frames[0].save(orbit_webp, save_all=True, append_images=walkthrough_frames[1:], duration=42, loop=0)
        print(f"[GSplatRenderer] Walkthrough animation saved to {orbit_gif}, {orbit_webp}, and {orbit_mp4}")

    # Optionally synchronize renders if artifact directory is specified via env var
    art_dir = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
    if art_dir and os.path.exists(art_dir):
        import shutil
        for f in novel_paths:
            shutil.copy(f, os.path.join(art_dir, os.path.basename(f)))
        shutil.copy(comp_path, os.path.join(art_dir, os.path.basename(comp_path)))
        shutil.copy(orbit_gif, os.path.join(art_dir, os.path.basename(orbit_gif)))
        shutil.copy(orbit_webp, os.path.join(art_dir, os.path.basename(orbit_webp)))
        shutil.copy(orbit_mp4, os.path.join(art_dir, os.path.basename(orbit_mp4)))
        print("[GSplatRenderer] All renders and animations synchronized to artifact directory.")

    return novel_paths, orbit_gif


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=15000, help="Training iterations")
    parser.add_argument("--init_pts", type=int, default=100000, help="Max initial points")
    args = parser.parse_args()

    trainer = GSplatTrainer(max_init_points=args.init_pts)
    out_ply = trainer.train(total_iterations=args.iters)
    render_gsplat_views_and_orbit(ply_path=out_ply)
