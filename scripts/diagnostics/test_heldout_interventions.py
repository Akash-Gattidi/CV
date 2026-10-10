import os
import sys
import json
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plyfile
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gsplat import rasterization, DefaultStrategy

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse.item())

def compute_ssim_val(pred, gt):
    from pipeline.gsplat_trainer import compute_ssim
    return compute_ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()

class ExperimentRunner:
    def __init__(self, cams_path="output/dust3r_double/cameras.json", points_path="output/dust3r_double/dust3r_points.ply"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(cams_path, "r") as f:
            all_cams = json.load(f)
        
        # Withhold every 5th frame for held-out test
        self.test_indices = [i for i in range(len(all_cams)) if i % 5 == 4]
        self.train_indices = [i for i in range(len(all_cams)) if i % 5 != 4]
        print(f"[Dataset] Total: {len(all_cams)} | Train: {len(self.train_indices)} | Test (Held-out): {len(self.test_indices)}")
        print(f"  - Held-out indices: {self.test_indices}")
        
        self.train_data = []
        for idx in self.train_indices:
            c = all_cams[idx]
            img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
            c2w = np.array(c["c2w"], dtype=np.float32)
            self.train_data.append({
                "id": idx,
                "img": torch.tensor(img, dtype=torch.float32, device=self.device),
                "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=self.device)[None],
                "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=self.device),
                "width": c["width"],
                "height": c["height"],
                "cam_pos": c2w[:3, 3]
            })
            
        self.test_data = []
        for idx in self.test_indices:
            c = all_cams[idx]
            img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
            c2w = np.array(c["c2w"], dtype=np.float32)
            self.test_data.append({
                "id": idx,
                "img": torch.tensor(img, dtype=torch.float32, device=self.device),
                "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=self.device)[None],
                "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=self.device),
                "width": c["width"],
                "height": c["height"],
                "cam_pos": c2w[:3, 3]
            })

        # Frame 33 for qualitative evaluation
        c33 = all_cams[33]
        img33 = cv2.cvtColor(cv2.imread(c33["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w33 = np.array(c33["c2w"], dtype=np.float32)
        self.frame33_data = {
            "id": 33,
            "img": torch.tensor(img33, dtype=torch.float32, device=self.device),
            "w2c": torch.tensor(np.linalg.inv(c2w33), dtype=torch.float32, device=self.device)[None],
            "K": torch.tensor([[[c33["fx"], 0, c33["cx"]], [0, c33["fy"], c33["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=self.device),
            "width": c33["width"],
            "height": c33["height"],
            "c2w": c2w33
        }
        
        # Load seeds
        ply_data = plyfile.PlyData.read(points_path)
        self.pts = np.stack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']], axis=-1).astype(np.float32)
        self.colors = (np.stack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)
        print(f"[Seeds] Loaded {len(self.pts):,} points from {points_path}")

        from scipy.spatial import KDTree
        tree = KDTree(self.pts)
        dists, _ = tree.query(self.pts, k=min(4, len(self.pts)))
        mean_dists = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.1)[:, None]
        self.init_log_scales = np.log(mean_dists).repeat(3, axis=-1).astype(np.float32)
        self.init_opacities = np.full((len(self.pts),), 1.386, dtype=np.float32)
        self.init_quats = np.zeros((len(self.pts), 4), dtype=np.float32)
        self.init_quats[:, 0] = 1.0
        self.scene_center = np.median(self.pts, axis=0)
        self.scene_scale = float(np.percentile(np.linalg.norm(self.pts - self.scene_center, axis=1), 90))

    def evaluate_model(self, params):
        psnrs = []
        ssims = []
        with torch.no_grad():
            for sample in self.test_data:
                pred, _, _ = rasterization(
                    means=params["means"],
                    quats=F.normalize(params["quats"], dim=-1),
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=params["colors"],
                    viewmats=sample["w2c"],
                    Ks=sample["K"],
                    width=sample["width"],
                    height=sample["height"],
                    near_plane=0.08,
                    packed=False
                )
                pred_img = pred[0].clamp(0.0, 1.0)
                gt_img = sample["img"]
                psnrs.append(compute_psnr(pred_img, gt_img))
                ssims.append(compute_ssim_val(pred_img, gt_img))
        return float(np.mean(psnrs)), float(np.mean(ssims))

    def render_frame33(self, params):
        with torch.no_grad():
            pred, _, _ = rasterization(
                means=params["means"],
                quats=F.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors"],
                viewmats=self.frame33_data["w2c"],
                Ks=self.frame33_data["K"],
                width=self.frame33_data["width"],
                height=self.frame33_data["height"],
                near_plane=0.08,
                packed=False
            )
        img = (pred[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def train_experiment(
        self,
        exp_name: str,
        total_iters: int = 4000,
        grow_grad2d: float = 0.00035,
        prune_opa: float = 0.02,
        refine_start: int = 200,
        refine_stop: int = 2000,
        reset_every: int = 1200,
        prune_cam_tube: bool = False,
        prune_max_scale: bool = False,
        reg_scale_weight: float = 0.0,
        reg_opa_weight: float = 0.0
    ):
        print(f"\n=================================================================")
        print(f"Running Experiment: {exp_name}")
        print(f"=================================================================")
        
        # Init params
        params = nn.ParameterDict({
            "means": nn.Parameter(torch.tensor(self.pts.copy(), dtype=torch.float32, device=self.device)),
            "scales": nn.Parameter(torch.tensor(self.init_log_scales.copy(), dtype=torch.float32, device=self.device)),
            "quats": nn.Parameter(torch.tensor(self.init_quats.copy(), dtype=torch.float32, device=self.device)),
            "opacities": nn.Parameter(torch.tensor(self.init_opacities.copy(), dtype=torch.float32, device=self.device)),
            "colors": nn.Parameter(torch.tensor(self.colors.copy(), dtype=torch.float32, device=self.device)),
        })

        optimizers = {
            "means": torch.optim.Adam([params["means"]], lr=1.6e-4, eps=1e-15),
            "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
            "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
            "opacities": torch.optim.Adam([params["opacities"]], lr=0.05, eps=1e-15),
            "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3, eps=1e-15),
        }

        strategy = DefaultStrategy(
            prune_opa=prune_opa,
            grow_grad2d=grow_grad2d,
            grow_scale3d=0.01,
            refine_start_iter=refine_start,
            refine_stop_iter=refine_stop,
            refine_every=100,
            reset_every=reset_every,
            absgrad=True,
            verbose=False
        )
        strategy_state = strategy.initialize_state(scene_scale=self.scene_scale)

        # Precompute camera trajectory for camera tube pruning
        train_cam_positions = torch.tensor(np.array([c["cam_pos"] for c in self.train_data]), dtype=torch.float32, device=self.device)

        t0 = time.time()
        for step in range(total_iters):
            lr_factor = math.exp(-2.0 * step / total_iters)
            optimizers["means"].param_groups[0]["lr"] = 1.6e-4 * max(0.05, lr_factor)

            sample = self.train_data[step % len(self.train_data)]
            render_colors, _, info = rasterization(
                means=params["means"],
                quats=F.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors"],
                viewmats=sample["w2c"],
                Ks=sample["K"],
                width=sample["width"],
                height=sample["height"],
                near_plane=0.08,
                packed=True,
                absgrad=True
            )

            pred_img = render_colors[0]
            gt_img = sample["img"]
            l1_loss = F.l1_loss(pred_img, gt_img)
            ssim_val = compute_ssim_val(pred_img, gt_img)
            loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

            # Regularization (3c)
            if reg_scale_weight > 0:
                loss = loss + reg_scale_weight * torch.mean(torch.exp(params["scales"]))
            if reg_opa_weight > 0:
                loss = loss + reg_opa_weight * torch.mean(torch.sigmoid(params["opacities"]))

            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            # Intervention 3b: Prune Gaussians very close to camera path or excessive scale during refine steps
            if (prune_cam_tube or prune_max_scale) and (step in range(refine_start, refine_stop, 200)):
                with torch.no_grad():
                    n_before = len(params["means"])
                    keep_mask = torch.ones(n_before, dtype=torch.bool, device=self.device)
                    
                    if prune_cam_tube:
                        # Find minimum distance to any camera position
                        # Batch compute distance
                        p = params["means"].detach()
                        # compute dist to camera trajectory in chunks
                        dists = torch.cdist(p, train_cam_positions) # (N, n_cams)
                        min_dist_to_cam = torch.min(dists, dim=1).values
                        # Camera tube threshold: 0.025 units
                        keep_mask = keep_mask & (min_dist_to_cam > 0.025)

                    if prune_max_scale:
                        max_s = torch.max(torch.exp(params["scales"].detach()), dim=-1).values
                        keep_mask = keep_mask & (max_s < 0.05)

                    if keep_mask.sum() < n_before:
                        pruned_count = n_before - keep_mask.sum().item()
                        for k in params.keys():
                            params[k] = nn.Parameter(params[k][keep_mask])
                        # re-init optimizers
                        optimizers = {
                            "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * max(0.05, lr_factor), eps=1e-15),
                            "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
                            "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
                            "opacities": torch.optim.Adam([params["opacities"]], lr=0.05, eps=1e-15),
                            "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3, eps=1e-15),
                        }
                        strategy_state = strategy.initialize_state(scene_scale=self.scene_scale)

        duration = time.time() - t0
        test_psnr, test_ssim = self.evaluate_model(params)
        final_gs = len(params["means"])
        print(f"Results for [{exp_name}]:")
        print(f"  Duration: {duration:.1f}s | Final GS: {final_gs:,}")
        print(f"  Held-out Test PSNR: {test_psnr:.2f} dB | Held-out SSIM: {test_ssim:.4f}")

        # Render Frame 33
        img33 = self.render_frame33(params)
        return {
            "name": exp_name,
            "duration": duration,
            "final_gs": final_gs,
            "test_psnr": test_psnr,
            "test_ssim": test_ssim,
            "img33": img33
        }

if __name__ == "__main__":
    runner = ExperimentRunner()
    results = []

    # 1. Baseline (Current M2 configuration)
    r_base = runner.train_experiment(
        exp_name="Baseline (M2 current)",
        total_iters=4000,
        grow_grad2d=0.00035,
        prune_opa=0.02,
        refine_start=200,
        refine_stop=2000,
        reset_every=1200
    )
    results.append(r_base)
    cv2.imwrite("output/renders_double/exp_baseline_f33.jpg", r_base["img33"])

    # 2. Intervention 3b: Prune large scale & camera tube
    r_3b = runner.train_experiment(
        exp_name="Intervention 3b (Prune Camera Tube & Max Scale)",
        total_iters=4000,
        grow_grad2d=0.00035,
        prune_opa=0.02,
        refine_start=200,
        refine_stop=2000,
        reset_every=1200,
        prune_cam_tube=True,
        prune_max_scale=True
    )
    results.append(r_3b)
    cv2.imwrite("output/renders_double/exp_3b_f33.jpg", r_3b["img33"])

    # 3. Intervention 3c: Scale and opacity regularization
    r_3c = runner.train_experiment(
        exp_name="Intervention 3c (Scale & Opacity Regularization)",
        total_iters=4000,
        grow_grad2d=0.00035,
        prune_opa=0.02,
        refine_start=200,
        refine_stop=2000,
        reset_every=1200,
        reg_scale_weight=0.01,
        reg_opa_weight=0.005
    )
    results.append(r_3c)
    cv2.imwrite("output/renders_double/exp_3c_f33.jpg", r_3c["img33"])

    # 4. Intervention 3d: Opacity reset tuned for large scene & finer densification
    r_3d = runner.train_experiment(
        exp_name="Intervention 3d (M1-style Schedule: grad2d=0.0002, 80% refine)",
        total_iters=4000,
        grow_grad2d=0.0002,
        prune_opa=0.008,
        refine_start=200,
        refine_stop=3200,
        reset_every=1000
    )
    results.append(r_3d)
    cv2.imwrite("output/renders_double/exp_3d_f33.jpg", r_3d["img33"])

    # Summary table
    print("\n=================================================================")
    print("EXPERIMENT COMPARISON TABLE (Held-out Test Set: 13 withheld frames)")
    print("=================================================================")
    print(f"{'Experiment':<45} | {'Final GS':<10} | {'Test PSNR':<10} | {'Test SSIM':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<45} | {r['final_gs']:<10,} | {r['test_psnr']:<10.2f} | {r['test_ssim']:<10.4f}")

    # Save results to json
    clean_results = [{k: v for k, v in r.items() if k != "img33"} for r in results]
    with open("output/renders_double/interventions_heldout_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
