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
import shutil

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gsplat import rasterization, DefaultStrategy
from pipeline.gsplat_trainer import compute_ssim

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return float(10.0 * math.log10(1.0 / max(mse.item(), 1e-10)))

def run_100k_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cams_path = "output/dust3r_double/cameras.json"
    points_path = "output/dust3r_double/dust3r_points_100k.ply"
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))

    with open(cams_path, "r") as f:
        all_cams = json.load(f)

    test_indices = [i for i in range(len(all_cams)) if i % 5 == 4]
    train_indices = [i for i in range(len(all_cams)) if i % 5 != 4]
    print(f"[Dataset] Total: {len(all_cams)} | Train: {len(train_indices)} | Test (Held-out): {len(test_indices)}")

    # Load stratified seeds
    ply_data = plyfile.PlyData.read(points_path)
    pts_all = np.stack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']], axis=-1).astype(np.float32)
    col_all = (np.stack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)
    print(f"[Seeds] Loaded {len(pts_all):,} stratified seed points from {points_path}")

    w, h = all_cams[0]["width"], all_cams[0]["height"]
    fx, fy, cx, cy = all_cams[0]["fx"], all_cams[0]["fy"], all_cams[0]["cx"], all_cams[0]["cy"]

    train_data = []
    train_cam_positions = []
    for idx in train_indices:
        c = all_cams[idx]
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w = np.array(c["c2w"], dtype=np.float32)
        train_cam_positions.append(c2w[:3, 3])
        train_data.append({
            "id": idx,
            "img": torch.tensor(img, dtype=torch.float32, device=device),
            "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
            "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=device),
            "width": c["width"], "height": c["height"]
        })
    train_cam_positions = np.array(train_cam_positions)

    test_data = []
    test_dists_to_train = []
    for idx in test_indices:
        c = all_cams[idx]
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w = np.array(c["c2w"], dtype=np.float32)
        pos = c2w[:3, 3]
        d_min = float(np.min(np.linalg.norm(train_cam_positions - pos, axis=1)))
        test_dists_to_train.append(d_min)
        test_data.append({
            "id": idx,
            "dist_to_train": d_min,
            "img": torch.tensor(img, dtype=torch.float32, device=device),
            "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
            "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=device),
            "width": c["width"], "height": c["height"]
        })

    from scipy.spatial import KDTree
    scene_center = np.median(pts_all, axis=0)
    scene_scale = float(np.percentile(np.linalg.norm(pts_all - scene_center, axis=1), 90))

    seeds = [42, 123, 999]
    all_runs = []
    best_params = None
    best_psnr = -1.0
    best_per_view_metrics = []

    for s in seeds:
        print(f"\n=======================================================")
        print(f"Training Seed {s} (5,000 iterations, M1 exact verified schedule)")
        print(f"=======================================================")
        np.random.seed(s)
        torch.manual_seed(s)

        tree = KDTree(pts_all)
        dists, _ = tree.query(pts_all, k=min(4, len(pts_all)))
        mean_d = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.1)[:, None]
        init_s = np.log(mean_d).repeat(3, axis=-1).astype(np.float32)
        init_op = np.full((len(pts_all),), 1.386, dtype=np.float32)
        init_q = np.zeros((len(pts_all), 4), dtype=np.float32)
        init_q[:, 0] = 1.0

        params = nn.ParameterDict({
            'means': nn.Parameter(torch.tensor(pts_all.copy(), dtype=torch.float32, device=device)),
            'scales': nn.Parameter(torch.tensor(init_s.copy(), dtype=torch.float32, device=device)),
            'quats': nn.Parameter(torch.tensor(init_q.copy(), dtype=torch.float32, device=device)),
            'opacities': nn.Parameter(torch.tensor(init_op.copy(), dtype=torch.float32, device=device)),
            'colors': nn.Parameter(torch.tensor(col_all.copy(), dtype=torch.float32, device=device)),
        })

        opts = {
            'means': torch.optim.Adam([params['means']], lr=1.6e-4, eps=1e-15),
            'scales': torch.optim.Adam([params['scales']], lr=5e-3, eps=1e-15),
            'quats': torch.optim.Adam([params['quats']], lr=1e-3, eps=1e-15),
            'opacities': torch.optim.Adam([params['opacities']], lr=0.05, eps=1e-15),
            'colors': torch.optim.Adam([params['colors']], lr=2.5e-3, eps=1e-15),
        }

        # Exact verified schedule: 50% refine, 50% anneal, grow_grad2d=0.00035, prune_opa=0.02
        strategy = DefaultStrategy(
            prune_opa=0.02,
            grow_grad2d=0.00035,
            grow_scale3d=0.01,
            refine_start_iter=200,
            refine_stop_iter=2500,
            refine_every=100,
            reset_every=1200,
            absgrad=True,
            verbose=False
        )
        st_state = strategy.initialize_state(scene_scale=scene_scale)

        t0 = time.time()
        for step in range(5000):
            lr_fac = math.exp(-2.0 * step / 5000)
            opts['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
            samp = train_data[step % len(train_data)]
            rc, _, info = rasterization(
                means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
                opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=samp['w2c'],
                Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True
            )
            l1 = F.l1_loss(rc[0], samp['img'])
            ss = compute_ssim(rc[0].unsqueeze(0), samp['img'].unsqueeze(0)).item()
            loss = 0.8 * l1 + 0.2 * (1.0 - ss)
            strategy.step_pre_backward(params, opts, st_state, step, info)
            loss.backward()
            strategy.step_post_backward(params, opts, st_state, step, info, packed=True)
            for opt in opts.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % 1000 == 0:
                print(f"  Step {step+1:4d}: Loss = {loss.item():.4f}, SSIM = {ss:.3f}, Gaussians = {len(params['means']):,}", flush=True)

        train_dur = time.time() - t0

        # Evaluate held-out test views
        psnrs, ssims = [], []
        run_view_metrics = []
        with torch.no_grad():
            for test_s in test_data:
                rc, _, _ = rasterization(
                    means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
                    opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=test_s['w2c'],
                    Ks=test_s['K'], width=test_s['width'], height=test_s['height'], near_plane=0.08, packed=False
                )
                pred = rc[0].clamp(0.0, 1.0)
                p_val = compute_psnr(pred, test_s['img'])
                s_val = compute_ssim(pred.unsqueeze(0), test_s['img'].unsqueeze(0)).item()
                psnrs.append(p_val)
                ssims.append(s_val)
                run_view_metrics.append({
                    "cam_id": test_s["id"],
                    "dist_to_train": test_s["dist_to_train"],
                    "psnr": p_val,
                    "ssim": s_val
                })

        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))
        final_gs = len(params['means'])
        print(f"--> Seed {s} Summary: Time = {train_dur:.1f}s | Final GS = {final_gs:,} | Held-out PSNR = {mean_psnr:.2f} dB | Held-out SSIM = {mean_ssim:.4f}")
        all_runs.append({
            "seed": s, "final_gs": final_gs, "heldout_psnr": mean_psnr, "heldout_ssim": mean_ssim
        })

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_params = {k: v.detach().clone() for k, v in params.items()}
            best_per_view_metrics = run_view_metrics

    # Analyze quality drop with distance from capture path
    dists = np.array([m["dist_to_train"] for m in best_per_view_metrics])
    p_scores = np.array([m["psnr"] for m in best_per_view_metrics])
    s_scores = np.array([m["ssim"] for m in best_per_view_metrics])

    # Quantify distance vs PSNR slope
    poly = np.polyfit(dists, p_scores, 1)
    slope = float(poly[0])
    corr = float(np.corrcoef(dists, p_scores)[0, 1])

    # Distance bins
    near_mask = dists < np.median(dists)
    far_mask = dists >= np.median(dists)

    dist_analysis = {
        "slope_dB_per_unit_distance": slope,
        "correlation": corr,
        "near_views": {
            "mean_dist": float(np.mean(dists[near_mask])),
            "mean_psnr": float(np.mean(p_scores[near_mask])),
            "mean_ssim": float(np.mean(s_scores[near_mask]))
        },
        "far_views": {
            "mean_dist": float(np.mean(dists[far_mask])),
            "mean_psnr": float(np.mean(p_scores[far_mask])),
            "mean_ssim": float(np.mean(s_scores[far_mask]))
        },
        "per_view_details": best_per_view_metrics
    }

    # Render Frame 33 and Off-Path Doorway View
    K_cam = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    # Frame 33
    c33 = all_cams[33]
    gt33 = cv2.imread(c33["img_path"])
    c2w33 = np.array(c33["c2w"], dtype=np.float32)
    w2c33_t = torch.tensor(np.linalg.inv(c2w33), dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        rc33, _, _ = rasterization(
            means=best_params['means'], quats=F.normalize(best_params['quats'], dim=-1), scales=torch.exp(best_params['scales']),
            opacities=torch.sigmoid(best_params['opacities']), colors=best_params['colors'], viewmats=w2c33_t,
            Ks=K_cam, width=w, height=h, near_plane=0.08, packed=False
        )
    f33_render = cv2.cvtColor((rc33[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    f33_side_by_side = np.hstack([gt33, f33_render])
    
    out_f33_local = "output/renders_double/frame_33_stratified_vs_gt.jpg"
    out_f33_art = os.path.join(art_dir, "frame_33_stratified_vs_gt.jpg")
    cv2.imwrite(out_f33_local, f33_side_by_side)
    shutil.copy(out_f33_local, out_f33_art)

    # Off-path doorway view:
    # Camera 34 & 36 frame the doorway crossing.
    # We take Camera 36 and introduce an off-path lateral offset of 0.02 units perpendicular to view direction
    c36 = all_cams[36]
    c2w36 = np.array(c36["c2w"], dtype=np.float64)
    rot36 = c2w36[:3, :3]
    # Lateral shift right by 0.020 units
    pos36_off = c2w36[:3, 3] + rot36[:, 0] * 0.020 + rot36[:, 1] * (-0.005)
    c_door_off = np.eye(4, dtype=np.float32)
    c_door_off[:3, :3] = rot36
    c_door_off[:3, 3] = pos36_off
    w2c_off_t = torch.tensor(np.linalg.inv(c_door_off), dtype=torch.float32, device=device)[None]

    gt36 = cv2.imread(c36["img_path"])
    with torch.no_grad():
        rc_off, _, _ = rasterization(
            means=best_params['means'], quats=F.normalize(best_params['quats'], dim=-1), scales=torch.exp(best_params['scales']),
            opacities=torch.sigmoid(best_params['opacities']), colors=best_params['colors'], viewmats=w2c_off_t,
            Ks=K_cam, width=w, height=h, near_plane=0.08, packed=False
        )
    door_off_render = cv2.cvtColor((rc_off[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    door_off_side_by_side = np.hstack([gt36, door_off_render])

    out_door_local = "output/renders_double/offpath_doorway_stratified_vs_gt.jpg"
    out_door_art = os.path.join(art_dir, "offpath_doorway_stratified_vs_gt.jpg")
    cv2.imwrite(out_door_local, door_off_side_by_side)
    shutil.copy(out_door_local, out_door_art)

    summary = {
        "baseline_7k": {
            "gaussians": 134105,
            "heldout_psnr": 20.33,
            "heldout_ssim": 0.6672
        },
        "stratified_100k_runs": all_runs,
        "mean_metrics": {
            "final_gs": int(np.mean([r["final_gs"] for r in all_runs])),
            "heldout_psnr": float(np.mean([r["heldout_psnr"] for r in all_runs])),
            "heldout_ssim": float(np.mean([r["heldout_ssim"] for r in all_runs])),
            "std_psnr": float(np.std([r["heldout_psnr"] for r in all_runs])),
            "std_ssim": float(np.std([r["heldout_ssim"] for r in all_runs]))
        },
        "distance_analysis": dist_analysis,
        "artifacts": {
            "frame_33": out_f33_art,
            "offpath_doorway": out_door_art
        }
    }

    out_summary_path = "output/renders_double/m2_100k_seeds_benchmark_summary.json"
    with open(out_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("      3-SEED STRATIFIED BENCHMARK COMPLETE             ")
    print("=======================================================")
    print(f"Baseline (7k seeds):        20.33 dB PSNR | 0.6672 SSIM (134k GS)")
    print(f"Stratified (100k seeds):    {summary['mean_metrics']['heldout_psnr']:.2f} +/- {summary['mean_metrics']['std_psnr']:.2f} dB PSNR | {summary['mean_metrics']['heldout_ssim']:.4f} +/- {summary['mean_metrics']['std_ssim']:.4f} SSIM ({summary['mean_metrics']['final_gs']:,} GS)")
    print(f"Quality drop with distance: {slope:.2f} dB / unit distance (r = {corr:.2f})")
    print(f"Artifacts saved to: {art_dir}")
    print("=======================================================\n")

if __name__ == "__main__":
    run_100k_benchmark()
