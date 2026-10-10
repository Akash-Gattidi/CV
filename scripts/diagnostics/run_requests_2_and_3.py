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
from scipy.spatial import KDTree

PROJECT_ROOT = os.path.abspath('.')
DUST3R_PATH = os.path.join(PROJECT_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if DUST3R_PATH not in sys.path: sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path: sys.path.insert(0, CROCO_PATH)

from gsplat import rasterization, DefaultStrategy
from pipeline.gsplat_trainer import compute_ssim

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    if mse == 0: return 100.0
    return float(10.0 * math.log10(1.0 / max(mse.item(), 1e-10)))

def pearson_depth_loss(pred_depth, gt_depth, mask):
    if mask.sum() < 100:
        return torch.tensor(0.0, device=pred_depth.device)
    p = pred_depth[mask]
    g = gt_depth[mask]
    p_m = p - p.mean()
    g_m = g - g.mean()
    cov = (p_m * g_m).sum()
    var_p = (p_m ** 2).sum()
    var_g = (g_m ** 2).sum()
    r = cov / (torch.sqrt(var_p * var_g) + 1e-7)
    return 1.0 - r

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))
    os.makedirs("output/renders_double", exist_ok=True)
    os.makedirs("output/splats", exist_ok=True)

    with open("output/dust3r_double/cameras.json", "r") as f:
        double_cams = json.load(f)

    test_indices = set([i for i in range(len(double_cams)) if i % 5 == 4])
    train_indices = [i for i in range(len(double_cams)) if i % 5 != 4]

    train_data = []
    train_cam_positions = []
    for idx in train_indices:
        c = double_cams[idx]
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
    for idx in sorted(list(test_indices)):
        c = double_cams[idx]
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w = np.array(c["c2w"], dtype=np.float32)
        pos = c2w[:3, 3]
        d_min = float(np.min(np.linalg.norm(train_cam_positions - pos, axis=1)))
        test_data.append({
            "id": idx,
            "dist": d_min,
            "img": torch.tensor(img, dtype=torch.float32, device=device),
            "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
            "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=device),
            "width": c["width"], "height": c["height"]
        })

    # Load 36k deduplicated seeds
    ply = plyfile.PlyData.read("output/dust3r_double/dust3r_points_train_only_dedup.ply")
    dedup_pts = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)
    dedup_col = np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1).astype(np.float32) / 255.0

    w_d, h_d = double_cams[0]["width"], double_cams[0]["height"]
    fx_d, fy_d, cx_d, cy_d = double_cams[0]["fx"], double_cams[0]["fy"], double_cams[0]["cx"], double_cams[0]["cy"]
    K_d = torch.tensor([[[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    # =========================================================================
    # TASK 2: RENDER FRAME 33 FROM CLEAN 7K BASELINE
    # =========================================================================
    print("======================================================================", flush=True)
    print("  TASK 2: TRAIN CLEAN 7K BASELINE & RENDER FRAME 33", flush=True)
    print("======================================================================", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    idx_7k = np.random.choice(len(dedup_pts), min(7407, len(dedup_pts)), replace=False)
    pts_7k = dedup_pts[idx_7k]
    col_7k = dedup_col[idx_7k]

    tree_7k = KDTree(pts_7k)
    dists_7k, _ = tree_7k.query(pts_7k, k=4)
    mean_d_7k = np.clip(np.mean(dists_7k[:, 1:], axis=1), 1e-4, 0.1)[:, None]
    init_s_7k = np.log(mean_d_7k).repeat(3, axis=-1).astype(np.float32)
    init_op_7k = np.full((len(pts_7k),), 1.386, dtype=np.float32)
    init_q_7k = np.zeros((len(pts_7k), 4), dtype=np.float32); init_q_7k[:, 0] = 1.0
    scene_center_7k = np.median(pts_7k, axis=0)
    scene_scale_7k = float(np.percentile(np.linalg.norm(pts_7k - scene_center_7k, axis=1), 90))

    params_7k = nn.ParameterDict({
        'means': nn.Parameter(torch.tensor(pts_7k, dtype=torch.float32, device=device)),
        'scales': nn.Parameter(torch.tensor(init_s_7k, dtype=torch.float32, device=device)),
        'quats': nn.Parameter(torch.tensor(init_q_7k, dtype=torch.float32, device=device)),
        'opacities': nn.Parameter(torch.tensor(init_op_7k, dtype=torch.float32, device=device)),
        'colors': nn.Parameter(torch.tensor(col_7k, dtype=torch.float32, device=device)),
    })
    opts_7k = {
        'means': torch.optim.Adam([params_7k['means']], lr=1.6e-4, eps=1e-15),
        'scales': torch.optim.Adam([params_7k['scales']], lr=5e-3, eps=1e-15),
        'quats': torch.optim.Adam([params_7k['quats']], lr=1e-3, eps=1e-15),
        'opacities': torch.optim.Adam([params_7k['opacities']], lr=0.05, eps=1e-15),
        'colors': torch.optim.Adam([params_7k['colors']], lr=2.5e-3, eps=1e-15),
    }
    strat_7k = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
        refine_start_iter=200, refine_stop_iter=2500, refine_every=100, reset_every=1200, absgrad=True, verbose=False
    )
    st_state_7k = strat_7k.initialize_state(scene_scale=scene_scale_7k)

    print("Training clean 7k baseline (5,000 iters)...", flush=True)
    t0 = time.time()
    for step in range(5000):
        lr_fac = math.exp(-2.0 * step / 5000)
        opts_7k['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
        samp = train_data[step % len(train_data)]
        rc, _, info = rasterization(
            means=params_7k['means'], quats=F.normalize(params_7k['quats'], dim=-1), scales=torch.exp(params_7k['scales']),
            opacities=torch.sigmoid(params_7k['opacities']), colors=params_7k['colors'], viewmats=samp['w2c'],
            Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True
        )
        loss = 0.8 * F.l1_loss(rc[0], samp['img']) + 0.2 * (1.0 - compute_ssim(rc[0].unsqueeze(0), samp['img'].unsqueeze(0)).item())
        strat_7k.step_pre_backward(params_7k, opts_7k, st_state_7k, step, info)
        loss.backward()
        strat_7k.step_post_backward(params_7k, opts_7k, st_state_7k, step, info, packed=True)
        for opt in opts_7k.values(): opt.step(); opt.zero_grad(set_to_none=True)

    print(f"7k baseline trained in {time.time()-t0:.1f}s | Final Gaussians: {len(params_7k['means']):,}", flush=True)

    # Render Frame 33 from 7k baseline
    c33 = double_cams[33]
    gt33 = cv2.imread(c33["img_path"])
    c2w33 = np.array(c33["c2w"], dtype=np.float32)
    w2c33_t = torch.tensor(np.linalg.inv(c2w33), dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        rc33_7k, _, _ = rasterization(
            means=params_7k['means'], quats=F.normalize(params_7k['quats'], dim=-1), scales=torch.exp(params_7k['scales']),
            opacities=torch.sigmoid(params_7k['opacities']), colors=params_7k['colors'], viewmats=w2c33_t,
            Ks=K_d, width=w_d, height=h_d, near_plane=0.08, packed=False
        )
    f33_render_7k = cv2.cvtColor((rc33_7k[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Load 36k seed Frame 33 render
    f33_render_36k_path = "output/renders_double/frame_33_stratified_vs_gt.jpg"
    if os.path.exists(f33_render_36k_path):
        f33_prev = cv2.imread(f33_render_36k_path)
        f33_render_36k = f33_prev[:, w_d:2*w_d] # right half was model
    else:
        f33_render_36k = f33_render_7k.copy()

    # Create 3-way side-by-side: Ground Truth | Clean 7k Baseline | 36k Deduplicated Seeds
    tri_f33 = np.hstack([gt33, f33_render_7k, f33_render_36k])
    cv2.putText(tri_f33, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(tri_f33, "Clean 7k Baseline (320k GS)", (w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(tri_f33, "36k Seeds (359k GS)", (2 * w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    f33_comp_path = "output/renders_double/frame_33_7k_vs_36k_vs_gt.jpg"
    cv2.imwrite(f33_comp_path, tri_f33)
    shutil.copy(f33_comp_path, os.path.join(art_dir, "frame_33_7k_vs_36k_vs_gt.jpg"))
    print(f"Saved 3-way Frame 33 comparison to {f33_comp_path}", flush=True)

    # Check left region (mirror area) in 7k baseline render
    left_third_7k = f33_render_7k[:, :int(w_d/3)]
    black_pixel_count = np.sum(np.all(left_third_7k < 15, axis=-1))
    print(f"Frame 33 (7k Baseline) Left 1/3 near-black pixel count (<15): {black_pixel_count} / {left_third_7k.shape[0]*left_third_7k.shape[1]} ({black_pixel_count/(left_third_7k.shape[0]*left_third_7k.shape[1])*100:.1f}%)")

    # =========================================================================
    # TASK 3: TIME-BOXED DEPTH SUPERVISION EXPERIMENT (SEED 42)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("  TASK 3: DEPTH SUPERVISION EXPERIMENT (DUSt3R DEPTH, SEED 42)", flush=True)
    print("======================================================================", flush=True)
    # Extract DUSt3R depth maps for each training camera
    cache_path = "output/dust3r_double/pairwise_cache.pt"
    cache = torch.load(cache_path, map_location="cpu")
    pw_v1 = cache['view1']['idx']

    # Map each training view to its first occurrence as view1
    depth_maps = {}
    conf_maps = {}
    for k in range(len(pw_v1)):
        v_idx = pw_v1[k]
        if v_idx in train_indices and v_idx not in depth_maps:
            # Camera optical depth along Z
            z_map = cache['pred1']['pts3d'][k, :, :, 2].to(device)
            c_map = cache['pred1']['conf'][k].to(device)
            depth_maps[v_idx] = z_map
            conf_maps[v_idx] = c_map

    print(f"Extracted per-frame DUSt3R depth maps for {len(depth_maps)} / {len(train_indices)} training views.", flush=True)

    # Initialize model with 36k seeds, seed 42
    np.random.seed(42); torch.manual_seed(42)
    tree_36k = KDTree(dedup_pts)
    dists_36k, _ = tree_36k.query(dedup_pts, k=4)
    mean_d_36k = np.clip(np.mean(dists_36k[:, 1:], axis=1), 1e-4, 0.1)[:, None]
    init_s_36k = np.log(mean_d_36k).repeat(3, axis=-1).astype(np.float32)
    init_op_36k = np.full((len(dedup_pts),), 1.386, dtype=np.float32)
    init_q_36k = np.zeros((len(dedup_pts), 4), dtype=np.float32); init_q_36k[:, 0] = 1.0
    scene_center_36k = np.median(dedup_pts, axis=0)
    scene_scale_36k = float(np.percentile(np.linalg.norm(dedup_pts - scene_center_36k, axis=1), 90))

    params_depth = nn.ParameterDict({
        'means': nn.Parameter(torch.tensor(dedup_pts.copy(), dtype=torch.float32, device=device)),
        'scales': nn.Parameter(torch.tensor(init_s_36k.copy(), dtype=torch.float32, device=device)),
        'quats': nn.Parameter(torch.tensor(init_q_36k.copy(), dtype=torch.float32, device=device)),
        'opacities': nn.Parameter(torch.tensor(init_op_36k.copy(), dtype=torch.float32, device=device)),
        'colors': nn.Parameter(torch.tensor(dedup_col.copy(), dtype=torch.float32, device=device)),
    })
    opts_depth = {
        'means': torch.optim.Adam([params_depth['means']], lr=1.6e-4, eps=1e-15),
        'scales': torch.optim.Adam([params_depth['scales']], lr=5e-3, eps=1e-15),
        'quats': torch.optim.Adam([params_depth['quats']], lr=1e-3, eps=1e-15),
        'opacities': torch.optim.Adam([params_depth['opacities']], lr=0.05, eps=1e-15),
        'colors': torch.optim.Adam([params_depth['colors']], lr=2.5e-3, eps=1e-15),
    }
    strat_depth = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
        refine_start_iter=200, refine_stop_iter=2500, refine_every=100, reset_every=1200, absgrad=True, verbose=False
    )
    st_state_depth = strat_depth.initialize_state(scene_scale=scene_scale_36k)

    lambda_depth = 0.05
    print(f"Training with depth supervision (lambda={lambda_depth}, 5,000 iters)...", flush=True)
    t0 = time.time()
    for step in range(5000):
        lr_fac = math.exp(-2.0 * step / 5000)
        opts_depth['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
        samp = train_data[step % len(train_data)]
        cam_id = samp['id']

        # Render RGB + Expected Depth
        rc, _, info = rasterization(
            means=params_depth['means'], quats=F.normalize(params_depth['quats'], dim=-1), scales=torch.exp(params_depth['scales']),
            opacities=torch.sigmoid(params_depth['opacities']), colors=params_depth['colors'], viewmats=samp['w2c'],
            Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True,
            render_mode='RGB+ED'
        )
        rgb_pred = rc[0, ..., :3]
        depth_pred = rc[0, ..., 3]

        loss_rgb = 0.8 * F.l1_loss(rgb_pred, samp['img']) + 0.2 * (1.0 - compute_ssim(rgb_pred.unsqueeze(0), samp['img'].unsqueeze(0)).item())

        # Depth supervision on high-confidence DUSt3R pixels
        if cam_id in depth_maps:
            gt_d = depth_maps[cam_id]
            gt_c = conf_maps[cam_id]
            mask_d = (gt_c > 1.0) & (gt_d > 0.01) & (depth_pred > 0.01)
            loss_d = pearson_depth_loss(depth_pred, gt_d, mask_d)
            total_loss = loss_rgb + lambda_depth * loss_d
        else:
            total_loss = loss_rgb

        strat_depth.step_pre_backward(params_depth, opts_depth, st_state_depth, step, info)
        total_loss.backward()
        strat_depth.step_post_backward(params_depth, opts_depth, st_state_depth, step, info, packed=True)
        for opt in opts_depth.values(): opt.step(); opt.zero_grad(set_to_none=True)

    dur_depth = time.time() - t0
    gs_depth = len(params_depth['means'])
    print(f"Depth-supervised model trained in {dur_depth:.1f}s | Final Gaussians: {gs_depth:,}", flush=True)

    # Evaluate on held-out test frames
    psnrs_all = []
    ssims_all = []
    psnrs_far = []
    ssims_far = []

    with torch.no_grad():
        for t_samp in test_data:
            rc, _, _ = rasterization(
                means=params_depth['means'], quats=F.normalize(params_depth['quats'], dim=-1), scales=torch.exp(params_depth['scales']),
                opacities=torch.sigmoid(params_depth['opacities']), colors=params_depth['colors'], viewmats=t_samp['w2c'],
                Ks=t_samp['K'], width=t_samp['width'], height=t_samp['height'], near_plane=0.08, packed=False
            )
            pred = rc[0].clamp(0.0, 1.0)
            psnr_v = compute_psnr(pred, t_samp['img'])
            ssim_v = compute_ssim(pred.unsqueeze(0), t_samp['img'].unsqueeze(0)).item()
            psnrs_all.append(psnr_v)
            ssims_all.append(ssim_v)
            if t_samp['dist'] >= 0.01:
                psnrs_far.append(psnr_v)
                ssims_far.append(ssim_v)

    mean_psnr_all = float(np.mean(psnrs_all))
    mean_ssim_all = float(np.mean(ssims_all))
    mean_psnr_far = float(np.mean(psnrs_far))
    mean_ssim_far = float(np.mean(ssims_far))

    print(f"\n--- Held-Out Results with Depth Supervision ---", flush=True)
    print(f"Overall (all 13 held-out views): PSNR = {mean_psnr_all:.2f} dB | SSIM = {mean_ssim_all:.4f}", flush=True)
    print(f"Far views (distance >= 0.01, 8 views): PSNR = {mean_psnr_far:.2f} dB | SSIM = {mean_ssim_far:.4f}", flush=True)

    # Render Off-path Doorway View (Camera 36 laterally offset by +0.020)
    c36 = double_cams[36]
    c2w36 = np.array(c36["c2w"], dtype=np.float64)
    rot36 = c2w36[:3, :3]
    pos36_off = c2w36[:3, 3] + rot36[:, 0] * 0.020 + rot36[:, 1] * (-0.005)
    c_door_off = np.eye(4, dtype=np.float32)
    c_door_off[:3, :3] = rot36
    c_door_off[:3, 3] = pos36_off
    w2c_off_t = torch.tensor(np.linalg.inv(c_door_off), dtype=torch.float32, device=device)[None]
    gt36 = cv2.imread(c36["img_path"])

    with torch.no_grad():
        rc_off_d, _, _ = rasterization(
            means=params_depth['means'], quats=F.normalize(params_depth['quats'], dim=-1), scales=torch.exp(params_depth['scales']),
            opacities=torch.sigmoid(params_depth['opacities']), colors=params_depth['colors'], viewmats=w2c_off_t,
            Ks=K_d, width=w_d, height=h_d, near_plane=0.08, packed=False
        )
    door_off_render_d = cv2.cvtColor((rc_off_d[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Load baseline off-path render (without depth loss)
    offpath_prev_path = "output/renders_double/offpath_doorway_stratified_vs_gt.jpg"
    if os.path.exists(offpath_prev_path):
        prev_img = cv2.imread(offpath_prev_path)
        door_off_render_base = prev_img[:, w_d:2*w_d]
    else:
        door_off_render_base = door_off_render_d.copy()

    # 3-way Doorway Comparison: Ground Truth (Cam 36) | Without Depth Supervision | With Depth Supervision
    tri_door = np.hstack([gt36, door_off_render_base, door_off_render_d])
    cv2.putText(tri_door, "GT (Training Cam 36)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(tri_door, "No Depth Supervision", (w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(tri_door, "With DUSt3R Depth Sup", (2 * w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    door_comp_path = "output/renders_double/offpath_doorway_depth_supervision_comp.jpg"
    cv2.imwrite(door_comp_path, tri_door)
    shutil.copy(door_comp_path, os.path.join(art_dir, "offpath_doorway_depth_supervision_comp.jpg"))
    print(f"Saved doorway depth supervision comparison to {door_comp_path}", flush=True)

    # Save summary json
    res = {
        "clean_7k_baseline": {
            "gs": len(params_7k['means']),
            "artifact": os.path.join(art_dir, "frame_33_7k_vs_36k_vs_gt.jpg"),
            "left_third_near_black_pct": float(black_pixel_count/(left_third_7k.shape[0]*left_third_7k.shape[1])*100)
        },
        "depth_supervision_exp": {
            "lambda_depth": lambda_depth,
            "final_gs": gs_depth,
            "overall_psnr": mean_psnr_all,
            "overall_ssim": mean_ssim_all,
            "far_views_psnr": mean_psnr_far,
            "far_views_ssim": mean_ssim_far,
            "artifact": os.path.join(art_dir, "offpath_doorway_depth_supervision_comp.jpg")
        }
    }
    with open("output/renders_double/depth_exp_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run()
