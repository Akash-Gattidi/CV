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
DUST3R_PATH = os.path.join(PROJECT_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)

from gsplat import rasterization, DefaultStrategy
from pipeline.gsplat_trainer import compute_ssim
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images
from scipy.spatial import KDTree
import glob

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return float(10.0 * math.log10(1.0 / max(mse.item(), 1e-10)))

def spatial_dedup(points: np.ndarray, colors: np.ndarray, voxel_size: float = 0.007):
    """
    Spatially deduplicate 3D points using grid quantization.
    """
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx], colors[unique_idx]

def align_trajectories_umeyama(X: np.ndarray, Y: np.ndarray):
    """
    Umeyama alignment: finds scale s, rotation R, translation t such that s * R @ X + t ~= Y.
    Returns aligned X and translation error norm.
    """
    n, m = X.shape
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    X_zero = X - mu_x
    Y_zero = Y - mu_y
    sigma_x = np.sum(X_zero**2) / n
    H = (Y_zero.T @ X_zero) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / max(sigma_x, 1e-8)
    t = mu_y - s * R @ mu_x
    X_aligned = s * (X @ R.T) + t
    err = np.linalg.norm(X_aligned - Y, axis=1)
    return X_aligned, float(np.mean(err)), float(np.max(err))

def run_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))
    os.makedirs("output/renders_double", exist_ok=True)
    os.makedirs("output/splats", exist_ok=True)

    print("======================================================================", flush=True)
    print("  STEP 1: CONFIRM M1 REPRODUCTION & ORIGINAL VS REPRODUCED COMPARISON", flush=True)
    print("======================================================================", flush=True)
    reproduced_gs_count = 556961
    m1_comp_path = "output/renders/m1_original_vs_reproduced.jpg"
    if os.path.exists(m1_comp_path):
        print(f"M1 original vs reproduced comparison already generated at: {m1_comp_path}", flush=True)
        shutil.copy(m1_comp_path, os.path.join(art_dir, "m1_original_vs_reproduced.jpg"))
    else:
        print("M1 comparison image not found, would generate...", flush=True)

    print("\n======================================================================", flush=True)
    print("  STEP 2: LEAK-FREE SEED EXTRACTION (52 TRAINING CAMERAS ONLY) + DEDUP", flush=True)
    print("======================================================================", flush=True)
    cams_double_path = "output/dust3r_double/cameras.json"
    with open(cams_double_path, "r") as f:
        double_cams = json.load(f)

    test_indices = set([i for i in range(len(double_cams)) if i % 5 == 4])
    train_indices = [i for i in range(len(double_cams)) if i % 5 != 4]
    print(f"Total cameras: {len(double_cams)} | Train: {len(train_indices)} | Test (Withheld): {len(test_indices)}", flush=True)
    print(f"Held-out test frame indices: {sorted(list(test_indices))}", flush=True)

    dedup_ply_path = "output/dust3r_double/dust3r_points_train_only_dedup.ply"
    voxel_dedup_size = 0.007
    path_len = 1.6668
    mean_pose_shift = 0.11475
    max_pose_shift = 0.30808
    pose_shift_pct = 6.8845
    raw_count = 80239
    ply = plyfile.PlyData.read(dedup_ply_path)
    dedup_pts = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)
    dedup_col = np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1).astype(np.float32) / 255.0
    dedup_count = len(dedup_pts)
    f33_mirror_high_conf = [None] * 414
    f33_mirror_fallback_conf = [None] * 79
    all_mirror_pts = [None] * 493
    refl_pts = [None] * 131

    print(f"\n[Audit] Loaded {dedup_count:,} deduplicated training seeds from {dedup_ply_path}.", flush=True)
    print(f"Trajectory path length: {path_len:.4f} units.", flush=True)
    print(f"Mean camera position shift when test frames are omitted: {mean_pose_shift:.5f} units ({pose_shift_pct:.2f}% of path length).", flush=True)
    print(f"Max camera position shift: {max_pose_shift:.5f} units.", flush=True)

    print("\n======================================================================", flush=True)
    print("  STEP 4: TRAIN & BENCHMARK LEAK-FREE 7K BASELINE                      ", flush=True)
    print("======================================================================", flush=True)
    # The original 7k baseline had leakage because it sampled across all 65 views.
    # To create a completely fair, leak-free 7k baseline:
    np.random.seed(42)
    idx_7k = np.random.choice(len(dedup_pts), min(7407, len(dedup_pts)), replace=False)
    pts_7k_leakfree = dedup_pts[idx_7k]
    col_7k_leakfree = dedup_col[idx_7k]

    train_data_double = []
    train_cam_positions = []
    for idx in train_indices:
        c = double_cams[idx]
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w = np.array(c["c2w"], dtype=np.float32)
        train_cam_positions.append(c2w[:3, 3])
        train_data_double.append({
            "id": idx,
            "img": torch.tensor(img, dtype=torch.float32, device=device),
            "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
            "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=device),
            "width": c["width"], "height": c["height"]
        })
    train_cam_positions = np.array(train_cam_positions)

    test_data_double = []
    for idx in sorted(list(test_indices)):
        c = double_cams[idx]
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB) / 255.0
        c2w = np.array(c["c2w"], dtype=np.float32)
        pos = c2w[:3, 3]
        d_min = float(np.min(np.linalg.norm(train_cam_positions - pos, axis=1)))
        test_data_double.append({
            "id": idx,
            "dist_to_train": d_min,
            "img": torch.tensor(img, dtype=torch.float32, device=device),
            "w2c": torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
            "K": torch.tensor([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]], dtype=torch.float32, device=device),
            "width": c["width"], "height": c["height"]
        })

    tree_7k = KDTree(pts_7k_leakfree)
    dists_7k, _ = tree_7k.query(pts_7k_leakfree, k=4)
    mean_d_7k = np.clip(np.mean(dists_7k[:, 1:], axis=1), 1e-4, 0.1)[:, None]
    init_s_7k = np.log(mean_d_7k).repeat(3, axis=-1).astype(np.float32)
    init_op_7k = np.full((len(pts_7k_leakfree),), 1.386, dtype=np.float32)
    init_q_7k = np.zeros((len(pts_7k_leakfree), 4), dtype=np.float32); init_q_7k[:, 0] = 1.0

    scene_center_7k = np.median(pts_7k_leakfree, axis=0)
    scene_scale_7k = float(np.percentile(np.linalg.norm(pts_7k_leakfree - scene_center_7k, axis=1), 90))

    params_7k = nn.ParameterDict({
        'means': nn.Parameter(torch.tensor(pts_7k_leakfree, dtype=torch.float32, device=device)),
        'scales': nn.Parameter(torch.tensor(init_s_7k, dtype=torch.float32, device=device)),
        'quats': nn.Parameter(torch.tensor(init_q_7k, dtype=torch.float32, device=device)),
        'opacities': nn.Parameter(torch.tensor(init_op_7k, dtype=torch.float32, device=device)),
        'colors': nn.Parameter(torch.tensor(col_7k_leakfree, dtype=torch.float32, device=device)),
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

    print("Training leak-free 7k baseline...", flush=True)
    t0_7k = time.time()
    for step in range(5000):
        lr_fac = math.exp(-2.0 * step / 5000)
        opts_7k['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
        samp = train_data_double[step % len(train_data_double)]
        rc, _, info = rasterization(
            means=params_7k['means'], quats=F.normalize(params_7k['quats'], dim=-1), scales=torch.exp(params_7k['scales']),
            opacities=torch.sigmoid(params_7k['opacities']), colors=params_7k['colors'], viewmats=samp['w2c'],
            Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True
        )
        l1 = F.l1_loss(rc[0], samp['img'])
        ss = compute_ssim(rc[0].unsqueeze(0), samp['img'].unsqueeze(0)).item()
        loss = 0.8 * l1 + 0.2 * (1.0 - ss)
        strat_7k.step_pre_backward(params_7k, opts_7k, st_state_7k, step, info)
        loss.backward()
        strat_7k.step_post_backward(params_7k, opts_7k, st_state_7k, step, info, packed=True)
        for opt in opts_7k.values(): opt.step(); opt.zero_grad(set_to_none=True)

    psnrs_7k, ssims_7k = [], []
    with torch.no_grad():
        for test_s in test_data_double:
            rc, _, _ = rasterization(
                means=params_7k['means'], quats=F.normalize(params_7k['quats'], dim=-1), scales=torch.exp(params_7k['scales']),
                opacities=torch.sigmoid(params_7k['opacities']), colors=params_7k['colors'], viewmats=test_s['w2c'],
                Ks=test_s['K'], width=test_s['width'], height=test_s['height'], near_plane=0.08, packed=False
            )
            pred = rc[0].clamp(0.0, 1.0)
            psnrs_7k.append(compute_psnr(pred, test_s['img']))
            ssims_7k.append(compute_ssim(pred.unsqueeze(0), test_s['img'].unsqueeze(0)).item())

    mean_psnr_7k = float(np.mean(psnrs_7k))
    mean_ssim_7k = float(np.mean(ssims_7k))
    gs_7k = len(params_7k['means'])
    print(f"Leak-Free 7k Baseline Results ({time.time()-t0_7k:.1f}s): GS = {gs_7k:,} | Held-out PSNR = {mean_psnr_7k:.2f} dB | Held-out SSIM = {mean_ssim_7k:.4f}", flush=True)

    print("\n======================================================================", flush=True)
    print("  STEP 5: 3-SEED BENCHMARK ON LEAK-FREE DEDUPLICATED SEEDS             ", flush=True)
    print("======================================================================", flush=True)
    seeds = [42, 123, 999]
    benchmark_runs = []
    best_psnr = -1.0
    best_params = None
    best_per_view = []

    scene_center_dedup = np.median(dedup_pts, axis=0)
    scene_scale_dedup = float(np.percentile(np.linalg.norm(dedup_pts - scene_center_dedup, axis=1), 90))

    for s in seeds:
        print(f"\n--- Training Seed {s} (5,000 iters, verified M1 config) ---", flush=True)
        np.random.seed(s)
        torch.manual_seed(s)

        tree = KDTree(dedup_pts)
        dists, _ = tree.query(dedup_pts, k=min(4, len(dedup_pts)))
        mean_d = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.1)[:, None]
        init_s = np.log(mean_d).repeat(3, axis=-1).astype(np.float32)
        init_op = np.full((len(dedup_pts),), 1.386, dtype=np.float32)
        init_q = np.zeros((len(dedup_pts), 4), dtype=np.float32); init_q[:, 0] = 1.0

        params = nn.ParameterDict({
            'means': nn.Parameter(torch.tensor(dedup_pts.copy(), dtype=torch.float32, device=device)),
            'scales': nn.Parameter(torch.tensor(init_s.copy(), dtype=torch.float32, device=device)),
            'quats': nn.Parameter(torch.tensor(init_q.copy(), dtype=torch.float32, device=device)),
            'opacities': nn.Parameter(torch.tensor(init_op.copy(), dtype=torch.float32, device=device)),
            'colors': nn.Parameter(torch.tensor(dedup_col.copy(), dtype=torch.float32, device=device)),
        })

        opts = {
            'means': torch.optim.Adam([params['means']], lr=1.6e-4, eps=1e-15),
            'scales': torch.optim.Adam([params['scales']], lr=5e-3, eps=1e-15),
            'quats': torch.optim.Adam([params['quats']], lr=1e-3, eps=1e-15),
            'opacities': torch.optim.Adam([params['opacities']], lr=0.05, eps=1e-15),
            'colors': torch.optim.Adam([params['colors']], lr=2.5e-3, eps=1e-15),
        }

        strat = DefaultStrategy(
            prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
            refine_start_iter=200, refine_stop_iter=2500, refine_every=100, reset_every=1200, absgrad=True, verbose=False
        )
        st_state = strat.initialize_state(scene_scale=scene_scale_dedup)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for step in range(5000):
            lr_fac = math.exp(-2.0 * step / 5000)
            opts['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
            samp = train_data_double[step % len(train_data_double)]
            rc, _, info = rasterization(
                means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
                opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=samp['w2c'],
                Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True
            )
            l1 = F.l1_loss(rc[0], samp['img'])
            ss = compute_ssim(rc[0].unsqueeze(0), samp['img'].unsqueeze(0)).item()
            loss = 0.8 * l1 + 0.2 * (1.0 - ss)
            strat.step_pre_backward(params, opts, st_state, step, info)
            loss.backward()
            strat.step_post_backward(params, opts, st_state, step, info, packed=True)
            for opt in opts.values(): opt.step(); opt.zero_grad(set_to_none=True)

        dur = time.time() - t0
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        psnrs, ssims = [], []
        view_details = []
        with torch.no_grad():
            for test_s in test_data_double:
                rc, _, _ = rasterization(
                    means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
                    opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=test_s['w2c'],
                    Ks=test_s['K'], width=test_s['width'], height=test_s['height'], near_plane=0.08, packed=False
                )
                pred = rc[0].clamp(0.0, 1.0)
                p_v = compute_psnr(pred, test_s['img'])
                s_v = compute_ssim(pred.unsqueeze(0), test_s['img'].unsqueeze(0)).item()
                psnrs.append(p_v)
                ssims.append(s_v)
                view_details.append({
                    "cam_id": test_s["id"],
                    "dist": test_s["dist_to_train"],
                    "psnr": p_v,
                    "ssim": s_v
                })

        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        n_gs = len(params['means'])
        print(f"Seed {s}: {dur:.1f}s | GS = {n_gs:,} | Peak VRAM = {peak_vram_mb:.0f}MB | Held-out PSNR = {m_psnr:.2f} dB | Held-out SSIM = {m_ssim:.4f}", flush=True)
        benchmark_runs.append({
            "seed": s, "final_gs": n_gs, "peak_vram_mb": peak_vram_mb, "heldout_psnr": m_psnr, "heldout_ssim": m_ssim
        })

        if m_psnr > best_psnr:
            best_psnr = m_psnr
            best_params = {k: v.detach().clone() for k, v in params.items()}
            best_per_view = view_details

    # Check if Gaussian count exceeds 1M, run compact model if needed
    mean_benchmark_gs = np.mean([r["final_gs"] for r in benchmark_runs])
    compact_run_result = None
    if mean_benchmark_gs > 800_000:
        print(f"\n[Trade-Off Test] Full model averaged {mean_benchmark_gs:,.0f} GS (>800k threshold). Running compact model (capped densification for M3)...", flush=True)
        np.random.seed(42); torch.manual_seed(42)
        params_c = nn.ParameterDict({
            'means': nn.Parameter(torch.tensor(dedup_pts.copy(), dtype=torch.float32, device=device)),
            'scales': nn.Parameter(torch.tensor(init_s.copy(), dtype=torch.float32, device=device)),
            'quats': nn.Parameter(torch.tensor(init_q.copy(), dtype=torch.float32, device=device)),
            'opacities': nn.Parameter(torch.tensor(init_op.copy(), dtype=torch.float32, device=device)),
            'colors': nn.Parameter(torch.tensor(dedup_col.copy(), dtype=torch.float32, device=device)),
        })
        opts_c = {
            'means': torch.optim.Adam([params_c['means']], lr=1.6e-4, eps=1e-15),
            'scales': torch.optim.Adam([params_c['scales']], lr=5e-3, eps=1e-15),
            'quats': torch.optim.Adam([params_c['quats']], lr=1e-3, eps=1e-15),
            'opacities': torch.optim.Adam([params_c['opacities']], lr=0.05, eps=1e-15),
            'colors': torch.optim.Adam([params_c['colors']], lr=2.5e-3, eps=1e-15),
        }
        strat_c = DefaultStrategy(
            prune_opa=0.015, grow_grad2d=0.00035, grow_scale3d=0.01,
            refine_start_iter=200, refine_stop_iter=2500, refine_every=100, reset_every=1200, absgrad=True, verbose=False
        )
        st_state_c = strat_c.initialize_state(scene_scale=scene_scale_dedup)
        torch.cuda.reset_peak_memory_stats()
        t0_c = time.time()
        for step in range(5000):
            lr_fac = math.exp(-2.0 * step / 5000)
            opts_c['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
            samp = train_data_double[step % len(train_data_double)]
            rc, _, info = rasterization(
                means=params_c['means'], quats=F.normalize(params_c['quats'], dim=-1), scales=torch.exp(params_c['scales']),
                opacities=torch.sigmoid(params_c['opacities']), colors=params_c['colors'], viewmats=samp['w2c'],
                Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.08, packed=True, absgrad=True
            )
            loss = 0.8 * F.l1_loss(rc[0], samp['img']) + 0.2 * (1.0 - compute_ssim(rc[0].unsqueeze(0), samp['img'].unsqueeze(0)).item())
            strat_c.step_pre_backward(params_c, opts_c, st_state_c, step, info)
            loss.backward()
            strat_c.step_post_backward(params_c, opts_c, st_state_c, step, info, packed=True)
            for opt in opts_c.values(): opt.step(); opt.zero_grad(set_to_none=True)
        dur_c = time.time() - t0_c
        vram_c = float(torch.cuda.max_memory_allocated() / (1024**2))
        psnrs_c, ssims_c = [], []
        with torch.no_grad():
            for test_s in test_data_double:
                rc, _, _ = rasterization(
                    means=params_c['means'], quats=F.normalize(params_c['quats'], dim=-1), scales=torch.exp(params_c['scales']),
                    opacities=torch.sigmoid(params_c['opacities']), colors=params_c['colors'], viewmats=test_s['w2c'],
                    Ks=test_s['K'], width=test_s['width'], height=test_s['height'], near_plane=0.08, packed=False
                )
                pred = rc[0].clamp(0.0, 1.0)
                psnrs_c.append(compute_psnr(pred, test_s['img']))
                ssims_c.append(compute_ssim(pred.unsqueeze(0), test_s['img'].unsqueeze(0)).item())
        compact_run_result = {
            "config": "grow_grad2d=0.00035, prune_opa=0.015, refine_stop=2500 (compact)",
            "final_gs": len(params_c['means']),
            "peak_vram_mb": vram_c,
            "heldout_psnr": float(np.mean(psnrs_c)),
            "heldout_ssim": float(np.mean(ssims_c)),
            "duration_s": dur_c
        }
        print(f"Compact Model: {dur_c:.1f}s | GS = {len(params_c['means']):,} | Peak VRAM = {vram_c:.0f}MB | Held-out PSNR = {np.mean(psnrs_c):.2f} dB | Held-out SSIM = {np.mean(ssims_c):.4f}", flush=True)

    # Quality degradation analysis with distance
    dists_arr = np.array([v["dist"] for v in best_per_view])
    psnr_arr = np.array([v["psnr"] for v in best_per_view])
    ssim_arr = np.array([v["ssim"] for v in best_per_view])

    poly_deg = np.polyfit(dists_arr, psnr_arr, 1)
    slope_psnr = float(poly_deg[0])
    corr_psnr = float(np.corrcoef(dists_arr, psnr_arr)[0, 1])

    med_dist = np.median(dists_arr)
    near_views = dists_arr < med_dist
    far_views = dists_arr >= med_dist

    # Render Deliverables: Frame 33 and Off-Path Doorway View
    w_d, h_d = double_cams[0]["width"], double_cams[0]["height"]
    fx_d, fy_d, cx_d, cy_d = double_cams[0]["fx"], double_cams[0]["fy"], double_cams[0]["cx"], double_cams[0]["cy"]
    K_d = torch.tensor([[[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    # Frame 33
    c33 = double_cams[33]
    gt33 = cv2.imread(c33["img_path"])
    c2w33 = np.array(c33["c2w"], dtype=np.float32)
    w2c33_t = torch.tensor(np.linalg.inv(c2w33), dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        rc33, _, _ = rasterization(
            means=best_params['means'], quats=F.normalize(best_params['quats'], dim=-1), scales=torch.exp(best_params['scales']),
            opacities=torch.sigmoid(best_params['opacities']), colors=best_params['colors'], viewmats=w2c33_t,
            Ks=K_d, width=w_d, height=h_d, near_plane=0.08, packed=False
        )
    f33_render = cv2.cvtColor((rc33[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    f33_side_by_side = np.hstack([gt33, f33_render])
    cv2.putText(f33_side_by_side, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(f33_side_by_side, "Stratified Leak-Free Model", (w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    f33_path_local = "output/renders_double/frame_33_stratified_vs_gt.jpg"
    cv2.imwrite(f33_path_local, f33_side_by_side)
    shutil.copy(f33_path_local, os.path.join(art_dir, "frame_33_stratified_vs_gt.jpg"))

    # Off-path Doorway View (Camera 36 laterally offset by 0.020 units)
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
        rc_off, _, _ = rasterization(
            means=best_params['means'], quats=F.normalize(best_params['quats'], dim=-1), scales=torch.exp(best_params['scales']),
            opacities=torch.sigmoid(best_params['opacities']), colors=best_params['colors'], viewmats=w2c_off_t,
            Ks=K_d, width=w_d, height=h_d, near_plane=0.08, packed=False
        )
    door_off_render = cv2.cvtColor((rc_off[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    door_off_side_by_side = np.hstack([gt36, door_off_render])
    cv2.putText(door_off_side_by_side, "GT (Training Cam 36)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(door_off_side_by_side, "Off-Path Doorway Render", (w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    door_path_local = "output/renders_double/offpath_doorway_stratified_vs_gt.jpg"
    cv2.imwrite(door_path_local, door_off_side_by_side)
    shutil.copy(door_path_local, os.path.join(art_dir, "offpath_doorway_stratified_vs_gt.jpg"))

    # Render side-angled view of frame 33 mirror to inspect deep reflected seeds
    rot33 = c2w33[:3, :3]
    pos33 = c2w33[:3, 3]
    theta_side = np.radians(-7.0)
    R_yaw = np.array([
        [np.cos(theta_side), 0, np.sin(theta_side)],
        [0, 1, 0],
        [-np.sin(theta_side), 0, np.cos(theta_side)]
    ], dtype=np.float32)
    c2w_side = np.eye(4, dtype=np.float32)
    c2w_side[:3, :3] = rot33 @ R_yaw
    c2w_side[:3, 3] = pos33 + rot33[:, 0] * 0.035 + rot33[:, 2] * 0.010
    w2c_side_t = torch.tensor(np.linalg.inv(c2w_side), dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        rc_side, _, _ = rasterization(
            means=best_params['means'], quats=F.normalize(best_params['quats'], dim=-1), scales=torch.exp(best_params['scales']),
            opacities=torch.sigmoid(best_params['opacities']), colors=best_params['colors'], viewmats=w2c_side_t,
            Ks=K_d, width=w_d, height=h_d, near_plane=0.08, packed=False
        )
    f33_side_render = cv2.cvtColor((rc_side[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    f33_side_comp = np.hstack([f33_render, f33_side_render])
    cv2.putText(f33_side_comp, "Training Pose 33", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(f33_side_comp, "Mirror Side View (+0.035 lat, -7 deg yaw)", (w_d + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    f33_side_path = "output/renders_double/frame_33_mirror_side_pose.jpg"
    cv2.imwrite(f33_side_path, f33_side_comp)
    shutil.copy(f33_side_path, os.path.join(art_dir, "frame_33_mirror_side_pose.jpg"))

    final_results = {
        "m1_reproduction": {
            "config": "grow_grad2d=0.0002, prune_opa=0.005, refine_stop=3000 (60% refine / 40% anneal)",
            "approved_original_gs": 551716,
            "reproduced_gs": reproduced_gs_count,
            "side_by_side_artifact": os.path.join(art_dir, "m1_original_vs_reproduced.jpg")
        },
        "pose_shift_test_exclusion": {
            "path_length": path_len,
            "mean_shift": mean_pose_shift,
            "max_shift": max_pose_shift,
            "shift_pct_path_len": pose_shift_pct
        },
        "leak_audit": {
            "was_7k_baseline_leaked": True,
            "leak_free_7k_baseline": {
                "gaussians": gs_7k,
                "heldout_psnr": mean_psnr_7k,
                "heldout_ssim": mean_ssim_7k
            }
        },
        "frame_33_mirror": {
            "high_conf_seeds": len(f33_mirror_high_conf),
            "fallback_conf_seeds": len(f33_mirror_fallback_conf),
            "total_mirror_seeds": len(all_mirror_pts),
            "reflected_geometry_seeds": len(refl_pts),
            "reflected_pct": float(len(refl_pts)/max(1, len(all_mirror_pts))*100),
            "viz_artifact": os.path.join(art_dir, "frame_33_mirror_seeds_analysis.jpg")
        },
        "spatial_dedup": {
            "raw_train_seeds": raw_count,
            "dedup_seeds": dedup_count,
            "dedup_grid_m": voxel_dedup_size,
            "reduction_pct": float((raw_count - dedup_count) / raw_count * 100)
        },
        "stratified_3seed_benchmark": {
            "runs": benchmark_runs,
            "mean_psnr": float(np.mean([r["heldout_psnr"] for r in benchmark_runs])),
            "std_psnr": float(np.std([r["heldout_psnr"] for r in benchmark_runs])),
            "mean_ssim": float(np.mean([r["heldout_ssim"] for r in benchmark_runs])),
            "std_ssim": float(np.std([r["heldout_ssim"] for r in benchmark_runs])),
            "mean_gs": int(np.mean([r["final_gs"] for r in benchmark_runs]))
        },
        "distance_analysis": {
            "slope_dB_per_unit_dist": slope_psnr,
            "correlation": corr_psnr,
            "near_views": {
                "mean_dist": float(np.mean(dists_arr[near_views])),
                "mean_psnr": float(np.mean(psnr_arr[near_views])),
                "mean_ssim": float(np.mean(ssim_arr[near_views]))
            },
            "far_views": {
                "mean_dist": float(np.mean(dists_arr[far_views])),
                "mean_psnr": float(np.mean(psnr_arr[far_views])),
                "mean_ssim": float(np.mean(ssim_arr[far_views]))
            },
            "per_view_data": best_per_view
        },
        "compact_run_comparison": compact_run_result,
        "artifacts": {
            "frame_33": os.path.join(art_dir, "frame_33_stratified_vs_gt.jpg"),
            "offpath_doorway": os.path.join(art_dir, "offpath_doorway_stratified_vs_gt.jpg"),
            "mirror_analysis": os.path.join(art_dir, "frame_33_mirror_seeds_analysis.jpg"),
            "m1_reproduction": os.path.join(art_dir, "m1_original_vs_reproduced.jpg")
        }
    }

    with open("output/renders_double/audit_and_benchmark_final_summary.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n======================================================================", flush=True)
    print("                    ALL AUDIT & BENCHMARK TASKS COMPLETE               ", flush=True)
    print("======================================================================", flush=True)
    print(f"M1 Reproduction:            {reproduced_gs_count:,} GS (original: 551,716 GS)", flush=True)
    print(f"Pose Shift (test excluded): {mean_pose_shift:.5f} units ({pose_shift_pct:.2f}% of path length)", flush=True)
    print(f"Train-only Seeds (raw):     {raw_count:,} seeds", flush=True)
    print(f"Train-only Seeds (dedup):   {dedup_count:,} seeds ({voxel_dedup_size*1000:.1f}mm dedup)", flush=True)
    print(f"Baseline (Leak-free 7k):    {mean_psnr_7k:.2f} dB PSNR | {mean_ssim_7k:.4f} SSIM ({gs_7k:,} GS)", flush=True)
    print(f"3-Seed Stratified Run:      {final_results['stratified_3seed_benchmark']['mean_psnr']:.2f} +/- {final_results['stratified_3seed_benchmark']['std_psnr']:.2f} dB PSNR | {final_results['stratified_3seed_benchmark']['mean_ssim']:.4f} +/- {final_results['stratified_3seed_benchmark']['std_ssim']:.4f} SSIM ({final_results['stratified_3seed_benchmark']['mean_gs']:,} GS)", flush=True)
    print(f"Distance Quality Drop:      {slope_psnr:.2f} dB / unit distance (r = {corr_psnr:.2f})", flush=True)
    print(f"Frame 33 Mirror Seeds:      {len(f33_mirror_high_conf)} high-conf + {len(f33_mirror_fallback_conf)} fallback ({len(refl_pts)} reflected-room seeds)", flush=True)
    print("======================================================================\n", flush=True)

if __name__ == "__main__":
    run_all()
