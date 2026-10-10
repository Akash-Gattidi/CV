import os
import sys
import json
import math
import cv2
import torch
import plyfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gsplat import rasterization, DefaultStrategy
from pipeline.gsplat_trainer import compute_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cams = json.load(open('output/dust3r/cameras.json'))
print('M1 Full cameras count:', len(cams))

train_data = []
for idx, c in enumerate(cams):
    img = cv2.cvtColor(cv2.imread(c['img_path']), cv2.COLOR_BGR2RGB) / 255.0
    c2w = np.array(c['c2w'], dtype=np.float32)
    train_data.append({
        'img': torch.tensor(img, dtype=torch.float32, device=device),
        'w2c': torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
        'K': torch.tensor([[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1.0]]], dtype=torch.float32, device=device),
        'width': c['width'], 'height': c['height']
    })

ply_path = 'output/dust3r/dust3r_points_dense.ply' if os.path.exists('output/dust3r/dust3r_points_dense.ply') else 'output/dust3r/dust3r_points.ply'
print(f"Loading seed points from: {ply_path}")
ply = plyfile.PlyData.read(ply_path)
pts_all = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)
col_all = (np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)
print(f"Total available seeds: {len(pts_all):,}")

from scipy.spatial import KDTree
scene_center = np.median(pts_all, axis=0)
scene_scale = float(np.percentile(np.linalg.norm(pts_all - scene_center, axis=1), 90))

# Test configurations
configs = [
    ("M1 Verified Config (50k seeds, Refine 2500 (50%), grad2d=0.00035, prune_opa=0.02, reset=1200)", 2500, 0.00035, 0.02, 1200, 50000),
    ("Sparse Seeds Test (5.7k seeds, Refine 2500, grad2d=0.00035, prune_opa=0.02, reset=1200)", 2500, 0.00035, 0.02, 1200, 5727),
]

for name, ref_stop, g2d, p_opa, reset_ev, n_seeds in configs:
    print(f"\n==========================================")
    print(f"Testing: {name}")
    print(f"==========================================")
    np.random.seed(42)
    torch.manual_seed(42)
    idx_sub = np.random.choice(len(pts_all), min(n_seeds, len(pts_all)), replace=False)
    pts = pts_all[idx_sub]
    col = col_all[idx_sub]
    print(f"Initialized with {len(pts):,} seeds.")

    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=min(4, len(pts)))
    mean_d = np.clip(np.mean(dists[:, 1:], axis=1), 1e-4, 0.1)[:, None]
    init_s = np.log(mean_d).repeat(3, axis=-1).astype(np.float32)
    init_op = np.full((len(pts),), 1.386, dtype=np.float32)
    init_q = np.zeros((len(pts), 4), dtype=np.float32)
    init_q[:, 0] = 1.0

    params = nn.ParameterDict({
        'means': nn.Parameter(torch.tensor(pts, dtype=torch.float32, device=device)),
        'scales': nn.Parameter(torch.tensor(init_s, dtype=torch.float32, device=device)),
        'quats': nn.Parameter(torch.tensor(init_q, dtype=torch.float32, device=device)),
        'opacities': nn.Parameter(torch.tensor(init_op, dtype=torch.float32, device=device)),
        'colors': nn.Parameter(torch.tensor(col, dtype=torch.float32, device=device)),
    })

    opts = {
        'means': torch.optim.Adam([params['means']], lr=1.6e-4, eps=1e-15),
        'scales': torch.optim.Adam([params['scales']], lr=5e-3, eps=1e-15),
        'quats': torch.optim.Adam([params['quats']], lr=1e-3, eps=1e-15),
        'opacities': torch.optim.Adam([params['opacities']], lr=0.05, eps=1e-15),
        'colors': torch.optim.Adam([params['colors']], lr=2.5e-3, eps=1e-15),
    }

    strategy = DefaultStrategy(
        prune_opa=p_opa,
        grow_grad2d=g2d,
        grow_scale3d=0.01,
        refine_start_iter=200,
        refine_stop_iter=ref_stop,
        refine_every=100,
        reset_every=reset_ev,
        absgrad=True,
        verbose=False
    )
    st_state = strategy.initialize_state(scene_scale=scene_scale)

    for step in range(5000):
        lr_fac = math.exp(-2.0 * step / 5000)
        opts['means'].param_groups[0]['lr'] = 1.6e-4 * max(0.05, lr_fac)
        samp = train_data[step % len(train_data)]
        rc, _, info = rasterization(
            means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
            opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=samp['w2c'],
            Ks=samp['K'], width=samp['width'], height=samp['height'], near_plane=0.1, packed=True, absgrad=True
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

        if (step + 1) in [1000, 2000, 2500, 3000, 4000, 5000]:
            print(f"  Step {step+1:4d}: Loss = {loss.item():.4f}, SSIM = {ss:.3f}, Gaussians = {len(params['means']):,}", flush=True)

    n_final = len(params['means'])
    print(f"--> Result [{name}]: Final GS = {n_final:,}\n", flush=True)
