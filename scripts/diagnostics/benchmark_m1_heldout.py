import os
import sys
PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import math
import cv2
import torch
import plyfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization, DefaultStrategy
from pipeline.gsplat_trainer import compute_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cams = json.load(open('output/dust3r/cameras.json'))
test_indices = [i for i in range(len(cams)) if i % 5 == 4]
train_indices = [i for i in range(len(cams)) if i % 5 != 4]
print(f"M1 Held-out Split: {len(train_indices)} train, {len(test_indices)} test (held-out: {test_indices})")

train_data = []
for idx in train_indices:
    c = cams[idx]
    img = cv2.cvtColor(cv2.imread(c['img_path']), cv2.COLOR_BGR2RGB) / 255.0
    c2w = np.array(c['c2w'], dtype=np.float32)
    train_data.append({
        'img': torch.tensor(img, dtype=torch.float32, device=device),
        'w2c': torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
        'K': torch.tensor([[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1.0]]], dtype=torch.float32, device=device),
        'width': c['width'], 'height': c['height']
    })

test_data = []
for idx in test_indices:
    c = cams[idx]
    img = cv2.cvtColor(cv2.imread(c['img_path']), cv2.COLOR_BGR2RGB) / 255.0
    c2w = np.array(c['c2w'], dtype=np.float32)
    test_data.append({
        'img': torch.tensor(img, dtype=torch.float32, device=device),
        'w2c': torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)[None],
        'K': torch.tensor([[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1.0]]], dtype=torch.float32, device=device),
        'width': c['width'], 'height': c['height']
    })

# Load seeds
ply = plyfile.PlyData.read('output/dust3r/dust3r_points.ply')
pts_all = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)
col_all = (np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1) / 255.0).astype(np.float32)

from scipy.spatial import KDTree
scene_center = np.median(pts_all, axis=0)
scene_scale = float(np.percentile(np.linalg.norm(pts_all - scene_center, axis=1), 90))

seeds = [42, 123]
m1_heldout_results = []

for s in seeds:
    np.random.seed(s)
    torch.manual_seed(s)
    idx_sub = np.random.choice(len(pts_all), min(50000, len(pts_all)), replace=False)
    pts = pts_all[idx_sub]
    col = col_all[idx_sub]

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

    # Evaluate held-out
    psnrs, ssims = [], []
    with torch.no_grad():
        for test_s in test_data:
            rc, _, _ = rasterization(
                means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
                opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=test_s['w2c'],
                Ks=test_s['K'], width=test_s['width'], height=test_s['height'], near_plane=0.1, packed=False
            )
            pred = rc[0].clamp(0.0, 1.0)
            mse = F.mse_loss(pred, test_s['img']).item()
            psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-10)))
            ssims.append(compute_ssim(pred.unsqueeze(0), test_s['img'].unsqueeze(0)).item())

    res = {
        'seed': s,
        'final_gs': len(params['means']),
        'heldout_psnr': float(np.mean(psnrs)),
        'heldout_ssim': float(np.mean(ssims))
    }
    m1_heldout_results.append(res)
    print(f"M1 Seed {s}: GS = {res['final_gs']:,} | Held-out PSNR = {res['heldout_psnr']:.2f} dB | Held-out SSIM = {res['heldout_ssim']:.4f}")

with open('output/splats/m1_heldout_seeds_benchmark.json', 'w') as f:
    json.dump(m1_heldout_results, f, indent=2)
