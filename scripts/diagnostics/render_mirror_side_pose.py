import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import shutil
import plyfile
from gsplat import rasterization

def load_ply_to_tensors(ply_path, device):
    ply = plyfile.PlyData.read(ply_path)
    v = ply['vertex']
    means = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
    opacities = np.array(v['opacity'], dtype=np.float32)[:, None]
    
    # SH or RGB
    if 'f_dc_0' in v:
        SH_C0 = 0.28209479177387814
        r = (np.array(v['f_dc_0']) * SH_C0 + 0.5).clip(0, 1)
        g = (np.array(v['f_dc_1']) * SH_C0 + 0.5).clip(0, 1)
        b = (np.array(v['f_dc_2']) * SH_C0 + 0.5).clip(0, 1)
        colors = np.stack([r, g, b], axis=-1).astype(np.float32)
    elif 'red' in v:
        colors = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.float32) / 255.0
    else:
        colors = np.ones_like(means) * 0.5

    return {
        'means': torch.tensor(means, device=device),
        'scales': torch.tensor(scales, device=device),
        'quats': torch.tensor(quats, device=device),
        'opacities': torch.tensor(opacities, device=device).squeeze(-1),
        'colors': torch.tensor(colors, device=device),
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))
    os.makedirs("output/renders_double", exist_ok=True)

    cams = json.load(open("output/dust3r_double/cameras.json"))
    c33 = cams[33]
    w, h = c33["width"], c33["height"]
    fx, fy, cx, cy = c33["fx"], c33["fy"], c33["cx"], c33["cy"]
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    # Find the latest/best trained model
    ply_candidates = [
        "output/splats/splat_seed_42.ply",
        "output/splats/splat_seed_123.ply",
        "output/splats/splat_seed_999.ply",
        "output/splats/room_splat.ply"
    ]
    target_ply = None
    for p in ply_candidates:
        if os.path.exists(p):
            target_ply = p
            break
            
    if not target_ply:
        print("No splat model found yet!")
        return

    print(f"Loading splat model: {target_ply}")
    params = load_ply_to_tensors(target_ply, device)

    # 1. Base Pose 33
    c2w33 = np.array(c33["c2w"], dtype=np.float32)
    rot33 = c2w33[:3, :3]
    pos33 = c2w33[:3, 3]

    # 2. Side-offset pose: step 0.035 units laterally (right) and yaw slightly left (-7 deg) toward the mirror
    theta = np.radians(-7.0)
    R_yaw = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float32)

    c2w_side = np.eye(4, dtype=np.float32)
    c2w_side[:3, :3] = rot33 @ R_yaw
    c2w_side[:3, 3] = pos33 + rot33[:, 0] * 0.035 + rot33[:, 2] * 0.010 # step right & slightly forward
    w2c_side = torch.tensor(np.linalg.inv(c2w_side), dtype=torch.float32, device=device)[None]

    # Render training pose 33
    w2c_orig = torch.tensor(np.linalg.inv(c2w33), dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        r_orig, _, _ = rasterization(
            means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
            opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=w2c_orig,
            Ks=K, width=w, height=h, near_plane=0.08, packed=False
        )
        r_side, _, _ = rasterization(
            means=params['means'], quats=F.normalize(params['quats'], dim=-1), scales=torch.exp(params['scales']),
            opacities=torch.sigmoid(params['opacities']), colors=params['colors'], viewmats=w2c_side,
            Ks=K, width=w, height=h, near_plane=0.08, packed=False
        )

    img_orig = cv2.cvtColor((r_orig[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_side = cv2.cvtColor((r_side[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    comp = np.hstack([img_orig, img_side])
    cv2.putText(comp, "Training Pose 33", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(comp, "Side-Angled View of Mirror (+0.035 lat, -7 deg yaw)", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    out_path = "output/renders_double/frame_33_mirror_side_pose.jpg"
    cv2.imwrite(out_path, comp)
    shutil.copy(out_path, os.path.join(art_dir, "frame_33_mirror_side_pose.jpg"))
    print(f"Saved side-angled mirror view to {out_path}")

if __name__ == "__main__":
    main()
