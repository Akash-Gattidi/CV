import os
import sys
import json
import glob
import time
import torch
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.dust3r_runner import DUSt3RGeometryEstimator
from pipeline.gsplat_trainer import GSplatTrainer, render_gsplat_views_and_orbit

def run_m2_reconstruction(
    frames_dir: str = "output/frames_double",
    dust3r_dir: str = "output/dust3r_double",
    splats_dir: str = "output/splats_double",
    renders_dir: str = "output/renders_double",
    train_iters: int = 5000
):
    os.makedirs(dust3r_dir, exist_ok=True)
    os.makedirs(splats_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    print(f"[M2 Pipeline] Found {len(frame_paths)} keyframes in {frames_dir}")
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # =========================================================================
    # Step 1: DUSt3R Geometry & Pose Estimation (Sliding Window swin-3-noncyclic)
    # =========================================================================
    cams_json = os.path.join(dust3r_dir, "cameras.json")
    dense_ply = os.path.join(dust3r_dir, "dust3r_points_dense.ply")

    if not os.path.exists(cams_json) or not os.path.exists(dense_ply):
        print("\n=======================================================")
        print("[M2 Pipeline] Step 1: DUSt3R Dense Geometry Estimation")
        print("  - Scene Graph: sliding-window (swin-3-noncyclic)")
        print("  - Target: Joint coordinate frame for Room A + Doorway + Room B")
        print("=======================================================")
        estimator = DUSt3RGeometryEstimator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )
        t_dust0 = time.time()
        results = estimator.run_reconstruction(
            image_paths=frame_paths,
            output_dir=dust3r_dir,
            scene_graph_type="swin-3-noncyclic",
            n_iters=60
        )
        dust_time = time.time() - t_dust0
        print(f"[M2 Pipeline] DUSt3R reconstruction completed in {dust_time:.1f}s")
    else:
        print(f"[M2 Pipeline] Cached DUSt3R cameras and points found in {dust3r_dir}.")

    # =========================================================================
    # Step 2: Unified Multi-Room Gaussian Splatting Training (gsplat)
    # =========================================================================
    print("\n=======================================================")
    print("[M2 Pipeline] Step 2: Unified Multi-Room Gaussian Splatting")
    print(f"  - Target Iterations: {train_iters}")
    print("  - Strategy: 2,500-step Densification + 2,500-step Annealing")
    print("=======================================================")
    out_splat_ply = os.path.join(splats_dir, "double_room_splat.ply")

    trainer = GSplatTrainer(
        cams_path=cams_json,
        points_path=dense_ply,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_init_points=100000
    )

    # Use our verified densification + annealing schedule
    from gsplat import DefaultStrategy
    trainer.strategy = DefaultStrategy(
        prune_opa=0.02,
        grow_grad2d=0.00035,
        grow_scale3d=0.01,
        refine_start_iter=200,
        refine_stop_iter=min(2500, int(train_iters * 0.5)),
        refine_every=100,
        reset_every=1200,
        absgrad=True,
        verbose=False
    )
    trainer.strategy_state = trainer.strategy.initialize_state(scene_scale=trainer.scene_scale)

    t_train0 = time.time()
    trainer.train(total_iterations=train_iters, output_ply=out_splat_ply)
    train_time = time.time() - t_train0
    print(f"[M2 Pipeline] Splat training completed in {train_time:.1f}s")

    # =========================================================================
    # Step 3: Multi-Room Renders: Doorway Crossing & Both Rooms
    # =========================================================================
    print("\n=======================================================")
    print("[M2 Pipeline] Step 3: Multi-Room Renders & Doorway Evaluation")
    print("=======================================================")
    render_m2_evaluation(
        ply_path=out_splat_ply,
        cams_path=cams_json,
        out_dir=renders_dir
    )

def render_m2_evaluation(
    ply_path: str,
    cams_path: str,
    out_dir: str
):
    import plyfile
    from gsplat import rasterization
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cams_path) as f:
        cams = json.load(f)

    ply_data = plyfile.PlyData.read(ply_path)
    v = ply_data['vertex']
    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32, device=device)
    scales = torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1), dtype=torch.float32, device=device)
    quats = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1), dtype=torch.float32, device=device)
    opacities = torch.tensor(v['opacity'], dtype=torch.float32, device=device)
    C0 = 0.28209479177387814
    sh0 = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = torch.tensor(np.clip(sh0 * C0 + 0.5, 0.0, 1.0), dtype=torch.float32, device=device)

    base = cams[0]
    fx, fy, cx, cy = base['fx'], base['fy'], base['cx'], base['cy']
    w, h = base['width'], base['height']
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]], dtype=torch.float32, device=device)

    def render_cam(c2w_mat):
        w2c = np.linalg.inv(c2w_mat)
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales), opacities=opacities, colors=colors,
                viewmats=w2c_t, Ks=K, width=w, height=h, near_plane=0.1, packed=False
            )
        return (render_colors[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # 1. Aligned Ground Truth vs Render Grid (covering Room A, Doorway, Room B)
    # Selected keyframes:
    # Frame 10: Room A (Bunk bed & study desk)
    # Frame 20: Room A (Colorful cabinets & mirror)
    # Frame 34: Doorway approach (standing at threshold looking through)
    # Frame 36: Doorway threshold crossing (pull-up bar & door frame)
    # Frame 38: Corridor / entering Room 2
    # Frame 45: Room B (Hourglass wardrobe & TV unit)
    # Frame 55: Room B (Living area & fan)
    # Frame 62: Room B looking back towards hallway
    eval_indices = [10, 20, 34, 36, 38, 45, 55, 62]
    gt_imgs = []
    rend_imgs = []
    for idx in eval_indices:
        c = cams[idx]
        gt = cv2.imread(c['img_path'])
        pred = render_cam(np.array(c['c2w'], dtype=np.float32))
        gt_imgs.append(gt)
        rend_imgs.append(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    gt_top = np.hstack(gt_imgs[:4])
    gt_bot = np.hstack(gt_imgs[4:])
    gt_grid = np.vstack([gt_top, gt_bot])

    rend_top = np.hstack(rend_imgs[:4])
    rend_bot = np.hstack(rend_imgs[4:])
    rend_grid = np.vstack([rend_top, rend_bot])

    full_grid = np.vstack([gt_grid, rend_grid])
    grid_path = os.path.join(out_dir, "m2_multiroom_comparison_grid.jpg")
    cv2.imwrite(grid_path, full_grid)
    print(f"[M2 Evaluation] Aligned comparison grid saved to {grid_path}")

    # 2. Doorway Crossing Sequence (Frames 32 to 40)
    door_crossing_frames = []
    for d_idx in range(32, 41):
        c = cams[d_idx]
        pred = render_cam(np.array(c['c2w'], dtype=np.float32))
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        gt = cv2.imread(c['img_path'])
        pair = np.hstack([gt, pred_bgr])
        p_path = os.path.join(out_dir, f"doorway_step_{d_idx:02d}.jpg")
        cv2.imwrite(p_path, pair)
        door_crossing_frames.append(pair)
    print(f"[M2 Evaluation] Doorway crossing frames (32->40) saved to {out_dir}")

    # 3. Full Multi-Room Walkthrough Animation (crossing Room A -> Doorway -> Room B)
    orbit_gif = os.path.join(out_dir, "multiroom_walkthrough.gif")
    orbit_webp = os.path.join(out_dir, "multiroom_walkthrough.webp")
    orbit_mp4 = os.path.join(out_dir, "multiroom_walkthrough.mp4")
    video_writer = cv2.VideoWriter(orbit_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    anim_frames = []

    for c in cams:
        pred = render_cam(np.array(c['c2w'], dtype=np.float32))
        anim_frames.append(Image.fromarray(pred))
        video_writer.write(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    video_writer.release()
    if anim_frames:
        anim_frames[0].save(orbit_gif, save_all=True, append_images=anim_frames[1:], duration=50, loop=0)
        anim_frames[0].save(orbit_webp, save_all=True, append_images=anim_frames[1:], duration=50, loop=0)
        print(f"[M2 Evaluation] Walkthrough animations saved to {orbit_gif}, {orbit_webp}, {orbit_mp4}")

    # 4. Sync assets to artifact directory if env var set
    art_dir = os.environ.get("ARTIFACTS_DIR")
    if art_dir and os.path.exists(art_dir):
        import shutil
        shutil.copy(grid_path, os.path.join(art_dir, "m2_multiroom_comparison_grid.jpg"))
        shutil.copy(orbit_webp, os.path.join(art_dir, "multiroom_walkthrough.webp"))
        shutil.copy(orbit_gif, os.path.join(art_dir, "multiroom_walkthrough.gif"))
        shutil.copy(orbit_mp4, os.path.join(art_dir, "multiroom_walkthrough.mp4"))
        for d_idx in [34, 36, 38, 40]:
            shutil.copy(os.path.join(out_dir, f"doorway_step_{d_idx:02d}.jpg"), os.path.join(art_dir, f"doorway_step_{d_idx:02d}.jpg"))
        print("[M2 Evaluation] Synced evaluation assets to artifact directory.")

if __name__ == "__main__":
    run_m2_reconstruction()
