import os
import sys
import json
import math
import cv2
import torch
import numpy as np
import plyfile

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.splat_trainer import GaussianModel, DifferentiableSplatRenderer

def load_splats(ply_path: str, device: str = "cuda") -> GaussianModel:
    data = plyfile.PlyData.read(ply_path)
    vertex = data['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    # Load SH DC
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)[:, :, None]
    
    # Load log scales
    log_scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    
    # Load rotations
    rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1)
    
    # Opacities
    opacities = vertex['opacity'][:, None]

    init_params = {
        "xyz": xyz,
        "sh_dc": sh_dc,
        "log_scales": log_scales,
        "rotations": rotations,
        "opacities": opacities
    }
    return GaussianModel(init_params, device=device)

def create_lookat_w2c(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, -1, 0])) -> np.ndarray:
    """
    Creates a 4x4 World-to-Camera matrix looking from cam_pos toward target.
    Camera conventions: +Z forward, +X right, -Y up (or OpenCV convention: +Z forward, +Y down, +X right).
    """
    forward = target - cam_pos
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-6:
        forward = np.array([0, 0, 1.0])
    else:
        forward = forward / norm_f

    # Right = forward x up
    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1.0, 0, 0])
    else:
        right = right / norm_r

    # True up = right x forward
    actual_up = np.cross(right, forward)

    # Rotation matrix camera-to-world: columns are right, down, forward
    # OpenCV convention: X right, Y down, Z forward
    R_c2w = np.stack([right, -actual_up, forward], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = cam_pos

    w2c = np.linalg.inv(c2w)
    return w2c

def render_novel_views_and_orbit(
    ply_path: str = "output/splats/room_splat.ply",
    cams_path: str = "output/dust3r/cameras.json",
    out_dir: str = "output/renders",
    n_views: int = 6,
    orbit_frames: int = 60,
    sharpness_multiplier: float = 0.35,
    opacity_boost: float = 1.6
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading splats from {ply_path}...")
    gaussians = load_splats(ply_path, device=device)
    renderer = DifferentiableSplatRenderer(bg_color=(0.95, 0.95, 0.95))

    with open(cams_path, "r") as f:
        cams = json.load(f)

    # Use first camera intrinsics
    base_cam = cams[0]
    fx, fy = base_cam["fx"], base_cam["fy"]
    cx, cy = base_cam["cx"], base_cam["cy"]
    w, h = base_cam["width"], base_cam["height"]

    # Calculate room center and camera trajectory center
    all_cam_pos = np.array([[c["c2w"][0][3], c["c2w"][1][3], c["c2w"][2][3]] for c in cams], dtype=np.float32)
    xyz_points = gaussians.get_xyz.detach().cpu().numpy()
    center = np.median(xyz_points, axis=0)


    # Calculate average camera distance and height
    cam_center = np.mean(all_cam_pos, axis=0)
    radius = np.mean(np.linalg.norm(all_cam_pos[:, :2] - center[:2], axis=1))
    if radius < 0.2:
        radius = 0.5
    cam_z = np.median(all_cam_pos[:, 2])

    print(f"Scene Center: {center}, Orbit Radius: {radius:.2f}m, Cam Height Z: {cam_z:.2f}")
    print(f"Sharpness Multiplier: {sharpness_multiplier} | Opacity Boost: {opacity_boost}")

    # 1. Render 6 Novel Views distributed around the room
    novel_paths = []
    print(f"\nRendering {n_views} novel views around the room...")
    angles = np.linspace(0, 2 * math.pi, n_views, endpoint=False)
    
    for i, angle in enumerate(angles):
        # Position on orbit ring
        pos = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            cam_z + 0.1 * math.sin(angle * 2) # Slight vertical variation
        ])
        w2c = create_lookat_w2c(pos, center)
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)

        with torch.no_grad():
            projected = renderer.project_gaussians(gaussians, w2c_t, fx, fy, cx, cy, w, h)
            if projected is not None:
                rendered_rgb = renderer.render(
                    projected, w, h,
                    sharpness_multiplier=sharpness_multiplier,
                    opacity_boost=opacity_boost
                )
                img_np = (rendered_rgb.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_np = np.full((h, w, 3), 240, dtype=np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, f"novel_view_{i+1:02d}.jpg")
        cv2.imwrite(out_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        novel_paths.append(out_path)
        print(f"Rendered novel view {i+1}/{n_views}: {out_path}")

    # 2. Render smooth flythrough / orbit animation
    print(f"\nRendering {orbit_frames}-frame orbit walkthrough animation...")
    orbit_gif_path = os.path.join(out_dir, "room_orbit.gif")
    orbit_webp_path = os.path.join(out_dir, "room_orbit.webp")

    from PIL import Image
    orbit_pil_frames = []
    orbit_angles = np.linspace(0, 2 * math.pi, orbit_frames, endpoint=False)
    for i, angle in enumerate(orbit_angles):
        pos = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            cam_z + 0.05 * math.sin(angle)
        ])
        w2c = create_lookat_w2c(pos, center)
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)

        with torch.no_grad():
            projected = renderer.project_gaussians(gaussians, w2c_t, fx, fy, cx, cy, w, h)
            if projected is not None:
                rendered_rgb = renderer.render(
                    projected, w, h,
                    sharpness_multiplier=sharpness_multiplier,
                    opacity_boost=opacity_boost
                )
                img_np = (rendered_rgb.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_np = np.full((h, w, 3), 240, dtype=np.uint8)

        orbit_pil_frames.append(Image.fromarray(img_np))

    # Save animated GIF and WebP
    if orbit_pil_frames:
        orbit_pil_frames[0].save(
            orbit_gif_path,
            save_all=True,
            append_images=orbit_pil_frames[1:],
            duration=50,
            loop=0
        )
        orbit_pil_frames[0].save(
            orbit_webp_path,
            save_all=True,
            append_images=orbit_pil_frames[1:],
            duration=50,
            loop=0
        )
        print(f"Orbit animation saved to: {orbit_gif_path} and {orbit_webp_path}")

    # Copy to artifact folder for IDE preview
    art_dir = r"C:\Users\akash\.gemini\antigravity-ide\brain\98e73429-616d-4959-84c9-efc1ac00bb95"
    if os.path.exists(art_dir):
        import shutil, glob
        for f in glob.glob(os.path.join(out_dir, "*.*")):
            shutil.copy(f, os.path.join(art_dir, os.path.basename(f)))
        print("Updated renders copied to artifact directory.")

    return novel_paths, orbit_gif_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharpness", type=float, default=0.35, help="Gaussian radius sharpness factor (lower is sharper)")
    parser.add_argument("--opacity", type=float, default=1.6, help="Opacity multiplier boost")
    parser.add_argument("--views", type=int, default=6, help="Number of novel views")
    args = parser.parse_args()

    render_novel_views_and_orbit(
        sharpness_multiplier=args.sharpness,
        opacity_boost=args.opacity,
        n_views=args.views
    )
