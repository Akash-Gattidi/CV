import os
import json
import numpy as np
import plyfile
import cv2
import shutil

def main():
    art_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.abspath("."), "output", "renders"))
    os.makedirs("output/renders_double", exist_ok=True)

    # 1. Load Camera 33
    cams = json.load(open("output/dust3r_double/cameras.json"))
    c33 = cams[33]
    c2w = np.array(c33["c2w"], dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = c33["fx"], c33["fy"], c33["cx"], c33["cy"]
    w, h = c33["width"], c33["height"]

    # 2. Load Frame 33 image
    img_bgr = cv2.imread("output/frames_double/frame_0033.jpg")
    img_h, img_w = img_bgr.shape[:2]
    # In case camera intrinsics differ in scale from image resolution
    scale_x = img_w / w
    scale_y = img_h / h

    # 3. Load stratified seeds
    ply = plyfile.PlyData.read("output/dust3r_double/dust3r_points_100k_stratified.ply")
    pts_world = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)
    rgb = np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1).astype(np.uint8)

    # 4. Project into Frame 33
    pts_cam = (w2c[:3, :3] @ pts_world.T + w2c[:3, 3:4]).T
    valid_z = pts_cam[:, 2] > 0.01
    u = (fx * pts_cam[valid_z, 0] / pts_cam[valid_z, 2]) + cx
    v = (fy * pts_cam[valid_z, 1] / pts_cam[valid_z, 2]) + cy
    z = pts_cam[valid_z, 2]
    colors = rgb[valid_z]

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_f = u[in_frame] * scale_x
    v_f = v[in_frame] * scale_y
    z_f = z[in_frame]
    colors_f = colors[in_frame]

    # Left third (mirror region)
    w_third = img_w / 3.0
    mirror_mask = u_f < w_third

    u_m = u_f[mirror_mask]
    v_m = v_f[mirror_mask]
    z_m = z_f[mirror_mask]
    colors_m = colors_f[mirror_mask]

    # Subsets:
    # A: Surface seeds (Z <= 0.1362)
    mask_surface = z_m <= 0.1362
    # B: Intermediate depth (Z in 0.1362 to 0.1763) -> the 182 seeds
    mask_intermediate = (z_m > 0.1362) & (z_m <= 0.1763)
    # C: Deep virtual seeds (Z >= 0.34) -> the 28 seeds
    mask_deep = z_m >= 0.34

    print(f"Total mirror seeds: {len(z_m)}")
    print(f"Surface seeds (Z <= 0.1362): {np.sum(mask_surface)}")
    print(f"Intermediate seeds (0.1362 < Z <= 0.1763): {np.sum(mask_intermediate)}")
    print(f"Deep virtual seeds (Z >= 0.34): {np.sum(mask_deep)}")

    # Analyze where intermediate seeds project vertically and horizontally
    u_inter = u_m[mask_intermediate]
    v_inter = v_m[mask_intermediate]
    z_inter = z_m[mask_intermediate]
    rgb_inter = colors_m[mask_intermediate]

    print("\n--- Intermediate Seeds (Z 0.136 - 0.176) Spatial Distribution ---")
    print(f"U (horizontal, 0..{w_third:.1f}): min={u_inter.min():.1f}, mean={u_inter.mean():.1f}, max={u_inter.max():.1f}")
    print(f"V (vertical, 0..{img_h}): min={v_inter.min():.1f}, mean={v_inter.mean():.1f}, max={v_inter.max():.1f}")
    print(f"RGB mean: R={rgb_inter[:,0].mean():.1f}, G={rgb_inter[:,1].mean():.1f}, B={rgb_inter[:,2].mean():.1f}")

    # Vertical bands in Frame 33:
    # v in [0, 0.3*H]: top wall / mirror frame top
    # v in [0.3*H, 0.7*H]: middle glass / reflected interior shelves
    # v in [0.7*H, H]: bottom cabinet / desk surface / lower frame
    top_band = np.sum(v_inter < 0.3 * img_h)
    mid_band = np.sum((v_inter >= 0.3 * img_h) & (v_inter < 0.7 * img_h))
    bot_band = np.sum(v_inter >= 0.7 * img_h)
    print(f"Vertical bands: Top (0-30%)={top_band}, Middle/Mirror center (30-70%)={mid_band}, Bottom (70-100%)={bot_band}")

    # Plot overlay on Frame 33 image
    overlay = img_bgr.copy()
    # Dim background slightly for contrast
    dimmed = (overlay * 0.75).astype(np.uint8)

    # Draw surface seeds in Cyan
    for ui, vi in zip(u_m[mask_surface], v_m[mask_surface]):
        cv2.circle(dimmed, (int(round(ui)), int(round(vi))), 2, (255, 255, 0), -1)

    # Draw intermediate seeds in Bright Yellow / Orange
    for ui, vi in zip(u_inter, v_inter):
        cv2.circle(dimmed, (int(round(ui)), int(round(vi))), 3, (0, 165, 255), -1)

    # Draw deep seeds in Bright Magenta
    for ui, vi in zip(u_m[mask_deep], v_m[mask_deep]):
        cv2.circle(dimmed, (int(round(ui)), int(round(vi))), 4, (255, 0, 255), -1)

    # Add vertical dividing line at u = w_third
    cv2.line(dimmed, (int(round(w_third)), 0), (int(round(w_third)), img_h), (0, 255, 0), 2)
    cv2.putText(dimmed, "Left 1/3 (Mirror / Cabinet)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(dimmed, "Cyan: Surface (Z<=0.136)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(dimmed, f"Orange: Intermediate (Z 0.136-0.176, N={np.sum(mask_intermediate)})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(dimmed, f"Magenta: Deep Reflected (Z>=0.34, N={np.sum(mask_deep)})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    out_img_path = "output/renders_double/frame_33_mirror_seeds_overlay.jpg"
    cv2.imwrite(out_img_path, dimmed)
    shutil.copy(out_img_path, os.path.join(art_dir, "frame_33_mirror_seeds_overlay.jpg"))
    print(f"\nSaved annotated seed overlay to {out_img_path}")

if __name__ == "__main__":
    main()
