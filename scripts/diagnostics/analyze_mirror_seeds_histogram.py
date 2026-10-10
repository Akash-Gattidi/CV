import os
import sys
import json
import numpy as np
import plyfile

# Load double room cameras
cams = json.load(open("output/dust3r_double/cameras.json"))
c33 = cams[33]
c2w = np.array(c33["c2w"], dtype=np.float32)
w2c = np.linalg.inv(c2w)
fx, fy, cx, cy = c33["fx"], c33["fy"], c33["cx"], c33["cy"]
w, h = c33["width"], c33["height"]

# Load the stratified seeds
ply = plyfile.PlyData.read("output/dust3r_double/dust3r_points_100k_stratified.ply")
pts_world = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1).astype(np.float32)

# Project all seeds into Frame 33 frustum
pts_cam = (w2c[:3, :3] @ pts_world.T + w2c[:3, 3:4]).T
valid_z = pts_cam[:, 2] > 0.01
u = (fx * pts_cam[valid_z, 0] / pts_cam[valid_z, 2]) + cx
v = (fy * pts_cam[valid_z, 1] / pts_cam[valid_z, 2]) + cy
z = pts_cam[valid_z, 2]

in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
u_f = u[in_frame]
v_f = v[in_frame]
z_f = z[in_frame]

# Left third: mirror region
w_third = w / 3.0
mirror_mask = u_f < w_third
wall_mask = u_f >= w_third

z_mirror = z_f[mirror_mask]
z_wall = z_f[wall_mask]

print(f"Frame 33 Total Stratified Seeds in Frustum: {len(z_f)}")
print(f"  - Left Third (Mirror Region): {len(z_mirror)} seeds")
print(f"  - Right Two-Thirds (Physical Room / Wall): {len(z_wall)} seeds")

z_min = float(np.percentile(z_f, 1))
z_max = float(np.percentile(z_f, 99))
z_span = z_max - z_min

z_wall_front = float(np.percentile(z_wall, 20)) # front surface of door frame / wall
z_wall_median = float(np.median(z_wall))

print(f"\nDepth Distribution (Camera Frame, arbitrary DUSt3R scale):")
print(f"  Z_min (1st pct): {z_min:.4f}")
print(f"  Z_max (99th pct): {z_max:.4f}")
print(f"  Total Depth Span: {z_span:.4f}")
print(f"  Physical Wall Plane (front edge / door jamb): {z_wall_front:.4f} (fraction of span: {(z_wall_front - z_min)/z_span*100:.1f}%)")
print(f"  Physical Wall Plane (median): {z_wall_median:.4f} (fraction of span: {(z_wall_median - z_min)/z_span*100:.1f}%)")

# Histogram of mirror seeds
hist, bin_edges = np.histogram(z_mirror, bins=10)
print(f"\nMirror Seeds Depth Histogram (10 bins from {z_mirror.min():.4f} to {z_mirror.max():.4f}):")
for i in range(len(hist)):
    print(f"  Bin {i+1} [{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}]: {hist[i]:4d} seeds")

# Seeds on surface vs behind wall
surface_seeds = np.sum(z_mirror <= z_wall_median)
refl_seeds = np.sum(z_mirror > z_wall_median)
print(f"\nSurface vs Reflected Split on ACTUAL Training Seeds (using Z_wall = {z_wall_median:.4f}):")
print(f"  - On Surface / Frame (Z <= {z_wall_median:.4f}): {surface_seeds} seeds ({surface_seeds/len(z_mirror)*100:.1f}%)")
print(f"  - Behind Wall Plane (Z > {z_wall_median:.4f}): {refl_seeds} seeds ({refl_seeds/len(z_mirror)*100:.1f}%)")
print(f"  - Maximum distance behind wall plane: {z_mirror.max() - z_wall_median:.4f} units (out of {z_span:.4f} total scene span)")
