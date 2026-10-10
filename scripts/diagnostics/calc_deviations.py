import json
import numpy as np

cams = json.load(open('output/dust3r_double/cameras.json'))
positions = np.array([np.array(c['c2w'])[:3, 3] for c in cams])

# Total camera path length
segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_path_length = np.sum(segment_lengths)

print(f"Total Camera Path Length: {total_path_length:.4f} units (sum of {len(segment_lengths)} segments)")

# Novel poses used earlier:
pairs = [
    (34, 35, 0.5, 'Novel 01 (between 34 & 35)'),
    (35, 36, 0.5, 'Novel 02 (between 35 & 36)'),
    (36, 37, 0.4, 'Novel 03 (between 36 & 37)'),
    (37, 38, 0.5, 'Novel 04 (between 37 & 38)'),
    (38, 39, 0.5, 'Novel 05 (between 38 & 39)'),
    (39, 40, 0.5, 'Novel 06 (between 39 & 40)'),
    (40, 41, 0.5, 'Novel 07 (between 40 & 41)')
]

for idx, (i1, i2, alpha, name) in enumerate(pairs):
    p1 = positions[i1]
    p2 = positions[i2]
    # base pos + sway
    sway = 0.006 * (-1.0 if idx % 2 == 0 else 1.0)
    # rot X
    c1 = np.array(cams[i1]['c2w'])
    pos_novel = (1 - alpha) * p1 + alpha * p2 + c1[:3, 0] * sway
    
    # Distance to nearest training pose
    dists = np.linalg.norm(positions - pos_novel, axis=1)
    nearest_idx = np.argmin(dists)
    min_dist = dists[nearest_idx]
    fraction = min_dist / total_path_length
    
    print(f"{name}:")
    print(f"  Nearest Training Cam: {nearest_idx} (dist = {min_dist:.5f})")
    print(f"  Deviation Fraction of Path Length: {fraction * 100:.2f}% ({fraction:.4f})")
