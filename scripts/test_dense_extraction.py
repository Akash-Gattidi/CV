import os
import sys
import json
import torch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DUST3R_PATH = os.path.join(PROJECT_ROOT, "dust3r_repo")
CROCO_PATH = os.path.join(DUST3R_PATH, "croco")
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images

def test_dense():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "checkpoints/dust3r_512"
    print("Loading DUSt3R model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(device)
    model.eval()

    with open("output/dust3r/cameras.json", "r") as f:
        cams = json.load(f)

    # Test pair (0, 1)
    imgs = load_images([cams[0]["img_path"], cams[1]["img_path"]], size=512)
    from dust3r.inference import inference
    output = inference([(imgs[0], imgs[1])], model, device, batch_size=1, verbose=False)

    p3d = output["pred1"]["pts3d"][0].detach().cpu().numpy()
    conf = output["pred1"]["conf"][0].detach().cpu().numpy()
    print(f"Pred shape: {p3d.shape}, Conf min={conf.min():.2f}, max={conf.max():.2f}")
    mask = conf > 1.5
    print(f"Confident points in frame 0: {mask.sum()} / {mask.size} ({mask.sum()/mask.size*100:.1f}%)")


if __name__ == "__main__":
    test_dense()
