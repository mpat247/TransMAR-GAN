import os
import os.path as path
import numpy as np
import random
import torch
import torch.utils.data as udata
from PIL import Image
import cv2

def _to_minus1_1(x):
    # x expected in [0,1]
    return (x * 2.0 - 1.0).astype(np.float32)

def _normalize_to_0_1(img):
    # robust per-slice min-max; assumes HU-like grayscale
    imin, imax = img.min(), img.max()
    if imax <= imin:
        return np.zeros_like(img, dtype=np.float32)
    x = (img - imin) / (imax - imin)
    return x.astype(np.float32)

def _augment(*args, hflip=True, vflip=True):
    h = hflip and (random.random() < 0.5)
    v = vflip and (random.random() < 0.5)
    out = []
    for a in args:
        if h: a = a[:, ::-1]
        if v: a = a[::-1, :]
        out.append(a)
    return out

def _to_chw1(x):
    # (H,W) -> (1,H,W)
    return np.transpose(np.expand_dims(x, 2), (2, 0, 1)).astype(np.float32)

def _linear_interp_masked_rowwise(img, mask):
    """Basic image-domain 'LI': interpolate masked pixels along rows."""
    out = img.copy()
    H, W = img.shape
    for r in range(H):
        row = out[r]
        msk = mask[r].astype(bool)
        if not msk.any():
            continue
        xs = np.arange(W)
        known = ~msk
        if known.sum() >= 2:
            out[r, msk] = np.interp(xs[msk], xs[known], row[known])
        else:
            # fallback: leave as is; we'll fix by column pass or inpainting
            pass
    return out

def _linear_interp_masked_colwise(img, mask):
    out = img.copy()
    H, W = img.shape
    xs = np.arange(H)
    for c in range(W):
        col = out[:, c]
        msk = mask[:, c].astype(bool)
        if not msk.any(): 
            continue
        known = ~msk
        if known.sum() >= 2:
            out[msk, c] = np.interp(xs[msk], xs[known], col[known])
    return out

def _telea_inpaint(img01, mask, radius=3):
    # expects img in [0,1], mask in {0,1}
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    m8 = (mask > 0).astype(np.uint8) * 255
    out8 = cv2.inpaint(img8, m8, radius, cv2.INPAINT_TELEA)
    return (out8.astype(np.float32) / 255.0)

class SpineWebTrainDataset(udata.Dataset):
    """
    Returns (LI, B, O) each as torch.Tensor float32 in [-1,1], shape (1, 64, 64).
    LI = artifact input (same as O here, i.e., no sinogram LI)
    B  = pseudo-clean target obtained by mask-based interpolation/inpainting
    O  = artifact image
    """
    def __init__(self, root_dir, split_dir="train", image_size=(416,416), patch_size=64,
                 max_hu_artifact=3000, dilate_iter=1):
        super().__init__()
        self.patch = patch_size
        self.image_size = image_size
        # Collect file paths
        self.artifact_dir = path.join(root_dir, split_dir, "artifact")
        if not path.isdir(self.artifact_dir):
            raise FileNotFoundError(f"Artifact dir not found: {self.artifact_dir}")
        # gather .npy files
        self.items = []
        for dp,_,files in os.walk(self.artifact_dir):
            for f in files:
                if f.endswith(".npy"):
                    self.items.append(path.join(dp, f))
        if len(self.items) == 0:
            raise RuntimeError(f"No .npy slices found under {self.artifact_dir}")
        self.max_hu_artifact = max_hu_artifact
        self.dilate_iter = dilate_iter
        self.rng = np.random.RandomState(123)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f = self.items[idx]
        img = np.load(f)  # raw HU or pre-normalized int
        # resize to model's base size (same as your first dataset before 64x64 crops)
        img = np.array(Image.fromarray(img).resize(self.image_size, Image.BILINEAR)).astype(np.float32)

        # 1) Build metal mask M by threshold (SpineWeb used config['max_hu'])
        M = (img > self.max_hu_artifact).astype(np.uint8)
        if self.dilate_iter > 0:
            kernel = np.ones((3,3), np.uint8)
            M = cv2.dilate(M, kernel, iterations=self.dilate_iter)

        # 2) Create pseudo-clean target by image-domain interpolation + inpaint fallback
        img01 = _normalize_to_0_1(img)  # [0,1]
        # do row-wise then col-wise interpolation
        interp = _linear_interp_masked_rowwise(img01, M)
        # fill remaining masked pixels by column-wise interp
        rem_mask = (M > 0) & (np.abs(interp - img01) < 1e-8)  # crude check for untouched
        if rem_mask.any():
            interp2 = _linear_interp_masked_colwise(interp, rem_mask.astype(np.uint8))
        else:
            interp2 = interp
        # Telea fallback if mask remains
        rem_mask2 = (M > 0) & (np.abs(interp2 - img01) < 1e-8)
        if rem_mask2.any():
            interp2[rem_mask2] = _telea_inpaint(interp2, rem_mask2.astype(np.uint8))[rem_mask2]

        B_full = _to_minus1_1(np.clip(interp2, 0, 1))   # pseudo-clean in [-1,1]
        O_full = _to_minus1_1(np.clip(img01,  0, 1))    # artifact in [-1,1]
        LI_full = O_full                                # no sinogram LI -> use artifact as input

        # 3) Random 64x64 crop + flips to match your first dataset
        H, W = O_full.shape
        ph = pw = self.patch
        if H == ph:
            r, c = 0, 0
        else:
            r = self.rng.randint(0, H - ph)
            c = self.rng.randint(0, W - pw)
        O = O_full[r:r+ph, c:c+pw]
        B = B_full[r:r+ph, c:c+pw]
        LI = LI_full[r:r+ph, c:c+pw]

        O, B, LI = _augment(O, B, LI)

        # 4) (H,W) -> (1,H,W) tensors in [-1,1]
        O = _to_chw1(O)
        B = _to_chw1(B)
        LI = _to_chw1(LI)

        return torch.from_numpy(LI.copy()), torch.from_numpy(B.copy()), torch.from_numpy(O.copy())

