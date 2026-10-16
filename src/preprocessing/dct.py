"""Luma-channel block DCT (Phase 4)."""
import cv2
import numpy as np
import torch


def block_dct(frames: torch.Tensor | np.ndarray, block: int = 8) -> torch.Tensor:
    """8x8 block DCT over the luma channel.

    frames: [N,3,H,W] torch tensor (as from align_crop, any value range)
            or [N,H,W,3] numpy array (RGB, uint8 or float).
    Returns float32 tensor [N, (H/block)*(W/block), block*block], blocks
    in row-major (top-left to bottom-right) order.
    """
    if isinstance(frames, torch.Tensor):
        arr = frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
    else:
        arr = np.asarray(frames).astype(np.float32)

    n, h, w, _ = arr.shape
    if h % block != 0 or w % block != 0:
        raise ValueError(f"H,W ({h},{w}) must both be divisible by block={block}")

    rows, cols = h // block, w // block
    out = np.empty((n, rows * cols, block * block), dtype=np.float32)

    for i in range(n):
        r, g, b = arr[i, ..., 0], arr[i, ..., 1], arr[i, ..., 2]
        luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)  # BT.601

        token = 0
        for by in range(rows):
            for bx in range(cols):
                tile = luma[by * block : (by + 1) * block, bx * block : (bx + 1) * block]
                out[i, token] = cv2.dct(tile).flatten()
                token += 1

    return torch.from_numpy(out)
