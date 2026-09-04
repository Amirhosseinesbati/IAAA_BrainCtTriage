"""Training-only paired translation; deployment model and decoder are unchanged."""
from __future__ import annotations
import torch


def overlap_slices(height: int, width: int, dx: int, dy: int):
    if not isinstance(dx, int) or not isinstance(dy, int) or abs(dx) >= width or abs(dy) >= height:
        raise ValueError('Invalid integer translation')
    source = (slice(max(0, -dy), height-max(0, dy)), slice(max(0, -dx), width-max(0, dx)))
    target = (slice(max(0, dy), height-max(0, -dy)), slice(max(0, dx), width-max(0, -dx)))
    return source, target


def translate_image(images, dx, dy):
    source, target = overlap_slices(*images.shape[-2:], dx, dy)
    moved = torch.zeros_like(images)
    moved[..., target[0], target[1]] = images[..., source[0], source[1]]
    return moved


def translated_targets(keypoints, masks, is_target, dx, dy, image_size, sigma):
    """Regenerate Gaussians, including tails newly visible after translation."""
    if dx % 4 or dy % 4 or sigma <= 0 or image_size % 4:
        raise ValueError('Expected quarter-resolution-compatible geometry')
    moved = keypoints + keypoints.new_tensor([dx, dy])
    finite = torch.isfinite(keypoints).all((1, 2)) & torch.isfinite(moved).all((1, 2))
    inside = ((keypoints >= 0) & (keypoints < image_size) & (moved >= 0) & (moved < image_size)).all((1, 2))
    positive_valid = (is_target > .5) & (masks > .5).all(1) & finite & inside
    eligible = (is_target <= .5) | positive_valid
    size = image_size//4
    grid = torch.arange(size, device=keypoints.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    centers = torch.nan_to_num(moved.float())/4
    dist = (xx-centers[..., 0, None, None]).square()+(yy-centers[..., 1, None, None]).square()
    gaussian = (-dist/(2*sigma*sigma)).exp()
    gaussian = gaussian * positive_valid[:, None, None, None]
    return moved, gaussian, eligible, positive_valid


def consistency_js(first_logits, second_logits, positive_valid, dx, dy):
    """Symmetric JS of overlap-renormalized heatmaps; gradients to both views."""
    if dx % 4 or dy % 4 or first_logits.shape != second_logits.shape:
        raise ValueError('Incompatible heatmap translation')
    if not positive_valid.any():
        return (first_logits.sum()+second_logits.sum())*0
    source, target = overlap_slices(*first_logits.shape[-2:], dx//4, dy//4)
    a = first_logits[positive_valid, :, source[0], source[1]].float().flatten(2)
    b = second_logits[positive_valid, :, target[0], target[1]].float().flatten(2)
    lp, lq = a.log_softmax(-1), b.log_softmax(-1)
    lm = torch.logaddexp(lp, lq)-0.6931471805599453
    return (.5*(lp.exp()*(lp-lm)+lq.exp()*(lq-lm))).sum(-1).mean()


def combine_losses(first, second, consistency, weight):
    if weight < 0:
        raise ValueError('Negative consistency weight')
    supervised = (first+second)*.5
    # Exact zero-weight control; no zero*NaN or consistency autograd path.
    return supervised if weight == 0 else supervised+weight*consistency
