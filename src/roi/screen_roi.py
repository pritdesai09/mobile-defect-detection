"""
Screen ROI detection for live phone photos.

Problem this solves: the memory bank only knows what normal *screen* patches
look like, so fingers, hands, desk and bezel in a full phone photo are all
"anomalous". We therefore find the screen first, warp it to a flat landscape
rectangle, and produce a validity mask so that anything that is not screen
(including a finger lying over the screen) is never scored.

Methods:
  "sam"     SAM 2 prompted with a point at the screen centre (default: image
            centre). Among SAM's multimask outputs, the most rectangular
            plausible-size mask is taken as the screen. The quadrilateral comes
            from its convex hull (so a finger notch doesn't shrink the quad),
            and the mask itself, warped, is the validity mask (so the finger
            notch IS excluded from scoring).
  "opencv"  classical fallback: largest convex 4-sided contour.
  "manual"  a box (x0, y0, x1, y1) or quad you provide.
  "auto"    sam if a segmenter is given, else opencv, else the full image.

Output is always landscape (long side horizontal) to match the MSD reference
orientation; a margin_frac border is invalidated to suppress bezel/edge glare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ScreenROI:
    image: Image.Image           # warped, landscape screen
    valid_mask: np.ndarray       # (H, W) bool in warped coordinates
    quad: np.ndarray             # (4, 2) tl, tr, br, bl in ORIGINAL image coordinates
    method: str
    homography: np.ndarray       # original -> warped
    score: float | None = None   # rectangularity of the detected screen mask


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl (clockwise in image coordinates)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # clockwise starting near -pi (left side)
    start = int(np.argmin(pts.sum(1)))    # tl has smallest x+y
    return np.roll(pts, -start, axis=0)


def rectangularity(mask: np.ndarray) -> float:
    import cv2

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return float(mask.sum() / max(w * h, 1.0))


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    import cv2

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(4, 2))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))


def warp_quad(image_np: np.ndarray, quad: np.ndarray, out_long_side: int | None = None):
    """Perspective-warp quad to an upright landscape rectangle. Returns (warped, H, quad_used)."""
    import cv2

    q = order_quad(quad)
    tl, tr, br, bl = q
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    if h > w:  # portrait phone: rotate the point order so the long edge becomes the top edge
        q = np.array([bl, tl, tr, br], np.float32)
        w, h = h, w
    if out_long_side:
        s = out_long_side / w
        w, h = out_long_side, h * s
    W, H = max(8, int(round(w))), max(8, int(round(h)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, M, q


def _warp_mask(mask: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.warpPerspective(mask.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0


def _erode_margin(valid: np.ndarray, margin_frac: float) -> np.ndarray:
    from src.postprocess.regions import border_valid_mask

    out = valid & border_valid_mask(valid.shape, margin_frac)
    if margin_frac > 0 and not valid.all():
        from scipy.ndimage import binary_erosion

        it = max(1, int(round(min(valid.shape) * margin_frac * 0.5)))
        out &= binary_erosion(valid, iterations=it)
    return out


def detect_screen_mask_sam(image_np: np.ndarray, segmenter, point_xy=None,
                           min_area_frac: float = 0.05, max_area_frac: float = 0.95):
    h, w = image_np.shape[:2]
    pt = np.array([point_xy if point_xy is not None else (w / 2, h / 2)], np.float32)
    segmenter.set_image(image_np)
    masks, scores = segmenter.predict(pt, np.array([1]), None, multimask_output=True)
    best, best_s = None, -1.0
    for m, s in zip(masks, scores):
        frac = m.mean()
        if not (min_area_frac <= frac <= max_area_frac):
            continue
        r = rectangularity(m)
        score = r * (0.5 + 0.5 * float(s)) * min(1.0, frac / 0.2)
        if score > best_s:
            best, best_s = m, score
    return best, (rectangularity(best) if best is not None else None)


def detect_screen_quad_opencv(image_np: np.ndarray, min_area_frac: float = 0.08):
    import cv2

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    cands = []
    edges = cv2.morphologyEx(cv2.Canny(gray, 30, 100), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for binimg in (edges, otsu, 255 - otsu):
        cnts, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_frac * h * w or area > 0.98 * h * w:
                continue
            hull = cv2.convexHull(c)
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [hull], -1, 1, -1)
            q = quad_from_mask(m)
            if q is None:
                continue
            qa = cv2.contourArea(q.astype(np.float32))
            rect = float(area / max(qa, 1.0))
            cands.append((min(rect, 1.0) * (area / (h * w)) ** 0.5, q, m.astype(bool)))
    if not cands:
        return None, None
    _, q, m = max(cands, key=lambda t: t[0])
    return q, m


def detect_screen(
    image: Image.Image,
    method: str = "auto",
    segmenter=None,
    manual_box: tuple[int, int, int, int] | None = None,
    manual_quad: np.ndarray | None = None,
    point_xy: tuple[float, float] | None = None,
    margin_frac: float = 0.03,
    out_long_side: int | None = 1036,
) -> ScreenROI:
    image_np = np.asarray(image.convert("RGB"))
    h, w = image_np.shape[:2]
    screen_mask, score, used = None, None, method

    if method == "manual" or manual_box is not None or manual_quad is not None:
        used = "manual"
        if manual_quad is not None:
            quad = order_quad(manual_quad)
        elif manual_box is not None:
            x0, y0, x1, y1 = manual_box
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        else:
            raise ValueError("method='manual' needs manual_box or manual_quad")
    else:
        quad = None
        if method in ("sam", "auto") and segmenter is not None:
            screen_mask, score = detect_screen_mask_sam(image_np, segmenter, point_xy)
            if screen_mask is not None:
                quad, used = quad_from_mask(screen_mask), "sam"
        if quad is None and method in ("opencv", "auto", "sam"):
            quad, screen_mask = detect_screen_quad_opencv(image_np)
            used = "opencv" if quad is not None else used
            screen_mask = None  # opencv mask is just the hull; no occlusion information
        if quad is None:
            used = "full_image"
            quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

    if screen_mask is not None:
        # Internal holes in the screen mask are content, glare or DEFECTS, not occlusions: fill
        # them so they are scored. Only occluders entering from the screen edge (fingers) stay
        # invalid. Limitation: an occluder fully inside the screen is scored (and may be flagged).
        from scipy.ndimage import binary_fill_holes, binary_opening

        screen_mask = binary_fill_holes(binary_opening(binary_fill_holes(screen_mask), iterations=2))
    warped, M, quad = warp_quad(image_np, quad, out_long_side)
    size = (warped.shape[1], warped.shape[0])
    valid = _warp_mask(screen_mask, M, size) if screen_mask is not None else np.ones(warped.shape[:2], bool)
    valid = _erode_margin(valid, margin_frac)
    return ScreenROI(Image.fromarray(warped), valid, quad, used, M, score)
