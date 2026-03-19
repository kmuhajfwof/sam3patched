"""
SAM3 Interactive Collectors — Point, BBox, Multi-Region, and Interactive Segmentation

Point/BBox editor widgets adapted from ComfyUI-KJNodes
Original: https://github.com/kijai/ComfyUI-KJNodes
Author: kijai
License: Apache 2.0
"""

import asyncio
import gc
import hashlib
import logging
import json
import io
import base64
import threading

import numpy as np
import torch
from PIL import Image

try:
    import server
    from aiohttp import web

    _SERVER_AVAILABLE = True
except Exception:
    server = None
    _SERVER_AVAILABLE = False
from .utils import comfy_image_to_pil, visualize_masks_on_image, masks_to_comfy_mask, pil_to_comfy_image

log = logging.getLogger("sam3")

# ---------------------------------------------------------------------------
# Interactive segmentation cache — keyed by node unique_id
# ---------------------------------------------------------------------------
_INTERACTIVE_CACHE = {}

# Serializes GPU work from parallel per-prompt requests
_SEGMENT_LOCK = threading.Lock()


class SAM3PointCollector:
    """
    Interactive Point Collector for SAM3

    Displays image canvas in the node where users can click to add:
    - Positive points (Left-click) - green circles
    - Negative points (Shift+Left-click or Right-click) - red circles

    Outputs point arrays to feed into SAM3Segmentation node.
    """

    # Class-level cache for output results
    _cache = {}

    # OpenPose body keypoint indices to use for positive points
    # 0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist, 5=LShoulder,
    # 6=LElbow, 7=LWrist, 8=RHip, 9=RKnee, 10=RAnkle, 11=LHip,
    # 12=LKnee, 13=LAnkle  (skip 14-17 eyes/ears — too close to nose)
    _POSE_KEYPOINT_INDICES = list(range(14))

    # Skeleton limb connections (pairs of keypoint indices) for midpoint generation
    # Each pair defines a body segment; we'll add a midpoint for denser coverage
    _SKELETON_LIMBS = [
        (1, 2),  # Neck → RShoulder
        (1, 5),  # Neck → LShoulder
        (2, 3),  # RShoulder → RElbow
        (3, 4),  # RElbow → RWrist
        (5, 6),  # LShoulder → LElbow
        (6, 7),  # LElbow → LWrist
        (1, 8),  # Neck → RHip (right torso)
        (1, 11),  # Neck → LHip (left torso)
        (8, 9),  # RHip → RKnee
        (9, 10),  # RKnee → RAnkle
        (11, 12),  # LHip → LKnee
        (12, 13),  # LKnee → LAnkle
        (2, 5),  # RShoulder → LShoulder (chest line)
        (8, 11),  # RHip → LHip (hip line)
    ]

    # body_points indices from WanAnimatePreprocess: [0, 1, 2, 5, 8, 11, 10, 13]
    # = Nose, Neck, RShoulder, LShoulder, RHip, LHip, RAnkle, LAnkle
    _BODY_POINTS_NAMES = {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "LShoulder",
        4: "RHip",
        5: "LHip",
        6: "RAnkle",
        7: "LAnkle",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to display in interactive canvas. Left-click to add positive points (green), Shift+Left-click or Right-click to add negative points (red). Points are automatically normalized to image dimensions."
                    },
                ),
                "points_store": ("STRING", {"multiline": False, "default": "{}"}),
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "auto_points_from_pose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, automatically place positive/negative points from detection inputs (bboxes, body_points, or pose_data).",
                    },
                ),
                "negative_radius_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.3,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Multiplier for negative point distance outside person bbox. Higher = farther from body.",
                    },
                ),
                "pose_confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum confidence for a pose keypoint to be used as a positive point.",
                    },
                ),
            },
            "optional": {
                "bboxes": (
                    "BBOX",
                    {
                        "tooltip": "Person bounding box from YOLO detector (e.g. Pose and Face Detection node). Gives precise body boundaries for negative point placement."
                    },
                ),
                "body_points": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "key_frame_body_points JSON from Pose and Face Detection. Used as positive points on the body.",
                    },
                ),
                "pose_data": (
                    "POSEDATA",
                    {"tooltip": "Pose data from ta pose smoother. Fallback when bboxes/body_points are not connected."},
                ),
            },
        }

    RETURN_TYPES = ("SAM3_POINTS_PROMPT", "SAM3_POINTS_PROMPT")
    RETURN_NAMES = ("positive_points", "negative_points")
    FUNCTION = "collect_points"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        points_store,
        coordinates,
        neg_coordinates,
        auto_points_from_pose=False,
        negative_radius_multiplier=1.5,
        pose_confidence_threshold=0.3,
        bboxes=None,
        body_points=None,
        pose_data=None,
    ):
        import hashlib

        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        h.update(str(auto_points_from_pose).encode())
        h.update(str(negative_radius_multiplier).encode())
        h.update(str(pose_confidence_threshold).encode())
        if auto_points_from_pose:
            if bboxes is not None:
                h.update(str(bboxes).encode())
            if body_points is not None:
                h.update(body_points.encode())
            if pose_data is not None:
                h.update(cls._hash_pose_data(pose_data))
        result = h.hexdigest()
        log.debug(f"IS_CHANGED SAM3PointCollector: shape={image.shape}, auto={auto_points_from_pose}")
        return result

    def collect_points(
        self,
        image,
        points_store,
        coordinates,
        neg_coordinates,
        auto_points_from_pose=False,
        negative_radius_multiplier=1.5,
        pose_confidence_threshold=0.3,
        bboxes=None,
        body_points=None,
        pose_data=None,
    ):
        """
        Collect points from interactive canvas or automatically from detection inputs.

        Priority for auto mode:
        1. bboxes (BBOX from YOLO) + body_points — best quality
        2. bboxes (BBOX from YOLO) alone — grid inside bbox
        3. pose_data (POSEDATA) — skeleton keypoints fallback
        """
        # Create cache key from inputs
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(coordinates.encode())
        h.update(neg_coordinates.encode())
        h.update(str(auto_points_from_pose).encode())
        h.update(str(negative_radius_multiplier).encode())
        h.update(str(pose_confidence_threshold).encode())
        if auto_points_from_pose:
            if bboxes is not None:
                h.update(str(bboxes).encode())
            if body_points is not None:
                h.update(body_points.encode())
            if pose_data is not None:
                h.update(self._hash_pose_data(pose_data))
        cache_key = h.hexdigest()

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]

        # Check if we have cached result
        if cache_key in SAM3PointCollector._cache:
            cached_result, cached_ui_extra = SAM3PointCollector._cache[cache_key]
            log.info(f"CACHE HIT - key={cache_key[:8]}, ui_extra keys: {list(cached_ui_extra.keys())}")
            img_base64 = self.tensor_to_base64(image)
            ui = {"bg_image": [img_base64]}
            ui.update(cached_ui_extra)
            log.info(f"CACHE HIT - returning UI keys: {list(ui.keys())}")
            return {"ui": ui, "result": cached_result}

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")
        log.info(f"Image dimensions: {img_width}x{img_height}")

        positive_points = {"points": [], "labels": []}
        negative_points = {"points": [], "labels": []}
        ui_extra = {}

        # --- Parse manual/merged coordinates from the widget ---
        # (these contain user-edited points, or auto-points merged from previous run)
        try:
            pos_coords = json.loads(coordinates) if coordinates and coordinates.strip() else []
            neg_coords = json.loads(neg_coordinates) if neg_coordinates and neg_coordinates.strip() else []
        except json.JSONDecodeError:
            pos_coords = []
            neg_coords = []

        has_widget_points = len(pos_coords) > 0 or len(neg_coords) > 0

        # --- Auto mode ---
        if auto_points_from_pose:
            auto_pos = None
            auto_neg = None

            # Priority 1: BBOX from YOLO (+optional body_points)
            if bboxes is not None:
                auto_pos, auto_neg = self._extract_bbox_points(
                    bboxes, body_points, img_width, img_height, negative_radius_multiplier
                )
                if auto_pos is not None:
                    log.info(f"Auto BBOX: {len(auto_pos['points'])} pos, {len(auto_neg['points'])} neg")

            # Priority 2: pose_data fallback
            if auto_pos is None and pose_data is not None:
                auto_pos, auto_neg = self._extract_pose_points(
                    pose_data, img_width, img_height, negative_radius_multiplier, pose_confidence_threshold
                )
                if auto_pos is not None:
                    log.info(f"Auto pose: {len(auto_pos['points'])} pos, {len(auto_neg['points'])} neg")

            if auto_pos is not None:
                # Send auto-points to JS for display/merging (pixel coordinates)
                auto_pos_pixels = [{"x": p[0] * img_width, "y": p[1] * img_height} for p in auto_pos["points"]]
                auto_neg_pixels = [{"x": p[0] * img_width, "y": p[1] * img_height} for p in auto_neg["points"]]
                ui_extra["auto_positive"] = [json.dumps(auto_pos_pixels)]
                ui_extra["auto_negative"] = [json.dumps(auto_neg_pixels)]

                if has_widget_points:
                    # Widget already has points (user may have edited them) — use widget values
                    log.info(f"Using widget points (user-edited): {len(pos_coords)} pos, {len(neg_coords)} neg")
                    for p in pos_coords:
                        positive_points["points"].append([p["x"] / img_width, p["y"] / img_height])
                        positive_points["labels"].append(1)
                    for n in neg_coords:
                        negative_points["points"].append([n["x"] / img_width, n["y"] / img_height])
                        negative_points["labels"].append(0)
                else:
                    # First run — use auto-generated points directly
                    log.info(f"First run, using auto-generated points")
                    positive_points = auto_pos
                    negative_points = auto_neg
            else:
                log.warning("Auto points: no valid detection data, falling back to manual")
                # Fall through to manual mode below

        # --- Manual mode (fallback if auto didn't produce points) ---
        if not positive_points["points"] and not auto_points_from_pose:
            log.info(f"Manual mode: {len(pos_coords)} pos, {len(neg_coords)} neg")
            for p in pos_coords:
                positive_points["points"].append([p["x"] / img_width, p["y"] / img_height])
                positive_points["labels"].append(1)
            for n in neg_coords:
                negative_points["points"].append([n["x"] / img_width, n["y"] / img_height])
                negative_points["labels"].append(0)
        elif not positive_points["points"] and auto_points_from_pose:
            # Auto mode but nothing worked — still try widget
            for p in pos_coords:
                positive_points["points"].append([p["x"] / img_width, p["y"] / img_height])
                positive_points["labels"].append(1)
            for n in neg_coords:
                negative_points["points"].append([n["x"] / img_width, n["y"] / img_height])
                negative_points["labels"].append(0)

        log.info(f"Output: {len(positive_points['points'])} pos, " f"{len(negative_points['points'])} neg")

        result = (positive_points, negative_points)
        SAM3PointCollector._cache[cache_key] = (result, ui_extra)

        img_base64 = self.tensor_to_base64(image)
        ui = {"bg_image": [img_base64]}
        ui.update(ui_extra)
        log.info(
            f"Returning UI keys: {list(ui.keys())}, result: {len(result[0]['points'])} pos, {len(result[1]['points'])} neg"
        )
        return {"ui": ui, "result": result}

    # ------------------------------------------------------------------
    # Pose-data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pose_attr(obj, key, default=None):
        """Read attribute from dict or object transparently."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _hash_pose_data(pose_data):
        """Return bytes hash fragment for the first frame of pose_data."""
        try:
            pd = pose_data
            if isinstance(pd, dict) and "pose_data" in pd:
                pd = pd["pose_data"]
            metas = SAM3PointCollector._pose_attr(pd, "pose_metas") or SAM3PointCollector._pose_attr(pd, "frames")
            if metas and len(metas) > 0:
                kps = SAM3PointCollector._pose_attr(metas[0], "kps_body")
                if kps is not None:
                    return np.asarray(kps).tobytes()
        except Exception:
            pass
        return b"pose_fallback"

    def _extract_bbox_points(self, bboxes, body_points_json, img_width, img_height, neg_radius_mult):
        """
        Generate high-quality SAM3 point prompts using YOLO bbox + body keypoints.

        Strategy for POSITIVE points:
        - Use body keypoints that fall inside the bbox
        - Add midpoints between adjacent keypoints for denser body coverage
        - If no keypoints available, use anatomical grid inside bbox

        Strategy for NEGATIVE points:
        - Place points just outside each side of the bbox (close boundary)
        - Add corner points far from the body (guaranteed background)
        - All negative points MUST be outside the person bbox
        """
        if not bboxes or len(bboxes) == 0:
            return None, None

        bbox = bboxes[0]
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        else:
            return None, None

        if x2 <= x1 or y2 <= y1:
            return None, None

        bw, bh = x2 - x1, y2 - y1
        log.info(f"BBOX person: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  size={bw:.0f}x{bh:.0f}")

        # ─── POSITIVE points ───────────────────────────────────────────
        positive = {"points": [], "labels": []}
        body_kps = []  # pixel coords of all body keypoints

        # Shrink bbox by 5% to avoid edge artifacts
        margin_x, margin_y = bw * 0.05, bh * 0.05
        inner_x1, inner_y1 = x1 + margin_x, y1 + margin_y
        inner_x2, inner_y2 = x2 - margin_x, y2 - margin_y

        if body_points_json is not None and body_points_json.strip():
            try:
                bp_list = json.loads(body_points_json)
                for bp in bp_list:
                    px, py = float(bp["x"]), float(bp["y"])
                    if inner_x1 <= px <= inner_x2 and inner_y1 <= py <= inner_y2:
                        body_kps.append((px, py))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Add midpoints between adjacent body keypoints for denser coverage
        # body_points indices: [Nose, Neck, RShoulder, LShoulder, RHip, LHip, RAnkle, LAnkle]
        # Semantic connections between these 8 points:
        _bp_limbs = [
            (1, 2),  # Neck → RShoulder
            (1, 3),  # Neck → LShoulder
            (2, 4),  # RShoulder → RHip (right torso)
            (3, 5),  # LShoulder → LHip (left torso)
            (4, 5),  # RHip → LHip
            (4, 6),  # RHip → RAnkle (right leg approx)
            (5, 7),  # LHip → LAnkle (left leg approx)
            (2, 3),  # RShoulder → LShoulder (chest)
            (0, 1),  # Nose → Neck
        ]

        midpoints = []
        if len(body_kps) >= 2:
            for i, j in _bp_limbs:
                if i < len(body_kps) and j < len(body_kps):
                    mx = (body_kps[i][0] + body_kps[j][0]) / 2.0
                    my = (body_kps[i][1] + body_kps[j][1]) / 2.0
                    if inner_x1 <= mx <= inner_x2 and inner_y1 <= my <= inner_y2:
                        midpoints.append((mx, my))

        # Combine keypoints + midpoints, deduplicate nearby points
        all_pos_px = body_kps + midpoints

        # Fallback: anatomical grid inside bbox (if no body keypoints)
        if not all_pos_px:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            all_pos_px = [
                (cx, y1 + bh * 0.15),  # head area
                (cx, y1 + bh * 0.30),  # chest
                (x1 + bw * 0.35, y1 + bh * 0.30),  # right shoulder area
                (x1 + bw * 0.65, y1 + bh * 0.30),  # left shoulder area
                (cx, y1 + bh * 0.50),  # waist
                (x1 + bw * 0.35, y1 + bh * 0.55),  # right hip area
                (x1 + bw * 0.65, y1 + bh * 0.55),  # left hip area
                (x1 + bw * 0.40, y1 + bh * 0.75),  # right thigh
                (x1 + bw * 0.60, y1 + bh * 0.75),  # left thigh
            ]

        # Deduplicate: remove points too close to each other (< 3% of bbox diagonal)
        min_dist = np.sqrt(bw**2 + bh**2) * 0.03
        deduped = []
        for px, py in all_pos_px:
            too_close = False
            for ex, ey in deduped:
                if np.sqrt((px - ex) ** 2 + (py - ey) ** 2) < min_dist:
                    too_close = True
                    break
            if not too_close:
                deduped.append((px, py))

        for px, py in deduped:
            nx, ny = px / img_width, py / img_height
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                positive["points"].append([nx, ny])
                positive["labels"].append(1)

        if not positive["points"]:
            return None, None

        # ─── NEGATIVE points ──────────────────────────────────────────
        negative = {"points": [], "labels": []}

        # Gap between bbox edge and negative point (scaled by multiplier)
        gap = max(bw, bh) * 0.1 * neg_radius_mult

        # Strategy 1: Close boundary — 4 points just outside each bbox edge midpoint
        neg_candidates = [
            ((x1 + x2) / 2.0, max(0, y1 - gap)),  # above
            ((x1 + x2) / 2.0, min(img_height - 1, y2 + gap)),  # below
            (max(0, x1 - gap), (y1 + y2) / 2.0),  # left
            (min(img_width - 1, x2 + gap), (y1 + y2) / 2.0),  # right
        ]

        # Strategy 2: Diagonal outside — 4 points at bbox corner diagonals
        diag_gap = gap * 0.7
        neg_candidates += [
            (max(0, x1 - diag_gap), max(0, y1 - diag_gap)),  # top-left
            (min(img_width - 1, x2 + diag_gap), max(0, y1 - diag_gap)),  # top-right
            (min(img_width - 1, x2 + diag_gap), min(img_height - 1, y2 + diag_gap)),  # bottom-right
            (max(0, x1 - diag_gap), min(img_height - 1, y2 + diag_gap)),  # bottom-left
        ]

        for px, py in neg_candidates:
            # CRITICAL: skip any negative point that falls inside the person bbox
            if x1 <= px <= x2 and y1 <= py <= y2:
                continue
            nx, ny = px / img_width, py / img_height
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                negative["points"].append([nx, ny])
                negative["labels"].append(0)

        log.info(
            f"BBOX auto-points: {len(positive['points'])} pos "
            f"({len(body_kps)} kps + {len(midpoints)} mids), "
            f"{len(negative['points'])} neg, gap={gap:.0f}px"
        )
        return positive, negative

    def _extract_pose_points(self, pose_data, img_width, img_height, neg_radius_mult, conf_threshold):
        """
        Generate SAM3 point prompts from POSEDATA skeleton.

        Positive: confident keypoints + midpoints between skeleton limbs.
        Negative: points just outside the skeleton bounding box on all 8 sides.
        """
        pd = pose_data
        if isinstance(pd, dict) and "pose_data" in pd:
            pd = pd["pose_data"]

        metas = self._pose_attr(pd, "pose_metas") or self._pose_attr(pd, "frames")
        if not metas or not isinstance(metas, list) or len(metas) == 0:
            return None, None

        meta = metas[0]
        kps_body = self._pose_attr(meta, "kps_body")
        kps_body_p = self._pose_attr(meta, "kps_body_p")
        pose_w = self._pose_attr(meta, "width", img_width)
        pose_h = self._pose_attr(meta, "height", img_height)

        if kps_body is None:
            return None, None

        kps_body = np.asarray(kps_body, dtype=np.float64)
        if kps_body.ndim != 2 or kps_body.shape[1] < 2:
            return None, None

        n_kps = kps_body.shape[0]
        if kps_body_p is None:
            kps_body_p = np.ones(n_kps, dtype=np.float64)
        else:
            kps_body_p = np.asarray(kps_body_p, dtype=np.float64).reshape(-1)

        scale_x = img_width / max(int(pose_w), 1)
        scale_y = img_height / max(int(pose_h), 1)
        kps_scaled = kps_body.copy()
        kps_scaled[:, 0] *= scale_x
        kps_scaled[:, 1] *= scale_y

        # Build a set of "valid" keypoint pixel positions (indexed by kp index)
        valid_kp = {}  # idx → (x, y) in pixels
        for idx in self._POSE_KEYPOINT_INDICES:
            if idx >= n_kps:
                continue
            if kps_body_p[idx] < conf_threshold:
                continue
            x, y = float(kps_scaled[idx, 0]), float(kps_scaled[idx, 1])
            if x <= 0 or y <= 0:
                continue
            if 0 < x < img_width and 0 < y < img_height:
                valid_kp[idx] = (x, y)

        if not valid_kp:
            return None, None

        # Positive: keypoints
        all_pos_px = list(valid_kp.values())

        # Positive: midpoints between skeleton limbs where both ends are valid
        for i, j in self._SKELETON_LIMBS:
            if i in valid_kp and j in valid_kp:
                mx = (valid_kp[i][0] + valid_kp[j][0]) / 2.0
                my = (valid_kp[i][1] + valid_kp[j][1]) / 2.0
                if 0 < mx < img_width and 0 < my < img_height:
                    all_pos_px.append((mx, my))

        # Deduplicate nearby points
        valid_arr = np.array(list(valid_kp.values()))
        bbox_diag = float(np.linalg.norm(valid_arr.max(axis=0) - valid_arr.min(axis=0)))
        min_dist = bbox_diag * 0.03
        deduped = []
        for px, py in all_pos_px:
            if not any(np.sqrt((px - ex) ** 2 + (py - ey) ** 2) < min_dist for ex, ey in deduped):
                deduped.append((px, py))

        positive = {"points": [], "labels": []}
        for px, py in deduped:
            positive["points"].append([px / img_width, py / img_height])
            positive["labels"].append(1)

        # Compute body bbox and torso size for negative point scaling
        bbox_min = valid_arr.min(axis=0)
        bbox_max = valid_arr.max(axis=0)
        bw = bbox_max[0] - bbox_min[0]
        bh = bbox_max[1] - bbox_min[1]

        NECK, RHIP, LHIP = 1, 8, 11
        torso_size = 0.0
        if NECK in valid_kp and (RHIP in valid_kp or LHIP in valid_kp):
            hip_pts = [valid_kp[i] for i in (RHIP, LHIP) if i in valid_kp]
            mid_hip = np.mean(hip_pts, axis=0)
            torso_size = float(np.linalg.norm(np.array(valid_kp[NECK]) - mid_hip))
        if torso_size < 10:
            torso_size = bbox_diag / 4.0

        # Negative: points just outside the body bbox on 8 sides
        gap = max(bw, bh) * 0.12 * neg_radius_mult
        x1, y1 = bbox_min[0], bbox_min[1]
        x2, y2 = bbox_max[0], bbox_max[1]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        neg_candidates = [
            (cx, max(0, y1 - gap)),  # top
            (cx, min(img_height - 1, y2 + gap)),  # bottom
            (max(0, x1 - gap), cy),  # left
            (min(img_width - 1, x2 + gap), cy),  # right
            (max(0, x1 - gap * 0.7), max(0, y1 - gap * 0.7)),  # top-left
            (min(img_width - 1, x2 + gap * 0.7), max(0, y1 - gap * 0.7)),  # top-right
            (min(img_width - 1, x2 + gap * 0.7), min(img_height - 1, y2 + gap * 0.7)),  # bottom-right
            (max(0, x1 - gap * 0.7), min(img_height - 1, y2 + gap * 0.7)),  # bottom-left
        ]

        negative = {"points": [], "labels": []}
        for px, py in neg_candidates:
            if x1 <= px <= x2 and y1 <= py <= y2:
                continue
            nx, ny = px / img_width, py / img_height
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                negative["points"].append([nx, ny])
                negative["labels"].append(0)

        log.info(
            f"Pose auto-points: {len(positive['points'])} pos "
            f"({len(valid_kp)} kps + {len(deduped) - len(valid_kp)} mids), "
            f"{len(negative['points'])} neg, torso={torso_size:.0f}px"
        )
        return positive, negative

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64


class SAM3BBoxCollector:
    """
    Interactive BBox Collector for SAM3

    Displays image canvas in the node where users can click and drag to add:
    - Positive bounding boxes (Left-click and drag) - cyan rectangles
    - Negative bounding boxes (Shift+Left-click and drag or Right-click and drag) - red rectangles

    Outputs bbox arrays to feed into SAM3Segmentation node.
    """

    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to display in interactive canvas. Click and drag to draw positive bboxes (cyan), Shift+Click/Right-click and drag to draw negative bboxes (red). Bounding boxes are automatically normalized to image dimensions."
                    },
                ),
                "bboxes": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_bboxes": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("positive_bboxes", "negative_bboxes")
    FUNCTION = "collect_bboxes"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True  # Makes node executable even without outputs connected

    @classmethod
    def IS_CHANGED(cls, image, bboxes, neg_bboxes):
        # Return hash based on actual bbox content, not object identity
        # This ensures downstream nodes don't re-run when bboxes haven't changed
        import hashlib

        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        result = h.hexdigest()
        log.debug(f"IS_CHANGED SAM3BBoxCollector: shape={image.shape}, bboxes={bboxes}, neg_bboxes={neg_bboxes}")
        log.debug(f"IS_CHANGED SAM3BBoxCollector: returning hash={result}")
        return result

    def collect_bboxes(self, image, bboxes, neg_bboxes):
        """
        Collect bounding boxes from interactive canvas

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            bboxes: Positive BBoxes JSON array (hidden widget)
            neg_bboxes: Negative BBoxes JSON array (hidden widget)

        Returns:
            Tuple of (positive_bboxes, negative_bboxes) as separate SAM3_BOXES_PROMPT outputs
        """
        # Create cache key from inputs
        import hashlib

        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(bboxes.encode())
        h.update(neg_bboxes.encode())
        cache_key = h.hexdigest()

        # Check if we have cached result
        if cache_key in SAM3BBoxCollector._cache:
            cached = SAM3BBoxCollector._cache[cache_key]
            log.info(f"CACHE HIT - returning cached result for key={cache_key[:8]}")
            # Still need to return UI update
            img_base64 = self.tensor_to_base64(image)
            return {"ui": {"bg_image": [img_base64]}, "result": cached}  # Return the SAME objects

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse bboxes from JSON
        try:
            pos_bbox_list = json.loads(bboxes) if bboxes and bboxes.strip() else []
            neg_bbox_list = json.loads(neg_bboxes) if neg_bboxes and neg_bboxes.strip() else []
        except json.JSONDecodeError:
            pos_bbox_list = []
            neg_bbox_list = []

        log.info(f"Collected {len(pos_bbox_list)} positive, {len(neg_bbox_list)} negative bboxes")

        # Get image dimensions for normalization
        img_height, img_width = image.shape[1], image.shape[2]
        log.info(f"Image dimensions: {img_width}x{img_height}")

        # Convert to SAM3_BOXES_PROMPT format with boxes and labels
        positive_boxes = []
        positive_labels = []
        negative_boxes = []
        negative_labels = []

        # Add positive bboxes (label = True)
        for bbox in pos_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox["x1"] / img_width
            y1_norm = bbox["y1"] / img_height
            x2_norm = bbox["x2"] / img_width
            y2_norm = bbox["y2"] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            positive_boxes.append([center_x, center_y, width, height])
            positive_labels.append(True)  # Positive boxes
            log.info(
                f"  Positive BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})"
            )

        # Add negative bboxes (label = False)
        for bbox in neg_bbox_list:
            # Normalize bbox coordinates to 0-1 range
            x1_norm = bbox["x1"] / img_width
            y1_norm = bbox["y1"] / img_height
            x2_norm = bbox["x2"] / img_width
            y2_norm = bbox["y2"] / img_height

            # Convert from [x1, y1, x2, y2] to [center_x, center_y, width, height]
            # SAM3 expects boxes in center format
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            width = x2_norm - x1_norm
            height = y2_norm - y1_norm

            negative_boxes.append([center_x, center_y, width, height])
            negative_labels.append(False)  # Negative boxes
            log.info(
                f"  Negative BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}) -> center=({center_x:.3f}, {center_y:.3f}) size=({width:.3f}, {height:.3f})"
            )

        log.info(f"Output: {len(positive_boxes)} positive, {len(negative_boxes)} negative bboxes")

        # Format as SAM3_BOXES_PROMPT (dict with 'boxes' and 'labels' keys)
        positive_prompt = {"boxes": positive_boxes, "labels": positive_labels}
        negative_prompt = {"boxes": negative_boxes, "labels": negative_labels}

        # Cache the result
        result = (positive_prompt, negative_prompt)
        SAM3BBoxCollector._cache[cache_key] = result

        # Send image back to widget as base64
        img_base64 = self.tensor_to_base64(image)

        return {"ui": {"bg_image": [img_base64]}, "result": result}

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        # Convert from [B, H, W, C] to PIL Image
        # Take first image if batch
        img_array = tensor[0].cpu().numpy()
        # Convert from 0-1 float to 0-255 uint8
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64


class SAM3MultiRegionCollector:
    """
    Interactive Multi-Region Collector for SAM3

    Displays image canvas in the node where users can:
    - Click/Right-click: Add positive/negative POINTS
    - Shift + Click/Drag: Add positive/negative BOXES

    Supports multiple prompt regions via tab bar.
    Each prompt region has its own set of points and boxes.

    Outputs a list of prompts for multi-object segmentation.
    """

    # Class-level cache for output results
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to display in interactive canvas. Click to add points, Shift+drag to draw boxes. Use tab bar to manage multiple prompt regions."
                    },
                ),
                "multi_prompts_store": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("SAM3_MULTI_PROMPTS",)
    RETURN_NAMES = ("multi_prompts",)
    FUNCTION = "collect_prompts"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, image, multi_prompts_store):
        import hashlib

        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        return h.hexdigest()

    def collect_prompts(self, image, multi_prompts_store):
        """
        Collect multiple prompt regions from interactive canvas.

        Args:
            image: ComfyUI image tensor [B, H, W, C]
            multi_prompts_store: JSON string containing all prompt regions

        Returns:
            List of prompt dicts, each with positive/negative points/boxes
        """
        import hashlib

        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        cache_key = h.hexdigest()

        # Check cache
        if cache_key in SAM3MultiRegionCollector._cache:
            cached = SAM3MultiRegionCollector._cache[cache_key]
            log.info(f"CACHE HIT - returning cached result for key={cache_key[:8]}")
            img_base64 = self.tensor_to_base64(image)
            return {"ui": {"bg_image": [img_base64]}, "result": cached}

        log.info(f"CACHE MISS - computing new result for key={cache_key[:8]}")

        # Parse stored prompts
        try:
            raw_prompts = json.loads(multi_prompts_store) if multi_prompts_store.strip() else []
        except json.JSONDecodeError:
            raw_prompts = []

        img_height, img_width = image.shape[1], image.shape[2]
        log.info(f"Image dimensions: {img_width}x{img_height}")
        log.info(f"Processing {len(raw_prompts)} prompt regions")

        # Convert to normalized output format
        multi_prompts = []
        for idx, raw_prompt in enumerate(raw_prompts):
            prompt = {
                "id": idx,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }

            # Normalize positive points
            for pt in raw_prompt.get("positive_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["positive_points"]["points"].append([norm_x, norm_y])
                prompt["positive_points"]["labels"].append(1)

            # Normalize negative points
            for pt in raw_prompt.get("negative_points", []):
                norm_x = pt["x"] / img_width
                norm_y = pt["y"] / img_height
                prompt["negative_points"]["points"].append([norm_x, norm_y])
                prompt["negative_points"]["labels"].append(0)

            # Normalize positive boxes (convert x1,y1,x2,y2 to center format)
            for box in raw_prompt.get("positive_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["positive_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["positive_boxes"]["labels"].append(True)

            # Normalize negative boxes
            for box in raw_prompt.get("negative_boxes", []):
                x1_norm = box["x1"] / img_width
                y1_norm = box["y1"] / img_height
                x2_norm = box["x2"] / img_width
                y2_norm = box["y2"] / img_height
                cx = (x1_norm + x2_norm) / 2
                cy = (y1_norm + y2_norm) / 2
                w = x2_norm - x1_norm
                h = y2_norm - y1_norm
                prompt["negative_boxes"]["boxes"].append([cx, cy, w, h])
                prompt["negative_boxes"]["labels"].append(False)

            # Count items for logging
            pos_pts = len(prompt["positive_points"]["points"])
            neg_pts = len(prompt["negative_points"]["points"])
            pos_boxes = len(prompt["positive_boxes"]["boxes"])
            neg_boxes = len(prompt["negative_boxes"]["boxes"])
            log.info(
                f"  Prompt {idx}: {pos_pts} pos pts, {neg_pts} neg pts, {pos_boxes} pos boxes, {neg_boxes} neg boxes"
            )

            # Only include prompts with content
            if (
                prompt["positive_points"]["points"]
                or prompt["negative_points"]["points"]
                or prompt["positive_boxes"]["boxes"]
                or prompt["negative_boxes"]["boxes"]
            ):
                multi_prompts.append(prompt)

        log.info(f"Output: {len(multi_prompts)} non-empty prompts")

        # Cache and return
        result = (multi_prompts,)
        SAM3MultiRegionCollector._cache[cache_key] = result
        img_base64 = self.tensor_to_base64(image)

        return {"ui": {"bg_image": [img_base64]}, "result": result}

    def tensor_to_base64(self, tensor):
        """Convert ComfyUI image tensor to base64 string for JavaScript widget"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64


class SAM3InteractiveCollector:
    """
    Interactive Collector with live segmentation preview.

    Same multi-region prompt UI as SAM3MultiRegionCollector, but also takes
    a SAM3 model and runs segmentation directly.  The widget has a "Run"
    button that calls a custom API route for instant mask overlay without
    having to queue the full workflow.
    """

    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL", {"tooltip": "SAM3 model from LoadSAM3Model node."}),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to segment. Draw points/boxes on the canvas, then click Run for a live mask preview."
                    },
                ),
                "multi_prompts_store": ("STRING", {"multiline": False, "default": "[]"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "SAM3_MULTI_PROMPTS")
    RETURN_NAMES = ("masks", "visualization", "multi_prompts")
    FUNCTION = "segment"
    CATEGORY = "SAM3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, sam3_model, image, multi_prompts_store, unique_id=None):
        h = hashlib.md5()
        h.update(str(image.shape).encode())
        h.update(multi_prompts_store.encode())
        return h.hexdigest()

    # -- helpers reused by both segment() and the API route ----------------

    @staticmethod
    def _parse_raw_prompts(raw_prompts, img_w, img_h):
        """Normalize raw JS prompts (pixel coords) to model format."""
        multi_prompts = []
        for idx, raw in enumerate(raw_prompts):
            prompt = {
                "id": idx,
                "positive_points": {"points": [], "labels": []},
                "negative_points": {"points": [], "labels": []},
                "positive_boxes": {"boxes": [], "labels": []},
                "negative_boxes": {"boxes": [], "labels": []},
            }
            for pt in raw.get("positive_points", []):
                prompt["positive_points"]["points"].append([pt["x"] / img_w, pt["y"] / img_h])
                prompt["positive_points"]["labels"].append(1)
            for pt in raw.get("negative_points", []):
                prompt["negative_points"]["points"].append([pt["x"] / img_w, pt["y"] / img_h])
                prompt["negative_points"]["labels"].append(0)
            for box in raw.get("positive_boxes", []):
                x1n, y1n = box["x1"] / img_w, box["y1"] / img_h
                x2n, y2n = box["x2"] / img_w, box["y2"] / img_h
                prompt["positive_boxes"]["boxes"].append([(x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n])
                prompt["positive_boxes"]["labels"].append(True)
            for box in raw.get("negative_boxes", []):
                x1n, y1n = box["x1"] / img_w, box["y1"] / img_h
                x2n, y2n = box["x2"] / img_w, box["y2"] / img_h
                prompt["negative_boxes"]["boxes"].append([(x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n])
                prompt["negative_boxes"]["labels"].append(False)
            has_content = (
                prompt["positive_points"]["points"]
                or prompt["negative_points"]["points"]
                or prompt["positive_boxes"]["boxes"]
                or prompt["negative_boxes"]["boxes"]
            )
            if has_content:
                multi_prompts.append(prompt)
        return multi_prompts

    @staticmethod
    def _run_prompts(model, state, multi_prompts, img_w, img_h):
        """Run predict_inst for each prompt, return stacked masks + scores."""
        all_masks = []
        all_scores = []
        for prompt in multi_prompts:
            pts, labels = [], []
            for pt in prompt["positive_points"]["points"]:
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(1)
            for pt in prompt["negative_points"]["points"]:
                pts.append([pt[0] * img_w, pt[1] * img_h])
                labels.append(0)
            box_array = None
            pos_boxes = prompt.get("positive_boxes", {}).get("boxes", [])
            if pos_boxes:
                cx, cy, w, h = pos_boxes[0]
                box_array = np.array(
                    [
                        (cx - w / 2) * img_w,
                        (cy - h / 2) * img_h,
                        (cx + w / 2) * img_w,
                        (cy + h / 2) * img_h,
                    ]
                )
            point_coords = np.array(pts) if pts else None
            point_labels = np.array(labels) if labels else None
            if point_coords is None and box_array is None:
                continue
            masks_np, scores_np, _ = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                mask_input=None,
                multimask_output=True,
                normalize_coords=True,
            )
            best_idx = np.argmax(scores_np)
            all_masks.append(torch.from_numpy(masks_np[best_idx]).float())
            all_scores.append(scores_np[best_idx])
        return all_masks, all_scores

    # -- main execution (workflow queue) -----------------------------------

    def segment(self, sam3_model, image, multi_prompts_store, unique_id=None):
        import comfy.model_management

        comfy.model_management.load_models_gpu([sam3_model])
        pil_image = comfy_image_to_pil(image)
        img_w, img_h = pil_image.size

        processor = sam3_model.processor
        model = processor.model  # The actual nn.Module with predict_inst

        # Sync processor device after model load
        if hasattr(processor, "sync_device_with_model"):
            processor.sync_device_with_model()

        state = processor.set_image(pil_image)

        # Cache for the API route
        _INTERACTIVE_CACHE[str(unique_id)] = {
            "sam3_model": sam3_model,
            "model": model,
            "processor": processor,
            "state": state,
            "pil_image": pil_image,
            "img_size": (img_w, img_h),
        }

        # Parse prompts
        try:
            raw_prompts = json.loads(multi_prompts_store) if multi_prompts_store.strip() else []
        except json.JSONDecodeError:
            raw_prompts = []
        multi_prompts = self._parse_raw_prompts(raw_prompts, img_w, img_h)

        # Run segmentation
        all_masks, all_scores = self._run_prompts(model, state, multi_prompts, img_w, img_h)

        if not all_masks:
            empty_mask = torch.zeros(1, img_h, img_w)
            vis_tensor = pil_to_comfy_image(pil_image)
            img_b64 = self._tensor_to_base64(image)
            return {
                "ui": {"bg_image": [img_b64]},
                "result": (empty_mask, vis_tensor, multi_prompts),
            }

        masks = torch.stack(all_masks, dim=0)
        scores = torch.tensor(all_scores)

        # Bounding boxes for visualization
        boxes_list = []
        for i in range(masks.shape[0]):
            coords = torch.where(masks[i] > 0)
            if len(coords[0]) > 0:
                boxes_list.append(
                    [coords[1].min().item(), coords[0].min().item(), coords[1].max().item(), coords[0].max().item()]
                )
            else:
                boxes_list.append([0, 0, 0, 0])
        boxes = torch.tensor(boxes_list).float()

        comfy_masks = masks_to_comfy_mask(masks)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        img_b64 = self._tensor_to_base64(image)
        overlay_b64 = self._pil_to_base64(vis_image)

        return {
            "ui": {"bg_image": [img_b64], "overlay_image": [overlay_b64]},
            "result": (comfy_masks, vis_tensor, multi_prompts),
        }

    @staticmethod
    def _tensor_to_base64(tensor):
        arr = tensor[0].cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _pil_to_base64(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Custom API route for live interactive segmentation
# ---------------------------------------------------------------------------


def _run_segment_sync(cached, raw_prompts):
    """Blocking helper — called from the async route via run_in_executor."""
    sam3_model = cached["sam3_model"]  # ModelPatcher for GPU management
    model = cached["model"]  # processor.model with predict_inst
    state = cached["state"]
    pil_image = cached["pil_image"]
    img_w, img_h = cached["img_size"]

    import comfy.model_management

    comfy.model_management.load_models_gpu([sam3_model])

    multi_prompts = SAM3InteractiveCollector._parse_raw_prompts(raw_prompts, img_w, img_h)
    if not multi_prompts:
        return {"error": "No valid prompts", "num_masks": 0}

    all_masks, all_scores = SAM3InteractiveCollector._run_prompts(model, state, multi_prompts, img_w, img_h)
    if not all_masks:
        return {"error": "No masks generated", "num_masks": 0}

    masks = torch.stack(all_masks, dim=0)
    scores = torch.tensor(all_scores)

    boxes_list = []
    for i in range(masks.shape[0]):
        coords = torch.where(masks[i] > 0)
        if len(coords[0]) > 0:
            boxes_list.append(
                [coords[1].min().item(), coords[0].min().item(), coords[1].max().item(), coords[0].max().item()]
            )
        else:
            boxes_list.append([0, 0, 0, 0])
    boxes = torch.tensor(boxes_list).float()

    vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
    buf = io.BytesIO()
    vis_image.save(buf, format="JPEG", quality=80)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"overlay": overlay_b64, "num_masks": len(all_masks)}


def _run_segment_sync_one(cached, raw_prompt, prompt_name):
    """Run segmentation for a single named prompt. Thread-safe via _SEGMENT_LOCK."""
    log.info("Prompt '%s' dispatched", prompt_name)

    sam3_model = cached["sam3_model"]
    model = cached["model"]
    state = cached["state"]
    pil_image = cached["pil_image"]
    img_w, img_h = cached["img_size"]

    import comfy.model_management

    comfy.model_management.load_models_gpu([sam3_model])

    multi_prompts = SAM3InteractiveCollector._parse_raw_prompts([raw_prompt], img_w, img_h)
    if not multi_prompts:
        log.info("Prompt '%s' result ready (no valid points/boxes)", prompt_name)
        return {"error": "No valid prompt content", "num_masks": 0}

    with _SEGMENT_LOCK:
        all_masks, all_scores = SAM3InteractiveCollector._run_prompts(model, state, multi_prompts, img_w, img_h)

    log.info("Prompt '%s' result ready", prompt_name)

    if not all_masks:
        return {"error": "No masks generated", "num_masks": 0}

    return {"num_masks": len(all_masks)}


if _SERVER_AVAILABLE:

    @server.PromptServer.instance.routes.post("/sam3/interactive_segment_one")
    async def _interactive_segment_one_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        node_id = str(body.get("node_id", ""))
        raw_prompt = body.get("prompt", {})
        prompt_name = str(body.get("prompt_name", "Prompt"))

        cached = _INTERACTIVE_CACHE.get(node_id)
        if not cached:
            return web.json_response(
                {"error": "Model not loaded. Queue the workflow first (Ctrl+Enter)."},
                status=400,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _run_segment_sync_one, cached, raw_prompt, prompt_name)
            return web.json_response(result)
        except Exception as exc:
            log.exception("Interactive segmentation (single prompt '%s') failed", prompt_name)
            return web.json_response({"error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.post("/sam3/interactive_segment")
    async def _interactive_segment_handler(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        node_id = str(body.get("node_id", ""))
        raw_prompts = body.get("prompts", [])

        cached = _INTERACTIVE_CACHE.get(node_id)
        if not cached:
            return web.json_response(
                {"error": "Model not loaded. Queue the workflow first (Ctrl+Enter)."},
                status=400,
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _run_segment_sync, cached, raw_prompts)
            return web.json_response(result)
        except Exception as exc:
            log.exception("Interactive segmentation failed")
            return web.json_response({"error": str(exc)}, status=500)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SAM3PointCollector": SAM3PointCollector,
    "SAM3BBoxCollector": SAM3BBoxCollector,
    "SAM3MultiRegionCollector": SAM3MultiRegionCollector,
    "SAM3InteractiveCollector": SAM3InteractiveCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3PointCollector": "SAM3 Point Collector",
    "SAM3BBoxCollector": "SAM3 BBox Collector",
    "SAM3MultiRegionCollector": "SAM3 Multi-Region Collector",
    "SAM3InteractiveCollector": "SAM3 Interactive Collector",
}
