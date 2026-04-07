#!/usr/bin/env python3
"""
Falcon Perception integration for the vision agent.

Wraps Falcon Perception (loaded via mlx-vlm fork) with helper utilities
for grounded visual reasoning. The agent can call:

  - ground_expression(query)        → list of mask metadata
  - extreme_mask(slot, direction)   → find topmost/bottommost/leftmost/rightmost
  - count(slot)                     → number of instances
  - bbox_query(mask_id)             → bounding box for a specific mask

After grounding, the masks are stored per-image in a session state so
the agent can refer back to them by ID.

Requires the YasserdahouML/mlx-vlm fork (grounded_reasoning branch) for
Falcon Perception MLX support:

  pip install git+https://github.com/YasserdahouML/mlx-vlm.git@grounded_reasoning
"""

import os
from PIL import Image
import numpy as np


def _image_region(cx, cy):
    """Return a human-readable region string from normalized coordinates."""
    h = "left" if cx < 0.33 else ("center" if cx < 0.67 else "right")
    v = "top" if cy < 0.33 else ("middle" if cy < 0.67 else "bottom")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    return f"{v}-{h}"


def _detection_to_metadata(det, orig_h, orig_w, mask_id, slot):
    """Convert a Falcon Perception detection to agent-friendly metadata."""
    cx = det["xy"]["x"]
    cy = det["xy"]["y"]
    bw = det["hw"]["w"]
    bh = det["hw"]["h"]

    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0, cx + bw / 2)
    y2 = min(1.0, cy + bh / 2)

    mask_np = None
    if "mask" in det:
        try:
            mask_np = np.array(det["mask"]).astype(bool)
            if not mask_np.any():
                return None
            # True pixel centroid (more accurate than bbox midpoint)
            yx = np.argwhere(mask_np)
            cy = float(yx[:, 0].mean()) / orig_h
            cx = float(yx[:, 1].mean()) / orig_w
            area_fraction = round(float(mask_np.sum()) / (orig_h * orig_w), 4)
        except Exception:
            mask_np = None
            area_fraction = round(float(bw * bh), 4)
    else:
        area_fraction = round(float(bw * bh), 4)

    return {
        "id": mask_id,
        "slot": slot,
        "area_fraction": area_fraction,
        "centroid_norm": {"x": round(cx, 4), "y": round(cy, 4)},
        "bbox_norm": {
            "x1": round(x1, 4), "y1": round(y1, 4),
            "x2": round(x2, 4), "y2": round(y2, 4),
        },
        "image_region": _image_region(cx, cy),
        "_mask_np": mask_np,  # internal — not serialized to JSON
    }


def metadata_to_dict(meta):
    """Strip private fields for JSON serialization."""
    return {k: v for k, v in meta.items() if not k.startswith("_")}


class FalconPerceptionTools:
    """Stateful Falcon Perception wrapper for the agent.

    Maintains a session of grounded masks per image. Agent calls
    `ground(query)` first, then operates on returned mask IDs.
    """

    def __init__(self, fp_model=None, fp_processor=None):
        self.fp_model = fp_model
        self.fp_processor = fp_processor

        # Session state for the current image
        self.current_image = None
        self.current_image_path = None
        self.image_size = None  # (W, H)
        self.masks = {}          # global id → metadata
        self.slots = {}          # slot name → list of mask ids
        self._next_id = 1

    @classmethod
    def load(cls, model_path="/Users/bigneek/models/falcon-perception"):
        """Load Falcon Perception from disk via mlx-vlm."""
        from mlx_vlm import load
        print(f"Loading Falcon Perception from {model_path}...")
        fp_model, fp_processor = load(model_path)
        print(f"  Loaded {type(fp_model).__name__}")
        return cls(fp_model=fp_model, fp_processor=fp_processor)

    def set_image(self, image_path):
        """Set the active image for this session and clear masks."""
        if isinstance(image_path, str):
            self.current_image = Image.open(image_path).convert("RGB")
            self.current_image_path = image_path
        else:
            self.current_image = image_path.convert("RGB")
            self.current_image_path = None
        self.image_size = self.current_image.size  # (W, H)
        self.masks = {}
        self.slots = {}
        self._next_id = 1

    def ground(self, query, slot=None, max_new_tokens=1024):
        """Run Falcon Perception with `query`, store masks under `slot`.

        Returns: a JSON-serializable list of mask metadata.
        """
        if self.current_image is None:
            return {"error": "no active image — call set_image first"}

        if self.fp_model is None:
            return {"error": "Falcon Perception not loaded"}

        if slot is None:
            slot = query.replace(" ", "_")[:32]

        try:
            detections = self.fp_model.generate_perception(
                self.fp_processor,
                image=self.current_image,
                query=query,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            return {"error": f"Falcon Perception failed: {e}"}

        W, H = self.image_size
        new_ids = []
        for det in detections:
            meta = _detection_to_metadata(det, H, W, mask_id=self._next_id, slot=slot)
            if meta is not None:
                self.masks[self._next_id] = meta
                new_ids.append(self._next_id)
                self._next_id += 1

        self.slots.setdefault(slot, []).extend(new_ids)

        return {
            "slot": slot,
            "count": len(new_ids),
            "mask_ids": new_ids,
            "masks": [metadata_to_dict(self.masks[i]) for i in new_ids],
        }

    def extreme(self, slot, direction):
        """Return the mask in `slot` that is most extreme in `direction`.

        direction: 'topmost', 'bottommost', 'leftmost', 'rightmost',
                   'largest', 'smallest'
        """
        if slot not in self.slots:
            return {"error": f"unknown slot '{slot}'. Available: {list(self.slots.keys())}"}
        ids = self.slots[slot]
        if not ids:
            return {"error": f"slot '{slot}' is empty"}

        masks = [self.masks[i] for i in ids]
        key_fn = {
            "topmost":     lambda m: m["centroid_norm"]["y"],          # smallest y
            "bottommost":  lambda m: -m["centroid_norm"]["y"],         # largest y
            "leftmost":    lambda m: m["centroid_norm"]["x"],          # smallest x
            "rightmost":   lambda m: -m["centroid_norm"]["x"],         # largest x
            "smallest":    lambda m: m["area_fraction"],               # smallest area
            "largest":     lambda m: -m["area_fraction"],              # largest area
        }
        if direction not in key_fn:
            return {"error": f"unknown direction '{direction}'. Use: {list(key_fn.keys())}"}

        winner = min(masks, key=key_fn[direction])
        return {
            "direction": direction,
            "slot": slot,
            "winner": metadata_to_dict(winner),
        }

    def count_slot(self, slot):
        """Return the number of masks in `slot`."""
        if slot not in self.slots:
            return {"error": f"unknown slot '{slot}'"}
        return {"slot": slot, "count": len(self.slots[slot])}

    def bbox(self, mask_id):
        """Return bounding box for a specific mask ID."""
        try:
            mid = int(mask_id)
        except Exception:
            return {"error": f"invalid mask id: {mask_id}"}
        if mid not in self.masks:
            return {"error": f"mask {mid} not found"}
        return {"mask_id": mid, **metadata_to_dict(self.masks[mid])}

    def annotate_image(self, mask_ids=None):
        """Draw bounding boxes + labels for masks on the current image.

        Returns a PIL Image with annotations.
        """
        from PIL import ImageDraw, ImageFont

        if self.current_image is None:
            return None

        img = self.current_image.copy()
        W, H = img.size
        draw = ImageDraw.Draw(img, "RGBA")

        # Color palette — high contrast for visibility
        COLORS = [
            (255, 64, 96),    # red
            (32, 200, 255),   # cyan
            (255, 200, 0),    # yellow
            (96, 255, 96),    # green
            (200, 96, 255),   # purple
            (255, 128, 0),    # orange
            (0, 200, 200),    # teal
            (255, 96, 255),   # magenta
        ]

        if mask_ids is None:
            mask_ids = sorted(self.masks.keys())

        for i, mid in enumerate(mask_ids):
            if mid not in self.masks:
                continue
            m = self.masks[mid]
            color = COLORS[i % len(COLORS)]
            color_fill = (*color, 60)  # translucent fill

            bbox = m["bbox_norm"]
            x1, y1 = bbox["x1"] * W, bbox["y1"] * H
            x2, y2 = bbox["x2"] * W, bbox["y2"] * H

            # Draw filled rectangle (translucent)
            draw.rectangle([x1, y1, x2, y2], fill=color_fill, outline=color, width=3)

            # Draw label background
            label = f"#{mid}"
            label_w = 36
            label_h = 24
            draw.rectangle(
                [x1, y1 - label_h, x1 + label_w, y1],
                fill=color, outline=color
            )

            # Draw label text (default font is small but readable)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except Exception:
                font = ImageFont.load_default()

            draw.text((x1 + 6, y1 - label_h + 4), label, fill=(255, 255, 255), font=font)

        return img
