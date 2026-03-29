import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import cv2
import numpy as np

RelativeXYXY = tuple[float, float, float, float]


@dataclass
class Annotation:
    bbox: RelativeXYXY
    label: str
    score: float


T = TypeVar("T")


@dataclass
class Sample(Generic[T]):
    file_name: str
    annotations: list[T]


def _intersects(a: RelativeXYXY, b: RelativeXYXY) -> bool:
    return a[2] > b[0] and a[0] < b[2] and a[3] > b[1] and a[1] < b[3]


def _inside_unit(b: RelativeXYXY) -> bool:
    return (
        0.0 <= b[0] <= 1.0
        and 0.0 <= b[1] <= 1.0
        and 0.0 <= b[2] <= 1.0
        and 0.0 <= b[3] <= 1.0
    )


def _sample_xy(
    w: float,
    h: float,
    allow_on_border: bool,
) -> tuple[float, float]:
    x = random.uniform(0.0, 1.0 - w * (not allow_on_border))
    y = random.uniform(0.0, 1.0 - h * (not allow_on_border))
    return x, y


def distribution_count(minimum=1, average=5):
    return max(0, int(np.random.poisson(average))) + minimum


def distribution_size():
    log_min, log_max = math.log(0.001), math.log(0.05)
    area = math.exp(random.uniform(log_min, log_max))

    log_ratio = random.uniform(math.log(0.5), math.log(2.0))
    ratio = math.exp(log_ratio)

    w = (area * ratio) ** 0.5
    h = (area / ratio) ** 0.5
    return w, h


def draw_object(
    max_attempts,
    allow_on_border,
    allow_overlaps,
    annotations,
    generate_size,
):
    for _ in range(max_attempts):
        w, h = generate_size()
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)

        if w >= 1.0 or h >= 1.0:
            continue

        x, y = _sample_xy(w, h, allow_on_border)
        bbox = (x, y, x + w, y + h)

        if not allow_on_border and not _inside_unit(bbox):
            continue

        if not allow_overlaps and any(
            _intersects(bbox, ann.bbox) for ann in annotations
        ):
            continue
        return bbox
    return None


def make_objects(
    draw_count: Callable[[], int],
    draw_size: Callable[[], tuple[float, float]],
    n_samples: int,
    allow_overlaps: bool = False,
    allow_on_border: bool = False,
    max_attempts: int = 100,
) -> list[Sample[Annotation]]:
    output = []
    for i in range(n_samples):
        n_objects = draw_count()
        annotations: list[Annotation] = []

        for i in range(n_objects):
            bbox = draw_object(
                max_attempts=max_attempts,
                allow_on_border=allow_on_border,
                allow_overlaps=allow_overlaps,
                annotations=annotations,
                generate_size=draw_size,
            )

            if bbox is None:
                continue

            annotations.append(
                Annotation(
                    bbox=bbox,
                    label=f"{i % 10}",
                    score=0.0,
                )
            )

        output.append(Sample(file_name=f"{i}.png", annotations=annotations))

    return output


def box_to_pixels(bbox, w, h):
    x1, y1, x2, y2 = bbox

    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)

    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2
    bw = px2 - px1
    bh = py2 - py1

    return px1, py1, px2, py2, cx, cy, bw, bh


def fit_to_bbox(cx, cy, bw, bh, pts):
    rx = bw / 2.0
    ry = bh / 2.0

    out = []
    for x, y in pts:
        px = cx + x * rx
        py = cy + y * ry
        out.append([int(px), int(py)])
    return np.array(out, dtype=np.int32).reshape((-1, 1, 2))


def circle(frame, bbox, label, color, thickness):
    h, w = frame.shape[:2]
    px1, py1, px2, py2, cx, cy, bw, bh = box_to_pixels(bbox, w, h)

    radius = min(bw, bh) // 2
    cv2.circle(frame, (cx, cy), radius, color, thickness)
    return frame


def cross(frame, bbox, label, color, thickness):
    h, w = frame.shape[:2]
    px1, py1, px2, py2, cx, cy, bw, bh = box_to_pixels(bbox, w, h)

    cv2.line(frame, (px1, py1), (px2, py2), color, thickness)
    cv2.line(frame, (px1, py2), (px2, py1), color, thickness)
    return frame


def ngon(frame, bbox, label, color, thickness):
    h, w = frame.shape[:2]
    px1, py1, px2, py2, cx, cy, bw, bh = box_to_pixels(bbox, w, h)

    n = max(3, int(label))

    # unit circle points
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n - math.pi / 2
        x = math.cos(theta)
        y = math.sin(theta)
        pts.append((x, y))

    pts = fit_to_bbox(cx, cy, bw, bh, pts)

    if thickness < 0:
        cv2.fillPoly(frame, [pts], color)
    else:
        cv2.polylines(frame, [pts], True, color, thickness)

    return frame


_SHAPES = {
    1: circle,
    2: cross,
}


def render_sample(
    frame: np.ndarray,
    sample: Sample[Annotation],
    color_map: dict[int, tuple[int, int, int]] | None = None,
    thickness: int = 2,
    fill: bool = False,
):
    def get_color(label: int):
        if color_map and label in color_map:
            return color_map[label]
        return (
            int((37 * label) % 255),
            int((17 * label) % 255),
            int((97 * label) % 255),
        )

    for ann in sample.annotations:
        label = int(ann.label)
        color = get_color(label)
        render_fn = _SHAPES.get(label, ngon)
        draw_thickness = -1 if fill else thickness
        frame = render_fn(
            frame,
            ann.bbox,
            label,
            color,
            draw_thickness,
        )

    return frame
