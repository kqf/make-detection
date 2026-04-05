import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import cv2
import numpy as np
from dataclasses_json import dataclass_json

RelativeXYXY = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    label: int
    score: float


T = TypeVar("T")


@dataclass_json
@dataclass
class Sample(Generic[T]):
    file_name: str
    annotations: list[T]


def save_samples(path: pathlib.Path, samples: list[Sample]):
    with path.open("w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in samples], f, indent=4)  # type: ignore


def _intersects(a: RelativeXYXY, b: RelativeXYXY) -> bool:
    return a[2] > b[0] and a[0] < b[2] and a[3] > b[1] and a[1] < b[3]


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, 0.0, None)
    inter_h = np.clip(y2 - y1, 0.0, None)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


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
    max_iou,
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
        bbox = x, y, x + w, y + h

        if not allow_on_border and not _inside_unit(bbox):
            continue

        boxes = np.array(
            [a.bbox for a in annotations],
            dtype=np.float32,
        ).reshape(-1, 4)

        if np.any(_iou(np.array(bbox, dtype=np.float32), boxes) > max_iou):
            continue

        return tuple(bbox)

    return None


def make_objects(
    draw_count: Callable[[], int],
    draw_size: Callable[[], tuple[float, float]],
    n_samples: int,
    max_iou: float = 0,
    allow_on_border: bool = False,
    max_attempts: int = 100,
    n_classes: int = 1,
) -> list[Sample[Annotation]]:
    output = []
    for i in range(n_samples):
        n_objects = draw_count()
        annotations: list[Annotation] = []

        for i in range(n_objects):
            bbox = draw_object(
                max_attempts=max_attempts,
                allow_on_border=allow_on_border,
                max_iou=max_iou,
                annotations=annotations,
                generate_size=draw_size,
            )

            if bbox is None:
                continue

            annotations.append(
                Annotation(
                    bbox=bbox,
                    label=i % n_classes,
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

    axes = (bw // 2, bh // 2)
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, color, thickness)
    return frame


def cross(frame, bbox, label, color, thickness):
    thickness = max(2, thickness)
    h, w = frame.shape[:2]
    px1, py1, px2, py2, cx, cy, bw, bh = box_to_pixels(bbox, w, h)

    cv2.line(frame, (px1, py1), (px2, py2), color, thickness)
    cv2.line(frame, (px1, py2), (px2, py1), color, thickness)
    return frame


def map_to_bbox(cx, cy, bw, bh, pts):
    sx = bw / 2.0
    sy = bh / 2.0

    return np.array(
        [[int(cx + x * sx), int(cy + y * sy)] for x, y in pts],
        dtype=np.int32,
    ).reshape((-1, 1, 2))


def ngon(frame, bbox, label, color, thickness):
    h, w = frame.shape[:2]
    _, _, _, _, cx, cy, bw, bh = box_to_pixels(bbox, w, h)

    n = max(3, label)

    pts = []
    rotation = math.pi / 6
    for i in range(n):
        theta = 2 * math.pi * i / n - math.pi / 2 + rotation
        pts.append((math.cos(theta), math.sin(theta)))

    pts = map_to_bbox(cx, cy, bw, bh, pts)

    if thickness < 0:
        cv2.fillPoly(frame, [pts], color)
    else:
        cv2.polylines(frame, [pts], True, color, thickness)

    return frame


_SHAPES = {
    0: circle,
    1: cross,
}


def render_sample(
    frame: np.ndarray,
    sample: Sample[Annotation],
    color_map: dict[int, tuple[int, int, int]] | None = None,
    thickness: int = 2,
    fill: bool = True,
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
        frame = _SHAPES.get(ann.label, ngon)(
            frame,
            ann.bbox,
            ann.label,
            get_color(ann.label),
            thickness=-1 if fill else thickness,
        )

    return frame


def make_detection_task(
    annotations: pathlib.Path,
    resolution: tuple[int, int],  # h, w
    images_subfolder: pathlib.Path = pathlib.Path("images"),
    n_samples: int = 1000,
    n_classes: int = 1,
    max_iou: float = 0,
    allow_on_border: bool = False,
    max_attempts: int = 100,
    draw_count: Callable[[], int] = distribution_count,
    draw_size: Callable[[], tuple[float, float]] = distribution_size,
) -> pathlib.Path:
    annotations.parent.mkdir(exist_ok=True, parents=True)
    samples = make_objects(
        draw_count=draw_count,
        draw_size=draw_size,
        n_samples=n_samples,
        max_iou=max_iou,
        allow_on_border=allow_on_border,
        max_attempts=max_attempts,
        n_classes=n_classes,
    )
    save_samples(annotations, samples)

    images = annotations / images_subfolder
    for i, sample in enumerate(samples):
        image = np.full((*resolution, 3), 255, dtype=np.uint8)
        image = render_sample(image, sample)
        cv2.imwrite(str(images / f"{i}.png"), image)
    return annotations
