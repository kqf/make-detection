import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

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

        for _ in range(n_objects):
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
                    label="1",
                    score=0.0,
                )
            )

        output.append(Sample(file_name=f"{i}.png", annotations=annotations))

    return output
