import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

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


def make_objects(
    distribution_count: Callable[[], int],
    distribution_size: Callable[[], tuple[float, float]],
    n_samples: int,
    allow_overlaps: bool,
    allow_on_border: bool,
) -> list[Sample[Annotation]]:
    output = []
    max_attempts = 100

    for i in range(n_samples):
        n_objects = distribution_count()
        annotations: list[Annotation] = []

        for _ in range(n_objects):
            for _ in range(max_attempts):
                w, h = distribution_size()
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

                annotations.append(
                    Annotation(
                        bbox=bbox,
                        label=1,
                        score=0.0,
                    )
                )
                break

        output.append(Sample(file_name=f"{i}.png", annotations=annotations))

    return output
