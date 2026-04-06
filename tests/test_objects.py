import cv2
import numpy as np
import pytest

from dadinhos.objects import (
    Annotation,
    Sample,
    distribution_count,
    distribution_size,
    load_samples,
    make_detection_task,
    make_objects,
    render_sample,
)


def plot(frame: np.ndarray, sample: Sample[Annotation]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for ann in sample.annotations:
        x1, y1, x2, y2 = ann.bbox
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            img,
            f"{ann.label}",
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    return img


@pytest.mark.skip("")
def test_objects():
    samples = make_objects(
        n_samples=10,
        draw_count=distribution_count,
        draw_size=distribution_size,
    )
    # sourcery skip: no-loop-in-tests
    for sample in samples:
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        image = render_sample(image, sample)
        image = plot(image, sample=sample)
        cv2.imshow("Sample", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_generates(tmp_path):
    annotations = make_detection_task(
        tmp_path / "data" / "annotations.json",
        resolution=(480, 640),
        n_samples=10,
    )
    for sample in load_samples(annotations):
        print(annotations.parent / sample.file_name)
        image = cv2.imread(annotations.parent / sample.file_name)
        print(image)
        image = render_sample(image, sample)
        image = plot(image, sample=sample)
        cv2.imshow("Sample", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
