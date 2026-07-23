from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def threshold_to_suffix(threshold: float) -> str:
    text = format(threshold, 'g')
    return text.replace('-', 'neg').replace('.', 'p')


def build_output_path(input_filepath: str | Path, threshold: float, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_filepath)
    destination_dir = Path(output_dir) if output_dir is not None else input_path.parent
    suffix = threshold_to_suffix(threshold)
    return destination_dir / f"{input_path.stem}_segmented_{suffix}.npy"


def normalize_thresholds(thresholds: Iterable[float] | None) -> List[float]:
    if thresholds is None:
        return [0.3, 0.5, 0.7]
    return [float(value) for value in thresholds]


def plan_segmentations(input_filepath: str | Path, thresholds: Sequence[float] | None = None, output_dir: str | Path | None = None) -> List[Tuple[float, Path]]:
    normalized = normalize_thresholds(thresholds)
    return [(threshold, build_output_path(input_filepath, threshold, output_dir)) for threshold in normalized]


if __name__ == '__main__':
    sample_input = Path('sample.npy')
    for threshold, output_path in plan_segmentations(sample_input):
        print(f"threshold={threshold} -> {output_path}")
