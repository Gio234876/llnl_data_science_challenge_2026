#!/usr/bin/env python3
"""Measure 3D centerline-to-boundary radii and plot their histogram."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


def load_binary(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")
    array = np.load(path)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {array.shape}")
    return array.astype(bool, copy=False)


def measure(
    mask: np.ndarray,
    skeleton: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask.shape != skeleton.shape:
        raise ValueError(
            f"mask and skeleton shapes differ: {mask.shape} versus {skeleton.shape}"
        )
    outside = skeleton & ~mask
    if np.any(outside):
        raise ValueError(
            f"{int(np.count_nonzero(outside))} skeleton voxels lie outside the mask"
        )

    coordinates = np.argwhere(skeleton)
    if coordinates.size == 0:
        raise ValueError("skeleton contains no foreground voxels")

    distance_map = distance_transform_edt(mask, sampling=spacing)
    radii = distance_map[tuple(coordinates.T)]
    return coordinates, radii, distance_map


def parse_bins(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        bins = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bins must be 'auto' or a positive integer"
        ) from exc
    if bins <= 0:
        raise argparse.ArgumentTypeError("bins must be positive")
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path)
    parser.add_argument("skeleton", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--bins", type=parse_bins, default="auto")
    parser.add_argument("--overlay-slice", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spacing = tuple(float(value) for value in args.spacing)
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("all --spacing values must be finite and positive")

    mask = load_binary(args.mask, "mask")
    skeleton = load_binary(args.skeleton, "skeleton")
    coordinates, radii, distance_map = measure(mask, skeleton, spacing)

    if args.overlay_slice is None:
        slice_counts = np.bincount(coordinates[:, 0], minlength=mask.shape[0])
        overlay_slice = int(np.argmax(slice_counts))
    else:
        overlay_slice = args.overlay_slice
    if not 0 <= overlay_slice < mask.shape[0]:
        raise ValueError(
            f"overlay slice {overlay_slice} is outside [0, {mask.shape[0] - 1}]"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "euclidean_radii.npy", radii)
    radius_map = np.zeros(mask.shape, dtype=np.float32)
    radius_map[tuple(coordinates.T)] = radii.astype(np.float32)
    np.save(args.output_dir / "radius_map.npy", radius_map)

    with (args.output_dir / "euclidean_radii.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("z", "y", "x", "radius"))
        for coordinate, radius in zip(coordinates, radii):
            writer.writerow((*map(int, coordinate), float(radius)))

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(radii, bins=args.bins, color="#3569a8", edgecolor="white")
    axis.set_xlabel("3D Euclidean radius")
    axis.set_ylabel("Frequency")
    axis.set_title("Centerline-to-Boundary Radius Distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "radius_histogram.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(mask[overlay_slice], cmap="gray", interpolation="nearest")
    overlay = np.ma.masked_where(~skeleton[overlay_slice], skeleton[overlay_slice])
    axis.imshow(overlay, cmap="autumn", interpolation="nearest", alpha=0.9)
    axis.set_title(f"Segmentation + Skeleton, z={overlay_slice}")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "segmentation_skeleton_overlay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    units = "voxels" if spacing == (1.0, 1.0, 1.0) else "physical spacing units"
    summary = {
        "mask": str(args.mask),
        "skeleton": str(args.skeleton),
        "measurement": "3D Euclidean distance to nearest segmentation background",
        "spacing_zyx": spacing,
        "units": units,
        "skeleton_voxels": int(radii.size),
        "overlay_slice": overlay_slice,
        "radius_statistics": {
            "minimum": float(np.min(radii)),
            "mean": float(np.mean(radii)),
            "median": float(np.median(radii)),
            "standard_deviation": float(np.std(radii)),
            "maximum": float(np.max(radii)),
        },
    }
    with (args.output_dir / "radius_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
