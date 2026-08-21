#!/usr/bin/env python3
"""Measure and classify registered lattice struts from a segmented CT volume."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "outputs/detect_missing_struts_test_40000"
DEFAULT_GRAPH = (
    ROOT
    / "data/missing_struts/registered_jsons"
    / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/recognize_thin_or_thick_struts"

THRESHOLD_RAW_INTENSITY = 40_000
UNIT_CELL_MM = 4.56
REGISTERED_UNIT_CELL_VOXELS = 78.97761899
VOXEL_SIZE_UM = UNIT_CELL_MM * 1_000 / REGISTERED_UNIT_CELL_VOXELS
NOMINAL_DIAMETER_UM = 350.0
NOMINAL_RADIUS_UM = NOMINAL_DIAMETER_UM / 2
NOMINAL_RADIUS_VOXELS = NOMINAL_RADIUS_UM / VOXEL_SIZE_UM
CAD_TOLERANCE_FRACTION = 0.20
THIN_LIMIT_VOXELS = NOMINAL_RADIUS_VOXELS * (1 - CAD_TOLERANCE_FRACTION)
THICK_LIMIT_VOXELS = NOMINAL_RADIUS_VOXELS * (1 + CAD_TOLERANCE_FRACTION)

TRIM_FRACTION = 0.20
LONGITUDINAL_SAMPLES = 25
ANGLE_PAIRS = 16
RAY_STEP_VOXELS = 0.25
MAX_RAY_DISTANCE_VOXELS = 10.0
MIN_VALID_ANGLE_FRACTION = 0.75
MIN_VALID_CROSS_SECTION_FRACTION = 0.70
MIN_SKELETON_COVERAGE = 0.70
MAX_UNSUPPORTED_RUN_FRACTION = 0.20
MISSING_DEFECT_THRESHOLD = 0.50
MAX_ROBUST_VARIATION = 0.20
SKELETON_ASSOCIATION_TOLERANCE = 5.5845609949697455


def orientation_plane(direction_xyz: np.ndarray) -> str:
    """Assign a strut to the coordinate plane it most nearly occupies."""
    if direction_xyz.shape != (3,) or not np.all(np.isfinite(direction_xyz)):
        raise ValueError("direction_xyz must contain three finite values")
    if np.linalg.norm(direction_xyz) == 0:
        raise ValueError("strut direction cannot be zero")
    # The component closest to zero is the plane normal.
    return ("YZ", "XZ", "XY")[int(np.argmin(np.abs(direction_xyz)))]


def plane_orientation_calibration(
    rows: list[dict], nominal_radius_voxels: float
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Return plane medians, CAD/median factors, and finite sample counts."""
    medians: dict[str, float] = {}
    factors: dict[str, float] = {}
    counts: dict[str, int] = {}
    for plane in ("XY", "XZ", "YZ"):
        radii = np.asarray(
            [
                row["radius_voxels_raw"]
                for row in rows
                if row["orientation_plane"] == plane
                and row["radius_voxels_raw"] is not None
            ],
            dtype=float,
        )
        counts[plane] = int(radii.size)
        if radii.size == 0:
            raise ValueError(f"cannot calibrate {plane}: no finite strut radii")
        plane_median = float(np.median(radii))
        if not np.isfinite(plane_median) or plane_median <= 0:
            raise ValueError(f"cannot calibrate {plane}: invalid median radius")
        medians[plane] = plane_median
        factors[plane] = float(nominal_radius_voxels / plane_median)
    return medians, factors, counts


def perpendicular_basis(direction_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction_xyz / np.linalg.norm(direction_xyz)
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.8
        else np.array([0.0, 1.0, 0.0])
    )
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def ray_boundary_distances(
    mask: np.ndarray,
    center_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    crop_origin_zyx: np.ndarray,
) -> np.ndarray:
    """Measure center-to-background distances along several rays."""
    distances = np.arange(
        RAY_STEP_VOXELS,
        MAX_RAY_DISTANCE_VOXELS + RAY_STEP_VOXELS / 2,
        RAY_STEP_VOXELS,
        dtype=np.float32,
    )
    points_xyz = (
        center_xyz[None, None, :]
        + directions_xyz[:, None, :] * distances[None, :, None]
    )
    points_zyx = points_xyz[..., ::-1] - crop_origin_zyx
    indices = np.rint(points_zyx).astype(np.int32)
    inside = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=-1)
    material = np.zeros(inside.shape, dtype=bool)
    valid_indices = indices[inside]
    material[inside] = mask[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    material &= inside
    background_seen = ~material
    has_boundary = background_seen.any(axis=1)
    first_background = np.argmax(background_seen, axis=1)
    result = np.full(len(directions_xyz), np.nan, dtype=np.float32)
    result[has_boundary] = np.maximum(
        RAY_STEP_VOXELS / 2,
        distances[first_background[has_boundary]] - RAY_STEP_VOXELS / 2,
    )
    return result


def measure_cross_section(
    mask: np.ndarray,
    center_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    crop_origin_zyx: np.ndarray,
) -> float:
    first, second = perpendicular_basis(direction_xyz)
    angles = np.linspace(0, np.pi, ANGLE_PAIRS, endpoint=False)
    positive = (
        np.cos(angles)[:, None] * first[None, :]
        + np.sin(angles)[:, None] * second[None, :]
    )
    directions = np.concatenate([positive, -positive], axis=0)
    ray_distances = ray_boundary_distances(
        mask, center_xyz, directions, crop_origin_zyx
    )
    pair_radii = (
        ray_distances[:ANGLE_PAIRS] + ray_distances[ANGLE_PAIRS:]
    ) / 2
    valid = np.isfinite(pair_radii)
    if valid.mean() < MIN_VALID_ANGLE_FRACTION:
        return float("nan")
    return float(np.median(pair_radii[valid]))


def load_score_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["strut_id"]): row for row in csv.DictReader(handle)}


def classify_strut(
    median_radius: float,
    valid_fraction: float,
    skeleton_coverage: float,
    defect_score: float,
    longest_unsupported_fraction: float,
    robust_variation: float,
) -> tuple[str, list[str]]:
    reasons = []
    if valid_fraction < MIN_VALID_CROSS_SECTION_FRACTION:
        reasons.append("low_support")
    if skeleton_coverage < MIN_SKELETON_COVERAGE:
        reasons.append("low_skeleton_coverage")
    if (
        defect_score >= MISSING_DEFECT_THRESHOLD
        or longest_unsupported_fraction > MAX_UNSUPPORTED_RUN_FRACTION
    ):
        reasons.append("disconnected")
    if np.isfinite(robust_variation) and robust_variation > MAX_ROBUST_VARIATION:
        reasons.append("excessive_variation")
    if not np.isfinite(median_radius):
        reasons.append("no_valid_measurement")
    if reasons:
        return "uncertain", sorted(set(reasons))
    if median_radius < THIN_LIMIT_VOXELS:
        return "thin", []
    if median_radius > THICK_LIMIT_VOXELS:
        return "thick", []
    return "normal", []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--web-output",
        type=Path,
        help="Optional JSON export path for an external visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.source / "analysis_metadata.json").read_text())
    source_threshold = float(metadata["threshold_raw_intensity"])
    if not np.isfinite(source_threshold) or not 0 <= source_threshold <= 65_535:
        raise ValueError("source segmentation threshold is invalid")
    crop_origin_zyx = np.asarray(metadata["crop_origin_zyx"], dtype=np.float32)
    mask = np.load(args.source / "segmentation_mask.npy", mmap_mode="r")
    skeleton = np.load(args.source / "skeleton.npy", mmap_mode="r")
    if mask.ndim != 3 or skeleton.ndim != 3 or mask.shape != skeleton.shape:
        raise ValueError("mask and skeleton must be matching 3D arrays")
    if np.any(skeleton & ~mask):
        raise ValueError("skeleton contains voxels outside segmentation")

    graph = json.loads(args.graph.read_text())
    junctions = np.asarray([j["position"] for j in graph["junctions"]])
    score_rows = load_score_rows(args.source / "strut_scores.csv")

    skeleton_zyx = np.argwhere(skeleton)
    skeleton_xyz = (skeleton_zyx + crop_origin_zyx).astype(np.float32)[:, ::-1]
    skeleton_tree = cKDTree(skeleton_xyz)
    sample_parameters = np.linspace(
        TRIM_FRACTION, 1 - TRIM_FRACTION, LONGITUDINAL_SAMPLES
    )

    results = []
    for index, strut in enumerate(graph["struts"]):
        strut_id = int(strut["id"])
        endpoint0 = junctions[int(strut["junction0"])]
        endpoint1 = junctions[int(strut["junction1"])]
        direction = endpoint1 - endpoint0
        registered_centers = (
            endpoint0[None, :]
            + sample_parameters[:, None] * direction[None, :]
        )
        association_distances, skeleton_indices = skeleton_tree.query(
            registered_centers, workers=-1
        )
        associated = (
            association_distances <= SKELETON_ASSOCIATION_TOLERANCE
        )
        local_radii = np.full(LONGITUDINAL_SAMPLES, np.nan, dtype=np.float32)
        for sample_index in np.flatnonzero(associated):
            center = skeleton_xyz[skeleton_indices[sample_index]]
            center_zyx = np.rint(center[::-1] - crop_origin_zyx).astype(int)
            if not mask[tuple(center_zyx)]:
                continue
            local_radii[sample_index] = measure_cross_section(
                mask, center, direction, crop_origin_zyx
            )

        valid = np.isfinite(local_radii)
        valid_fraction = float(valid.mean())
        measured = local_radii[valid]
        if len(measured):
            median_radius = float(np.median(measured))
            radius_mad = float(np.median(np.abs(measured - median_radius)))
            robust_variation = (
                1.4826 * radius_mad / median_radius
                if median_radius > 0
                else float("inf")
            )
            radius_p10, radius_p90 = map(
                float, np.percentile(measured, [10, 90])
            )
        else:
            median_radius = radius_mad = robust_variation = float("nan")
            radius_p10 = radius_p90 = float("nan")

        score = score_rows[strut_id]
        skeleton_coverage = float(score["skeleton_coverage"])
        defect_score = float(score["defect_score"])
        longest_unsupported = float(score["longest_unsupported_fraction"])
        confidence = max(
            0.0,
            min(
                1.0,
                valid_fraction
                * min(1.0, skeleton_coverage / MIN_SKELETON_COVERAGE)
                * max(0.0, 1 - defect_score)
                * max(
                    0.0,
                    1
                    - (
                        robust_variation / MAX_ROBUST_VARIATION
                        if np.isfinite(robust_variation)
                        else 1
                    ),
                ),
            ),
        )
        finite_or_none = lambda value: (
            float(value) if np.isfinite(value) else None
        )
        results.append(
            {
                "strut_id": strut_id,
                "junction0": int(strut["junction0"]),
                "junction1": int(strut["junction1"]),
                "unit_cell_edge_idx": int(strut["unit_cell_edge_idx"]),
                "endpoint0_xyz": endpoint0.tolist(),
                "endpoint1_xyz": endpoint1.tolist(),
                "orientation_plane": orientation_plane(direction),
                "radius_voxels_raw": finite_or_none(median_radius),
                "radius_p10_voxels": finite_or_none(radius_p10),
                "radius_p90_voxels": finite_or_none(radius_p90),
                "radius_mad_voxels": finite_or_none(radius_mad),
                "robust_variation": finite_or_none(robust_variation),
                "valid_cross_sections": int(valid.sum()),
                "valid_fraction": valid_fraction,
                "skeleton_coverage": skeleton_coverage,
                "defect_score": defect_score,
                "longest_unsupported_fraction": longest_unsupported,
                "confidence": confidence,
            }
        )
        if (index + 1) % 500 == 0:
            print(f"Measured {index + 1:,}/{len(graph['struts']):,} struts")

    plane_medians, correction_factors, plane_counts = (
        plane_orientation_calibration(results, NOMINAL_RADIUS_VOXELS)
    )
    for row in results:
        raw_radius = row["radius_voxels_raw"]
        factor = correction_factors[row["orientation_plane"]]
        corrected_radius = None if raw_radius is None else raw_radius * factor
        classification, uncertainty_reasons = classify_strut(
            float("nan") if corrected_radius is None else corrected_radius,
            row["valid_fraction"],
            row["skeleton_coverage"],
            row["defect_score"],
            row["longest_unsupported_fraction"],
            (
                float("nan")
                if row["robust_variation"] is None
                else row["robust_variation"]
            ),
        )
        row["orientation_correction_factor"] = factor
        row["radius_voxels"] = corrected_radius
        row["radius_um"] = (
            None if corrected_radius is None else corrected_radius * VOXEL_SIZE_UM
        )
        row["diameter_um"] = (
            None
            if corrected_radius is None
            else corrected_radius * VOXEL_SIZE_UM * 2
        )
        row["classification"] = classification
        row["uncertainty_reasons"] = uncertainty_reasons

    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["classification"] for row in results)
    summary = {
        "measurement": (
            "median multi-angle opposing-boundary radius over the trimmed "
            "central 60% of each registered strut"
        ),
        "threshold_raw_intensity": source_threshold,
        "voxel_size_um": VOXEL_SIZE_UM,
        "nominal_diameter_um": NOMINAL_DIAMETER_UM,
        "nominal_radius_voxels": NOMINAL_RADIUS_VOXELS,
        "cad_tolerance_fraction": CAD_TOLERANCE_FRACTION,
        "orientation_correction": {
            "method": "CAD nominal radius divided by median measured radius in each coordinate plane",
            "applied_before_cad_tolerance_rule": True,
            "plane_median_radius_voxels": plane_medians,
            "plane_correction_factors": correction_factors,
            "finite_strut_counts": plane_counts,
        },
        "thin_limit_radius_voxels": THIN_LIMIT_VOXELS,
        "thick_limit_radius_voxels": THICK_LIMIT_VOXELS,
        "thin_limit_diameter_um": NOMINAL_DIAMETER_UM
        * (1 - CAD_TOLERANCE_FRACTION),
        "thick_limit_diameter_um": NOMINAL_DIAMETER_UM
        * (1 + CAD_TOLERANCE_FRACTION),
        "uncertainty_rules": {
            "minimum_valid_cross_section_fraction": (
                MIN_VALID_CROSS_SECTION_FRACTION
            ),
            "minimum_skeleton_coverage": MIN_SKELETON_COVERAGE,
            "maximum_unsupported_run_fraction": (
                MAX_UNSUPPORTED_RUN_FRACTION
            ),
            "missing_defect_threshold": MISSING_DEFECT_THRESHOLD,
            "maximum_robust_variation": MAX_ROBUST_VARIATION,
        },
        "strut_count": len(results),
        "classification_counts": dict(counts),
    }
    (args.output / "strut_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    csv_fields = [
        key
        for key in results[0]
        if key not in {"endpoint0_xyz", "endpoint1_xyz", "uncertainty_reasons"}
    ]
    with (args.output / "strut_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in results)

    payload = {"summary": summary, "struts": results}
    (args.output / "classified_struts.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    if args.web_output is not None:
        args.web_output.parent.mkdir(parents=True, exist_ok=True)
        args.web_output.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
