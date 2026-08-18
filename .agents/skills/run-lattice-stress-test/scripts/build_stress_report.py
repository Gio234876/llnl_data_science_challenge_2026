"""Build an illustrated Markdown report from lattice continuum stress-test artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def display(value: Any, digits: int = 6) -> str:
    if value is None:
        return "Not reported"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def relative_link(target: Path, report: Path) -> str:
    return Path(os.path.relpath(target.resolve(), report.parent.resolve())).as_posix()


def add_parameter_lines(lines: list[str], params: dict[str, Any]) -> None:
    for section in (
        "loading_parameters",
        "settling_parameters",
        "material_parameters",
        "defect_parameters",
        "geometry_parameters_mm",
    ):
        values = params.get(section)
        if not isinstance(values, dict):
            continue
        lines.extend([f"### {section.replace('_', ' ').title()}", ""])
        for key, value in values.items():
            lines.append(f"- `{key}`: {display(value)}")
        lines.append("")


def build_report(args: argparse.Namespace) -> None:
    tif_path = args.tif.resolve()
    registered_path = args.registered_json.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tif_path.is_file() or tif_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Input TIFF is missing or has an invalid extension: {tif_path}")
    if not registered_path.is_file():
        raise FileNotFoundError(f"Registered JSON not found: {registered_path}")

    registered = load_json(registered_path)
    junctions = registered.get("junctions")
    if not isinstance(junctions, list) or not junctions:
        raise ValueError("Registered JSON must contain a nonempty 'junctions' list")

    json_paths = {
        "Parameters": args.parameters.resolve(),
        "Axes": args.axes.resolve(),
        "Materials": args.materials.resolve(),
        "Effective model": args.model.resolve(),
        "FEA inputs": args.fea_inputs.resolve(),
        "FEA results": args.fea_results.resolve(),
        "Postprocessed summary": args.summary.resolve(),
    }
    payloads = {name: load_json(path) for name, path in json_paths.items()}
    params = payloads["Parameters"]
    axes = payloads["Axes"]
    materials = payloads["Materials"]
    model = payloads["Effective model"]
    inputs = payloads["FEA inputs"]
    results = payloads["FEA results"]
    summary = payloads["Postprocessed summary"]

    figures = summary.get("figures", {})
    required_figures = (
        ("Force–displacement response", "force_displacement_png"),
        ("Stress–strain response", "stress_strain_png"),
    )
    resolved_figures: list[tuple[str, Path]] = []
    for title, key in required_figures:
        raw_path = figures.get(key) if isinstance(figures, dict) else None
        if not raw_path:
            raise ValueError(f"Summary JSON does not declare figure '{key}'")
        figure_path = Path(raw_path)
        if not figure_path.is_absolute():
            figure_path = args.summary.parent / figure_path
        figure_path = figure_path.resolve()
        if not figure_path.is_file():
            sibling_fallback = (args.summary.parent / figure_path.name).resolve()
            if sibling_fallback.is_file():
                figure_path = sibling_fallback
        if not figure_path.is_file():
            raise FileNotFoundError(f"Declared figure not found: {figure_path}")
        report_figure_path = (output_path.parent / figure_path.name).resolve()
        if figure_path != report_figure_path:
            shutil.copy2(figure_path, report_figure_path)
        resolved_figures.append((title, report_figure_path))

    specimen = (
        params.get("specimen_name")
        or results.get("specimen_name")
        or inputs.get("specimen_name")
        or tif_path.stem
    )
    result_summary = summary.get("summary", {})
    partition = nested(summary, "deformation_partition", "final_step_displacement_fraction", default={})
    final_strains = nested(summary, "deformation_partition", "final_step_region_strains", default={})
    checks = summary.get("consistency_checks", {})
    solver_summary = results.get("results_summary", {})
    solver_materials = results.get("materials", {})
    main_modulus = (
        solver_summary.get("equivalent_specimen_modulus_gpa_main_regime")
        or solver_summary.get("equivalent_specimen_modulus_gpa")
        or nested(solver_materials, "equivalent_specimen_modulus_gpa_main_regime")
        or nested(solver_materials, "equivalent_specimen_modulus_gpa")
    )

    lines = [
        f"# Stress Test Report: {specimen}",
        "",
        "## Executive summary",
        "",
        "This report documents a simplified specimen-scale continuum compression test generated through the repository MCP workflow. It is not a voxel-resolved local stress analysis.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Maximum displacement (mm) | {display(result_summary.get('max_displacement_mm'))} |",
        f"| Maximum engineering strain | {display(result_summary.get('max_engineering_strain'))} |",
        f"| Maximum engineering stress (MPa) | {display(result_summary.get('max_engineering_stress_mpa'))} |",
        f"| Maximum reaction force (N) | {display(result_summary.get('max_reaction_force_n'))} |",
        f"| Main-regime/model modulus (GPa) | {display(main_modulus)} |",
        f"| Full-curve fitted modulus (GPa) | {display(result_summary.get('effective_specimen_modulus_gpa'))} |",
        f"| Number of load steps | {display(result_summary.get('num_steps'))} |",
        "",
        "For settling runs, the full-curve fitted modulus is obtained across a piecewise response and should not be treated as identical to the solver's main-regime modulus.",
        "",
        "## Input provenance",
        "",
        f"- CT TIFF: `{tif_path}`",
        f"- Registered lattice JSON: `{registered_path}`",
        f"- Registered junction count: {len(junctions)}",
        f"- Registered strut count: {len(registered.get('struts', []))}",
        f"- Registered unit-cell count: {len(registered.get('unit_cells', []))}",
        f"- Axes method: {display(axes.get('method'))}",
        f"- Junctions used for axis fit: {display(nested(axes, 'diagnostics', 'num_junctions_used'))}",
        "",
        "## Test configuration",
        "",
        f"- Pipeline: {display(params.get('pipeline', results.get('model_type')))}",
        f"- Loading axis: {display(nested(model, 'loading_axis', 'name', default=nested(inputs, 'geometry', 'loading_axis', 'name')))}",
        f"- Material assignment method: {display(materials.get('method'))}",
        f"- Effective properties estimated: {display(nested(materials, 'metadata', 'effective_properties_estimated'))}",
        "",
    ]
    add_parameter_lines(lines, params)

    lines.extend(["## Response graphs", ""])
    for title, figure_path in resolved_figures:
        lines.extend([f"### {title}", "", f"![{title}](<{relative_link(figure_path, output_path)}>)", ""])

    lines.extend([
        "## Deformation partition",
        "",
        "| Region | Final displacement fraction | Final strain |",
        "|---|---:|---:|",
    ])
    for key, label in (
        ("top_skin", "Top skin"),
        ("bottom_skin", "Bottom skin"),
        ("lattice_interior", "Lattice interior"),
    ):
        lines.append(f"| {label} | {display(partition.get(key))} | {display(final_strains.get(key))} |")
    lines.extend([
        "",
        f"Dominant final-strain region: **{display(nested(summary, 'deformation_partition', 'dominant_strain_region'))}**.",
        "",
        "## Consistency checks",
        "",
        f"- Maximum displacement reconstruction residual (mm): {display(checks.get('max_displacement_reconstruction_residual_mm'))}",
        f"- Axis-fit determinant: {display(nested(axes, 'diagnostics', 'determinant'))}",
        "",
        "## Assumptions and limitations",
        "",
    ])
    assumptions = results.get("assumptions", [])
    if isinstance(assumptions, list):
        for assumption in assumptions:
            lines.append(f"- {assumption}")
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    lines.extend([
        "- The TIFF is retained as source provenance; this simplified continuum workflow does not solve stresses directly on its voxels.",
        "- Geometry, defect percentages, and effective properties are user-supplied or MCP-estimated parameters and must be interpreted accordingly.",
        "",
        "## Artifact inventory",
        "",
        f"- Input TIFF: [{tif_path.name}](<{relative_link(tif_path, output_path)}>)",
        f"- Registered JSON: [{registered_path.name}](<{relative_link(registered_path, output_path)}>)",
    ])
    for label, path in json_paths.items():
        lines.append(f"- {label}: [{path.name}](<{relative_link(path, output_path)}>)")
    for title, figure_path in resolved_figures:
        lines.append(f"- {title}: [{figure_path.name}](<{relative_link(figure_path, output_path)}>)")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved stress-test report to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", type=Path, required=True)
    parser.add_argument("--registered-json", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fea-inputs", type=Path, required=True)
    parser.add_argument("--fea-results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_report(parse_args())
