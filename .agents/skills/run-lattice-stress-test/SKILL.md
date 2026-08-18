---
name: run-lattice-stress-test
description: Run the repository's MCP-based simplified continuum compression stress test for one X-ray CT lattice specimen, using a .tif/.tiff volume plus its registered lattice .json and stress-test parameters. Use when Codex must prepare inputs, run the linear or settling solver, postprocess results, save force-displacement and stress-strain graphs, or create a specimen-named Markdown report under Stress_Tests/Reports.
---

# Run a lattice stress test

Process exactly one TIFF/registered-JSON pair per invocation. Keep every generated artifact together and preserve the input files unchanged.

## Required inputs

- One `.tif` or `.tiff` CT volume.
- One registered lattice `.json` containing a nonempty `junctions` list; usable junctions contain three-value `position` and `indices` fields.
- Stress-test parameters, preferably a JSON following `Emma_Stress_Tests/Trial6/trial6_parameters.json`.

The TIFF and registered JSON alone do not contain all physical geometry, defect, material, and loading values required by the continuum MCP tools. If no parameter JSON is supplied, obtain the missing values from the user. Never silently invent geometry or defect percentages. MCP defaults may be used only when the user accepts them, and the report must label them as defaults.

Required parameter groups are:

- `geometry_parameters_mm`: lattice sizes x/y/z and top/bottom skin thicknesses.
- `defect_parameters`: intentional missing, measured/unintentional missing, and broken/disconnected strut percentages.
- `loading_parameters`: loading axis, maximum displacement, and number of steps.
- `settling_parameters` when using the settling solver.
- `material_parameters`; bulk values have MCP defaults and effective lattice values may be estimated by the MCP tool when omitted.

Treat `intentional_missing_struts_percent` as nominal missing, `unintentional_missing_struts_percent` as measured missing, and `broken_disconnected_struts_percent` as measured disconnected. Record this mapping.

## Output layout

Use two output locations unless the user explicitly selects alternatives:

- Create working solver artifacts in `<tiff-directory>/stress_test/<tiff-stem>/`.
- Create report deliverables in `<repository-root>/Stress_Tests/Reports/<tiff-stem>/`.

Use `<tiff-stem>` as the artifact prefix. Name the report `<tiff-stem>_stress_test_report.md`. The report directory must contain the Markdown report plus only these two copied report figures:

- `<figure-prefix>_force_displacement.png`
- `<figure-prefix>_stress_strain.png`

Keep JSON, NPZ, execution logs, helper scripts, and any deformation-partition PNG in the working solver directory, not in `Stress_Tests/Reports`.

Do not copy or convert the full TIFF for the continuum workflow. It is provenance/context for the registered geometry and can be very large. Validate its extension and existence without loading the entire volume.

## MCP workflow

Call the configured segmentation/stress MCP tools in this order. After every call, stop if the result contains `error`, record the failing stage, and preserve completed artifacts.

1. Call `compute_specimen_axes` with the registered JSON and save `<prefix>_axes.json`.
2. Call `assign_effective_material_properties` and save `<prefix>_materials.json`.
3. Call `build_effective_specimen_model` with the axes and physical dimensions; honor the supplied loading axis and save `<prefix>_effective_model.json`.
4. Call `export_continuum_fea_inputs` and save the prefix `<prefix>_continuum_fea_inputs`.
5. Prefer `run_continuum_fea_with_settling` when settling parameters are supplied; otherwise call `run_continuum_fea`. Save the prefix `<prefix>_continuum_fea_results_with_settling` or `<prefix>_continuum_fea_results`.
6. Call `postprocess_fea_results` on the result `.npz`, save `<prefix>_continuum_fea_summary.json`, set `save_pngs=true`, and set `figure_prefix` to `<prefix>_continuum_figures`.
7. Verify that the summary and these two report graphs exist:
   - `<figure-prefix>_force_displacement.png`
   - `<figure-prefix>_stress_strain.png`
8. Run `scripts/build_stress_report.py` with an output path under `<repository-root>/Stress_Tests/Reports/<tiff-stem>/`. The builder copies only the two required report graphs beside the Markdown file and embeds only those graphs. Do not embed or copy the deformation-partition PNG.

Do not use `export_voxel_fea_inputs` as a solver. The repository exports voxel FEA inputs but currently provides no voxel FEA solve tool.

## Create the report

Run:

```powershell
python .agents/skills/run-lattice-stress-test/scripts/build_stress_report.py --tif <volume.tif> --registered-json <registered.json> --parameters <parameters.json> --axes <axes.json> --materials <materials.json> --model <effective_model.json> --fea-inputs <continuum_fea_inputs.json> --fea-results <continuum_fea_results.json> --summary <continuum_fea_summary.json> --output Stress_Tests/Reports/<tiff-stem>/<tiff-stem>_stress_test_report.md
```

If parameters were supplied interactively instead of through a file, save them as `<prefix>_stress_test_parameters.json` in the working directory and pass that file. The report builder validates required artifacts, copies the two report PNGs into the report directory, uses relative image links, and includes provenance, configuration, numerical results, assumptions, and an artifact inventory.

Report both modulus values when available:

- Label the solver manifest's equivalent modulus as the main-regime/model modulus.
- Label the postprocessor's modulus as a full-curve fitted modulus. For settling runs it comes from a fit across a piecewise response and is not interchangeable with the main-regime value.

## Quality and stopping rules

- Run one specimen only and make at most three remediation attempts for failed MCP stages.
- Do not change source TIFF/JSON or overwrite unrelated runs.
- Verify every path returned by an MCP tool before reporting success.
- Embed only the force-displacement and stress-strain PNGs in the report using relative paths.
- Clearly distinguish computed results, MCP estimates/defaults, and user-supplied inputs.
- State that this is a simplified 1D effective-continuum compression model, not a voxel-resolved local stress solution.
- Finish only when the report and its two PNGs exist under `Stress_Tests/Reports/<tiff-stem>/`, or write a partial failure report identifying the failed stage and existing artifacts.
