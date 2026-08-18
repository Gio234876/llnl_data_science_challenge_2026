# Stress Test Report: 210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices

## Executive summary

This report documents a simplified specimen-scale continuum compression test generated through the repository MCP workflow. It is not a voxel-resolved local stress analysis.

| Metric | Result |
|---|---:|
| Maximum displacement (mm) | 0.125 |
| Maximum engineering strain | 0.00245098 |
| Maximum engineering stress (MPa) | 0.905427 |
| Maximum reaction force (N) | 1544.52 |
| Main-regime/model modulus (GPa) | 0.416007 |
| Full-curve fitted modulus (GPa) | 0.393109 |
| Number of load steps | 26 |

For settling runs, the full-curve fitted modulus is obtained across a piecewise response and should not be treated as identical to the solver's main-regime modulus.

## Input provenance

- CT TIFF: `C:\Users\Emmag\llnl_data_science_challenge_2026\data\missing_struts\tif_stacks\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- Registered lattice JSON: `C:\Users\Emmag\llnl_data_science_challenge_2026\data\missing_struts\registered_jsons\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json`
- Registered junction count: 10206
- Registered strut count: 18468
- Registered unit-cell count: 729
- Axes method: aligned_json
- Junctions used for axis fit: 10206

## Test configuration

- Pipeline: simplified_continuum_with_settling
- Loading axis: r3
- Material assignment method: assign_effective_material_properties
- Effective properties estimated: False

### Loading Parameters

- `loading_axis_name`: r3
- `max_displacement_mm`: 0.125
- `num_displacement_steps`: 25

### Settling Parameters

- `settling_displacement_mm`: 0.02
- `settling_stiffness_scale`: 0.3

### Material Parameters

- `bulk_youngs_modulus_gpa`: 85
- `bulk_poissons_ratio`: 0.31
- `bulk_density_kg_per_m3`: 4650
- `effective_lattice_youngs_modulus_gpa`: 0.334954
- `effective_lattice_poissons_ratio`: 0.342
- `effective_lattice_density_kg_per_m3`: 240

### Defect Parameters

- `intentional_missing_struts_percent`: 0.493
- `unintentional_missing_struts_percent`: 0.4504
- `broken_disconnected_struts_percent`: 0.724

### Geometry Parameters Mm

- `lattice_size_x`: 41.365
- `lattice_size_y`: 41.239
- `lattice_size_z`: 41.024
- `top_skin_thickness`: 4.987
- `bottom_skin_thickness`: 4.989

## Response graphs

### Force–displacement response

![Force–displacement response](<210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_force_displacement.png>)

### Stress–strain response

![Stress–strain response](<210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_stress_strain.png>)

## Deformation partition

| Region | Final displacement fraction | Final strain |
|---|---:|---:|
| Top skin | 0.000478576 | 1.06521e-05 |
| Bottom skin | 0.000478768 | 1.06521e-05 |
| Lattice interior | 0.999043 | 0.00270314 |

Dominant final-strain region: **lattice_interior**.

## Consistency checks

- Maximum displacement reconstruction residual (mm): 0.014
- Axis-fit determinant: 1

## Assumptions and limitations

- small-strain layered response after settling
- initial bedding-in/settling is represented by a reduced initial stiffness
- uniform axial stress after engagement
- perfect bonding between layers
- uniform cross-sectional area
- no explicit local defect-resolution
- This summary is for a single simplified continuum specimen model.
- The reported effective modulus comes from the linear stress-strain slope of the solved response.
- The deformation partition shows how much of the imposed displacement is taken up by the skins versus the effective lattice interior.
- The TIFF is retained as source provenance; this simplified continuum workflow does not solve stresses directly on its voxels.
- Geometry, defect percentages, and effective properties are user-supplied or MCP-estimated parameters and must be interpreted accordingly.

## Artifact inventory

- Input TIFF: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif](<../../../data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif>)
- Registered JSON: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json](<../../../data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json>)
- Parameters: [trial6_parameters.json](<../../Trial6/trial6_parameters.json>)
- Axes: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_axes.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_axes.json>)
- Materials: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_materials.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_materials.json>)
- Effective model: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_effective_model.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_effective_model.json>)
- FEA inputs: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_inputs.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_inputs.json>)
- FEA results: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_results_with_settling.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_results_with_settling.json>)
- Postprocessed summary: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_summary.json](<../../../data/missing_struts/tif_stacks/stress_test/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_fea_summary.json>)
- Force–displacement response: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_force_displacement.png](<210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_force_displacement.png>)
- Stress–strain response: [210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_stress_strain.png](<210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_continuum_figures_stress_strain.png>)
