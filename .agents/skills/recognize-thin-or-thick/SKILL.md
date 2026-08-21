---
name: recognize-thin-or-thick
description: Process multi-page CT TIFF stacks or 3D NumPy volumes by segmenting, skeletonizing, overlaying the centerline on the segmentation, measuring the orientation-independent 3D Euclidean distance from every skeleton voxel to the nearest segmentation boundary, and plotting a radius-frequency histogram. Use for recognizing thin or thick struts, analyzing radius distributions, or processing data/missing_struts/tif_stacks.
---

# Recognize Thin or Thick

## Workflow

1. Resolve the input directory. Use `data/missing_struts/tif_stacks`; do not silently substitute `data/missing_structs`, which is not the repository path.
2. For each `.tif` or `.tiff` stack:
   - Invoke `convert_tif_stack_to_npy()` to create a 3D `.npy` volume.
   - Obtain the segmentation threshold from the user or an established project value. Do not invent it.
   - Invoke `segment_ct_dataset()` on the volume to create a binary mask.
   - Invoke `skeletonize()` on the mask to create the centerline.
3. Run `scripts/measure_euclidean_radii.py` with the mask and skeleton when a
   per-skeleton-voxel radius distribution is needed.
4. Run `scripts/classify_struts.py` for the registered CAD-strut workflow: it
   samples 25 trimmed cross-sections per strut, measures 16 opposing boundary
   direction pairs per cross-section, and calculates each whole-strut median.
5. Verify that all arrays are 3D, shapes match, and every sampled skeleton voxel is inside the segmentation.
6. After measuring each whole-strut median radius, assign its CAD direction to
   the nearest coordinate plane (XY, XZ, or YZ, using the smallest absolute
   direction component as the plane normal). Compute the median measured strut
   radius separately within each plane.
7. For each plane, compute `correction factor = CAD nominal radius in voxels /
   plane median measured radius`. Multiply every raw strut radius in that plane
   by this factor. Preserve both the raw radius and corrected radius.
8. Apply the CAD ±20% thin/normal/thick rule to the corrected radius, never to
   the raw radius. Correction must occur before classification.
9. Repeat the measurement and calibration across a justified segmentation-
   threshold sensitivity band, then return output paths and a concise summary.

Segmentation must precede skeletonization because `skeletonize()` accepts a binary mask, not raw CT intensities.

## Current Project Restart Baseline

For the current LLNL lattice, preserve this user-established checkpoint before
testing any correction or normalization:

- segment at the established raw-intensity threshold of 40,000;
- measure the median whole-strut diameter from 25 trimmed cross-sections and
  16 opposing boundary-direction pairs;
- compare that measured diameter directly with the established 350 μm nominal
  diameter;
- label below 280 μm thin, 280–420 μm normal, and above 420 μm thick;
- keep missing, disconnected, unsupported, or unstable measurements uncertain.

Preserve the saved uncorrected baseline outputs. New runs must apply the XY/XZ/YZ
orientation calibration before the radius-first ±20% result and write to a new
output directory so corrected and historical counts remain comparable.

## Coordinate-Plane Orientation Correction

This is a scanner-orientation correction, not a replacement of the CAD nominal
radius. Use all finite whole-strut median radii in each plane to estimate its
median. The correction factor for plane `p` is:

`factor[p] = nominal_CAD_radius_voxels / median(raw_strut_radius_voxels[p])`

For every strut in that plane:

`corrected_radius_voxels = raw_radius_voxels * factor[p]`

Record the plane label, raw radius, plane median, correction factor, corrected
radius, and finite sample count. Fail explicitly if any plane has no finite
measurements or has a non-positive median; do not silently use factor 1.0.

## Radius Definition

Treat the skeleton as the 3D centerline. Compute a Euclidean distance transform of the segmented foreground, respecting physical voxel spacing, and sample it at every skeleton voxel. Each sampled value is the shortest 3D distance from that centerline voxel to the segmentation background.

Use this orientation-independent radius instead of a fixed left/right image-axis scan. Note that lattice junctions can still have larger radii than ordinary strut sections.

At this specimen's roughly seven-voxel nominal diameter, a one-voxel boundary
change is large enough to cross a typical engineering tolerance. Binary-mask
radii therefore support screening, not an automatic physical thin/thick claim.

## Run the Measurement

From this skill directory, run:

```bash
python scripts/measure_euclidean_radii.py \
  MASK.npy SKELETON.npy OUTPUT_DIR \
  --spacing Z Y X --bins auto
```

Use `--spacing 1 1 1` only when voxel units are acceptable. Use `--overlay-slice Z` to select a representative z-slice; otherwise, let the script select the slice with the most skeleton voxels.

## Run the Registered-Strut Classifier

From the repository root, run:

```bash
python .agents/skills/recognize-thin-or-thick/scripts/classify_struts.py \
  --source DETECTION_OUTPUT_DIR \
  --graph REGISTERED_CAD_GRAPH.json \
  --output OUTPUT_DIR
```

The source directory must contain `analysis_metadata.json`,
`segmentation_mask.npy`, `skeleton.npy`, and `strut_scores.csv`. The script
uses a CAD graph to exclude the junction ends by trimming each strut, measures
the cross-sections, applies XY/XZ/YZ calibration, and writes classifications.

## Outputs

Expect:

- `euclidean_radii.npy`: one nearest-boundary radius per skeleton voxel;
- `radius_map.npy`: radii stored at their 3D skeleton coordinates;
- `euclidean_radii.csv`: centerline coordinates and radii;
- `radius_histogram.png`: radius on the x-axis and frequency on the y-axis;
- `segmentation_skeleton_overlay.png`: segmentation with the skeleton overlaid;
- `radius_summary.json`: measurement settings and descriptive statistics.

## Guardrails

- Preserve source TIFF files and raw converted volumes.
- Save generated data in a separate output directory.
- Reject skeleton voxels outside the segmentation.
- Use physical `(z, y, x)` spacing whenever it is known; do not compare voxel radii with physical reference values.
- Report voxel units when spacing is left at `(1, 1, 1)`.
- Do not label measurements thin or thick until the user supplies a classification threshold or reference distribution; the histogram is the evidence used to choose it.
- Do not use a specimen-wide or local empirical median as though it were the
  physical CAD diameter. Relative peer ratios must be labeled as local
  screening results.
- Use plane medians only to estimate multiplicative orientation bias. Anchor
  every factor to the CAD nominal radius and apply the ±20% limits afterward.
- For a confirmed whole-strut label, require longitudinal persistence, a
  classification margin larger than segmentation/boundary precision, and the
  same verdict across the threshold-sensitivity band. Otherwise report an
  unresolved candidate.
- Treat coplanar clustering as a warning to inspect the raw orthogonal CT
  views, not as proof that calls are false. Horizontal CAD struts naturally
  occupy build planes.
