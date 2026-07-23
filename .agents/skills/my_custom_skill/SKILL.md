---
name: threshold-optimizer
description: Runs CT segmentation across multiple threshold values and saves each mask for side-by-side comparison.
---

# Threshold Optimizer Protocol

You are the **Threshold Optimizer** for this project. When this skill is active, evaluate a 3D CT `.npy` volume by running segmentation at several threshold values and saving each result as a separate output file.

## Goal
Given an input volumetric `.npy` file, generate multiple segmentation masks using the MCP tool `segment_ct_dataset()` so the user can compare how threshold choice changes the extracted structure.

## Workflow

### Step 1: Validate the Input
- Confirm the user has provided or clearly implied a source `.npy` volume.
- Check that the file exists before doing any processing.
- If the user does not specify thresholds, use this default set:
  - `0.3`
  - `0.5`
  - `0.7`
- If the user provides thresholds, use those values instead.

### Step 2: Build Output Paths
- Save outputs alongside the input file unless the user asks for a different destination.
- For each threshold, create a separate filename derived from the source filename.
- Use a threshold-safe suffix in the output name:
  - `0.3` -> `0p3`
  - `0.5` -> `0p5`
  - `0.7` -> `0p7`
- Example:
  - Input: `sample.npy`
  - Outputs:
    - `sample_segmented_0p3.npy`
    - `sample_segmented_0p5.npy`
    - `sample_segmented_0p7.npy`

### Step 3: Run Segmentation
- Invoke `segment_ct_dataset()` once per threshold.
- Pass:
  - `input_filepath`: the source `.npy`
  - `output_filepath`: the threshold-specific output path
  - `threshold`: the numeric threshold value
- Do not overwrite unrelated files.

### Step 4: Verify Outputs
- Confirm that each expected output file was created.
- If any segmentation call fails, report which threshold failed and continue with the remaining thresholds when possible.

### Step 5: Summarize Results
- Return a concise summary including:
  - the input file used
  - the threshold values attempted
  - the output file created for each threshold
  - any failed thresholds
- If helpful, suggest the next step of visualizing slices from the generated masks for comparison.

## Technical Constraints
- Use the MCP tool `segment_ct_dataset()` for segmentation instead of implementing a custom thresholding script.
- Treat threshold values as user-controlled numeric inputs.
- Preserve the original input file.
- Only create threshold-specific segmentation outputs.
- If you create any temporary helper scripts, remove them before finishing.
