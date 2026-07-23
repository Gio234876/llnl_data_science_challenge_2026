# Segmentation Evaluation Rubric 1

You are evaluating a segmentation result image against a ground-truth segmentation image for a lattice-structure X-ray CT dataset.

## Inputs
- The first attached image is the ground-truth segmentation.
- The second attached image is the result image to evaluate.

## Evaluation Goal
Compare the result image to the ground truth and judge how well the result preserves the intended lattice segmentation.

## Criteria
1. Structural Integrity
   - Does the result preserve the connectivity of the lattice struts?
   - Are continuous struts captured where they appear in the ground truth?

2. False Positives / False Negatives
   - Over-segmentation: Are there extra segmented regions, halos, blobs, or noise not present in the ground truth?
   - Under-segmentation: Are struts, thin members, or parts of the lattice missing?

3. Topology
   - Are lattice nodes and junctions preserved?
   - Does the branching structure match the ground truth?

4. Noise and Artifacts
   - Does the result contain visible noise, speckling, disconnected debris, jagged artifacts, or other structures absent from the clean ground truth?

## Scoring Guide (0-5)
- 5: Identical or nearly identical to ground truth. No missing structures and no false positives.
- 4: Excellent match with only very minor differences.
- 3: Main topology is correct, but there is noticeable noise, mild over-segmentation, or some thin struts are missing.
- 2: Fair result, but significant differences are present, such as large missing regions, broken connectivity, or substantial false positives.
- 1: Major structural failure, severe topology errors, or excessive noise/artifacts.
- 0: Blank, unrelated, or unusable output.

## Evaluation Instructions
- Focus primarily on visible agreement with the ground truth at slice level.
- Weigh topology and structural connectivity more heavily than tiny pixel-level mismatches.
- Penalize both missing lattice structure and extra spurious segmented regions.
- Give a single integer score from 0 to 5.

## Output Format
Return only valid JSON with exactly these keys:
- "reasoning": a concise explanation of the score
- "score": an integer from 0 to 5

Example:
{"reasoning":"The lattice connectivity is mostly preserved, but several thin struts are missing and there is mild extra noise near some junctions.","score":3}
