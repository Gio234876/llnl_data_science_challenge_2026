from pathlib import Path
from fastmcp import FastMCP
from skeletonization import skeletonize_mask
from typing import Optional
import tifffile
import numpy as np
import trimesh
from skimage.measure import block_reduce, marching_cubes

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")

@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        import numpy as np

        data = np.load(input_filepath)
        segmented = (data >= threshold).astype(np.uint8)
        np.save(output_filepath, segmented)

        return f"Segmentation successful. Saved segmented dataset to: {output_filepath}"
    except Exception as error:
        return f"Error during segmentation: {error}"

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
          import os
          import numpy as np
          import matplotlib.pyplot as plt

          if axis not in (0, 1, 2):
              return f"Error: axis must be 0, 1, or 2, got {axis}."

          if not os.path.exists(input_filepath):
              return f"Error: input file not found: {input_filepath}"

          data = np.load(input_filepath, mmap_mode="r")

          if data.ndim != 3:
              return f"Error: expected a 3D array, got shape {data.shape}."

          if slice_index < 0 or slice_index >= data.shape[axis]:
              return (
                  f"Error: slice_index {slice_index} is out of bounds for axis {axis} "
                  f"with size {data.shape[axis]}."
              )

          if axis == 0:
              slice_data = data[slice_index, :, :]
          elif axis == 1:
              slice_data = data[:, slice_index, :]
          else:
              slice_data = data[:, :, slice_index]

          output_dir = os.path.dirname(output_filepath)
          if output_dir:
              os.makedirs(output_dir, exist_ok=True)

          plt.figure(figsize=(6, 6))
          plt.imshow(slice_data, cmap="gray")
          plt.title(f"Slice {slice_index} along axis {axis}")
          plt.axis("off")
          plt.tight_layout()
          plt.savefig(output_filepath, bbox_inches="tight", pad_inches=0)
          plt.close()

          return f"Saved slice visualization to {output_filepath}"
    except Exception as e:
        return f"Error visualizing slice: {e}"

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    from pathlib import Path
    try:
        input_path = Path(input_filepath).expanduser()
        output_path = Path(output_filepath).expanduser()

        if not input_path.is_file():
            return f"Error: input mask file not found at {input_path}"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        skeleton = skeletonize_mask(str(input_path), str(output_path))
        if skeleton is None:
            return f"Error skeletonizing mask from {input_path}"

        # np.save appends .npy when the supplied path does not already have it.
        saved_path = output_path if output_path.suffix == ".npy" else Path(f"{output_path}.npy")
        return f"Skeleton saved successfully to {saved_path}"
    except (OSError, ValueError, TypeError) as exc:
        return f"Error skeletonizing mask: {exc}"


#####################
## Histogram Tools ##
#####################


@mcp.tool()
def convert_tif_stack_to_npy(
    input_filepath: str,
    output_filepath: str,
    dtype: Optional[str] = None,
) -> dict:
    """
    Convert a single multi-page TIFF into a cached 3D NumPy volume.
    """

    from pathlib import Path
    import numpy as np
    
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"Invalid input TIFF file: {input_path}")

    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Input file must be a .tif or .tiff file: {input_path}")

    volume = tifffile.imread(str(input_path))

    if volume.ndim < 3:
        raise ValueError(f"Expected a multi-page TIFF with at least 3 dimensions, got shape {volume.shape}")

    if dtype is not None:
        volume = volume.astype(dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume)

    return {
        "result": f"Saved NumPy volume to {output_path}",
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "num_slices": int(volume.shape[0]),
        "input_file": str(input_path),
    }

@mcp.tool()
def ct_tif_to_stl(
    input_filepath: str,
    output_filepath: str,
    threshold: float,
) -> str:
    """
    Convert a 3D CT TIFF stack into a viewer-friendly STL mesh.

    This version keeps the original tool name/signature, but downsamples
    the binary mask by 4x before meshing so the STL is much smaller.
    """
    try:
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        if not input_path.exists():
            return f"Error: Input file not found: {input_filepath}"

        if input_path.suffix.lower() not in {".tif", ".tiff"}:
            return f"Error: Input file must be a .tif or .tiff file: {input_filepath}"

        volume = tifffile.imread(str(input_path))

        if volume.ndim != 3:
            return f"Error: Expected a 3D TIFF volume, but got shape {volume.shape}"

        mask = volume >= threshold
        foreground_voxels = int(np.count_nonzero(mask))

        if foreground_voxels == 0:
            return (
                f"Error: No foreground voxels found at threshold {threshold}. "
                f"No STL was created."
            )

        downsample_factor = 4
        mask = block_reduce(
            mask.astype(np.uint8, copy=False),
            block_size=(downsample_factor, downsample_factor, downsample_factor),
            func=np.max,
        ).astype(np.uint8)

        verts, faces, normals, values = marching_cubes(
            mask,
            level=0.5,
            spacing=(downsample_factor, downsample_factor, downsample_factor),
        )

        triangles = verts[faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        face_normals = np.cross(edge_a, edge_b)

        lengths = np.linalg.norm(face_normals, axis=1)
        valid = lengths > 0
        face_normals[valid] /= lengths[valid][:, None]
        face_normals[~valid] = 0.0

        facet_dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attr", "<u2"),
        ])

        facets = np.empty(len(faces), dtype=facet_dtype)
        facets["normal"] = face_normals.astype(np.float32, copy=False)
        facets["vertices"] = triangles.astype(np.float32, copy=False)
        facets["attr"] = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as handle:
            handle.write(b"Generated by CT Segmentation MCP".ljust(80, b" "))
            handle.write(np.uint32(len(faces)).tobytes())
            handle.write(facets.tobytes())

        return (
            f"Saved STL to {output_path}. "
            f"Original volume shape: {tuple(volume.shape)}. "
            f"Downsampled mask shape: {tuple(mask.shape)}. "
            f"Foreground voxels: {foreground_voxels}. "
            f"Vertices: {len(verts)}. "
            f"Faces: {len(faces)}. "
            f"Downsample factor: 4."
        )

    except Exception as exc:
        return f"Error converting TIFF to STL: {exc}"

@mcp.tool()
def compute_specimen_axes(
    aligned_json_filepath: str,
    output_filepath: Optional[str] = None,
) -> dict:
    """
    Compute the specimen center c and local axes / rotation matrix R
    from an aligned lattice JSON.

    Expected JSON structure:
    {
    "junctions": [
        {
        "id": 0,
        "position": [x_or_z, y, z_or_x],
        "indices": [i, j, k]
        },
        ...
    ]
    }

    The tool fits an affine map from lattice index space -> scan space,
    then extracts the nearest orthonormal rotation matrix.

    Args:
        aligned_json_filepath: Path to the aligned JSON file.
        output_filepath: Optional path to save the computed axes result as JSON.

    Returns:
        A dictionary containing:
        - method
        - center
        - rotation_matrix
        - axis_vectors
        - diagnostics
        - saved_to (if output_filepath provided)
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        def _normalize(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Cannot normalize a zero-length vector.")
            return vector / norm

        def _nearest_rotation(linear_part: np.ndarray) -> np.ndarray:
            """
            Compute the closest proper rotation matrix to linear_part
            using polar decomposition via SVD.
            """
            u, _, vt = np.linalg.svd(linear_part)
            rotation = u @ vt

            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1.0
                rotation = u @ vt

            return rotation

        json_path = Path(aligned_json_filepath)
        if not json_path.exists():
            return {"error": f"Aligned JSON file not found: {aligned_json_filepath}"}

        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        junctions = payload.get("junctions", [])
        if not junctions:
            return {"error": "Aligned JSON contains no 'junctions'."}

        positions = []
        indices = []

        for junction in junctions:
            position = junction.get("position")
            index_triplet = junction.get("indices")

            if position is None or index_triplet is None:
                continue

            if len(position) != 3 or len(index_triplet) != 3:
                continue

            positions.append(position)
            indices.append(index_triplet)

        if len(positions) < 4:
            return {"error": "Not enough valid junctions with both 'position' and 'indices'."}

        positions = np.asarray(positions, dtype=float)
        indices = np.asarray(indices, dtype=float)

        # Fit affine map:
        # positions ~= indices @ A.T + b
        design_matrix = np.column_stack([indices, np.ones(len(indices))])
        solution, _, _, _ = np.linalg.lstsq(design_matrix, positions, rcond=None)

        linear_part = solution[:3, :].T   # shape (3, 3)
        translation = solution[3, :]      # shape (3,)

        rotation_matrix = _nearest_rotation(linear_part)

        # Compute specimen center from the lattice-index bounding-box center.
        # This is usually better than using the mean of positions for a cube-like specimen.
        index_min = indices.min(axis=0)
        index_max = indices.max(axis=0)
        index_center = 0.5 * (index_min + index_max)

        center = index_center @ linear_part.T + translation

        r1 = _normalize(rotation_matrix[:, 0])
        r2 = _normalize(rotation_matrix[:, 1])
        r3 = _normalize(rotation_matrix[:, 2])

        # Basic diagnostics
        orthogonality = rotation_matrix.T @ rotation_matrix
        determinant = float(np.linalg.det(rotation_matrix))
        axis_scales = [
            float(np.linalg.norm(linear_part[:, 0])),
            float(np.linalg.norm(linear_part[:, 1])),
            float(np.linalg.norm(linear_part[:, 2])),
        ]

        result = {
            "method": "aligned_json",
            "aligned_json_filepath": str(json_path),
            "center": center.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "axis_vectors": {
                "r1": r1.tolist(),
                "r2": r2.tolist(),
                "r3": r3.tolist(),
            },
            "diagnostics": {
                "num_junctions_used": int(len(positions)),
                "index_min": index_min.tolist(),
                "index_max": index_max.tolist(),
                "index_center": index_center.tolist(),
                "estimated_axis_scales": axis_scales,
                "orthogonality_check": orthogonality.tolist(),
                "determinant": determinant,
            },
        }

        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

            result["saved_to"] = str(output_path)

        return result

    except Exception as error:
        return {"error": f"Error computing specimen axes: {error}"}


@mcp.tool()
def extract_boundary_voxels(
    mask_filepath: str,
    axes_filepath: str,
    output_filepath: Optional[str] = None,
    loading_axis: str = "r3",
    boundary_band_thickness: float = 1.0,
) -> dict:
    """
    Extract top and bottom boundary voxels from a segmented 3D mask using
    specimen-aligned axes.

    This tool:
    - loads a binary mask
    - loads specimen center/axes from compute_specimen_axes output
    - finds exposed surface voxels
    - projects those surface voxels onto the chosen loading axis
    - selects top and bottom boundary voxel sets within a thin band

    Args:
        mask_filepath: Path to a 3D binary .npy mask (solid=1, void=0).
        axes_filepath: Path to the JSON output from compute_specimen_axes.
        output_filepath: Optional path to save the boundary voxel sets as .npz.
        loading_axis: Which specimen axis to use for loading ("r1", "r2", or "r3").
        boundary_band_thickness: Thickness of the top/bottom selection band in voxel units.
            A value of 1.0 is a good default for slightly rough segmented faces.

    Returns:
        A dictionary containing:
        - method
        - loading_axis
        - boundary_band_thickness
        - top_voxel_count
        - bottom_voxel_count
        - exposed_voxel_count
        - projection_range
        - saved_to (if output_filepath provided)
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        def _normalize(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Cannot normalize a zero-length loading axis.")
            return vector / norm

        mask_path = Path(mask_filepath)
        if not mask_path.exists():
            return {"error": f"Mask file not found: {mask_filepath}"}

        axes_path = Path(axes_filepath)
        if not axes_path.exists():
            return {"error": f"Axes file not found: {axes_filepath}"}

        if boundary_band_thickness <= 0:
            return {"error": "boundary_band_thickness must be > 0."}

        if loading_axis not in {"r1", "r2", "r3"}:
            return {"error": f"loading_axis must be one of 'r1', 'r2', or 'r3', got {loading_axis}."}

        mask = np.load(mask_path)
        if mask.ndim != 3:
            return {"error": f"Expected a 3D mask, got shape {mask.shape}."}

        mask_bool = mask > 0
        if not np.any(mask_bool):
            return {"error": "Mask contains no foreground voxels."}

        with axes_path.open("r", encoding="utf-8") as handle:
            axes_payload = json.load(handle)

        axis_vectors = axes_payload.get("axis_vectors")
        center = axes_payload.get("center")

        if axis_vectors is None or center is None:
            return {"error": "Axes JSON must contain 'axis_vectors' and 'center'."}

        if loading_axis not in axis_vectors:
            return {"error": f"Axes JSON does not contain '{loading_axis}'."}

        axis = _normalize(np.asarray(axis_vectors[loading_axis], dtype=float))
        center = np.asarray(center, dtype=float)

        # Find exposed surface voxels using a 6-neighborhood.
        padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
        core = padded[1:-1, 1:-1, 1:-1]

        all_six_neighbors_filled = (
            padded[:-2, 1:-1, 1:-1] &
            padded[2:, 1:-1, 1:-1] &
            padded[1:-1, :-2, 1:-1] &
            padded[1:-1, 2:, 1:-1] &
            padded[1:-1, 1:-1, :-2] &
            padded[1:-1, 1:-1, 2:]
        )

        exposed_mask = core & (~all_six_neighbors_filled)
        exposed_coords = np.argwhere(exposed_mask)

        if len(exposed_coords) == 0:
            return {"error": "No exposed surface voxels were found."}

        # Project exposed voxel coordinates onto the loading axis.
        # We treat voxel index coordinates as voxel-center coordinates for this step.
        relative = exposed_coords.astype(float) - center
        projections = relative @ axis

        projection_min = float(projections.min())
        projection_max = float(projections.max())

        top_keep = (projection_max - projections) <= boundary_band_thickness
        bottom_keep = (projections - projection_min) <= boundary_band_thickness

        top_voxels = exposed_coords[top_keep]
        bottom_voxels = exposed_coords[bottom_keep]

        result = {
            "method": "surface_projection_band",
            "mask_filepath": str(mask_path),
            "axes_filepath": str(axes_path),
            "loading_axis": loading_axis,
            "loading_axis_vector": axis.tolist(),
            "boundary_band_thickness": float(boundary_band_thickness),
            "top_voxel_count": int(len(top_voxels)),
            "bottom_voxel_count": int(len(bottom_voxels)),
            "exposed_voxel_count": int(len(exposed_coords)),
            "projection_range": {
                "min": projection_min,
                "max": projection_max,
            },
        }

        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                output_path,
                top_voxels=top_voxels.astype(np.int32),
                bottom_voxels=bottom_voxels.astype(np.int32),
                exposed_voxels=exposed_coords.astype(np.int32),
                center=center.astype(float),
                loading_axis_vector=axis.astype(float),
                boundary_band_thickness=np.asarray([boundary_band_thickness], dtype=float),
            )

            result["saved_to"] = str(output_path)

        return result

    except Exception as error:
        return {"error": f"Error extracting boundary voxels: {error}"}


@mcp.tool()
def export_voxel_fea_inputs(
    mask_filepath: str,
    axes_filepath: str,
    boundary_filepath: str,
    output_prefix: str,
    voxel_spacing_axis0: float = 1.0,
    voxel_spacing_axis1: float = 1.0,
    voxel_spacing_axis2: float = 1.0,
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.30,
    num_displacement_steps: int = 10,
    max_displacement: float = 1.0,
) -> dict:
    """
    Export a voxel-based FEA input package from:
    - a binary specimen mask
    - specimen axes / center
    - top/bottom boundary voxel sets

    This tool also selects the specimen loading axis as the true specimen axis
    (r1, r2, or r3) most aligned with the TIFF bottom-to-top stack direction.

    Assumptions:
    - The TIFF stack axis is axis 0 of the volume.
    - Increasing axis-0 index corresponds to bottom -> top in the scan.
    - The boundary voxel file was created beforehand, ideally using the same loading axis.

    Args:
        mask_filepath: Path to the 3D binary .npy mask (solid=1, void=0).
        axes_filepath: Path to the JSON output from compute_specimen_axes.
        boundary_filepath: Path to the .npz output from extract_boundary_voxels.
        output_prefix: Prefix for output files. This tool writes:
            - <output_prefix>.npz
            - <output_prefix>.json
        voxel_spacing_axis0: Physical voxel spacing along array axis 0.
        voxel_spacing_axis1: Physical voxel spacing along array axis 1.
        voxel_spacing_axis2: Physical voxel spacing along array axis 2.
        youngs_modulus: Placeholder Young's modulus for the FEA material model.
        poissons_ratio: Placeholder Poisson's ratio for the FEA material model.
        num_displacement_steps: Number of prescribed displacement steps.
        max_displacement: Maximum prescribed displacement magnitude.

    Returns:
        A dictionary describing the exported voxel FEA package.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        def _normalize(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Cannot normalize a zero-length vector.")
            return vector / norm

        def _select_loading_axis_from_tif_stack(axis_vectors: dict) -> dict:
            """
            Select the true specimen axis most aligned with TIFF bottom-to-top.

            We assume the scan stack runs along array axis 0, and increasing axis-0
            index corresponds to bottom -> top.
            """
            scan_up = np.array([1.0, 0.0, 0.0], dtype=float)

            best_name = None
            best_alignment = None
            best_vector = None

            for axis_name in ("r1", "r2", "r3"):
                vector = np.asarray(axis_vectors[axis_name], dtype=float)
                vector = _normalize(vector)
                alignment = float(np.dot(vector, scan_up))

                if best_alignment is None or abs(alignment) > abs(best_alignment):
                    best_name = axis_name
                    best_alignment = alignment
                    best_vector = vector

            flipped = False
            if best_alignment < 0:
                best_vector = -best_vector
                flipped = True

            return {
                "loading_axis_name": best_name,
                "loading_axis_vector": best_vector.tolist(),
                "scan_up_vector": scan_up.tolist(),
                "scan_alignment": float(best_alignment),
                "flipped_to_match_tif_bottom_to_top": flipped,
            }

        if voxel_spacing_axis0 <= 0 or voxel_spacing_axis1 <= 0 or voxel_spacing_axis2 <= 0:
            return {"error": "All voxel spacings must be > 0."}

        if num_displacement_steps <= 0:
            return {"error": "num_displacement_steps must be > 0."}

        if poissons_ratio <= -1.0 or poissons_ratio >= 0.5:
            return {"error": "poissons_ratio must be between -1 and 0.5 (exclusive of 0.5)."}

        mask_path = Path(mask_filepath)
        axes_path = Path(axes_filepath)
        boundary_path = Path(boundary_filepath)

        if not mask_path.exists():
            return {"error": f"Mask file not found: {mask_filepath}"}
        if not axes_path.exists():
            return {"error": f"Axes file not found: {axes_filepath}"}
        if not boundary_path.exists():
            return {"error": f"Boundary file not found: {boundary_filepath}"}

        mask = np.load(mask_path)
        if mask.ndim != 3:
            return {"error": f"Expected a 3D mask, got shape {mask.shape}."}

        solid_voxels = np.argwhere(mask > 0)
        if len(solid_voxels) == 0:
            return {"error": "Mask contains no foreground voxels."}

        with axes_path.open("r", encoding="utf-8") as handle:
            axes_payload = json.load(handle)

        if "center" not in axes_payload or "axis_vectors" not in axes_payload:
            return {"error": "Axes JSON must contain 'center' and 'axis_vectors'."}

        center = np.asarray(axes_payload["center"], dtype=float)
        axis_vectors = axes_payload["axis_vectors"]

        for key in ("r1", "r2", "r3"):
            if key not in axis_vectors:
                return {"error": f"Axes JSON missing axis vector '{key}'."}

        loading_axis_info = _select_loading_axis_from_tif_stack(axis_vectors)
        loading_axis_vector = np.asarray(loading_axis_info["loading_axis_vector"], dtype=float)

        boundary_data = np.load(boundary_path)

        if "top_voxels" not in boundary_data or "bottom_voxels" not in boundary_data:
            return {"error": "Boundary file must contain 'top_voxels' and 'bottom_voxels'."}

        top_voxels = np.asarray(boundary_data["top_voxels"], dtype=np.int32)
        bottom_voxels = np.asarray(boundary_data["bottom_voxels"], dtype=np.int32)

        if len(top_voxels) == 0:
            return {"error": "Boundary file contains zero top voxels."}
        if len(bottom_voxels) == 0:
            return {"error": "Boundary file contains zero bottom voxels."}

        boundary_loading_axis_vector = None
        axis_consistency = None

        if "loading_axis_vector" in boundary_data:
            boundary_loading_axis_vector = _normalize(
                np.asarray(boundary_data["loading_axis_vector"], dtype=float)
            )
            alignment = float(np.dot(boundary_loading_axis_vector, loading_axis_vector))
            axis_consistency = {
                "dot_product": alignment,
                "abs_dot_product": abs(alignment),
                "likely_consistent": bool(abs(alignment) > 0.95),
            }

        voxel_spacing = np.array(
            [voxel_spacing_axis0, voxel_spacing_axis1, voxel_spacing_axis2],
            dtype=float,
        )

        # Physical coordinates of all solid voxel centers
        solid_voxel_centers_physical = solid_voxels.astype(float) * voxel_spacing[None, :]

        # Useful specimen measures along the chosen loading axis
        center_physical = center * voxel_spacing
        relative_physical = solid_voxel_centers_physical - center_physical[None, :]
        projections = relative_physical @ loading_axis_vector

        specimen_height = float(projections.max() - projections.min())

        displacement_steps = np.linspace(
            0.0,
            float(max_displacement),
            int(num_displacement_steps) + 1,
            dtype=float,
        )

        output_base = Path(output_prefix)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output_base.with_suffix(".npz")
        json_path = output_base.with_suffix(".json")

        np.savez_compressed(
            npz_path,
            solid_voxels=solid_voxels.astype(np.int32),
            top_voxels=top_voxels.astype(np.int32),
            bottom_voxels=bottom_voxels.astype(np.int32),
            center=center.astype(float),
            rotation_matrix=np.asarray(axes_payload.get("rotation_matrix"), dtype=float),
            voxel_spacing=voxel_spacing,
            loading_axis_vector=loading_axis_vector.astype(float),
            displacement_steps=displacement_steps,
        )

        manifest = {
            "method": "voxel_fea_export",
            "inputs": {
                "mask_filepath": str(mask_path),
                "axes_filepath": str(axes_path),
                "boundary_filepath": str(boundary_path),
            },
            "volume": {
                "shape": list(mask.shape),
                "solid_voxel_count": int(len(solid_voxels)),
                "voxel_spacing": voxel_spacing.tolist(),
            },
            "specimen": {
                "center_index_coordinates": center.tolist(),
                "center_physical_coordinates": center_physical.tolist(),
                "rotation_matrix": axes_payload.get("rotation_matrix"),
                "axis_vectors": axis_vectors,
                "selected_loading_axis": loading_axis_info,
                "estimated_height_along_loading_axis": specimen_height,
            },
            "boundary_sets": {
                "top_voxel_count": int(len(top_voxels)),
                "bottom_voxel_count": int(len(bottom_voxels)),
                "boundary_axis_consistency_check": axis_consistency,
            },
            "material": {
                "youngs_modulus": float(youngs_modulus),
                "poissons_ratio": float(poissons_ratio),
                "model_type": "linear_elastic_placeholder",
            },
            "loading": {
                "control_type": "displacement_control",
                "displacement_steps": displacement_steps.tolist(),
            },
            "outputs": {
                "npz_filepath": str(npz_path),
                "json_filepath": str(json_path),
            },
        }

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return {
            "result": "Voxel FEA input package exported successfully.",
            "npz_filepath": str(npz_path),
            "json_filepath": str(json_path),
            "solid_voxel_count": int(len(solid_voxels)),
            "top_voxel_count": int(len(top_voxels)),
            "bottom_voxel_count": int(len(bottom_voxels)),
            "selected_loading_axis": loading_axis_info,
            "estimated_height_along_loading_axis": specimen_height,
            "boundary_axis_consistency_check": axis_consistency,
        }

    except Exception as error:
        return {"error": f"Error exporting voxel FEA inputs: {error}"}


@mcp.tool()
def assign_effective_material_properties(
    specimen_name: str,
    nominal_missing_strut_percent: float,
    measured_missing_strut_percent: float,
    measured_disconnected_strut_percent: float,
    bulk_youngs_modulus_gpa: float = 85.0,
    bulk_poissons_ratio: float = 0.310,
    bulk_density_kg_per_m3: float = 4650.0,
    effective_youngs_modulus_gpa: Optional[float] = None,
    effective_poissons_ratio: Optional[float] = None,
    effective_density_kg_per_m3: Optional[float] = None,
    output_filepath: Optional[str] = None,
) -> dict:
    """
    Assign effective material properties for the simplified continuum model.

    This tool combines:
    - bulk Ti-5553 material properties
    - specimen-specific defect percentages
    - optional user-specified effective lattice properties

    It is intended for a simplified specimen-scale continuum model where:
    - top/bottom skins use bulk Ti-5553 properties
    - the lattice interior uses effective continuum properties

    If effective lattice properties are not provided explicitly, the tool
    derives a simple first-pass estimate from defect percentages.

    Args:
        specimen_name: Name/label of the specimen.
        nominal_missing_strut_percent: Nominal designed missing-strut percentage.
        measured_missing_strut_percent: Measured missing-strut percentage from CT.
        measured_disconnected_strut_percent: Measured disconnected-strut percentage from CT.
        bulk_youngs_modulus_gpa: Bulk Ti-5553 Young's modulus in GPa.
        bulk_poissons_ratio: Bulk Ti-5553 Poisson's ratio.
        bulk_density_kg_per_m3: Bulk Ti-5553 density in kg/m^3.
        effective_youngs_modulus_gpa: Optional user-provided effective lattice Young's modulus in GPa.
        effective_poissons_ratio: Optional user-provided effective lattice Poisson's ratio.
        effective_density_kg_per_m3: Optional user-provided effective lattice density in kg/m^3.
        output_filepath: Optional path to save the assigned properties as JSON.

    Returns:
        A dictionary containing:
        - specimen name
        - defect percentages
        - bulk skin properties
        - effective lattice properties
        - metadata about whether properties were user-specified or estimated
    """
    try:
        import json
        from pathlib import Path

        # Basic validation
        for name, value in [
            ("nominal_missing_strut_percent", nominal_missing_strut_percent),
            ("measured_missing_strut_percent", measured_missing_strut_percent),
            ("measured_disconnected_strut_percent", measured_disconnected_strut_percent),
        ]:
            if value < 0:
                return {"error": f"{name} must be >= 0, got {value}."}

        if bulk_youngs_modulus_gpa <= 0:
            return {"error": f"bulk_youngs_modulus_gpa must be > 0, got {bulk_youngs_modulus_gpa}."}

        if bulk_density_kg_per_m3 <= 0:
            return {"error": f"bulk_density_kg_per_m3 must be > 0, got {bulk_density_kg_per_m3}."}

        if bulk_poissons_ratio <= -1.0 or bulk_poissons_ratio >= 0.5:
            return {"error": f"bulk_poissons_ratio must be between -1 and 0.5 (exclusive of 0.5), got {bulk_poissons_ratio}."}

        if effective_youngs_modulus_gpa is not None and effective_youngs_modulus_gpa <= 0:
            return {"error": f"effective_youngs_modulus_gpa must be > 0, got {effective_youngs_modulus_gpa}."}

        if effective_density_kg_per_m3 is not None and effective_density_kg_per_m3 <= 0:
            return {"error": f"effective_density_kg_per_m3 must be > 0, got {effective_density_kg_per_m3}."}

        if effective_poissons_ratio is not None and (effective_poissons_ratio <= -1.0 or effective_poissons_ratio >= 0.5):
            return {"error": f"effective_poissons_ratio must be between -1 and 0.5 (exclusive of 0.5), got {effective_poissons_ratio}."}

        total_measured_defect_percent = measured_missing_strut_percent + measured_disconnected_strut_percent

        # First-pass estimation logic if effective properties are not supplied.
        # This is intentionally simple and traceable, not a calibrated constitutive model.
        #
        # We use the paper-informed range as a practical guide:
        # - effective Young's modulus is much smaller than bulk Ti-5553
        # - it decreases with increasing defect content
        # - effective Poisson's ratio for the lattice is around ~0.34 in the paper
        #
        # If user values are provided, they take precedence.
        estimated_effective = False

        if effective_youngs_modulus_gpa is None:
            estimated_effective = True

            # Simple baseline heuristic:
            # use a nominal no-defect lattice modulus near 0.34 GPa and soften it
            # with measured total defect content.
            baseline_effective_modulus_gpa = 0.342
            modulus_reduction_per_percent_defect_gpa = 0.006

            estimated_E = baseline_effective_modulus_gpa - (
                modulus_reduction_per_percent_defect_gpa * total_measured_defect_percent
            )

            # Keep within a physically sensible positive bound
            effective_youngs_modulus_gpa = max(0.05, estimated_E)

        if effective_poissons_ratio is None:
            estimated_effective = True

            # Paper values are around ~0.339 - 0.344, so use a stable default.
            effective_poissons_ratio = 0.342

        if effective_density_kg_per_m3 is None:
            estimated_effective = True

            # If no explicit lattice effective density is provided, use a first-pass
            # approximation by scaling bulk density by relative density from the paper's
            # ~240-271 kg/m^3 measured lattice bounding-box values.
            #
            # A representative midpoint approximation:
            effective_density_kg_per_m3 = 240.0

        result = {
            "method": "assign_effective_material_properties",
            "specimen_name": specimen_name,
            "defect_percentages": {
                "nominal_missing_strut_percent": float(nominal_missing_strut_percent),
                "measured_missing_strut_percent": float(measured_missing_strut_percent),
                "measured_disconnected_strut_percent": float(measured_disconnected_strut_percent),
                "total_measured_defect_percent": float(total_measured_defect_percent),
            },
            "bulk_skin_material": {
                "material_name": "Ti-5553",
                "youngs_modulus_gpa": float(bulk_youngs_modulus_gpa),
                "poissons_ratio": float(bulk_poissons_ratio),
                "density_kg_per_m3": float(bulk_density_kg_per_m3),
            },
            "effective_lattice_material": {
                "model_type": "effective_continuum_placeholder",
                "youngs_modulus_gpa": float(effective_youngs_modulus_gpa),
                "poissons_ratio": float(effective_poissons_ratio),
                "density_kg_per_m3": float(effective_density_kg_per_m3),
            },
            "metadata": {
                "effective_properties_user_supplied": bool(
                    effective_youngs_modulus_gpa is not None and
                    effective_poissons_ratio is not None and
                    effective_density_kg_per_m3 is not None and
                    not estimated_effective
                ),
                "effective_properties_estimated": bool(estimated_effective),
                "notes": [
                    "Bulk Ti-5553 properties are intended for the solid skins.",
                    "Effective lattice properties are intended for the simplified continuum interior region.",
                    "If these values are estimated, they should be treated as first-pass placeholders and refined later.",
                ],
            },
        }

        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

            result["saved_to"] = str(output_path)

        return result

    except Exception as error:
        return {"error": f"Error assigning effective material properties: {error}"}

@mcp.tool()
def build_effective_specimen_model(
    specimen_name: str,
    axes_filepath: str,
    lattice_size_x_mm: float,
    lattice_size_y_mm: float,
    lattice_size_z_mm: float,
    top_skin_thickness_mm: float,
    bottom_skin_thickness_mm: float,
    loading_axis_name: Optional[str] = None,
    output_filepath: Optional[str] = None,
) -> dict:
    """
    Build a simplified specimen-scale continuum model description.

    This tool creates a geometry/metadata description for a simplified specimen
    composed of:
    - a lattice interior region represented as an effective continuum block
    - a top solid skin
    - a bottom solid skin

    The model is aligned using the specimen axes from compute_specimen_axes.

    Args:
        specimen_name: Name/label of the specimen.
        axes_filepath: Path to the JSON output from compute_specimen_axes.
        lattice_size_x_mm: Lattice bounding-box size along specimen local x.
        lattice_size_y_mm: Lattice bounding-box size along specimen local y.
        lattice_size_z_mm: Lattice bounding-box size along specimen local z.
        top_skin_thickness_mm: Thickness of the top skin in mm.
        bottom_skin_thickness_mm: Thickness of the bottom skin in mm.
        loading_axis_name: Optional chosen loading axis ("r1", "r2", or "r3").
            If omitted, defaults to "r1".
        output_filepath: Optional path to save the simplified model as JSON.

    Returns:
        A dictionary describing the simplified specimen-scale geometry.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        def _normalize(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Cannot normalize a zero-length vector.")
            return vector / norm

        if lattice_size_x_mm <= 0 or lattice_size_y_mm <= 0 or lattice_size_z_mm <= 0:
            return {"error": "All lattice dimensions must be > 0."}

        if top_skin_thickness_mm < 0 or bottom_skin_thickness_mm < 0:
            return {"error": "Skin thicknesses must be >= 0."}

        if loading_axis_name is None:
            loading_axis_name = "r1"

        if loading_axis_name not in {"r1", "r2", "r3"}:
            return {"error": f"loading_axis_name must be one of 'r1', 'r2', or 'r3', got {loading_axis_name}."}

        axes_path = Path(axes_filepath)
        if not axes_path.exists():
            return {"error": f"Axes file not found: {axes_filepath}"}

        with axes_path.open("r", encoding="utf-8") as handle:
            axes_payload = json.load(handle)

        if "center" not in axes_payload or "axis_vectors" not in axes_payload:
            return {"error": "Axes JSON must contain 'center' and 'axis_vectors'."}

        center_index = np.asarray(axes_payload["center"], dtype=float)
        axis_vectors = axes_payload["axis_vectors"]

        for key in ("r1", "r2", "r3"):
            if key not in axis_vectors:
                return {"error": f"Axes JSON missing axis vector '{key}'."}

        r1 = _normalize(np.asarray(axis_vectors["r1"], dtype=float))
        r2 = _normalize(np.asarray(axis_vectors["r2"], dtype=float))
        r3 = _normalize(np.asarray(axis_vectors["r3"], dtype=float))

        rotation_matrix = np.column_stack([r1, r2, r3])

        loading_axis_vector = _normalize(np.asarray(axis_vectors[loading_axis_name], dtype=float))

        # Map local specimen sizes onto the chosen axes:
        # local x -> r1, local y -> r2, local z -> r3
        local_half_sizes = {
            "r1": 0.5 * lattice_size_x_mm,
            "r2": 0.5 * lattice_size_y_mm,
            "r3": 0.5 * lattice_size_z_mm,
        }

        # Determine which local size corresponds to the chosen loading axis.
        lattice_height_mm = float(2.0 * local_half_sizes[loading_axis_name])

        overall_height_mm = lattice_height_mm + top_skin_thickness_mm + bottom_skin_thickness_mm

        # The lattice region occupies the center; skins extend on +/- loading axis.
        # Define signed positions along the loading axis relative to specimen center.
        lattice_top_position_mm = lattice_height_mm / 2.0
        lattice_bottom_position_mm = -lattice_height_mm / 2.0

        top_skin_inner_position_mm = lattice_top_position_mm
        top_skin_outer_position_mm = lattice_top_position_mm + top_skin_thickness_mm

        bottom_skin_inner_position_mm = lattice_bottom_position_mm
        bottom_skin_outer_position_mm = lattice_bottom_position_mm - bottom_skin_thickness_mm

        result = {
            "method": "build_effective_specimen_model",
            "specimen_name": specimen_name,
            "axes_filepath": str(axes_path),
            "center_index_coordinates": center_index.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "axis_vectors": {
                "r1": r1.tolist(),
                "r2": r2.tolist(),
                "r3": r3.tolist(),
            },
            "loading_axis": {
                "name": loading_axis_name,
                "vector": loading_axis_vector.tolist(),
            },
            "geometry_mm": {
                "lattice_region_size": {
                    "x": float(lattice_size_x_mm),
                    "y": float(lattice_size_y_mm),
                    "z": float(lattice_size_z_mm),
                },
                "skin_thickness": {
                    "top": float(top_skin_thickness_mm),
                    "bottom": float(bottom_skin_thickness_mm),
                },
                "lattice_height_along_loading_axis": float(lattice_height_mm),
                "overall_height_along_loading_axis": float(overall_height_mm),
            },
            "regions": {
                "lattice_interior": {
                    "type": "effective_continuum_block",
                    "top_position_along_loading_axis_mm": float(lattice_top_position_mm),
                    "bottom_position_along_loading_axis_mm": float(lattice_bottom_position_mm),
                },
                "top_skin": {
                    "type": "solid_skin",
                    "inner_position_along_loading_axis_mm": float(top_skin_inner_position_mm),
                    "outer_position_along_loading_axis_mm": float(top_skin_outer_position_mm),
                },
                "bottom_skin": {
                    "type": "solid_skin",
                    "inner_position_along_loading_axis_mm": float(bottom_skin_inner_position_mm),
                    "outer_position_along_loading_axis_mm": float(bottom_skin_outer_position_mm),
                },
            },
            "boundary_faces": {
                "top_face_position_along_loading_axis_mm": float(top_skin_outer_position_mm),
                "bottom_face_position_along_loading_axis_mm": float(bottom_skin_outer_position_mm),
            },
            "notes": [
                "This is a simplified specimen-scale continuum description, not a voxel-resolved geometry.",
                "The lattice interior is represented as an effective continuum block.",
                "The top and bottom skins are represented as solid regions.",
            ],
        }

        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

            result["saved_to"] = str(output_path)

        return result

    except Exception as error:
        return {"error": f"Error building effective specimen model: {error}"}

@mcp.tool()
def export_continuum_fea_inputs(
    model_filepath: str,
    material_filepath: str,
    output_prefix: str,
    num_displacement_steps: int = 10,
    max_displacement_mm: float = 0.125,
) -> dict:
    """
    Export a simplified continuum FEA input package from:
    - a simplified specimen model description
    - assigned bulk/effective material properties

    This tool does not run the simulation. It packages:
    - simplified specimen geometry
    - loading direction
    - top/bottom face locations
    - material assignments
    - displacement-control schedule
    - derived reference quantities (height, area, engineering strain schedule)

    Args:
        model_filepath: Path to the JSON output from build_effective_specimen_model.
        material_filepath: Path to the JSON output from assign_effective_material_properties.
        output_prefix: Prefix for output files. This tool writes:
            - <output_prefix>.npz
            - <output_prefix>.json
        num_displacement_steps: Number of displacement-control steps.
        max_displacement_mm: Maximum imposed compression displacement in mm.

    Returns:
        A dictionary describing the exported continuum FEA package.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        model_path = Path(model_filepath)
        material_path = Path(material_filepath)

        if not model_path.exists():
            return {"error": f"Model file not found: {model_filepath}"}

        if not material_path.exists():
            return {"error": f"Material file not found: {material_filepath}"}

        if num_displacement_steps <= 0:
            return {"error": f"num_displacement_steps must be > 0, got {num_displacement_steps}."}

        if max_displacement_mm < 0:
            return {"error": f"max_displacement_mm must be >= 0, got {max_displacement_mm}."}

        with model_path.open("r", encoding="utf-8") as handle:
            model_payload = json.load(handle)

        with material_path.open("r", encoding="utf-8") as handle:
            material_payload = json.load(handle)

        # Validate model payload
        required_model_keys = [
            "specimen_name",
            "axis_vectors",
            "loading_axis",
            "geometry_mm",
            "regions",
            "boundary_faces",
        ]
        for key in required_model_keys:
            if key not in model_payload:
                return {"error": f"Model JSON missing required key '{key}'."}

        axis_vectors = model_payload["axis_vectors"]
        loading_axis = model_payload["loading_axis"]
        geometry_mm = model_payload["geometry_mm"]
        regions = model_payload["regions"]
        boundary_faces = model_payload["boundary_faces"]

        for key in ("r1", "r2", "r3"):
            if key not in axis_vectors:
                return {"error": f"Model JSON axis_vectors missing '{key}'."}

        if "name" not in loading_axis or "vector" not in loading_axis:
            return {"error": "Model JSON loading_axis must contain 'name' and 'vector'."}

        if "lattice_region_size" not in geometry_mm or "skin_thickness" not in geometry_mm:
            return {"error": "Model JSON geometry_mm must contain 'lattice_region_size' and 'skin_thickness'."}

        lattice_region_size = geometry_mm["lattice_region_size"]
        for key in ("x", "y", "z"):
            if key not in lattice_region_size:
                return {"error": f"Model JSON lattice_region_size missing '{key}'."}

        lattice_x = float(lattice_region_size["x"])
        lattice_y = float(lattice_region_size["y"])
        lattice_z = float(lattice_region_size["z"])

        if lattice_x <= 0 or lattice_y <= 0 or lattice_z <= 0:
            return {"error": "All lattice region dimensions must be > 0."}

        if "overall_height_along_loading_axis" not in geometry_mm:
            return {"error": "Model JSON geometry_mm missing 'overall_height_along_loading_axis'."}

        overall_height_mm = float(geometry_mm["overall_height_along_loading_axis"])
        if overall_height_mm <= 0:
            return {"error": f"overall_height_along_loading_axis must be > 0, got {overall_height_mm}."}

        if "top_face_position_along_loading_axis_mm" not in boundary_faces:
            return {"error": "Model JSON boundary_faces missing 'top_face_position_along_loading_axis_mm'."}

        if "bottom_face_position_along_loading_axis_mm" not in boundary_faces:
            return {"error": "Model JSON boundary_faces missing 'bottom_face_position_along_loading_axis_mm'."}

        # Validate material payload
        required_material_keys = [
            "specimen_name",
            "bulk_skin_material",
            "effective_lattice_material",
        ]
        for key in required_material_keys:
            if key not in material_payload:
                return {"error": f"Material JSON missing required key '{key}'."}

        bulk_skin_material = material_payload["bulk_skin_material"]
        effective_lattice_material = material_payload["effective_lattice_material"]

        for key in ("youngs_modulus_gpa", "poissons_ratio", "density_kg_per_m3"):
            if key not in bulk_skin_material:
                return {"error": f"bulk_skin_material missing '{key}'."}

        for key in ("youngs_modulus_gpa", "poissons_ratio", "density_kg_per_m3"):
            if key not in effective_lattice_material:
                return {"error": f"effective_lattice_material missing '{key}'."}

        loading_axis_name = loading_axis["name"]
        if loading_axis_name not in {"r1", "r2", "r3"}:
            return {"error": f"loading_axis name must be one of 'r1', 'r2', or 'r3', got {loading_axis_name}."}

        # Cross-sectional area perpendicular to the loading axis
        # Local x -> r1, local y -> r2, local z -> r3
        if loading_axis_name == "r1":
            reference_area_mm2 = lattice_y * lattice_z
            cross_section_dimensions_mm = {"dim_1": lattice_y, "dim_2": lattice_z}
        elif loading_axis_name == "r2":
            reference_area_mm2 = lattice_x * lattice_z
            cross_section_dimensions_mm = {"dim_1": lattice_x, "dim_2": lattice_z}
        else:  # r3
            reference_area_mm2 = lattice_x * lattice_y
            cross_section_dimensions_mm = {"dim_1": lattice_x, "dim_2": lattice_y}

        displacement_steps_mm = np.linspace(
            0.0,
            float(max_displacement_mm),
            int(num_displacement_steps) + 1,
            dtype=float,
        )
        engineering_strain = displacement_steps_mm / overall_height_mm

        top_face_position = float(boundary_faces["top_face_position_along_loading_axis_mm"])
        bottom_face_position = float(boundary_faces["bottom_face_position_along_loading_axis_mm"])

        output_base = Path(output_prefix)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output_base.with_suffix(".npz")
        json_path = output_base.with_suffix(".json")

        # Compact machine-readable bundle
        np.savez_compressed(
            npz_path,
            displacement_steps_mm=displacement_steps_mm.astype(float),
            engineering_strain=engineering_strain.astype(float),
            loading_axis_vector=np.asarray(loading_axis["vector"], dtype=float),
            rotation_matrix=np.asarray(model_payload["rotation_matrix"], dtype=float),
            center_index_coordinates=np.asarray(model_payload["center_index_coordinates"], dtype=float),
            top_face_position_along_loading_axis_mm=np.asarray([top_face_position], dtype=float),
            bottom_face_position_along_loading_axis_mm=np.asarray([bottom_face_position], dtype=float),
            overall_height_mm=np.asarray([overall_height_mm], dtype=float),
            reference_area_mm2=np.asarray([reference_area_mm2], dtype=float),
        )

        manifest = {
            "method": "export_continuum_fea_inputs",
            "inputs": {
                "model_filepath": str(model_path),
                "material_filepath": str(material_path),
            },
            "specimen_name": model_payload["specimen_name"],
            "geometry": {
                "center_index_coordinates": model_payload["center_index_coordinates"],
                "rotation_matrix": model_payload["rotation_matrix"],
                "axis_vectors": model_payload["axis_vectors"],
                "loading_axis": loading_axis,
                "lattice_region_size_mm": {
                    "x": lattice_x,
                    "y": lattice_y,
                    "z": lattice_z,
                },
                "skin_thickness_mm": geometry_mm["skin_thickness"],
                "overall_height_along_loading_axis_mm": overall_height_mm,
                "top_face_position_along_loading_axis_mm": top_face_position,
                "bottom_face_position_along_loading_axis_mm": bottom_face_position,
                "reference_cross_section_area_mm2": float(reference_area_mm2),
                "reference_cross_section_dimensions_mm": cross_section_dimensions_mm,
            },
            "regions": regions,
            "materials": {
                "top_skin": bulk_skin_material,
                "bottom_skin": bulk_skin_material,
                "lattice_interior": effective_lattice_material,
            },
            "loading": {
                "control_type": "displacement_control",
                "num_displacement_steps": int(num_displacement_steps),
                "max_displacement_mm": float(max_displacement_mm),
                "displacement_steps_mm": displacement_steps_mm.tolist(),
                "engineering_strain": engineering_strain.tolist(),
                "compression_direction_note": "Top face is driven toward bottom face along the selected loading axis.",
            },
            "outputs": {
                "npz_filepath": str(npz_path),
                "json_filepath": str(json_path),
            },
            "notes": [
                "This is a simplified continuum-model FEA package, not a voxel-resolved model.",
                "The lattice interior is assigned effective continuum properties.",
                "The top and bottom skins are assigned bulk Ti-5553 properties.",
            ],
        }

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return {
            "result": "Continuum FEA input package exported successfully.",
            "npz_filepath": str(npz_path),
            "json_filepath": str(json_path),
            "specimen_name": model_payload["specimen_name"],
            "loading_axis_name": loading_axis_name,
            "overall_height_along_loading_axis_mm": overall_height_mm,
            "reference_cross_section_area_mm2": float(reference_area_mm2),
            "num_displacement_steps": int(num_displacement_steps),
            "max_displacement_mm": float(max_displacement_mm),
        }

    except Exception as error:
        return {"error": f"Error exporting continuum FEA inputs: {error}"}

@mcp.tool()
def run_continuum_fea(
    fea_input_filepath: str,
    output_prefix: str,
) -> dict:
    """
    Run a simplified continuum compression solve from an exported continuum FEA package.

    This implementation uses a 1D layered axial model with three regions in series:
    - bottom skin
    - effective lattice interior
    - top skin

    Assumptions:
    - small-strain linear elasticity
    - displacement-controlled uniaxial compression
    - perfectly bonded layers
    - uniform cross-sectional area
    - uniform axial stress through the specimen
    - no local bending, shear, or explicit defect-resolution effects

    This is a practical specimen-scale continuum approximation for overall
    force-displacement and stress-strain behavior.

    Args:
        fea_input_filepath: Path to the JSON output from export_continuum_fea_inputs.
        output_prefix: Prefix for output files. This tool writes:
            - <output_prefix>.npz
            - <output_prefix>.json

    Returns:
        A dictionary containing the solve summary and output file paths.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        input_path = Path(fea_input_filepath)
        if not input_path.exists():
            return {"error": f"Continuum FEA input file not found: {fea_input_filepath}"}

        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        required_keys = ["geometry", "materials", "loading", "specimen_name"]
        for key in required_keys:
            if key not in payload:
                return {"error": f"Continuum FEA input JSON missing required key '{key}'."}

        geometry = payload["geometry"]
        materials = payload["materials"]
        loading = payload["loading"]

        for key in [
            "overall_height_along_loading_axis_mm",
            "reference_cross_section_area_mm2",
            "lattice_region_size_mm",
            "skin_thickness_mm",
            "loading_axis",
        ]:
            if key not in geometry:
                return {"error": f"Geometry section missing '{key}'."}

        for key in ["top_skin", "bottom_skin", "lattice_interior"]:
            if key not in materials:
                return {"error": f"Materials section missing '{key}'."}

        for key in ["displacement_steps_mm", "engineering_strain"]:
            if key not in loading:
                return {"error": f"Loading section missing '{key}'."}

        overall_height_mm = float(geometry["overall_height_along_loading_axis_mm"])
        area_mm2 = float(geometry["reference_cross_section_area_mm2"])

        if overall_height_mm <= 0:
            return {"error": f"overall_height_along_loading_axis_mm must be > 0, got {overall_height_mm}."}
        if area_mm2 <= 0:
            return {"error": f"reference_cross_section_area_mm2 must be > 0, got {area_mm2}."}

        lattice_region_size_mm = geometry["lattice_region_size_mm"]
        skin_thickness_mm = geometry["skin_thickness_mm"]

        loading_axis_name = geometry["loading_axis"]["name"]

        if loading_axis_name == "r1":
            lattice_height_mm = float(lattice_region_size_mm["x"])
        elif loading_axis_name == "r2":
            lattice_height_mm = float(lattice_region_size_mm["y"])
        elif loading_axis_name == "r3":
            lattice_height_mm = float(lattice_region_size_mm["z"])
        else:
            return {"error": f"Unsupported loading_axis name: {loading_axis_name}"}

        top_skin_thickness_mm = float(skin_thickness_mm["top"])
        bottom_skin_thickness_mm = float(skin_thickness_mm["bottom"])

        if lattice_height_mm <= 0:
            return {"error": f"Derived lattice height must be > 0, got {lattice_height_mm}."}
        if top_skin_thickness_mm < 0 or bottom_skin_thickness_mm < 0:
            return {"error": "Skin thicknesses must be >= 0."}

        top_skin = materials["top_skin"]
        bottom_skin = materials["bottom_skin"]
        lattice_interior = materials["lattice_interior"]

        for region_name, region in [
            ("top_skin", top_skin),
            ("bottom_skin", bottom_skin),
            ("lattice_interior", lattice_interior),
        ]:
            for key in ["youngs_modulus_gpa", "poissons_ratio", "density_kg_per_m3"]:
                if key not in region:
                    return {"error": f"Material region '{region_name}' missing '{key}'."}

        # Convert GPa -> N/mm^2 (MPa)
        # 1 GPa = 1000 MPa = 1000 N/mm^2
        E_top_n_per_mm2 = float(top_skin["youngs_modulus_gpa"]) * 1000.0
        E_bottom_n_per_mm2 = float(bottom_skin["youngs_modulus_gpa"]) * 1000.0
        E_lattice_n_per_mm2 = float(lattice_interior["youngs_modulus_gpa"]) * 1000.0

        if E_top_n_per_mm2 <= 0 or E_bottom_n_per_mm2 <= 0 or E_lattice_n_per_mm2 <= 0:
            return {"error": "All Young's modulus values must be > 0."}

        displacement_steps_mm = np.asarray(loading["displacement_steps_mm"], dtype=float)
        engineering_strain_input = np.asarray(loading["engineering_strain"], dtype=float)

        if displacement_steps_mm.ndim != 1:
            return {"error": "displacement_steps_mm must be a 1D array/list."}

        # 1D series compliance model:
        # delta_total = sigma * sum(t_i / E_i)
        # strain_total = delta_total / H_total
        # sigma = delta_total / sum(t_i / E_i)
        #
        # Equivalent modulus:
        # E_eff = H_total / sum(t_i / E_i)
        compliance_sum = (
            (bottom_skin_thickness_mm / E_bottom_n_per_mm2) +
            (lattice_height_mm / E_lattice_n_per_mm2) +
            (top_skin_thickness_mm / E_top_n_per_mm2)
        )

        if compliance_sum <= 0:
            return {"error": f"Computed compliance_sum must be > 0, got {compliance_sum}."}

        equivalent_modulus_n_per_mm2 = overall_height_mm / compliance_sum
        equivalent_modulus_gpa = equivalent_modulus_n_per_mm2 / 1000.0

        engineering_strain = displacement_steps_mm / overall_height_mm
        engineering_stress_n_per_mm2 = displacement_steps_mm / compliance_sum
        engineering_stress_mpa = engineering_stress_n_per_mm2
        reaction_force_n = engineering_stress_n_per_mm2 * area_mm2

        # Region-wise strain partitioning under common axial stress
        top_skin_strain = engineering_stress_n_per_mm2 / E_top_n_per_mm2
        bottom_skin_strain = engineering_stress_n_per_mm2 / E_bottom_n_per_mm2
        lattice_strain = engineering_stress_n_per_mm2 / E_lattice_n_per_mm2

        top_skin_displacement_mm = top_skin_strain * top_skin_thickness_mm
        bottom_skin_displacement_mm = bottom_skin_strain * bottom_skin_thickness_mm
        lattice_displacement_mm = lattice_strain * lattice_height_mm

        # Consistency check
        reconstructed_total_displacement_mm = (
            top_skin_displacement_mm +
            bottom_skin_displacement_mm +
            lattice_displacement_mm
        )

        max_displacement_residual_mm = float(
            np.max(np.abs(reconstructed_total_displacement_mm - displacement_steps_mm))
        )

        output_base = Path(output_prefix)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output_base.with_suffix(".npz")
        json_path = output_base.with_suffix(".json")

        np.savez_compressed(
            npz_path,
            displacement_steps_mm=displacement_steps_mm.astype(float),
            engineering_strain=engineering_strain.astype(float),
            engineering_stress_mpa=engineering_stress_mpa.astype(float),
            reaction_force_n=reaction_force_n.astype(float),
            top_skin_strain=top_skin_strain.astype(float),
            bottom_skin_strain=bottom_skin_strain.astype(float),
            lattice_strain=lattice_strain.astype(float),
            top_skin_displacement_mm=top_skin_displacement_mm.astype(float),
            bottom_skin_displacement_mm=bottom_skin_displacement_mm.astype(float),
            lattice_displacement_mm=lattice_displacement_mm.astype(float),
            reconstructed_total_displacement_mm=reconstructed_total_displacement_mm.astype(float),
        )

        manifest = {
            "method": "run_continuum_fea",
            "model_type": "1d_series_layer_compression",
            "input_fea_package": str(input_path),
            "specimen_name": payload["specimen_name"],
            "assumptions": [
                "small-strain linear elasticity",
                "uniform axial stress",
                "perfect bonding between layers",
                "uniform cross-sectional area",
                "no explicit local defect-resolution",
            ],
            "geometry": {
                "loading_axis": geometry["loading_axis"],
                "overall_height_along_loading_axis_mm": overall_height_mm,
                "reference_cross_section_area_mm2": area_mm2,
                "layer_thicknesses_mm": {
                    "bottom_skin": bottom_skin_thickness_mm,
                    "lattice_interior": lattice_height_mm,
                    "top_skin": top_skin_thickness_mm,
                },
            },
            "materials": {
                "top_skin": top_skin,
                "bottom_skin": bottom_skin,
                "lattice_interior": lattice_interior,
                "equivalent_specimen_modulus_gpa": float(equivalent_modulus_gpa),
            },
            "loading": {
                "control_type": "displacement_control",
                "displacement_steps_mm": displacement_steps_mm.tolist(),
                "engineering_strain": engineering_strain.tolist(),
            },
            "results_summary": {
                "max_engineering_stress_mpa": float(engineering_stress_mpa.max()),
                "max_reaction_force_n": float(reaction_force_n.max()),
                "equivalent_specimen_modulus_gpa": float(equivalent_modulus_gpa),
                "max_displacement_reconstruction_residual_mm": max_displacement_residual_mm,
            },
            "outputs": {
                "npz_filepath": str(npz_path),
                "json_filepath": str(json_path),
            },
        }

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return {
            "result": "Continuum compression solve completed successfully.",
            "npz_filepath": str(npz_path),
            "json_filepath": str(json_path),
            "specimen_name": payload["specimen_name"],
            "equivalent_specimen_modulus_gpa": float(equivalent_modulus_gpa),
            "max_engineering_stress_mpa": float(engineering_stress_mpa.max()),
            "max_reaction_force_n": float(reaction_force_n.max()),
            "max_displacement_reconstruction_residual_mm": max_displacement_residual_mm,
        }

    except Exception as error:
        return {"error": f"Error running continuum FEA: {error}"}

@mcp.tool()
def postprocess_fea_results(
    fea_results_filepath: str,
    output_filepath: Optional[str] = None,
    save_pngs: bool = True,
    figure_prefix: Optional[str] = None,
) -> dict:
    """
    Postprocess a single-specimen continuum FEA result package.

    This tool is intended for the output of run_continuum_fea. It summarizes:
    - force-displacement response
    - stress-strain response
    - equivalent specimen modulus
    - peak stress / force
    - region-wise deformation partitioning

    It can also save PNG figures for:
    - force vs displacement
    - stress vs strain
    - final-step deformation partition

    It does not compare multiple specimens; it focuses on one specimen only.

    Args:
        fea_results_filepath: Path to the .npz output from run_continuum_fea.
        output_filepath: Optional path to save the summary JSON.
        save_pngs: Whether to save PNG plots.
        figure_prefix: Optional prefix for PNG outputs. If omitted, uses the
            results file stem in the same folder.

    Returns:
        A dictionary containing a concise summary of the specimen response.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        results_path = Path(fea_results_filepath)
        if not results_path.exists():
            return {"error": f"FEA results file not found: {fea_results_filepath}"}

        data = np.load(results_path)

        required_arrays = [
            "displacement_steps_mm",
            "engineering_strain",
            "engineering_stress_mpa",
            "reaction_force_n",
            "top_skin_strain",
            "bottom_skin_strain",
            "lattice_strain",
            "top_skin_displacement_mm",
            "bottom_skin_displacement_mm",
            "lattice_displacement_mm",
            "reconstructed_total_displacement_mm",
        ]
        for key in required_arrays:
            if key not in data:
                return {"error": f"Results package missing required array '{key}'."}

        displacement_steps_mm = np.asarray(data["displacement_steps_mm"], dtype=float)
        engineering_strain = np.asarray(data["engineering_strain"], dtype=float)
        engineering_stress_mpa = np.asarray(data["engineering_stress_mpa"], dtype=float)
        reaction_force_n = np.asarray(data["reaction_force_n"], dtype=float)

        top_skin_strain = np.asarray(data["top_skin_strain"], dtype=float)
        bottom_skin_strain = np.asarray(data["bottom_skin_strain"], dtype=float)
        lattice_strain = np.asarray(data["lattice_strain"], dtype=float)

        top_skin_displacement_mm = np.asarray(data["top_skin_displacement_mm"], dtype=float)
        bottom_skin_displacement_mm = np.asarray(data["bottom_skin_displacement_mm"], dtype=float)
        lattice_displacement_mm = np.asarray(data["lattice_displacement_mm"], dtype=float)
        reconstructed_total_displacement_mm = np.asarray(
            data["reconstructed_total_displacement_mm"], dtype=float
        )

        n = len(displacement_steps_mm)
        if n == 0:
            return {"error": "Results arrays are empty."}

        same_length_arrays = [
            engineering_strain,
            engineering_stress_mpa,
            reaction_force_n,
            top_skin_strain,
            bottom_skin_strain,
            lattice_strain,
            top_skin_displacement_mm,
            bottom_skin_displacement_mm,
            lattice_displacement_mm,
            reconstructed_total_displacement_mm,
        ]
        if not all(len(arr) == n for arr in same_length_arrays):
            return {"error": "Results arrays do not all have the same length."}

        max_force_n = float(np.max(reaction_force_n))
        max_stress_mpa = float(np.max(engineering_stress_mpa))
        max_strain = float(np.max(engineering_strain))
        max_displacement_mm = float(np.max(displacement_steps_mm))

        if n >= 2 and np.max(engineering_strain) > 0:
            slope_mpa = float(np.polyfit(engineering_strain, engineering_stress_mpa, 1)[0])
            effective_modulus_gpa = slope_mpa / 1000.0
        else:
            slope_mpa = 0.0
            effective_modulus_gpa = 0.0

        final_total = float(reconstructed_total_displacement_mm[-1])
        if final_total > 0:
            top_fraction = float(top_skin_displacement_mm[-1] / final_total)
            bottom_fraction = float(bottom_skin_displacement_mm[-1] / final_total)
            lattice_fraction = float(lattice_displacement_mm[-1] / final_total)
        else:
            top_fraction = 0.0
            bottom_fraction = 0.0
            lattice_fraction = 0.0

        final_region_strains = {
            "top_skin": float(top_skin_strain[-1]),
            "bottom_skin": float(bottom_skin_strain[-1]),
            "lattice_interior": float(lattice_strain[-1]),
        }
        dominant_strain_region = max(final_region_strains, key=final_region_strains.get)

        displacement_residual_mm = np.abs(reconstructed_total_displacement_mm - displacement_steps_mm)
        max_residual_mm = float(np.max(displacement_residual_mm))

        response_points = []
        for i in range(n):
            response_points.append(
                {
                    "step": int(i),
                    "displacement_mm": float(displacement_steps_mm[i]),
                    "engineering_strain": float(engineering_strain[i]),
                    "engineering_stress_mpa": float(engineering_stress_mpa[i]),
                    "reaction_force_n": float(reaction_force_n[i]),
                }
            )

        png_outputs = {}

        if save_pngs:
            import matplotlib.pyplot as plt

            if figure_prefix is None:
                figure_base = results_path.with_suffix("")
            else:
                figure_base = Path(figure_prefix)
                figure_base.parent.mkdir(parents=True, exist_ok=True)

            force_disp_png = str(figure_base) + "_force_displacement.png"
            stress_strain_png = str(figure_base) + "_stress_strain.png"
            partition_png = str(figure_base) + "_deformation_partition.png"

            # Force-displacement
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(displacement_steps_mm, reaction_force_n, marker="o", linewidth=2)
            ax.set_xlabel("Displacement (mm)")
            ax.set_ylabel("Reaction Force (N)")
            ax.set_title("Force-Displacement Response")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(force_disp_png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Stress-strain
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(engineering_strain, engineering_stress_mpa, marker="o", linewidth=2)
            ax.set_xlabel("Engineering Strain")
            ax.set_ylabel("Engineering Stress (MPa)")
            ax.set_title("Stress-Strain Response")
            ax.grid(True, alpha=0.3)
            ax.text(
                0.02,
                0.98,
                f"Effective modulus = {effective_modulus_gpa:.3f} GPa",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            fig.tight_layout()
            fig.savefig(stress_strain_png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Final-step deformation partition
            labels = ["Top Skin", "Bottom Skin", "Lattice Interior"]
            values = [top_fraction, bottom_fraction, lattice_fraction]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Fraction of Total Displacement")
            ax.set_title("Final-Step Deformation Partition")
            ax.grid(True, axis="y", alpha=0.3)

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            fig.tight_layout()
            fig.savefig(partition_png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            png_outputs = {
                "force_displacement_png": force_disp_png,
                "stress_strain_png": stress_strain_png,
                "deformation_partition_png": partition_png,
            }

        result = {
            "method": "postprocess_fea_results",
            "input_results_filepath": str(results_path),
            "summary": {
                "num_steps": int(n),
                "max_displacement_mm": max_displacement_mm,
                "max_engineering_strain": max_strain,
                "max_engineering_stress_mpa": max_stress_mpa,
                "max_reaction_force_n": max_force_n,
                "effective_specimen_modulus_gpa": float(effective_modulus_gpa),
                "stress_strain_slope_mpa": float(slope_mpa),
            },
            "deformation_partition": {
                "final_step_displacement_fraction": {
                    "top_skin": top_fraction,
                    "bottom_skin": bottom_fraction,
                    "lattice_interior": lattice_fraction,
                },
                "final_step_region_strains": final_region_strains,
                "dominant_strain_region": dominant_strain_region,
            },
            "consistency_checks": {
                "max_displacement_reconstruction_residual_mm": max_residual_mm,
            },
            "response_curve": response_points,
            "figures": png_outputs,
            "notes": [
                "This summary is for a single simplified continuum specimen model.",
                "The reported effective modulus comes from the linear stress-strain slope of the solved response.",
                "The deformation partition shows how much of the imposed displacement is taken up by the skins versus the effective lattice interior.",
            ],
        }

        if output_filepath:
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            result["saved_to"] = str(output_path)

        return result

    except Exception as error:
        return {"error": f"Error postprocessing FEA results: {error}"}

@mcp.tool()
def run_continuum_fea_with_settling(
    fea_input_filepath: str,
    output_prefix: str,
    settling_displacement_mm: float = 0.02,
    settling_stiffness_scale: float = 0.3,
) -> dict:
    """
    Run a simplified continuum compression solve with an initial settling/bedding-in regime.

    This implementation extends the 1D layered axial model used in run_continuum_fea by
    adding an initial low-stiffness regime before the main linear elastic response.

    Model structure:
    - bottom skin
    - effective lattice interior
    - top skin

    Response law:
    - For small imposed displacement up to settling_displacement_mm:
        lower effective stiffness = settling_stiffness_scale * main stiffness
    - After settling_displacement_mm:
        main linear elastic stiffness is used

    Assumptions:
    - small-strain layered response
    - displacement-controlled uniaxial compression
    - initial settling/bedding-in is represented phenomenologically
    - no explicit local defect-resolution or hysteresis
    - no unloading/reloading cycle modeling

    Args:
        fea_input_filepath: Path to the JSON output from export_continuum_fea_inputs.
        output_prefix: Prefix for output files. This tool writes:
            - <output_prefix>.npz
            - <output_prefix>.json
        settling_displacement_mm: End of the initial low-stiffness regime.
        settling_stiffness_scale: Relative stiffness in the settling regime
            compared to the main elastic stiffness. Must satisfy 0 < scale <= 1.

    Returns:
        A dictionary containing the solve summary and output file paths.
    """
    try:
        import json
        import numpy as np
        from pathlib import Path

        input_path = Path(fea_input_filepath)
        if not input_path.exists():
            return {"error": f"Continuum FEA input file not found: {fea_input_filepath}"}

        if settling_displacement_mm < 0:
            return {"error": f"settling_displacement_mm must be >= 0, got {settling_displacement_mm}."}

        if settling_stiffness_scale <= 0 or settling_stiffness_scale > 1:
            return {
                "error": (
                    f"settling_stiffness_scale must satisfy 0 < scale <= 1, "
                    f"got {settling_stiffness_scale}."
                )
            }

        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        required_keys = ["geometry", "materials", "loading", "specimen_name"]
        for key in required_keys:
            if key not in payload:
                return {"error": f"Continuum FEA input JSON missing required key '{key}'."}

        geometry = payload["geometry"]
        materials = payload["materials"]
        loading = payload["loading"]

        for key in [
            "overall_height_along_loading_axis_mm",
            "reference_cross_section_area_mm2",
            "lattice_region_size_mm",
            "skin_thickness_mm",
            "loading_axis",
        ]:
            if key not in geometry:
                return {"error": f"Geometry section missing '{key}'."}

        for key in ["top_skin", "bottom_skin", "lattice_interior"]:
            if key not in materials:
                return {"error": f"Materials section missing '{key}'."}

        if "displacement_steps_mm" not in loading:
            return {"error": "Loading section missing 'displacement_steps_mm'."}

        overall_height_mm = float(geometry["overall_height_along_loading_axis_mm"])
        area_mm2 = float(geometry["reference_cross_section_area_mm2"])

        if overall_height_mm <= 0:
            return {"error": f"overall_height_along_loading_axis_mm must be > 0, got {overall_height_mm}."}
        if area_mm2 <= 0:
            return {"error": f"reference_cross_section_area_mm2 must be > 0, got {area_mm2}."}

        lattice_region_size_mm = geometry["lattice_region_size_mm"]
        skin_thickness_mm = geometry["skin_thickness_mm"]

        loading_axis_name = geometry["loading_axis"]["name"]

        if loading_axis_name == "r1":
            lattice_height_mm = float(lattice_region_size_mm["x"])
        elif loading_axis_name == "r2":
            lattice_height_mm = float(lattice_region_size_mm["y"])
        elif loading_axis_name == "r3":
            lattice_height_mm = float(lattice_region_size_mm["z"])
        else:
            return {"error": f"Unsupported loading_axis name: {loading_axis_name}"}

        top_skin_thickness_mm = float(skin_thickness_mm["top"])
        bottom_skin_thickness_mm = float(skin_thickness_mm["bottom"])

        top_skin = materials["top_skin"]
        bottom_skin = materials["bottom_skin"]
        lattice_interior = materials["lattice_interior"]

        for region_name, region in [
            ("top_skin", top_skin),
            ("bottom_skin", bottom_skin),
            ("lattice_interior", lattice_interior),
        ]:
            for key in ["youngs_modulus_gpa", "poissons_ratio", "density_kg_per_m3"]:
                if key not in region:
                    return {"error": f"Material region '{region_name}' missing '{key}'."}

        # Convert GPa -> N/mm^2
        E_top_n_per_mm2 = float(top_skin["youngs_modulus_gpa"]) * 1000.0
        E_bottom_n_per_mm2 = float(bottom_skin["youngs_modulus_gpa"]) * 1000.0
        E_lattice_n_per_mm2 = float(lattice_interior["youngs_modulus_gpa"]) * 1000.0

        if E_top_n_per_mm2 <= 0 or E_bottom_n_per_mm2 <= 0 or E_lattice_n_per_mm2 <= 0:
            return {"error": "All Young's modulus values must be > 0."}

        displacement_steps_mm = np.asarray(loading["displacement_steps_mm"], dtype=float)
        if displacement_steps_mm.ndim != 1:
            return {"error": "displacement_steps_mm must be a 1D array/list."}

        # Main layered elastic stiffness
        compliance_sum = (
            (bottom_skin_thickness_mm / E_bottom_n_per_mm2) +
            (lattice_height_mm / E_lattice_n_per_mm2) +
            (top_skin_thickness_mm / E_top_n_per_mm2)
        )

        if compliance_sum <= 0:
            return {"error": f"Computed compliance_sum must be > 0, got {compliance_sum}."}

        equivalent_modulus_n_per_mm2 = overall_height_mm / compliance_sum
        equivalent_modulus_gpa = equivalent_modulus_n_per_mm2 / 1000.0

        # Force-displacement main slope
        k_main_n_per_mm = area_mm2 / compliance_sum
        k_soft_n_per_mm = settling_stiffness_scale * k_main_n_per_mm

        # Piecewise force law
        reaction_force_n = np.zeros_like(displacement_steps_mm, dtype=float)

        for i, disp in enumerate(displacement_steps_mm):
            if disp <= settling_displacement_mm:
                reaction_force_n[i] = k_soft_n_per_mm * disp
            else:
                reaction_force_n[i] = (
                    k_soft_n_per_mm * settling_displacement_mm +
                    k_main_n_per_mm * (disp - settling_displacement_mm)
                )

        engineering_strain = displacement_steps_mm / overall_height_mm
        engineering_stress_mpa = reaction_force_n / area_mm2

        # Region-wise partition using the current axial stress state
        top_skin_strain = engineering_stress_mpa / E_top_n_per_mm2
        bottom_skin_strain = engineering_stress_mpa / E_bottom_n_per_mm2
        lattice_strain = engineering_stress_mpa / E_lattice_n_per_mm2

        top_skin_displacement_mm = top_skin_strain * top_skin_thickness_mm
        bottom_skin_displacement_mm = bottom_skin_strain * bottom_skin_thickness_mm
        lattice_displacement_mm = lattice_strain * lattice_height_mm

        reconstructed_total_displacement_mm = (
            top_skin_displacement_mm +
            bottom_skin_displacement_mm +
            lattice_displacement_mm
        )

        # This residual is expected to be nonzero during the settling regime because
        # the settling phase is phenomenological and not fully represented by the
        # elastic layered partition.
        displacement_residual_mm = np.abs(displacement_steps_mm - reconstructed_total_displacement_mm)
        max_residual_mm = float(np.max(displacement_residual_mm))

        output_base = Path(output_prefix)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output_base.with_suffix(".npz")
        json_path = output_base.with_suffix(".json")

        np.savez_compressed(
            npz_path,
            displacement_steps_mm=displacement_steps_mm.astype(float),
            engineering_strain=engineering_strain.astype(float),
            engineering_stress_mpa=engineering_stress_mpa.astype(float),
            reaction_force_n=reaction_force_n.astype(float),
            top_skin_strain=top_skin_strain.astype(float),
            bottom_skin_strain=bottom_skin_strain.astype(float),
            lattice_strain=lattice_strain.astype(float),
            top_skin_displacement_mm=top_skin_displacement_mm.astype(float),
            bottom_skin_displacement_mm=bottom_skin_displacement_mm.astype(float),
            lattice_displacement_mm=lattice_displacement_mm.astype(float),
            reconstructed_total_displacement_mm=reconstructed_total_displacement_mm.astype(float),
            displacement_residual_mm=displacement_residual_mm.astype(float),
        )

        manifest = {
            "method": "run_continuum_fea_with_settling",
            "model_type": "1d_series_layer_compression_with_initial_settling",
            "input_fea_package": str(input_path),
            "specimen_name": payload["specimen_name"],
            "assumptions": [
                "small-strain layered response after settling",
                "initial bedding-in/settling is represented by a reduced initial stiffness",
                "uniform axial stress after engagement",
                "perfect bonding between layers",
                "uniform cross-sectional area",
                "no explicit local defect-resolution",
            ],
            "geometry": {
                "loading_axis": geometry["loading_axis"],
                "overall_height_along_loading_axis_mm": overall_height_mm,
                "reference_cross_section_area_mm2": area_mm2,
                "layer_thicknesses_mm": {
                    "bottom_skin": bottom_skin_thickness_mm,
                    "lattice_interior": lattice_height_mm,
                    "top_skin": top_skin_thickness_mm,
                },
            },
            "materials": {
                "top_skin": top_skin,
                "bottom_skin": bottom_skin,
                "lattice_interior": lattice_interior,
                "equivalent_specimen_modulus_gpa_main_regime": float(equivalent_modulus_gpa),
            },
            "settling_model": {
                "settling_displacement_mm": float(settling_displacement_mm),
                "settling_stiffness_scale": float(settling_stiffness_scale),
                "main_stiffness_n_per_mm": float(k_main_n_per_mm),
                "settling_stiffness_n_per_mm": float(k_soft_n_per_mm),
            },
            "loading": {
                "control_type": "displacement_control",
                "displacement_steps_mm": displacement_steps_mm.tolist(),
                "engineering_strain": engineering_strain.tolist(),
            },
            "results_summary": {
                "max_engineering_stress_mpa": float(engineering_stress_mpa.max()),
                "max_reaction_force_n": float(reaction_force_n.max()),
                "equivalent_specimen_modulus_gpa_main_regime": float(equivalent_modulus_gpa),
                "max_displacement_residual_mm": max_residual_mm,
            },
            "outputs": {
                "npz_filepath": str(npz_path),
                "json_filepath": str(json_path),
            },
        }

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return {
            "result": "Continuum compression solve with settling completed successfully.",
            "npz_filepath": str(npz_path),
            "json_filepath": str(json_path),
            "specimen_name": payload["specimen_name"],
            "equivalent_specimen_modulus_gpa_main_regime": float(equivalent_modulus_gpa),
            "max_engineering_stress_mpa": float(engineering_stress_mpa.max()),
            "max_reaction_force_n": float(reaction_force_n.max()),
            "settling_displacement_mm": float(settling_displacement_mm),
            "settling_stiffness_scale": float(settling_stiffness_scale),
            "max_displacement_residual_mm": max_residual_mm,
        }

    except Exception as error:
        return {"error": f"Error running continuum FEA with settling: {error}"}


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
