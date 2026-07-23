from fastmcp import FastMCP
from skeletonization import skeletonize_mask
from typing import Optional
import tifffile

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

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
