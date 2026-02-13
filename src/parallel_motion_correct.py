#!/usr/bin/env python3
"""
Parallel Batch Motion Correction for Multi-TIFF and Raw Microscopy Images

This module provides parallel processing capabilities for motion correction
of multiple TIFF or RAW files simultaneously using joblib.

It implements a robust workflow for large files:
1. Load data (supports .raw and .tif)
2. Temporal binning (to reduce noise and size)
3. Motion correction on binned data
4. Interpolation of transforms to full frame rate
5. precise application of transforms to full resolution data

Usage:
    from parallel_motion_correct import parallel_batch_process
    
    files = ['recording1.raw', 'recording2.tif']
    results = parallel_batch_process(files, output_dir='./corrected/', n_jobs=4)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Union
import traceback
import mmap

import numpy as np
import tifffile
import zarr
from pystackreg import StackReg
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping of transformation type names to StackReg constants
TRANSFORM_TYPES = {
    'translation': StackReg.TRANSLATION,
    'rigid': StackReg.RIGID_BODY,
    'scaled': StackReg.SCALED_ROTATION,
    'affine': StackReg.AFFINE,
    'bilinear': StackReg.BILINEAR,
}

TransformType = Literal['translation', 'rigid', 'scaled', 'affine', 'bilinear']


def get_available_cores() -> int:
    """Return the number of available CPU cores."""
    return cpu_count()


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None
    n_frames: int = 0
    processing_time: float = 0.0


def loadRawFile(filepath, nFrames, nPixelsX, nPixelsY, datatype, load_count=None):
    """
    Load a raw binary file as a numpy array.
    
    Args:
        filepath: Path to the raw file
        nFrames: Number of frames (total in file, used for validaton/metadata if needed)
        nPixelsX: Width
        nPixelsY: Height
        datatype: numpy dtype (e.g. np.uint16)
        load_count: Optional number of frames to actually load
        
    Returns:
        numpy array with shape (frames, height, width)
    """
    # Calculate expected size
    dtype = np.dtype(datatype)
    expected_bytes = nFrames * nPixelsX * nPixelsY * dtype.itemsize
    file_size = os.path.getsize(filepath)
    
    # Calculate count of elements to read
    if load_count is not None:
        count = load_count * nPixelsX * nPixelsY
    else:
        count = -1
        
    with open(filepath, 'rb') as f:
        # Memory map for efficiency with large files
        data = np.fromfile(f, dtype=datatype, count=count)
        
    # Reshape
    # Raw files usually [frames, height, width] or [height, width, frames]
    # Adjust based on known format. Assuming [frames, height, width] or analogous
    
    actual_frames = data.size // (nPixelsX * nPixelsY)
    
    if load_count is not None and actual_frames != load_count and actual_frames != nFrames:
         # Only warn if we didn't get what we asked for, and it wasn't just EOF
         pass

    return data[:actual_frames * nPixelsX * nPixelsY].reshape((actual_frames, nPixelsY, nPixelsX))


def get_raw_file_info(
    filepath: Path,
    dtype: np.dtype = np.uint16
) -> tuple[int, int, int]:
    """
    Infer frame count and dimensions from raw file.
    
    First tries to read dimensions from preview images (ChanC_Preview.tif or ChanA_Preview.tif)
    in the same directory. Falls back to guessing square dimensions if no preview found.
    
    Args:
        filepath: Path to raw file
        dtype: Data type (default uint16)
        
    Returns:
        Tuple of (n_frames, height, width)
        
    Raises:
        ValueError: If dimensions cannot be inferred
    """
    filepath = Path(filepath)
    folder = filepath.parent
    file_size = os.path.getsize(filepath)
    bytes_per_pixel = np.dtype(dtype).itemsize
    
    # Try to get dimensions from preview image (like movieTools.py does)
    preview_files = ['ChanC_Preview.tif', 'ChanA_Preview.tif']
    
    for preview_name in preview_files:
        preview_path = folder / preview_name
        if preview_path.exists():
            try:
                preview = tifffile.imread(preview_path)
                height = preview.shape[0]
                width = preview.shape[1]
                
                bytes_per_frame = width * height * bytes_per_pixel
                n_frames = file_size // bytes_per_frame
                
                remainder = file_size % bytes_per_frame
                if remainder > 0:
                    logger.warning(
                        f"File {filepath.name}: {remainder} bytes don't fit into complete frames. "
                        f"Using {n_frames} complete frames."
                    )
                
                logger.info(f"Got dimensions {width}x{height} from {preview_name}, {n_frames} frames")
                return (n_frames, height, width)
                
            except Exception as e:
                logger.warning(f"Failed to read preview {preview_name}: {e}")
                continue
    
    # Fallback: try common square dimensions
    logger.warning(f"No preview image found for {filepath.name}, guessing square dimensions")
    
    for dim in [512, 256, 1024, 128]:
        bytes_per_frame = dim * dim * bytes_per_pixel
        n_frames = file_size // bytes_per_frame
        remainder = file_size % bytes_per_frame
        
        if n_frames >= 1:
            if remainder > 0:
                logger.warning(
                    f"File {filepath.name}: {remainder} bytes ({remainder / bytes_per_frame:.2%} of a frame) "
                    f"don't fit into complete {dim}x{dim} frames. Using {n_frames} complete frames."
                )
            return (n_frames, dim, dim)
    
    raise ValueError(
        f"Could not infer dimensions for raw file: {filepath}. "
        f"Size: {file_size} bytes. No preview image found and file too small for common dimensions."
    )


def get_tiff_file_info(filepath: Path) -> tuple[int, int, int]:
    """
    Get frame count and dimensions from a TIFF file.
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        Tuple of (n_frames, height, width)
    """
    filepath = Path(filepath)
    
    with tifffile.TiffFile(filepath) as tif:
        # Check if it's a 3D stack
        if len(tif.series) > 0:
            series = tif.series[0]
            shape = series.shape
            
            if len(shape) >= 3:
                # 3D stack: (frames, height, width)
                return (shape[0], shape[1], shape[2])
            elif len(shape) == 2:
                # Single 2D image
                return (1, shape[0], shape[1])
        
        # Multi-page TIFF
        n_pages = len(tif.pages)
        if n_pages > 0:
            page = tif.pages[0]
            return (n_pages, page.shape[0], page.shape[1])
    
    raise ValueError(f"Could not determine dimensions for TIFF file: {filepath}")


def check_already_processed(
    input_path: Path,
    output_dir: Union[str, Path],
    temporal_bin: int = 1,
    apply_to_full: bool = True,
    save_binned: bool = False,
    suffix: str = '_corrected',
    n_frames: Optional[int] = None,
    output_name: Optional[str] = None,
    save_transforms: bool = True
) -> tuple[bool, str]:
    """
    Check if a file has already been processed with complete output.
    
    Verifies that output files exist AND match expected file sizes based on
    input dimensions. This prevents skipping files that were only partially
    processed (e.g., test runs with subset of frames).
    
    Args:
        input_path: Path to input file (raw or TIFF)
        output_dir: Directory where output files are saved
        temporal_bin: Temporal binning factor used
        apply_to_full: Whether full resolution output was requested
        save_binned: Whether binned output was also saved
        suffix: Output filename suffix (ignored if output_name is set)
        n_frames: If specified, expected number of frames (None = all frames in file)
        output_name: Custom base name for output files. If None, uses input stem + suffix.
        save_transforms: Whether transform files were saved (only checks if True)
        
    Returns:
        Tuple of (is_complete, reason_string)
        - is_complete: True if all expected outputs exist with correct sizes
        - reason_string: Description of what was checked/missing
    """
    input_path = Path(input_path)
    # If output_dir is None, check in the same directory as the input file
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Use custom output_name if provided, otherwise use input stem + suffix
    base_name = output_name if output_name else f"{input_path.stem}{suffix}"
    
    # Check if output directory exists
    if not output_dir.exists():
        return (False, f"Output directory not found: {output_dir}")
    
    try:
        # Get input file info based on file type
        if input_path.suffix.lower() == '.raw':
            total_frames, height, width = get_raw_file_info(input_path)
        elif input_path.suffix.lower() in ('.tif', '.tiff'):
            total_frames, height, width = get_tiff_file_info(input_path)
        else:
            return (False, f"Unsupported file format: {input_path.suffix}")
        
        # Apply n_frames limit if specified
        if n_frames is not None:
            total_frames = min(n_frames, total_frames)
        
        bytes_per_frame_uint16 = width * height * 2
        
        # Determine which output files should exist and their expected sizes
        expected_files = {}
        
        if apply_to_full or temporal_bin <= 1:
            # Full resolution output
            output_path = output_dir / f"{base_name}.tif"
            # TIFF has some overhead, but data should be approximately:
            expected_min_size = total_frames * bytes_per_frame_uint16
            expected_files[output_path] = expected_min_size
        else:
            # Binned output only
            n_bins = total_frames // temporal_bin
            output_path = output_dir / f"{base_name}_binned{temporal_bin}x.tif"
            expected_min_size = n_bins * bytes_per_frame_uint16
            expected_files[output_path] = expected_min_size
        
        # If save_binned and apply_to_full, also check binned output
        if save_binned and apply_to_full and temporal_bin > 1:
            n_bins = total_frames // temporal_bin
            binned_path = output_dir / f"{base_name}_binned{temporal_bin}x.tif"
            expected_min_size = n_bins * bytes_per_frame_uint16
            expected_files[binned_path] = expected_min_size
        
        # Also check for transforms file (only if save_transforms was enabled)
        if save_transforms:
            if temporal_bin > 1:
                transform_path = output_dir / f"{base_name}_transforms_bin{temporal_bin}x.npy"
            else:
                transform_path = output_dir / f"{base_name}_transforms.npy"
        else:
            transform_path = None
        
        # Check all expected files
        missing = []
        too_small = []
        
        for filepath, expected_min in expected_files.items():
            if not filepath.exists():
                missing.append(filepath.name)
            else:
                actual_size = os.path.getsize(filepath)
                # Allow 20% tolerance for TIFF overhead/compression differences
                # But require at least 80% of expected size to catch partial processing
                if actual_size < expected_min * 0.8:
                    too_small.append(
                        f"{filepath.name} ({actual_size / 1e6:.1f}MB < expected {expected_min * 0.8 / 1e6:.1f}MB)"
                    )
        
        if transform_path is not None and not transform_path.exists():
            missing.append(transform_path.name)
        
        if missing:
            return (False, f"Missing: {', '.join(missing)}")
        
        if too_small:
            return (False, f"Incomplete: {', '.join(too_small)}")
        
        return (True, f"Complete ({len(expected_files)} output files verified)")
        
    except Exception as e:
        return (False, f"Error checking: {str(e)}")


def load_raw_chunk(
    filepath: Path,
    start_frame: int,
    end_frame: int,
    width: int = 512,
    height: int = 512,
    dtype: np.dtype = np.uint16
) -> np.ndarray:
    """
    Load a specific range of frames from a raw file.
    
    Uses file seeking to avoid loading the entire file into memory.
    
    Args:
        filepath: Path to raw file
        start_frame: First frame to load (0-indexed)
        end_frame: Last frame to load (exclusive)
        width: Frame width in pixels
        height: Frame height in pixels
        dtype: Data type (default uint16)
        
    Returns:
        numpy array with shape (n_frames, height, width)
    """
    filepath = Path(filepath)
    bytes_per_pixel = np.dtype(dtype).itemsize
    bytes_per_frame = width * height * bytes_per_pixel
    
    offset = start_frame * bytes_per_frame
    n_frames = end_frame - start_frame
    count = n_frames * width * height
    
    with open(filepath, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=count)
    
    return data.reshape((n_frames, height, width))


def load_binned_frames(
    filepath: Path,
    bin_size: int,
    width: int = 512,
    height: int = 512,
    dtype: np.dtype = np.uint16,
    total_frames: Optional[int] = None,
    progress_bar: bool = False
) -> np.ndarray:
    """
    Load and temporally bin frames by averaging every bin_size frames.
    
    Memory-efficient: loads only bin_size frames at a time, averages them,
    then frees memory before loading the next group.
    
    Args:
        filepath: Path to raw file
        bin_size: Number of frames to average together
        width: Frame width in pixels
        height: Frame height in pixels
        dtype: Data type of raw file (default uint16)
        total_frames: Total frames in file (inferred if None)
        progress_bar: Show progress bar
        
    Returns:
        numpy array with shape (n_bins, height, width) as uint16
    """
    filepath = Path(filepath)
    
    # Infer total frames if not provided
    if total_frames is None:
        total_frames, height, width = get_raw_file_info(filepath, dtype)
    
    n_bins = total_frames // bin_size
    
    if n_bins == 0:
        # File smaller than one bin - just average all frames
        all_frames = load_raw_chunk(filepath, 0, total_frames, width, height, dtype)
        return np.mean(all_frames, axis=0, keepdims=True).astype(np.uint16)
    
    # Pre-allocate output array
    binned_stack = np.zeros((n_bins, height, width), dtype=np.float32)
    
    iterator = range(n_bins)
    if progress_bar:
        iterator = tqdm(iterator, desc="Loading binned frames", unit="bin")
    
    for i in iterator:
        start = i * bin_size
        end = start + bin_size
        
        # Load only bin_size frames
        chunk = load_raw_chunk(filepath, start, end, width, height, dtype)
        
        # Average the frames
        binned_stack[i] = np.mean(chunk, axis=0)
        
        # Explicit memory release
        del chunk
    
    return binned_stack.astype(np.uint16)


def load_tiff_chunk(
    filepath: Path,
    start_frame: int,
    end_frame: int
) -> np.ndarray:
    """
    Load a specific range of frames from a TIFF file efficiently.
    
    Uses zarr for efficient partial reading without loading the entire file.
    Includes robust fallback to memmap if zarr fails or returns zeros (common on Windows).
    
    Args:
        filepath: Path to TIFF file
        start_frame: First frame to load (0-indexed)
        end_frame: Last frame to load (exclusive)
        
    Returns:
        numpy array with shape (n_frames, height, width)
    """
    filepath = Path(filepath)
    
    try:
        with tifffile.TiffFile(filepath) as tif:
            # Check if it's a 3D stack
            if len(tif.series) > 0:
                series = tif.series[0]
                if len(series.shape) >= 3:
                    try:
                        # Primary method: use zarr
                        store = tif.aszarr()
                        z = zarr.open(store, mode='r')
                        data = np.array(z[start_frame:end_frame])
                        
                        # Check for suspicious zeros (if expected data shouldn't be empty)
                        if data.size > 0 and np.max(data) == 0:
                            # Only warn if it's suspicious (e.g. valid specific types)
                            # For now, treat all-zeros as potential failure of zarr mapping on Windows
                            logger.warning(f"Zarr loader returned all zeros for {filepath.name} [{start_frame}:{end_frame}]. Trying fallback.")
                            raise ValueError("Zarr returned all zeros")
                            
                        return data
                    except Exception as e:
                        logger.warning(f"Zarr loading failed/suspicious for {filepath.name}: {e}. Falling back to memmap.")
                        # Fallback: use memmap
                        # Note: memmap might keep file open, but it's more robust on Windows than zarr in some envs
                        return tifffile.memmap(filepath)[start_frame:end_frame]
            
            # Fallback for multi-page TIFF or if series check failed
            return tifffile.imread(filepath, key=range(start_frame, end_frame))
            
    except Exception as e:
        logger.error(f"Failed to load chunk from {filepath} [{start_frame}:{end_frame}]: {e}")
        # Last resort fallback: try reading full file memmapped and slicing
        try:
             return tifffile.memmap(filepath)[start_frame:end_frame]
        except:
             raise e


def load_binned_frames_tiff(
    filepath: Path,
    bin_size: int,
    total_frames: Optional[int] = None,
    progress_bar: bool = False
) -> np.ndarray:
    """
    Load and temporally bin frames from a TIFF file by averaging every bin_size frames.
    
    Memory-efficient: loads only bin_size frames at a time, averages them,
    then frees memory before loading the next group.
    
    Args:
        filepath: Path to TIFF file
        bin_size: Number of frames to average together
        total_frames: Total frames in file (inferred if None)
        progress_bar: Show progress bar
        
    Returns:
        numpy array with shape (n_bins, height, width) as uint16
    """
    filepath = Path(filepath)
    
    # Infer total frames and dimensions if not provided
    if total_frames is None:
        total_frames, height, width = get_tiff_file_info(filepath)
    else:
        _, height, width = get_tiff_file_info(filepath)
    
    n_bins = total_frames // bin_size
    
    if n_bins == 0:
        # File smaller than one bin - just average all frames
        all_frames = load_tiff_chunk(filepath, 0, total_frames)
        return np.mean(all_frames, axis=0, keepdims=True).astype(np.uint16)
    
    # Pre-allocate output array
    binned_stack = np.zeros((n_bins, height, width), dtype=np.float32)
    
    iterator = range(n_bins)
    if progress_bar:
        iterator = tqdm(iterator, desc="Loading binned frames (TIFF)", unit="bin")
    
    for i in iterator:
        start = i * bin_size
        end = start + bin_size
        
        # Load only bin_size frames
        chunk = load_tiff_chunk(filepath, start, end)
        
        # Average the frames
        binned_stack[i] = np.mean(chunk, axis=0)
        
        # Explicit memory release
        del chunk
    
    return binned_stack.astype(np.uint16)


def load_data(
    filepath: Path,
    n_frames: Optional[int] = None
) -> np.ndarray:
    """
    Load data from TIFF or RAW file.
    
    Args:
        filepath: Path to file
        n_frames: Number of frames to load (for TIFF). For RAW, tries to infer or requires parse.
        
    Returns:
        numpy array (frames, height, width)
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() in ('.tif', '.tiff'):
        return load_tiff_partial(filepath, n_frames=n_frames)
        
    elif filepath.suffix.lower() == '.raw':
        # Parse filename for dimensions if possible: e.g. Image_001_001.raw
        # This is tricky without metadata. 
        # Strategy: Look for XML or assume standard size? 
        # User prompt implies: "Process all frames in the recording".
        # We need dimensions. 
        # HEURISTIC: Try to find metadata file or assume 512x512 or try to guess.
        # BETTER HEURISTIC for this specific task: 
        # The user provided code often uses hardcoded dims or reads metadata.
        # Let's assume standard 512x512 like the notebook shows, or calculate.
        # A common format in this lab seems to contain dimensions in a companion file 
        # OR we just try to reshape to Square frames.
        
        # Let's try to infer from file size assuming square frames
        file_size = os.path.getsize(filepath)
        dtype = np.uint16 # Most common for microscopy
        
        # Try 512x512
        bytes_per_frame_512 = 512 * 512 * 2
        if file_size % bytes_per_frame_512 == 0:
            frames = file_size // bytes_per_frame_512
            return loadRawFile(filepath, frames, 512, 512, np.uint16, load_count=n_frames)
            
        # Try 256x256
        bytes_per_frame_256 = 256 * 256 * 2
        if file_size % bytes_per_frame_256 == 0:
            frames = file_size // bytes_per_frame_256
            return loadRawFile(filepath, frames, 256, 256, np.uint16, load_count=n_frames)
            
        raise ValueError(f"Could not infer dimensions for raw file: {filepath}. Size: {file_size}")
        
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_tiff_partial(
    filepath: Path,
    n_frames: Optional[int] = None,
    start_frame: int = 0
) -> np.ndarray:
    """Load a subset of frames from a TIFF file efficiently."""
    with tifffile.TiffFile(filepath) as tif:
        # Check if it's a 3D stack stored as single page
        if len(tif.series) > 0:
            series = tif.series[0]
            if len(series.shape) >= 3:
                # 3D stack - use zarr for efficient partial reading
                store = tif.aszarr()
                z = zarr.open(store, mode='r')
                total_frames = z.shape[0]
                
                if n_frames is None:
                    end_frame = total_frames
                else:
                    end_frame = min(start_frame + n_frames, total_frames)
                
                return np.array(z[start_frame:end_frame])
        
        # Multi-page TIFF
        if n_frames is None:
            return tifffile.imread(filepath)
        else:
            end_frame = start_frame + n_frames
            return tifffile.imread(filepath, key=range(start_frame, end_frame))


def temporal_bin_stack(
    stack: np.ndarray, 
    bin_size: int
) -> np.ndarray:
    """Apply temporal binning to a stack."""
    if bin_size <= 1:
        return stack
    
    n_frames = stack.shape[0]
    n_bins = n_frames // bin_size
    
    if n_bins == 0:
        return np.mean(stack, axis=0, keepdims=True).astype(stack.dtype)
    
    # Reshape and average
    # Limit to full bins
    binned = stack[:n_bins * bin_size].reshape(n_bins, bin_size, *stack.shape[1:])
    binned = np.mean(binned, axis=1).astype(stack.dtype)
    
    return binned


def compute_intensity_offset(
    original: np.ndarray,
    corrected: np.ndarray,
    n_sample_frames: int = 100,
    margin_fraction: float = 0.2
) -> float:
    """
    Compute global intensity offset between original and corrected stacks.
    
    Bicubic interpolation during motion correction can cause intensity shifts.
    This function computes the mean difference in a central ROI (avoiding borders
    where zeros are introduced) to determine a global correction offset.
    
    Args:
        original: Original stack (n_frames, height, width)
        corrected: Motion-corrected stack (n_frames, height, width)
        n_sample_frames: Number of frames to sample for computing offset
        margin_fraction: Fraction of image to exclude from each edge (0.2 = use central 60%)
        
    Returns:
        Global offset value. Subtract this from corrected stack to match original intensity.
    """
    n_frames = min(len(original), len(corrected), n_sample_frames)
    height, width = original.shape[1:3]
    
    # Define central ROI (avoiding borders where zeros are introduced)
    y_margin = int(height * margin_fraction)
    x_margin = int(width * margin_fraction)
    
    y_start, y_end = y_margin, height - y_margin
    x_start, x_end = x_margin, width - x_margin
    
    # Compute mean difference in central ROI using first n_sample_frames
    original_roi = original[:n_frames, y_start:y_end, x_start:x_end].astype(np.float64)
    corrected_roi = corrected[:n_frames, y_start:y_end, x_start:x_end].astype(np.float64)
    
    diff = corrected_roi - original_roi
    offset = diff.mean()
    
    return offset


def interpolate_transforms(
    transforms: np.ndarray,
    target_frames: int,
    kind: str = 'linear'
) -> np.ndarray:
    """
    Interpolate transformation matrices to a new number of frames.
    
    Args:
        transforms: Source transformation matrices (N, 3, 3)
        target_frames: Number of frames to interpolate to
        kind: Interpolation kind (linear, cubic, etc.)
        
    Returns:
        Interpolated transformation matrices (target_frames, 3, 3)
    """
    source_frames = len(transforms)
    
    # Create time points for source
    # We assume the transforms correspond to the CENTER of the bins
    # Source: 0, 1, 2...
    # Target range: 0 to source_frames-1
    x_source = np.linspace(0, source_frames - 1, source_frames)
    
    # Target time points map to the same range
    x_target = np.linspace(0, source_frames - 1, target_frames)
    
    # Reshape transforms to (N, 9) for easier interpolation
    transforms_flat = transforms.reshape(source_frames, -1)
    
    # Interpolate each element of the matrix
    f = interp1d(x_source, transforms_flat, kind=kind, axis=0, fill_value="extrapolate")
    transforms_interp_flat = f(x_target)
    
    # Reshape back to (target_frames, 3, 3)
    transforms_interp = transforms_interp_flat.reshape(target_frames, 3, 3)
    
    return transforms_interp


def apply_transforms(
    stack: np.ndarray,
    transforms: np.ndarray,
    progress_bar: bool = False
) -> np.ndarray:
    """Apply transformation matrices to a stack."""
    n_frames = stack.shape[0]
    
    if len(transforms) != n_frames:
        # Try to resize transforms if valid
        logger.warning(f"Transform count ({len(transforms)}) != frames ({n_frames}). Interpolating.")
        transforms = interpolate_transforms(transforms, n_frames)
    
    sr = StackReg(StackReg.RIGID_BODY)
    
    corrected_stack = np.zeros_like(stack, dtype=np.float64)
    original_dtype = stack.dtype
    
    iterator = range(n_frames)
    if progress_bar:
        iterator = tqdm(iterator, desc="Applying transforms", unit="frame")
        
    for i in iterator:
        frame = stack[i].astype(np.float64)
        corrected_stack[i] = sr.transform(frame, transforms[i])
        
    # Convert back to original dtype
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    else:
        max_val = np.finfo(original_dtype).max
        
    corrected_stack = np.clip(corrected_stack, 0, max_val).astype(original_dtype)
    return corrected_stack


def apply_transforms_to_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    transforms: np.ndarray,
    chunk_size: int = 500,
    correct_intensity: bool = False,
    intensity_offset: float = 0.0,
    progress_bar: bool = False,
    n_frames: Optional[int] = None,
    transform_type: TransformType = 'rigid'
) -> Path:
    """
    Apply pre-computed transformation matrices to a TIFF or RAW file.
    
    This function streams data in chunks and writes directly to a TIFF file,
    enabling memory-efficient processing of arbitrarily large files.
    
    Args:
        input_path: Path to input TIFF or RAW file
        output_path: Path for output TIFF file
        transforms: Pre-computed transformation matrices (n_frames, 3, 3).
                    If length differs from input frames, will be interpolated.
        chunk_size: Number of frames to process at a time (default 500)
        correct_intensity: If True, apply intensity offset correction
        intensity_offset: Offset value to subtract (only used if correct_intensity=True)
        progress_bar: Show progress bar during processing
        n_frames: Number of frames to process (None = all frames)
    
    Returns:
        Path to the output file
    
    Example:
        # After computing transforms on file A, apply them to file B:
        >>> transforms = np.load('transforms.npy')
        >>> apply_transforms_to_file(
        ...     'channel_B.tif',
        ...     'channel_B_corrected.tif',
        ...     transforms
        ... )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file type
    is_raw = input_path.suffix.lower() == '.raw'
    is_tiff = input_path.suffix.lower() in ('.tif', '.tiff')
    
    if not (is_raw or is_tiff):
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Get file dimensions
    if is_raw:
        total_frames, height, width = get_raw_file_info(input_path)
    else:
        total_frames, height, width = get_tiff_file_info(input_path)
    
    # Limit frames if requested
    if n_frames is not None:
        total_frames = min(n_frames, total_frames)
    
    # Interpolate transforms if needed
    if len(transforms) != total_frames:
        logger.info(f"Interpolating transforms from {len(transforms)} to {total_frames} frames")
        transforms = interpolate_transforms(transforms, total_frames)
    
    # Estimate if we need BigTIFF
    output_size_estimate = total_frames * height * width * 2  # uint16
    bigtiff = output_size_estimate > 2 * 1024**3
    
    # Create StackReg instance for transforms
    # Create StackReg instance for transforms
    sr = StackReg(TRANSFORM_TYPES[transform_type])
    
    logger.info(f"Applying transforms to {input_path.name}: {total_frames} frames @ {height}x{width}")
    
    # Stream processing with TiffWriter
    with tifffile.TiffWriter(output_path, bigtiff=bigtiff) as tif:
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        iterator = range(0, total_frames, chunk_size)
        if progress_bar:
            iterator = tqdm(iterator, total=n_chunks, desc="Applying transforms", unit="chunk")
        
        for start in iterator:
            end = min(start + chunk_size, total_frames)
            
            # Load chunk from file (RAW or TIFF)
            if is_raw:
                chunk = load_raw_chunk(input_path, start, end, width, height)
            else:  # TIFF
                chunk = load_tiff_chunk(input_path, start, end)
            
            # Apply transforms to this chunk
            chunk_transforms = transforms[start:end]
            corrected_chunk = np.zeros_like(chunk, dtype=np.float64)
            
            for i, (frame, tmat) in enumerate(zip(chunk, chunk_transforms)):
                corrected_chunk[i] = sr.transform(frame.astype(np.float64), tmat)

            # Debug check for zeros
            if np.max(corrected_chunk) == 0:
                 logger.warning(f"Chunk starting at {start} became all zeros after transform!")
            
            # Apply intensity correction if needed
            if correct_intensity and abs(intensity_offset) >= 1.0:
                corrected_chunk = corrected_chunk - intensity_offset
            
            # Convert back to uint16
            corrected_chunk = np.clip(corrected_chunk, 0, 65535).astype(np.uint16)
            
            # Write chunk to TIFF (appends automatically)
            tif.write(corrected_chunk, contiguous=True)
            
            # Explicit memory release
            del chunk, corrected_chunk
    
    logger.info(f"Saved corrected file: {output_path}")
    return output_path


def motion_correct_stack(
    stack: np.ndarray,
    transform_type: TransformType = 'rigid',
    reference_frames: Optional[int] = None,
    progress_bar: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Apply motion correction to an image stack using StackReg."""
    n_frames = stack.shape[0]
    sr = StackReg(TRANSFORM_TYPES[transform_type])
    
    # Compute reference image centered around brightest frame
    if reference_frames is None or reference_frames <= 1:
        # Single reference frame: use the brightest frame
        frame_means = np.mean(stack, axis=(1, 2))
        brightest_idx = int(np.argmax(frame_means))
        reference = stack[brightest_idx].astype(np.float64)
        logger.info(f"Using brightest frame as reference: frame {brightest_idx} (mean intensity: {frame_means[brightest_idx]:.1f})")
    else:
        # Multiple reference frames: center around brightest frame
        n_ref = min(reference_frames, n_frames)
        frame_means = np.mean(stack, axis=(1, 2))
        brightest_idx = int(np.argmax(frame_means))
        
        # Compute indices centered around brightest frame
        half = n_ref // 2
        start_idx = brightest_idx - half
        end_idx = start_idx + n_ref
        
        # Clamp to valid range
        if start_idx < 0:
            start_idx = 0
            end_idx = n_ref
        elif end_idx > n_frames:
            end_idx = n_frames
            start_idx = n_frames - n_ref
        
        reference = np.mean(stack[start_idx:end_idx], axis=0).astype(np.float64)
        logger.info(f"Brightest frame: {brightest_idx} (mean intensity: {frame_means[brightest_idx]:.1f}). Using reference frames {start_idx}-{end_idx-1}")
    
    transformation_matrices = []
    original_dtype = stack.dtype
    corrected_stack = np.zeros_like(stack, dtype=np.float64)
    
    iterator = range(n_frames)
    if progress_bar:
        iterator = tqdm(iterator, desc="Motion correction", unit="frame")
    
    for i in iterator:
        frame = stack[i].astype(np.float64)
        tmat = sr.register(reference, frame)
        transformation_matrices.append(tmat)
        corrected_stack[i] = sr.transform(frame, tmat)
    
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    else:
        max_val = np.finfo(original_dtype).max
    corrected_stack = np.clip(corrected_stack, 0, max_val).astype(original_dtype)
    
    return corrected_stack, np.array(transformation_matrices)


def process_single_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    transform_type: TransformType = 'rigid',
    reference_frames: Optional[int] = None,
    temporal_bin: int = 1,
    n_frames: Optional[int] = None,
    save_transforms: bool = True,
    suffix: str = '_corrected',
    progress_bar: bool = False,
    apply_to_full: bool = True,
    save_binned: bool = False,
    output_name: Optional[str] = None,
    correct_intensity: bool = False
) -> ProcessingResult:
    """
    Process a single file with binning-interpolation workflow.
    
    Args:
        input_path: Path to input file
        output_dir: Output directory (may be overridden to per-recording 'corrected' folder)
        transform_type: Type of transformation for registration
        reference_frames: Number of frames to average for reference image
        temporal_bin: Temporal binning factor for registration
        n_frames: Number of frames to load (None = all)
        save_transforms: Whether to save transformation matrices
        suffix: Suffix for output filename (ignored if output_name is set)
        progress_bar: Show progress bars
        apply_to_full: If True, interpolate transforms and apply to full resolution data.
                       If False, save the corrected binned stack directly (faster, smaller).
        save_binned: If True (and apply_to_full is True), also save the corrected binned stack.
        output_name: Custom base name for output files (without extension). If None, uses
                     input filename stem + suffix. Example: 'my_output' -> 'my_output.tif'
        correct_intensity: If True, compute and apply a global intensity offset correction
                           to compensate for bicubic interpolation artifacts.
    
    Returns:
        ProcessingResult with status and output path
    """
    import time
    start_time = time.time()
    
    input_path = Path(input_path)
    # If output_dir is None, save in the same directory as the input file
    if output_dir is None:
        local_output_dir = input_path.parent
    else:
        local_output_dir = Path(output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        stack = load_data(input_path, n_frames=n_frames)
        original_frames = stack.shape[0]
        
        # 2. Binning
        if temporal_bin > 1:
            stack_binned = temporal_bin_stack(stack, temporal_bin)
        else:
            stack_binned = stack
            
        # 3. Register (Motion Correction on binned)
        corrected_binned, transforms_binned = motion_correct_stack(
            stack_binned,
            transform_type=transform_type,
            reference_frames=reference_frames,
            progress_bar=progress_bar
        )
        
        # 4. Decide output based on apply_to_full flag
        if apply_to_full and temporal_bin > 1:
            # Interpolate transforms and apply to full resolution
            transforms_full = interpolate_transforms(transforms_binned, original_frames)
            corrected_output = apply_transforms(stack, transforms_full, progress_bar=progress_bar)
            output_frames = original_frames
            transforms_to_save = transforms_full
        else:
            # Use corrected binned stack directly
            corrected_output = corrected_binned
            output_frames = corrected_binned.shape[0]
            transforms_to_save = transforms_binned
        
        # 4b. Apply intensity correction if requested
        intensity_offset = 0.0
        if correct_intensity:
            if apply_to_full and temporal_bin > 1:
                intensity_offset = compute_intensity_offset(stack, corrected_output)
            else:
                intensity_offset = compute_intensity_offset(stack_binned, corrected_output)
            
            if abs(intensity_offset) >= 1.0:
                logger.info(f"Applying intensity correction: offset = {intensity_offset:.2f}")
                corrected_output = corrected_output.astype(np.float64) - intensity_offset
                corrected_output = np.clip(corrected_output, 0, 65535).astype(np.uint16)
        
        # 5. Save corrected stack
        # Use custom output_name if provided, otherwise use input stem + suffix
        base_name = output_name if output_name else f"{input_path.stem}{suffix}"
        # Add _binned suffix if not applying to full
        if apply_to_full or temporal_bin <= 1:
            output_path = local_output_dir / f"{base_name}.tif"
        else:
            output_path = local_output_dir / f"{base_name}_binned{temporal_bin}x.tif"
        
        bigtiff = corrected_output.nbytes > 2 * 1024**3
        tifffile.imwrite(
            output_path,
            corrected_output,
            bigtiff=bigtiff,
            metadata={
                'motion_correction': {
                    'method': 'pystackreg',
                    'transform_type': transform_type,
                    'reference_frames': reference_frames,
                    'temporal_bin': temporal_bin,
                    'apply_to_full': apply_to_full,
                    'source_file': str(input_path),
                    'original_shape': stack.shape,
                    'output_frames': output_frames
                }
            }
        )
        
        # 5b. Optionally save binned stack as well when applying to full
        binned_output_path = None
        if save_binned and apply_to_full and temporal_bin > 1:
            binned_output_path = local_output_dir / f"{base_name}_binned{temporal_bin}x.tif"
            bigtiff_binned = corrected_binned.nbytes > 2 * 1024**3
            tifffile.imwrite(
                binned_output_path,
                corrected_binned,
                bigtiff=bigtiff_binned,
                metadata={
                    'motion_correction': {
                        'method': 'pystackreg',
                        'transform_type': transform_type,
                        'reference_frames': reference_frames,
                        'temporal_bin': temporal_bin,
                        'is_binned': True,
                        'source_file': str(input_path),
                        'original_shape': stack.shape,
                        'binned_frames': corrected_binned.shape[0]
                    }
                }
            )
        
        # 6. Save transforms (always save original binned transforms with bin factor in name)
        if save_transforms:
            if temporal_bin > 1:
                transform_path = local_output_dir / f"{base_name}_transforms_bin{temporal_bin}x.npy"
            else:
                transform_path = local_output_dir / f"{base_name}_transforms.npy"
            np.save(transform_path, transforms_binned)
        
        # 7. Save processing parameters to text file
        params_path = local_output_dir / f"{base_name}_parameters.txt"
        from datetime import datetime
        with open(params_path, 'w') as f:
            f.write(f"Motion Correction Parameters\n")
            f.write(f"============================\n")
            f.write(f"Processed: {datetime.now().isoformat()}\n\n")
            f.write(f"Source file: {input_path}\n")
            f.write(f"Output file: {output_path}\n\n")
            f.write(f"--- Processing Settings ---\n")
            f.write(f"Transform type: {transform_type}\n")
            f.write(f"Reference frames: {reference_frames}\n")
            f.write(f"Temporal binning: {temporal_bin}x\n")
            f.write(f"Apply to full resolution: {apply_to_full}\n")
            f.write(f"Intensity correction: {correct_intensity}\n")
            if correct_intensity:
                f.write(f"Intensity offset applied: {intensity_offset:.4f}\n")
            f.write(f"\n")
            f.write(f"--- Data Info ---\n")
            f.write(f"Original frames: {original_frames}\n")
            f.write(f"Binned frames (for registration): {len(transforms_binned)}\n")
            f.write(f"Output frames: {output_frames}\n")
            f.write(f"Original shape: {stack.shape}\n\n")
            f.write(f"--- Transform Matrix ---\n")
            f.write(f"Transform matrix file: {transform_path.name if save_transforms else 'Not saved'}\n")
            f.write(f"Transform matrix shape: {transforms_binned.shape}\n")
            f.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            n_frames=output_frames,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ProcessingResult(
            input_path=input_path,
            output_path=None,
            success=False,
            error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            processing_time=processing_time
        )


def process_single_file_chunked(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    transform_type: TransformType = 'rigid',
    reference_frames: Optional[int] = None,
    temporal_bin: int = 1,
    n_frames: Optional[int] = None,
    save_transforms: bool = True,
    suffix: str = '_corrected',
    progress_bar: bool = False,
    apply_to_full: bool = True,
    save_binned: bool = False,
    chunk_size: int = 500,
    output_name: Optional[str] = None,
    correct_intensity: bool = False,
    additional_files: Optional[list[Union[str, Path]]] = None,
    register_on_companion: bool = False
) -> ProcessingResult:
    """
    Memory-efficient chunked processing of a single file.
    
    Uses two-phase approach:
    1. Load binned frames efficiently (memory: ~1-2 GB)
    2. Apply transforms in chunks to full resolution (memory: ~500 MB per chunk)
    
    This allows processing files of any size with constant memory usage.
    
    Args:
        input_path: Path to input file
        output_dir: Output directory (overridden to per-recording 'corrected' folder)
        transform_type: Type of transformation for registration
        reference_frames: Number of frames to average for reference image
        temporal_bin: Temporal binning factor for registration
        n_frames: Number of frames to process (None = all)
        save_transforms: Whether to save transformation matrices
        suffix: Suffix for output filename (ignored if output_name is set)
        progress_bar: Show progress bars
        apply_to_full: If True, apply transforms to full resolution data
        save_binned: If True (and apply_to_full), also save corrected binned stack
        chunk_size: Number of frames to process at a time in Phase 2
        output_name: Custom base name for output files (without extension). If None, uses
                     input filename stem + suffix. Example: 'my_output' -> 'my_output.tif'
        correct_intensity: If True, compute and apply a global intensity offset correction
                           to compensate for bicubic interpolation artifacts.
        additional_files: Optional list of additional files (e.g., companion channels) to
                          apply the same transforms to. Each file will be processed using
                          the transforms computed from the primary input file.
        register_on_companion: If True and additional_files is provided, compute transforms
                               on the first companion file instead of the input file. The
                               input file then becomes an "additional file" that gets the
                               transforms applied to it.
    
    Returns:
        ProcessingResult with status and output path
    """
    import time
    start_time = time.time()
    
    input_path = Path(input_path)
    # If output_dir is None, save in the same directory as the input file
    if output_dir is None:
        local_output_dir = input_path.parent
    else:
        local_output_dir = Path(output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine file type
        is_raw = input_path.suffix.lower() == '.raw'
        is_tiff = input_path.suffix.lower() in ('.tif', '.tiff')
        
        # Get file dimensions based on type
        if is_raw:
            total_frames, height, width = get_raw_file_info(input_path)
        elif is_tiff:
            total_frames, height, width = get_tiff_file_info(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Limit frames if requested
        if n_frames is not None:
            total_frames = min(n_frames, total_frames)
        
        original_frames = total_frames
        original_shape = (original_frames, height, width)
        
        logger.info(f"Processing {input_path.name}: {original_frames} frames @ {height}x{width}")
        
        # Handle register_on_companion: swap registration source with companion
        registration_path = input_path
        files_to_correct = [input_path]
        if additional_files:
            files_to_correct.extend([Path(f) for f in additional_files])
        
        if register_on_companion and additional_files:
            # Use the first companion file for registration instead
            registration_path = Path(additional_files[0])
            logger.info(f"Computing transforms on companion file: {registration_path.name}")
        
        # ============================================================
        # PHASE 1: Load binned frames and compute transforms
        # Memory: ~1.6 GB for 3000 binned frames at 512x512
        # ============================================================
        
        # Determine registration source file type
        reg_is_raw = registration_path.suffix.lower() == '.raw'
        reg_is_tiff = registration_path.suffix.lower() in ('.tif', '.tiff')
        
        if temporal_bin > 1:
            logger.info(f"Phase 1: Loading binned frames ({temporal_bin}x binning) from {registration_path.name}...")
            if reg_is_raw:
                stack_binned = load_binned_frames(
                    registration_path,
                    bin_size=temporal_bin,
                    width=width,
                    height=height,
                    total_frames=original_frames,
                    progress_bar=progress_bar
                )
            else:  # TIFF
                stack_binned = load_binned_frames_tiff(
                    registration_path,
                    bin_size=temporal_bin,
                    total_frames=original_frames,
                    progress_bar=progress_bar
                )
        else:
            # No binning requested - need to load all for registration
            # This path may still use significant memory for very large files
            logger.warning("No temporal binning - loading full stack for registration")
            if reg_is_raw:
                stack_binned = load_raw_chunk(registration_path, 0, original_frames, width, height)
            else:  # TIFF
                stack_binned = load_tiff_chunk(registration_path, 0, original_frames)
        
        # Motion correct the binned stack
        logger.info(f"Phase 1: Computing transforms on {len(stack_binned)} binned frames...")
        corrected_binned, transforms_binned = motion_correct_stack(
            stack_binned,
            transform_type=transform_type,
            reference_frames=reference_frames,
            progress_bar=progress_bar
        )
        
        # Compute intensity offset if requested (before freeing binned data)
        intensity_offset = 0.0
        if correct_intensity:
            # Compute offset using the binned original and corrected frames
            # This is efficient because we already have these in memory
            intensity_offset = compute_intensity_offset(stack_binned, corrected_binned)
            if abs(intensity_offset) >= 1.0:
                logger.info(f"Will apply intensity correction: offset = {intensity_offset:.2f}")
        
        # Free binned stack if we're applying to full (save memory for Phase 2)
        if apply_to_full and temporal_bin > 1:
            # Keep corrected_binned if we need to save it
            if not save_binned:
                del corrected_binned
            del stack_binned
        
        # Interpolate transforms to full frame count
        if apply_to_full and temporal_bin > 1:
            transforms_full = interpolate_transforms(transforms_binned, original_frames)
        else:
            transforms_full = transforms_binned
        
        # ============================================================
        # PHASE 2: Apply transforms in chunks and stream to output
        # Memory: ~500 MB per chunk (chunk_size frames at 512x512)
        # ============================================================
        
        # Use custom output_name if provided, otherwise use input stem + suffix
        base_name = output_name if output_name else f"{input_path.stem}{suffix}"
        
        if apply_to_full and temporal_bin > 1:
            output_path = local_output_dir / f"{base_name}.tif"
            output_frames = original_frames
            
            logger.info(f"Phase 2: Applying transforms in chunks of {chunk_size} frames...")
            
            # Use the standalone function to apply transforms
            apply_transforms_to_file(
                input_path=input_path,
                output_path=output_path,
                transforms=transforms_full,
                chunk_size=chunk_size,
                correct_intensity=correct_intensity,
                intensity_offset=intensity_offset,
                progress_bar=progress_bar,
                n_frames=n_frames,
                transform_type=transform_type
            )
            
            # Optionally save binned corrected stack
            if save_binned:
                binned_output_path = local_output_dir / f"{base_name}_binned{temporal_bin}x.tif"
                bigtiff_binned = corrected_binned.nbytes > 2 * 1024**3
                tifffile.imwrite(binned_output_path, corrected_binned, bigtiff=bigtiff_binned)
                
        else:
            # Not applying to full - save binned stack directly
            output_path = local_output_dir / f"{base_name}_binned{temporal_bin}x.tif"
            output_frames = corrected_binned.shape[0]
            
            # Apply intensity correction if needed
            if correct_intensity and abs(intensity_offset) >= 1.0:
                corrected_binned = corrected_binned.astype(np.float64) - intensity_offset
                corrected_binned = np.clip(corrected_binned, 0, 65535).astype(np.uint16)
            
            bigtiff = corrected_binned.nbytes > 2 * 1024**3
            tifffile.imwrite(output_path, corrected_binned, bigtiff=bigtiff)
        
        # ============================================================
        # Save transforms and parameters
        # ============================================================
        
        if save_transforms:
            if temporal_bin > 1:
                transform_path = local_output_dir / f"{base_name}_transforms_bin{temporal_bin}x.npy"
            else:
                transform_path = local_output_dir / f"{base_name}_transforms.npy"
            np.save(transform_path, transforms_binned)
        
        # Save processing parameters
        params_path = local_output_dir / f"{base_name}_parameters.txt"
        from datetime import datetime
        with open(params_path, 'w') as f:
            f.write(f"Motion Correction Parameters (Chunked Processing)\n")
            f.write(f"=================================================\n")
            f.write(f"Processed: {datetime.now().isoformat()}\n\n")
            f.write(f"Source file: {input_path}\n")
            f.write(f"Output file: {output_path}\n\n")
            f.write(f"--- Processing Settings ---\n")
            f.write(f"Transform type: {transform_type}\n")
            f.write(f"Reference frames: {reference_frames}\n")
            f.write(f"Temporal binning: {temporal_bin}x\n")
            f.write(f"Apply to full resolution: {apply_to_full}\n")
            f.write(f"Chunk size: {chunk_size} frames\n")
            f.write(f"Intensity correction: {correct_intensity}\n")
            if correct_intensity:
                f.write(f"Intensity offset applied: {intensity_offset:.4f}\n")
            f.write(f"\n")
            f.write(f"--- Data Info ---\n")
            f.write(f"Original frames: {original_frames}\n")
            f.write(f"Binned frames (for registration): {len(transforms_binned)}\n")
            f.write(f"Output frames: {output_frames}\n")
            f.write(f"Original shape: {original_shape}\n\n")
            f.write(f"--- Transform Matrix ---\n")
            f.write(f"Transform matrix file: {transform_path.name if save_transforms else 'Not saved'}\n")
            f.write(f"Transform matrix shape: {transforms_binned.shape}\n")
            f.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
        
        # ============================================================
        # PHASE 3: Apply transforms to additional files (if any)
        # ============================================================
        
        additional_outputs = []
        if additional_files:
            logger.info(f"Phase 3: Applying transforms to {len(additional_files)} additional file(s)...")
            for add_file in additional_files:
                add_file = Path(add_file)
                if not add_file.exists():
                    logger.warning(f"Additional file not found, skipping: {add_file}")
                    continue
                
                # Generate output name for additional file
                add_output_name = f"{add_file.stem}{suffix}"
                add_output_path = local_output_dir / f"{add_output_name}.tif"
                
                try:
                    apply_transforms_to_file(
                        input_path=add_file,
                        output_path=add_output_path,
                        transforms=transforms_full,
                        chunk_size=chunk_size,
                        correct_intensity=correct_intensity,
                        intensity_offset=intensity_offset,
                        progress_bar=progress_bar,
                        n_frames=n_frames,
                        transform_type=transform_type
                    )
                    additional_outputs.append(add_output_path)
                    logger.info(f"Successfully processed additional file: {add_file.name}")
                except Exception as add_err:
                    logger.error(f"Failed to process additional file {add_file}: {add_err}")
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            n_frames=output_frames,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ProcessingResult(
            input_path=input_path,
            output_path=None,
            success=False,
            error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            processing_time=processing_time
        )


def parallel_batch_process(
    file_list: list[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    transform_type: TransformType = 'rigid',
    reference_frames: Optional[int] = None,
    temporal_bin: int = 1,
    n_frames: Optional[int] = None,
    save_transforms: bool = True,
    suffix: str = '_corrected',
    n_jobs: int = -1,
    verbose: int = 10,
    apply_to_full: bool = True,
    save_binned: bool = False,
    use_chunked: bool = True,
    chunk_size: int = 500,
    skip_existing: bool = False,
    output_name: Optional[str] = None,
    correct_intensity: bool = False,
    companion_suffix: Optional[str] = None,
    register_on_companion: bool = False,
    backend: str = 'loky'
) -> list[ProcessingResult]:
    """
    Process multiple files in parallel.
    
    Args:
        file_list: List of file paths to process
        output_dir: Output directory. If None, saves each file in the same directory
                    as its input file.
        transform_type: Type of transformation for registration
        reference_frames: Number of frames for reference image
        temporal_bin: Temporal binning factor
        n_frames: Number of frames to load per file (None = all frames)
        save_transforms: Save transformation matrices
        suffix: Output filename suffix (ignored if output_name is set)
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Joblib verbosity level
        apply_to_full: If True, interpolate and apply transforms to full resolution.
                       If False, save corrected binned stacks directly.
        save_binned: If True (and apply_to_full is True), also save corrected binned stacks.
        use_chunked: If True, use memory-efficient chunked processing for .raw files.
                     This allows processing files larger than available RAM.
        chunk_size: Number of frames per chunk when use_chunked=True (default 500).
                    Lower values use less memory but may be slower.
        skip_existing: If True, skip files that already have complete output files
                       with correct sizes. Incomplete outputs (e.g., from test runs 
                       with n_frames) will still be reprocessed.
        output_name: Custom base name for output files (without extension). If None,
                     uses input filename stem + suffix. Example: 'corrected_movie' ->
                     'corrected_movie.tif'. Note: When processing multiple files,
                     each file will use the same output_name, which may cause overwrites
                     unless files are in different directories.
        correct_intensity: If True, compute and apply a global intensity offset correction
                           to compensate for bicubic interpolation artifacts.
        companion_suffix: Optional suffix to identify companion files that should be
                          corrected using the same transforms. For example, if primary
                          file is '1-jumpCorrected.tif' and companion_suffix='-channel2',
                          the function will look for '1-jumpCorrected-channel2.tif' in
                          the same directory and apply the computed transforms to it.
        register_on_companion: If True (and companion_suffix is set), compute transforms
                               on the companion file instead of the primary file. Both
                                files still get corrected, but registration is based on
                                the companion channel.
        backend: Joblib backend ('loky', 'threading', 'multiprocessing'). 
                 'loky' is default but can fail on Windows in some envs. 
                 Try 'threading' if having issues.
    
    Returns:
        List of ProcessingResult objects
    """
    file_list = [Path(f) for f in file_list]
    # output_dir can be None to save each file in its source directory
    if output_dir is not None:
        output_dir = Path(output_dir)
    
    missing_files = [f for f in file_list if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Files not found: {missing_files}")
    
    # Filter out already processed files if skip_existing is enabled
    skipped_results = []
    files_to_process = []
    
    if skip_existing:
        logger.info("Checking for already processed files...")
        for f in file_list:
            is_complete, reason = check_already_processed(
                input_path=f,
                output_dir=output_dir,
                temporal_bin=temporal_bin,
                apply_to_full=apply_to_full,
                save_binned=save_binned,
                suffix=suffix,
                n_frames=n_frames,  # Check for same frame count as requested
                output_name=output_name,
                save_transforms=save_transforms
            )
            
            # Determine the expected output filename
            base_name = output_name if output_name else f"{f.stem}{suffix}"
            # Determine the actual output directory for this file
            file_output_dir = f.parent if output_dir is None else output_dir
            
            if is_complete:
                logger.info(f"Skipping {f.name}: {reason}")
                # Create a "skipped" result
                skipped_results.append(ProcessingResult(
                    input_path=f,
                    output_path=file_output_dir / f"{base_name}.tif",
                    success=True,
                    error_message=f"Skipped: {reason}",
                    n_frames=0,
                    processing_time=0.0
                ))
            else:
                logger.info(f"Will process {f.name}: {reason}")
                files_to_process.append(f)
        
        logger.info(f"Skipped {len(skipped_results)} already complete files, processing {len(files_to_process)}")
    else:
        files_to_process = file_list
    
    # If nothing to process, return early
    if not files_to_process:
        logger.info("All files already processed, nothing to do!")
        return skipped_results
    
    if n_jobs == -1:
        actual_n_jobs = cpu_count()
    elif n_jobs == -2:
        actual_n_jobs = max(1, cpu_count() - 1)
    elif n_jobs < 0:
        actual_n_jobs = max(1, cpu_count() + n_jobs + 1)
    else:
        actual_n_jobs = min(n_jobs, cpu_count())
    
    # Select processing function based on use_chunked flag
    process_func = process_single_file_chunked if use_chunked else process_single_file
    
    if use_chunked:
        logger.info(f"Using memory-efficient chunked processing (chunk_size={chunk_size})")
    
    logger.info(f"Processing {len(files_to_process)} files with {actual_n_jobs} parallel jobs (backend={backend})")
    
    if use_chunked:
        # Build list of additional files for each primary file if companion_suffix is set
        def get_companion_files(primary_file: Path) -> Optional[list[Path]]:
            if companion_suffix is None:
                return None
            # Construct companion filename: stem + companion_suffix + extension
            # Be careful not to double the suffix if it's already there (though less likely here)
            companion_name = f"{primary_file.stem}{companion_suffix}{primary_file.suffix}"
            companion_path = primary_file.parent / companion_name
            if companion_path.exists():
                return [companion_path]
            return None

        # Prepare arguments for chunked processing
        tasks = [
            delayed(process_single_file_chunked)(
                input_path=f,
                output_dir=output_dir,
                transform_type=transform_type,
                reference_frames=reference_frames,
                temporal_bin=temporal_bin,
                n_frames=n_frames,
                save_transforms=save_transforms,
                suffix=suffix,
                progress_bar=(i == 0),  # Only show progress bar for first file to avoid clutter
                apply_to_full=apply_to_full,
                save_binned=save_binned,
                chunk_size=chunk_size,
                output_name=output_name,
                correct_intensity=correct_intensity,
                additional_files=get_companion_files(f),
                register_on_companion=register_on_companion
            )
            for i, f in enumerate(files_to_process)
        ]
    else:
        # Prepare arguments for standard processing
        tasks = [
            delayed(process_single_file)(
                input_path=f,
                output_dir=output_dir,
                transform_type=transform_type,
                reference_frames=reference_frames,
                temporal_bin=temporal_bin,
                n_frames=n_frames,
                save_transforms=save_transforms,
                suffix=suffix,
                progress_bar=(i == 0),
                apply_to_full=apply_to_full,
                save_binned=save_binned,
                output_name=output_name,
                correct_intensity=correct_intensity
            )
            for i, f in enumerate(files_to_process)
        ]
    
    # Run parallel jobs
    results = Parallel(n_jobs=actual_n_jobs, verbose=verbose, backend=backend)(tasks)
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    # Combine with skipped results
    all_results = skipped_results + results
    
    if skip_existing and skipped_results:
        logger.info(f"\nProcessing complete: {successful} succeeded, {failed} failed, {len(skipped_results)} skipped (already complete)")
    else:
        logger.info(f"\nProcessing complete: {successful} succeeded, {failed} failed")
    
    if failed > 0:
        logger.warning("Failed files:")
        for r in results:
            if not r.success:
                logger.warning(f"  - {r.input_path}: {r.error_message}")
    
    return all_results


def find_files(
    directory: Union[str, Path],
    extensions: list[str] = ['.tif', '.tiff', '.raw'],
    recursive: bool = False
) -> list[Path]:
    """Find all matching files in a directory."""
    directory = Path(directory)
    extensions = [e.lower() for e in extensions]
    
    if recursive:
        files = list(directory.rglob('*'))
    else:
        files = list(directory.glob('*'))
    
    filtered = [f for f in files if f.suffix.lower() in extensions and f.is_file()]
    return sorted(filtered)


def process_directory_parallel(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    transform_type: TransformType = 'rigid',
    reference_frames: Optional[int] = None,
    temporal_bin: int = 1,
    n_frames: Optional[int] = None,
    recursive: bool = False,
    n_jobs: int = -1
) -> list[ProcessingResult]:
    """Find and process all files."""
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir / 'corrected'
    
    files = find_files(input_dir, recursive=recursive)
    
    if not files:
        logger.warning(f"No image files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(files)} files")
    
    return parallel_batch_process(
        file_list=files,
        output_dir=output_dir,
        transform_type=transform_type,
        reference_frames=reference_frames,
        temporal_bin=temporal_bin,
        n_frames=n_frames,
        n_jobs=n_jobs
    )


if __name__ == '__main__':
    # Add CLI if needed, for now mainly library usage
    pass
