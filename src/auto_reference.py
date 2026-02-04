"""
Automatic Reference Trace Generation for Out-of-Focus Detection.

This module provides utilities to automatically generate corrReference traces
by analyzing raw imaging files and finding the best ROI that shows clear
negative deflections when images go out of focus.
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import savgol_filter, argrelmin
from os.path import getsize, join, exists, dirname
import os
import matplotlib.pyplot as plt


def parse_experiment_xml(xml_path):
    """
    Parse ThorImage Experiment.xml to extract imaging parameters.
    
    Parameters
    ----------
    xml_path : str
        Path to the Experiment.xml file
        
    Returns
    -------
    dict
        Dictionary with keys: pixelX, pixelY, pixelSizeUM, widthUM, heightUM, frameRate
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    lsm = root.find('LSM')
    
    return {
        'pixelX': int(lsm.get('pixelX')),
        'pixelY': int(lsm.get('pixelY')),
        'pixelSizeUM': float(lsm.get('pixelSizeUM')),
        'widthUM': float(lsm.get('widthUM')),
        'heightUM': float(lsm.get('heightUM')),
        'frameRate': float(lsm.get('frameRate'))
    }


def get_raw_file_info(folder, filename='Image_001_001.raw'):
    """
    Get dimensions and frame count from a raw file.
    
    Parameters
    ----------
    folder : str
        Folder containing the raw file and preview image
    filename : str
        Name of the raw file
        
    Returns
    -------
    tuple
        (width, height, n_frames)
    """
    from skimage.io import imread
    
    fullpath = join(folder, filename)
    
    # Try to get dimensions from preview image
    preview_names = ['ChanC_Preview.tif', 'ChanA_Preview.tif']
    prev = None
    for pname in preview_names:
        ppath = join(folder, pname)
        if exists(ppath):
            prev = imread(ppath)
            break
    
    if prev is None:
        raise FileNotFoundError(f"No preview image found in {folder}")
    
    width = prev.shape[1]
    height = prev.shape[0]
    
    nbytes = getsize(fullpath)
    frame_size = width * height * 2  # 2 bytes per pixel (uint16)
    n_frames = int(nbytes / frame_size)
    
    return width, height, n_frames


def load_frames(file_handle, n_frames, height, width, start=0):
    """
    Load frames from an open raw file.
    
    Parameters
    ----------
    file_handle : file
        Open file handle to the raw file
    n_frames : int
        Number of frames to load
    height, width : int
        Frame dimensions
    start : int
        Starting frame index
        
    Returns
    -------
    ndarray
        3D array of shape (n_frames, height, width)
    """
    frame_size = width * height * 2
    offset = start * frame_size
    file_handle.seek(offset)
    
    data = file_handle.read(n_frames * frame_size)
    return np.frombuffer(data, dtype=np.uint16).reshape((n_frames, height, width))



def generate_grid_rois(height, width, grid_size, overlap=0.0):
    """
    Generate ROI bounds for a specific grid configuration.
    Uses 'Diagonal Shifted Grid' logic:
    Generates the base grid, plus copies shifted diagonally by (1-overlap) steps.
    """
    # Handle grid_size argument (int or tuple)
    if isinstance(grid_size, int):
        n_y = grid_size
        n_x = grid_size
    else:
        n_y, n_x = grid_size
        
    roi_bounds = []
    
    # Calculate base steps
    step_y_size = height / n_y
    step_x_size = width / n_x
    
    # Calculate number of shifts
    # overlap 0 -> 1 shift (0)
    # overlap 0.5 -> 2 shifts (0, 0.5)
    # overlap 0.75 -> 4 shifts (0, 0.25, 0.5, 0.75)
    if overlap >= 1.0 or overlap < 0:
        overlap = 0
        
    if overlap == 0:
        num_shifts = 1
    else:
        num_shifts = int(round(1 / (1.0 - overlap)))
        
    for k in range(num_shifts):
        # Calculate diagonal offset for this layer
        fraction = k / num_shifts
        offset_y = fraction * step_y_size
        offset_x = fraction * step_x_size
        
        # Generate grid ROIs for this offset
        # We iterate enough to cover the image, but only keep valid ones
        # Usually i goes up to n_y, but with offset we might lose the last row/col
        
        for i in range(n_y + 1):
             for j in range(n_x + 1):
                 y0 = int(offset_y + i * step_y_size)
                 x0 = int(offset_x + j * step_x_size)
                 y1 = int(offset_y + (i + 1) * step_y_size)
                 x1 = int(offset_x + (j + 1) * step_x_size)
                 
                 # Check if valid ROI (fully within image)
                 if y1 <= height and x1 <= width:
                     roi_bounds.append((y0, y1, x0, x1))
            
    return roi_bounds



def compute_traces_for_rois(folder, roi_bounds, sample_step=1, max_frames=None, 
                            filename='Image_001_001.raw', smooth=True, smooth_window=11, compute_std=True):
    """
    Compute mean intensity AND optionally spatial standard deviation traces for ROIs.
    Spatial Std is a robust measure for focus: OOF image = low variance (blurry).
    """
    width, height, total_frames = get_raw_file_info(folder, filename)
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        
    n_rois = len(roi_bounds)
    frame_indices = np.arange(0, total_frames, sample_step)
    n_sampled = len(frame_indices)
    
    fullpath = join(folder, filename)
    
    mean_image = None
    traces_mean = np.zeros((n_sampled, n_rois), dtype=np.float32)
    traces_std = None
    if compute_std:
        traces_std = np.zeros((n_sampled, n_rois), dtype=np.float32)
    
    # Try to load everything into memory
    try:
        with open(fullpath, 'rb') as f:
            full_data = load_frames(f, total_frames, height, width, 0)
            
        # Apply sample_step
        if sample_step > 1:
            full_data = full_data[::sample_step]
            
        mean_image = np.mean(full_data[:min(500, len(full_data))], axis=0)
        
        for roi_idx, (y0, y1, x0, x1) in enumerate(roi_bounds):
            # Vectorized mean and std over spatial dimensions (axes 1 and 2)
            roi_stack = full_data[:, y0:y1, x0:x1]
            traces_mean[:, roi_idx] = roi_stack.mean(axis=(1, 2))
            if compute_std:
                traces_std[:, roi_idx] = roi_stack.std(axis=(1, 2))
            
        # Clean up
        del full_data
            
    except (MemoryError, Exception) as e:
        print(f"Loading full movie failed ({e}), falling back to chunked processing...")
        return compute_traces_chunked(folder, roi_bounds, sample_step, max_frames, filename, width, height, total_frames, compute_std)

    # Apply smoothing if requested (only to mean trace usually, but maybe std too?)
    # Let's smooth both for consistency
    if smooth and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        for roi_idx in range(n_rois):
            traces_mean[:, roi_idx] = savgol_filter(traces_mean[:, roi_idx], smooth_window, 1)
            if compute_std:
                traces_std[:, roi_idx] = savgol_filter(traces_std[:, roi_idx], smooth_window, 1)
            
    return {
        'traces': traces_mean,
        'traces_std': traces_std,
        'n_frames': n_sampled,
        'mean_image': mean_image
    }


def compute_traces_chunked(folder, roi_bounds, sample_step, max_frames, filename, width, height, total_frames, compute_std=True):
    """Fallback chunked processor computing mean and std"""
    chunk_size = 500
    fullpath = join(folder, filename)
    
    n_rois = len(roi_bounds)
    frame_indices = np.arange(0, total_frames, sample_step)
    n_sampled = len(frame_indices)
    traces_mean = np.zeros((n_sampled, n_rois), dtype=np.float32)
    traces_std = None
    if compute_std:
        traces_std = np.zeros((n_sampled, n_rois), dtype=np.float32)
    mean_image = None
    
    with open(fullpath, 'rb') as f:
        # 1. Compute mean image
        n_preview = min(500, total_frames)
        preview_frames = load_frames(f, n_preview, height, width, 0)
        mean_image = np.mean(preview_frames, axis=0)
        
        # 2. Process chunks
        sampled_idx = 0
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            frames_in_chunk = load_frames(f, chunk_end - chunk_start, height, width, chunk_start)
            
            for local_idx in range(frames_in_chunk.shape[0]):
                global_idx = chunk_start + local_idx
                if global_idx % sample_step == 0:
                    frame = frames_in_chunk[local_idx]
                    for roi_idx, (y0, y1, x0, x1) in enumerate(roi_bounds):
                        roi_portion = frame[y0:y1, x0:x1]
                        traces_mean[sampled_idx, roi_idx] = roi_portion.mean()
                        if compute_std:
                            traces_std[sampled_idx, roi_idx] = roi_portion.std()
                    sampled_idx += 1
                    
    return {
        'traces': traces_mean,
        'traces_std': traces_std,
        'n_frames': n_sampled,
        'mean_image': mean_image
    }


def compute_grid_traces(folder, grid_size=4, sample_step=1, max_frames=None, 
                        filename='Image_001_001.raw', smooth=True, smooth_window=11, overlap=0.0):
    """
    Legacy/Single-grid wrapper. 
    Divides FOV into a grid and computes traces.
    """
    width, height, _ = get_raw_file_info(folder, filename)
    rois = generate_grid_rois(height, width, grid_size, overlap)
    
    result = compute_traces_for_rois(
        folder, rois, sample_step, max_frames, filename, smooth, smooth_window
    )
    
    result['roi_bounds'] = rois
    result['grid_size'] = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    result['sample_step'] = sample_step
    
    return result


def select_best_reference_roi(grid_data, minima_order=50, score_fraction=0.66):
    """
    Select the best ROI from the grid traces based on quality metrics.
    """
    traces = grid_data['traces']
    # Handle optional std traces if available (from new compute function)
    traces_std = grid_data.get('traces_std', None) 
    roi_bounds = grid_data['roi_bounds']
    
    n_rois = traces.shape[1]
    scores = np.zeros(n_rois)
    
    for i in range(n_rois):
        trace = traces[:, i]
        # Quality metric: Temporal Variance/Std of the Mean Trace
        # "Structure" in the sample leads to differences as it moves/defocuses.
        # A flat featureless ROI has low temporal variance.
        # WEIGHTING: We care more about the beginning of the trace (baseline).
        # We apply a linear weight decay from 1.0 to 0.5 over the trace.
        n_frames = len(trace)
        if n_frames > 0:
            weights = np.linspace(1.0, 0.5, n_frames)
            weighted_mean = np.average(trace, weights=weights)
            weighted_variance = np.average((trace - weighted_mean)**2, weights=weights)
            scores[i] = np.sqrt(weighted_variance)
        else:
            scores[i] = 0.0
        
    best_idx = np.argmax(scores)
    
    best_trace_std = None
    if traces_std is not None:
        best_trace_std = traces_std[:, best_idx]
        
    return {
        'best_roi_index': best_idx,
        'roi_bounds': roi_bounds[best_idx],
        'best_trace': traces[:, best_idx],
        'best_trace_std': best_trace_std,
        'scores': scores,
        'mean_image': grid_data.get('mean_image', None)
    }







def detect_oof_intervals(trace_mean, trace_std, threshold_fraction=0.6, min_duration=5, merge_gap=10):
    """
    Detect Out-Of-Focus (OOF) intervals using Spatial Standard Deviation (Blur detection).
    
    Logic:
    In-focus images have high spatial frequencies (high variance/std).
    OOF images are blurry (low variance/std).
    We detect periods where the spatial std drops significantly below the baseline.
    
    Parameters
    ----------
    trace_mean : ndarray
        Mean intensity trace (for reference/plotting)
    trace_std : ndarray
        Spatial standard deviation trace (KEY for focus detection)
    threshold_fraction : float
        Fraction of the baseline variability to consider "bad".
        e.g. 0.6 means if std < 0.6 * baseline_std, it's OOF.
    min_duration : int
        Minimum frames for an interval
    merge_gap : int
        Merge close intervals
        
    Returns
    -------
    list of tuples
        OOF intervals (start, end)
    """
    if trace_std is None or len(trace_std) == 0:
        return []
        
    s = pd.Series(trace_std)
    
    # 1. Establish robust baseline for "Good Focus"
    # Use the median of the upper half of variance values (assuming mostly in focus)
    median_std = s.median()
    upper_std_values = s[s >= median_std]
    
    if len(upper_std_values) > 0:
        baseline_focus = upper_std_values.median()
    else:
        baseline_focus = median_std
def detect_oof_intervals(trace_mean, trace_std, threshold_fraction=0.6, min_duration=2, merge_gap=5):
    """
    Detect Out-Of-Focus (OOF) intervals using a Hybrid approach:
    1. Spatial Blur (Low Spatial Std)
    2. Intensity Drops (Low Mean Intensity)
    
    This catches both focus loss (blur) and shutter/artifact events (dark frames).
    
    Parameters
    ----------
    trace_mean : ndarray
        Mean intensity trace
    trace_std : ndarray
        Spatial standard deviation trace
    threshold_fraction : float
        Fraction of baseline to define cutoff (e.g. 0.6).
        Applies to both mean and std baselines.
    min_duration : int
        Minimum frames to count as an event (lowered to catch spikes)
    merge_gap : int
        Merge close intervals
    """
    if trace_mean is None or len(trace_mean) == 0:
        return []
        
    # --- 1. Compute Baselines (Robust) ---
    def get_baseline(data):
        if data is None: return 0
        s = pd.Series(data)
        med = s.median()
        upper = s[s >= med]
        return upper.median() if len(upper) > 0 else med

    baseline_mean = get_baseline(trace_mean)
    mean_cutoff = baseline_mean * 0.7 # slightly more permissive for intensity
    
    is_oof_mean = trace_mean < mean_cutoff
    
    is_oof_std = np.zeros_like(trace_mean, dtype=bool)
    if trace_std is not None and len(trace_std) == len(trace_mean):
        baseline_std = get_baseline(trace_std)
        std_cutoff = baseline_std * threshold_fraction
        is_oof_std = trace_std < std_cutoff
        
    # Combine criteria: OOF if EITHER metric drops
    is_oof = is_oof_mean | is_oof_std
    
    # --- 2. Find Intervals ---
    oof_int = is_oof.astype(int)
    diffs = np.diff(oof_int, prepend=0, append=0)
    
    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]
    
    intervals = []
    for s_idx, e_idx in zip(start_indices, end_indices):
        if (e_idx - s_idx) >= min_duration:
            intervals.append((s_idx, e_idx))
            
    if not intervals:
        return []
        
    # --- 3. Merge ---
    merged = []
    if intervals:
        curr_s, curr_e = intervals[0]
        for next_s, next_e in intervals[1:]:
            if next_s - curr_e <= merge_gap:
                curr_e = next_e
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged.append((curr_s, curr_e))
        
    return merged


def find_cutoff_frame(trace, window=1000, deviation_thresh=0.20, recover_window=500):
    """
    Find the frame index after which the recording becomes unusable 
    (e.g., permanent focus loss or movement).
    
    Logic: Detects if the rolling median deviates significantly from the 
    baseline (initial part of trace) and fails to recover for the rest of the recording.
    
    Parameters
    ----------
    trace : ndarray
        1D intensity trace
    window : int
        Window size for rolling baseline (smoothing)
    deviation_thresh : float
        Fractional deviation (0.2 = 20%) to trigger "bad" state
    recover_window : int
        How long a "bad" state must persist to be considered permanent cutoff
        
    Returns
    -------
    int
        Index of the last usable frame. Returns len(trace) if no cutoff found.
    """
    # Ensure window isn't larger than trace
    window = min(window, len(trace) // 5 + 1)
    
    s = pd.Series(trace)
    
    # 1. Establish initial baseline from start (robust to noise)
    baseline_window = min(len(trace), 1000)
    baseline = s.iloc[:baseline_window].median()
    
    # 2. Compute rolling median to smooth out transient OOF events
    # min_periods needs to be small enough
    rolling = s.rolling(window=window, min_periods=max(1, window//10), center=True).median()
    
    # Fill nan edges
    rolling = rolling.bfill().ffill()
    
    # 3. Define acceptable range
    lower_bound = baseline * (1 - deviation_thresh)
    upper_bound = baseline * (1 + deviation_thresh)
    
    # 4. Identify frames where signal is "bad"
    is_bad = (rolling < lower_bound) | (rolling > upper_bound)
    
    # 5. Check for permanent failure
    # If we see a bad block that never extends to the end or close to it, it might be transient.
    # But if we see a bad state that persists until the very end, we backtrace to where it started.
    
    # Reverse iterating to find where the "end" bad state began
    # If the last frame is good, then the movie is usable till the end (mostly).
    if not is_bad.iloc[-1]:
        return len(trace)
    
    # Find the last time the signal was "good"
    good_indices = np.where(~is_bad)[0]
    
    if len(good_indices) == 0:
        return 0
        
    last_good = good_indices[-1]
    
    # The cutoff is shortly after the last good frame.
    return min(last_good + 1, len(trace))

def generate_reference_trace(folder, output_csv=None, grid_size=4, sample_step=1,
                              max_frames=None, smooth=True, smooth_window=11,
                              minima_order=50, filename='Image_001_001.raw', score_fraction=0.66,
                              overlap=0.5, detect_oof=True):
    """
    Main function to generate a corrReference trace automatically.
    
    Parameters
    ----------
    folder : str
        Experiment folder containing the raw file
    output_csv : str or None
        Path for output CSV. If None, saves to 'corrReference.csv' in folder
    grid_size : int, tuple, or list of tuples
        Grid divisions per axis. 
        - int: (N, N)
        - tuple: (ny, nx)
        - list: [(ny1, nx1), (ny2, nx2), ...] to search multiple grid scales
    sample_step : int
        Frame sampling step
    max_frames : int or None
        Maximum frames to analyze
    smooth : bool
        Apply Savitzky-Golay smoothing
    smooth_window : int
        Smoothing window size
    minima_order : int
        Order for minima detection
    filename : str
        Name of raw file
    score_fraction : float
        Fraction of trace to use for scoring (default 0.66)
    overlap : float
        Fraction of overlap between adjacent ROIs (0.0 to < 1.0).
        Replaces 'use_shifted_grid'. Default 0.5 provides 50% overlap.
    detect_oof : bool
        Whether to perform Out-Of-Focus (OOF) detection. Default is True.
        If False, OOF intervals will be empty.
        
    Returns
    -------
    dict
        {
            'roi_index': int,
            'roi_bounds': tuple,
            'trace': ndarray,
            'csv_path': str,
            'scores': ndarray,
            'mean_image': ndarray,
            'cutoff_frame': int,
            'oof_intervals': list of tuples
        }
    """
    # Normalize grid_size to a list of configurations
    if isinstance(grid_size, list):
        grids = grid_size
    else:
        grids = [grid_size]
        
    print(f"Generating ROI candidates for {folder}...")
    
    # Get image dimensions first to generate ROIs
    width, height, _ = get_raw_file_info(folder, filename)
    
    all_bounds = []
    
    # 1. Generate all ROI candidates from all grid configs
    for g_size in grids:
        print(f"  - Generating ROIs for grid: {g_size} with overlap {overlap}")
        rois = generate_grid_rois(height, width, g_size, overlap)
        all_bounds.extend(rois)
        
    print(f"Total ROIs to analyze: {len(all_bounds)}")
    
    # 2. Compute traces for ALL ROIs in one pass
    print(f"Computing traces for {len(all_bounds)} ROIs (single pass)...")
    
    trace_data = compute_traces_for_rois(
        folder, 
        all_bounds, 
        sample_step=sample_step, 
        max_frames=max_frames,
        filename=filename,
        smooth=smooth,
        smooth_window=smooth_window,
        compute_std=detect_oof
    )
    
    # Construct combined grid_data
    combined_grid_data = {
        'traces': trace_data['traces'],
        'traces_std': trace_data.get('traces_std'),
        'roi_bounds': all_bounds,
        'mean_image': trace_data['mean_image']
    }
    

    print(f"Selecting best reference ROI...")
    result = select_best_reference_roi(combined_grid_data, minima_order=minima_order, score_fraction=score_fraction)
    
    best_trace = result['best_trace']
    best_trace_std = result.get('best_trace_std', None)
    
    # Calculate cutoff frame (last usable frame)
    cutoff_frame = find_cutoff_frame(best_trace)
    
    # Detect OOF intervals on the FULL trace (no cutoff truncation)
    # This allows us to see jumps even in the cut-off region
    oof_intervals = []
    oof_intervals = []
    if detect_oof and best_trace_std is not None:
        # Hybrid detection: Spatial Blur OR Intensity Drop
        # Low min_duration to catch single-frame artifacts (spikes)
        oof_intervals = detect_oof_intervals(best_trace, best_trace_std, min_duration=2, merge_gap=5)
    
    # Create output DataFrame (matching existing format: Slice is 1-indexed)
    n_frames = len(best_trace)
    df_data = {
        'Slice': np.arange(1, n_frames + 1) * sample_step,
        'Mean': best_trace
    }
    # Add Std trace to CSV if available for debugging
    if best_trace_std is not None:
        df_data['Std'] = best_trace_std
        
    df = pd.DataFrame(df_data)
    
    # Save CSV
    if output_csv is None:
        output_csv = join(folder, 'corrReference.csv')
    
    df.to_csv(output_csv, index=False)
    print(f"Saved reference trace to {output_csv}")
    print(f"Best ROI: index {result['best_roi_index']} (global), bounds {result['roi_bounds']}")
    print(f"Detected cutoff frame: {cutoff_frame} (of {n_frames})")
    
    if detect_oof:
        print(f"Detected {len(oof_intervals)} OOF intervals (Spatial Blur)")
    
    return {
        'roi_index': result['best_roi_index'],
        'roi_bounds': result['roi_bounds'],
        'trace': best_trace,
        'trace_std': best_trace_std, 
        'csv_path': output_csv,
        'scores': result['scores'],
        'mean_image': result['mean_image'],
        'cutoff_frame': cutoff_frame,
        'oof_intervals': oof_intervals
    }

def plotResult(result, grid_size=4):
    scores = result['scores']
    mean_image = result.get('mean_image')
    cutoff_frame = result.get('cutoff_frame')
    
    # Handle grid_size tuple
    if isinstance(grid_size, tuple):
        grid_y, grid_x = grid_size
    else:
        grid_y = grid_x = grid_size
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    
    # Trace plot (top)
    ax_trace = fig.add_subplot(gs[0, :])
    trace = result['trace']
    frames = np.arange(len(trace))
    ax_trace.plot(frames, trace, 'b-', linewidth=0.5)
    
    # Draw cutoff frame line
    if cutoff_frame is not None and cutoff_frame < len(result['trace']):
        ax_trace.axvline(x=cutoff_frame, color='r', linestyle='--', linewidth=1.5, label='Usable Cutoff')
        
    # Draw OOF intervals as RED SEGMENTS superimposed on the trace
    oof_intervals = result.get('oof_intervals', [])
    if oof_intervals:
        # Add label for the first one only for legend
        start, end = oof_intervals[0]
        # oof_intervals are [start, end). Plot slice [start:end]
        ax_trace.plot(frames[start:end], trace[start:end], 'r-', linewidth=1.2, label='OOF Event')
        
        for start, end in oof_intervals[1:]:
             ax_trace.plot(frames[start:end], trace[start:end], 'r-', linewidth=1.2)
             
    ax_trace.legend(loc='upper right')
        
    ax_trace.set_xlabel('Frame')
    ax_trace.set_ylabel('Mean Intensity')
    
    # Optional: Plot Spatial Std on twin axis to show why OOF was detected
    trace_std = result.get('trace_std')
    if trace_std is not None:
        ax_std = ax_trace.twinx()
        ax_std.plot(trace_std, color='orange', alpha=0.3, linewidth=1, label='Spatial Std')
        ax_std.set_ylabel('Spatial Std', color='orange')
        ax_std.tick_params(axis='y', labelcolor='orange')
        # Add simpler legend manually or just leave it

    ax_trace.set_title(f'Reference Trace (ROI {result["roi_index"]})')
    ax_trace.grid(True, alpha=0.3)
    
    # Image plot (bottom)
    ax_img = fig.add_subplot(gs[1, :])
    
    if mean_image is not None:
        im = ax_img.imshow(mean_image, cmap='gray')
        plt.colorbar(im, ax=ax_img, label='Mean Intensity')
    
    # Determine grid positions for overlay
    # Re-calculate edges based on score array size logic or pass them through result?
    # Ideally should be passed through result, but for now we infer from grid_size + image shape
    # Assuming result['roi_bounds'] is correct and comes from the same grid logic
    
    y0, y1, x0, x1 = result['roi_bounds']
    
    # Draw the best ROI box
    import matplotlib.patches as patches
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='r', facecolor='none', label='Best ROI')
    ax_img.add_patch(rect)
    
    # Annotate score
    best_score = scores[result['roi_index']]
    ax_img.text(x0, y0-5, f"Score: {best_score:.1f}", color='red', fontsize=12, fontweight='bold')
    
    ax_img.set_title(f"Best ROI Location (Index {result['roi_index']})")
    ax_img.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def batch_generate_references(folder_list, skip_existing=True, plot=False, score_fraction=0.66, overlap=0.5, **kwargs):
    """
    Generate reference traces for multiple experiment folders.
    
    Parameters
    ----------
    folder_list : list of str
        List of experiment folder paths
    skip_existing : bool
        If True, skip folders where corrReference.csv already exists
    plot : bool
        If True, display the result plot for each folder
    score_fraction : float
        Fraction of trace to use for scoring (default 0.66)
    overlap : float
        Fraction of overlap between adjacent ROIs (0.0 to < 1.0).
        Default 0.5.
    **kwargs
        Additional arguments passed to generate_reference_trace
        
    Returns
    -------
    dict
        Dictionary mapping folder paths to results (or error messages)
    """
    results = {}
    
    for folder in folder_list:
        if skip_existing and (exists(join(folder, 'corrReference.csv')) or exists(join(folder, 'corrReference.npy'))):
            print(f"Skipping existing reference trace for {folder}")
            continue
        else:
            print(f"\n{'='*60}")
            print(f"Processing: {folder}")
            try:
                results[folder] = generate_reference_trace(folder, score_fraction=score_fraction, overlap=overlap, **kwargs)
                if plot:
                    plotResult(results[folder])
            except Exception as e:
                print(f"Error: {e}")
                results[folder] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        result = generate_reference_trace(folder)
        print(f"\nResult: {result}")
    else:
        print("Usage: python auto_reference.py <experiment_folder>")
