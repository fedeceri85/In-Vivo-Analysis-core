import pandas as pd
import os
import tqdm

import numpy as np


def generate_oof_labels(corrected_path, output_path,first_frame,last_frame,nChannels,spatialGaussian,temporalGaussian,save_parameters = True, dry_run=False):

    """
    Generates labels by identifying static (replaced) frames in the jump-corrected movie.
    According to user input: "removed frames have been replaced with a still image of the last good frame".
    Therefore, frames where the derivative of the mean intensity is zero are 'Out-of-Focus'.
    """
       
        
    # 2. Load Corrected Trace & Compute Derivative
    print(f"Processing: {corrected_path}")
    import tifffile
    
    # Load data. For reliability, we load the whole stack or use memmap.
    # Given typical sizes (3-4GB), memmap is safer if available, but simple 'imread' works if RAM allows.
    # We'll use memmap to be safe against huge files.
    with tifffile.TiffFile(corrected_path) as tif:
        # Check if series[0] is available (standard for OME-TIFF or multi-page)
        if len(tif.series) > 0:
                # asarray() with out='memmap' creates a memmap backed array
                data = tif.series[0].asarray(out='memmap')
        else:
                # Fallback for simple multipage tiff
                data = tifffile.imread(corrected_path)

    expected_frames = last_frame - first_frame + 1
    n_corr = data.shape[0]

    if n_corr != expected_frames:
        print(f"Warning: Expected {expected_frames} frames, but found {n_corr} frames in the corrected movie.")
    # Compute mean intensity trace
    # Optimization: Just sum over axes 1,2 then divide, or just use sum (derivative of sum is also 0 if image is identical)
    # Using sum is faster and avoids float division issues initially.
    trace = np.sum(data, axis=(1, 2), dtype=np.float64)
    
    # 3. Detect Static Frames (Zero Derivative)
    diff = np.diff(trace)
    
    # Threshold for "Zero". 
    # Since data is integers, difference should be exactly 0. 
    # But allow small epsilon if conversions happened.
    # The user said "derivative... is zero".
    is_static = np.abs(diff) < 1e-6
    
    # is_static[i] == True means trace[i+1] == trace[i].
    # This implies frame i+1 is a repeat of frame i.
    # So frame i+1 is the "bad" one (replaced).
    
    # Create mask of size n_corr
    mask = np.zeros(n_corr, dtype=bool)
    
    # If diff[i] is 0, then frame i+1 is static.
    static_indices = np.where(is_static)[0] + 1
    mask[static_indices] = True
    
    print(f"Found {len(static_indices)} static frames out of {n_corr} ({len(static_indices)/n_corr:.1%})")
    
    # Use the mask as-is from the corrected movie (no padding)
    # The Excel frame ranges specify which portion of the recording to use,
    # which should match the length of the corrected movie
    final_mask = mask
    
        
    # 5. Save Labels
    
    df = pd.DataFrame({
        'Frame': np.arange(len(final_mask)),
        'Label': final_mask.astype(int)
    })
    
    if not dry_run:
        df.to_csv(output_path, index=False)
        print(f"Saved labels to: {output_path}")

        if save_parameters:
             import json
             params = {
                 'first_frame': first_frame,
                 'last_frame': last_frame,
                 'nChannels': nChannels,
                 'spatialGaussian': spatialGaussian,
                 'temporalGaussian': temporalGaussian
             }
             params_path = os.path.join(os.path.split(output_path)[0], JSON_FILE)
             with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)

    return True


def compute_total_space(folder_list,file_name='1-jumpcorrected.tif'):
    total_size = 0
    for folder in folder_list:
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
        else:
            print(f"File not found: {file_path}")

    total_size_gb = total_size / (1024 ** 3)
    print(f"Total size of '{file_name}' across all folders: {total_size_gb:.2f} GB")
    return total_size_gb


def safe_remove_jumpCorrected(folder_list,processedFolder = 'processedMovies',
                              jumpcorrected_file = '1-jumpcorrected.tif',
                                motioncorrected_file = '1-jumpcorrected-mc.tif',
                                oof_labels_file = 'out_of_focus_labels.csv',
                                JSON_FILE = 'processing_parameters.json',
                                RAW_FILE = 'Image_001_001.raw',
                              dry_run=True):
    for folder in tqdm.tqdm(folder_list, desc="Processing folders"):
        jumpcorrected_path = os.path.join(folder, processedFolder, jumpcorrected_file)
        raw_path = os.path.join(folder, RAW_FILE)
        motioncorrected_path = os.path.join(folder, processedFolder, motioncorrected_file)
        oof_labels_path = os.path.join(folder, processedFolder, oof_labels_file)
        json_params_path = os.path.join(folder, processedFolder, JSON_FILE)

        if os.path.exists(jumpcorrected_path) and os.path.exists(raw_path) and os.path.exists(motioncorrected_path) and os.path.exists(oof_labels_path) and os.path.exists(json_params_path):
            if dry_run:
                print(f"[Dry Run] Would remove: {jumpcorrected_path}")
            else:
                os.remove(jumpcorrected_path)
                print(f"Removed: {jumpcorrected_path}")
        else:
            print(f"File not removed: {jumpcorrected_path}")


def safe_remove_motionCorrected(folder_list,processedFolder = 'processedMovies',
                                motioncorrected_file = '1-jumpcorrected-mc.tif',
                                oof_labels_file = 'out_of_focus_labels.csv',
                                JSON_FILE = 'processing_parameters.json',
                                RAW_FILE = 'Image_001_001.raw',
                                dry_run=True):
    for folder in tqdm.tqdm(folder_list, desc="Processing folders"):
        motioncorrected_path = os.path.join(folder, processedFolder, motioncorrected_file)
        raw_path = os.path.join(folder, RAW_FILE)
        oof_labels_path = os.path.join(folder, processedFolder, oof_labels_file)
        json_params_path = os.path.join(folder, processedFolder, JSON_FILE)

        if os.path.exists(motioncorrected_path) and os.path.exists(raw_path) and os.path.exists(oof_labels_path) and os.path.exists(json_params_path):
            if dry_run:
                print(f"[Dry Run] Would remove: {motioncorrected_path}")
            else:
                os.remove(motioncorrected_path)
                print(f"Removed: {motioncorrected_path}")
        else:
            print(f"File not removed: {motioncorrected_path}")