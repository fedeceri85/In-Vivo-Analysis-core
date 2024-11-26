## Sample code for the paper *In vivo spontaneoous Ca2+ activity in the pre-hearing mammalian cochlea* by De Faveri et al..

This repository contains a pipeline for analyzing calcium imaging data for spontaneous activity in the developing cochlea.

## Project Structure

- **notebooks/**: Jupyter notebooks for movie registration and segmentation and data processing 
- **parameters/**: Configuration and parameter files
- **src/**: Source code of python modules for analysis functions, and TraceExplorer and naparipy software (trace annotation)

## Key Features

- Support for multiple experiment types:
    - Calcium events in IHCs (Myo15 or Atoh1 with GCaMP6)
    - Calcium events in SGN (NeuroD or SNAP25 with GCaMP6)
    - Calcium waves (Pax2 with GCaMP6)

## Data Processing Pipeline

### Jump Correction:
- Uses an interactive interface to select frames of fluorescence traces
- Out of focus frames (due to breathing) of the original file are removed and replaced with the last in-focus frame before the "jump"
- Generates jump-corrected files with with "jumpCorrected.tif" suffix creating a "processedMovies" directory

Processed parameter files are saved in the `parameters` directory with corresponding naming conventions.

### Motion Correction:
- Uses CaImAn's MotionCorrect for non-rigid correction
- Generates motion-corrected files with "-mc.tif" suffix, saved in the "processedMovies" directory
- Corrects intensity values altered during motion correction

### Segmentation:
- Uses an interactive interface to segment the image, using an average projection of the "jumpCorrected-mc.tif" files
- Specific tools allow segmentation of IHCs, SGNs and calcium waves in Napari. ROI masks can be visualised as Napari layers
- Automatically generated ROIs can be manually adjusted and annotated (e.g. to label pillar/modiolar SGN terminals)
- Extracts fluorescence profiles from individual ROIs, saved as "traces.csv" files
- Saves ROIs and annotations files in the "processedMovies" directory

## Usage

The main processing pipeline is implemented in `batchMotionCorrect.ipynb`. Configure the `fileHeader` variable to specify which experiment types to process.

## Requirements

Required Python packages:
- numpy
- pandas
- scipy
- matplotlib
- caiman
- tifffile
- jupyter
- napari
- scikit-image
- xlrd
- openpyxl
- PyQt5
- pyqtgraph
- ipywidgets

## Data Organization

Experiment metadata is stored in Excel files in the root directory:
- `Myo15_IHCs.xlsx`
- `NeuroD_SGN.xlsx`
- `Pax2_Calciumwaves.xlsx`

