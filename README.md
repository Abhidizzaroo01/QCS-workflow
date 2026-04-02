# QCS-workflow

This repository contains an end-to-end workflow for histopathology image analysis (example-focused on HER2-like samples). It combines:

1. Whole-slide (WSI) patch extraction from `.ndpi`
2. Stain deconvolution (HED) into Hematoxylin (H) / Eosin (E) / DAB (D) channels
3. Nuclei segmentation (Cellpose on the H channel)
4. Membrane segmentation (either Cellpose on the DAB channel for strong staining, or DAB-OD driven membrane refinement for weak staining)
5. Patch-level quantification (per-cell table + summary metrics)

The implementation is split into reusable library modules under `dqcs/` and runnable example scripts under `example2/` and `examples/`.

## Project Layout

- `dqcs/` : core processing library
  - `dutils.py` : small image helpers (`img_check`)
  - `dpixel.py` : pixel-level optical density (OD) and geometry helpers
  - `dseg.py` : segmentation helpers (stain extraction, OD computation, Cellpose wrappers, membrane segmentation)
  - `danalysis.py` : annotation I/O (TXT + GeoJSON), filtering/refinement of outlines, and bioinformatics metrics
  - `dslide.py` : NDPI slide utilities (thumbnail, patch extraction, OD-based patch filtering)
  - `stainnorm.py` : stain normalization classes (based on `tiatoolbox`)
- `examples/` : smaller example scripts
- `example2/` : batch pipelines/scripts demonstrating the full workflow
- `.gitignore` : ignores large artifacts (images, segmentation outputs)

## What the Workflow Produces

Most scripts generate annotation files per input image/patch:

- Nuclei annotations:
  - `*_n.txt` (QuPath-like outline text)
  - `*_n.geojson` (GeoJSON FeatureCollection)
  - `*_ncenter.txt` (centers for each nucleus)
- Filtered / refined variants:
  - `*_nf*.txt/.geojson/.txt` (after nuclei filtering)
  - `*_nsf*.txt/.geojson/.txt` (after nuclei squeezing/refinement)
- Membrane annotations:
  - `*_m.txt/.geojson/*center.txt` (raw membrane outlines)
  - `*_mf.txt/.geojson/*center.txt` (DAB refined membrane outlines)

The “batch nucleus+membrane + bioinformatics” scripts additionally output:

- `*_cell_table.csv` : per-cell metrics (micron-based)
- `batch_bioinfo_summary.csv` : batch-level metrics aggregation

## Core Processing (Conceptual Pipeline)

### 1) Slide patch extraction (NDPI)
If you start from a whole-slide image, the `Slide` class in `dqcs/dslide.py` is used to:

- Open an `.ndpi` slide with `openslide`
- Extract square patches (`extract_patches_from_ndpi`)
- Optionally filter patches using OD thresholds (`filter_patches`)

### 2) Stain deconvolution (HED)
The segmentation code uses HED deconvolution via `skimage`:

- `get_hed(img_rgb)` : produces H / E / D pseudo-images
- `get_hd_clean(img_file, ...)` : same idea, but removes pixels depending on OD comparison between H and D to separate nuclei vs membrane signals

### 3) Optical density (OD) computation
OD is computed from image intensities (scaled and clipped) and then averaged or used for thresholding. Key functions:

- `get_OD_single_channel(...)` : OD for a single (DAB) channel
- `get_OD(img_rgb)` : OD computed on RGB (averaged across channels)

### 4) Nuclei segmentation (Cellpose)
The nuclei segmentation uses Cellpose on the enhanced H channel:

- `cellpose_seg(..., is_membrane=False)` in `dqcs/dseg.py`

Then outlines are filtered and optionally refined:

- `filter_nuclei_annotations(...)`
- `squeeze_nuclei_annotations(...)` (aligns outline points with high OD pixels)

### 5) Membrane segmentation (adaptive strategy)
`example2/Batch_Nuclei_Mem_Seg/batch_nm_seg.py` demonstrates the “adaptive flow”:

1. Compute DAB OD on the RAW (pre-gamma) DAB channel and estimate staining strength
2. If staining is “strong” (DAB OD avg > threshold), run membrane Cellpose and filter/pair membranes
3. If staining is “weak”, run a nuclei-primary approach:
   - filter nuclei
   - expand nuclei outlines to build synthetic membrane rings
   - refine membrane contours by keeping only boundary points with high DAB OD
   - finally run `membrane_seg(...)` which casts radial rays from nucleus centers and stops based on OD thresholds

Key library functions:

- `filter_membrane_by_shape(...)`
- `filter_membrane_by_dab_intensity(...)`
- `refine_outlines_by_dab_points(...)`
- `membrane_seg(...)`

### 6) Quantification (“bioinformatics” metrics)
`patch_bioinformatics_v2(...)` in `dqcs/danalysis.py` computes:

- A per-cell table (`pandas.DataFrame`) containing:
  - nucleus area in micron^2
  - membrane OD mean inside the membrane polygon
  - centroid coordinates (px and micron)
  - validity flag (area between thresholds)
- Summary metrics:
  - `q15` : 15th percentile of membrane OD among valid cells
  - `percent_positive_od10` : percent with membrane OD >= 10
  - `density_high_expressors` : count of membrane OD >= 25 per mm^2
  - `bsps_score` : binary spatial proximity score (neighbors within r=25 um among OD>=60 cells)
  - `csps_mean` : continuous proximity score mean (neighbors within r=50 um among cells above baseline)

## Setup & Run (Windows)

### 1) Create a Python environment
From the repository folder (`C:\Users\abhishekd_dizzaroo\Desktop\qcs`):

```powershell
cd "C:\Users\abhishekd_dizzaroo\Desktop\qcs"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies
The scripts import the following packages (as used in `dqcs/` and the example scripts):

```powershell
pip install numpy opencv-python scikit-image matplotlib pillow openslide-python
pip install pandas scipy
pip install cellpose
```

Notes:
- Cellpose uses PyTorch under the hood. If `pip install cellpose` fails due to Torch, install Torch first (CPU is fine for testing).
- OpenSlide on Windows sometimes requires installing the OpenSlide native binaries and ensuring they are on your system PATH.

### 3) Fix the `sys.path.append(...)` lines in your scripts
Some scripts currently hardcode Linux/WSL paths so Python can find the local `dqcs/` package. On Windows, you must update those lines to the absolute path where you stored this repository.

Update these files:

1) Nuclei-only segmentation:
   - File: `example2/Batch_Nuclei_Segmentation/batch_nuclei_seg.py`
   - Replace:
     `sys.path.append('/home/aktsh/codes/')`
   - With (example):
     `sys.path.append('C:/Users/<YOUR_USER>/Desktop/qcs')`

2) Nuclei + membrane segmentation + bioinformatics:
   - File: `example2/Batch_Nuclei_Mem_Seg/batch_nm_seg.py`
   - Replace:
     `sys.path.append('/mnt/c/Users/abhishekd_dizzaroo/Desktop/qcs')`
   - With (example):
     `sys.path.append('C:/Users/<YOUR_USER>/Desktop/qcs')`

Alternative (often cleaner): if you run commands from the repo root (`cd qcs`), you can usually remove the `sys.path.append(...)` lines entirely because the `dqcs/` folder is already in the repository.

### 4) Prepare your input images folder
The batch scripts use:
`path = '../images/'`

So you should place input tiles/patches here (relative to the repo root):
- `C:\Users\<YOU>\Desktop\images\<tp>\...`

Where `tp` is the tile subfolder used inside each script (for example `'0'`, `'3'`).

If your `images/` folder is elsewhere, update the `path = '../images/'` line inside the scripts accordingly.

## How to Run the Example Scripts

### A) WSI -> patches (NDPI)
Run:

```powershell
python .\examples\WSI_analysis\slide_analysis.py
```

This script loads `fn = '58_HER2.ndpi'`, creates a thumbnail, extracts patches, and filters patches by OD. Ensure the `.ndpi` file is available where the script expects it.

### B) Nuclei-only batch segmentation
Run:

```powershell
python .\example2\Batch_Nuclei_Segmentation\batch_nuclei_seg.py
```

The script creates an output folder `tpo = tp + '_nseg/'` and writes per-image nuclei annotations (TXT + GeoJSON) and visualization overlays.

### C) Nuclei + Membrane + Bioinformatics (adaptive)
Run:

```powershell
python .\example2\Batch_Nuclei_Mem_Seg\batch_nm_seg.py
```

This runs an adaptive pipeline based on DAB OD, writes intermediate nuclei/membrane outputs, and produces:
- `*_cell_table.csv`
- `batch_bioinfo_summary.csv` (after the batch finishes)


