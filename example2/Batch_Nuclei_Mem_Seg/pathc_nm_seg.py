
# #Importing image analysis llibraries
# import cv2
# from skimage import data, io
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt

# sys.path.append('/mnt/c/Users/admin/OneDrive/Desktop/Digital_Pathology/backend/qcs/')

# #from dqcs import dpixel
# from dqcs.dseg import cellpose_seg, get_hd_clean, get_hed, membrane_seg, squeeze_nuclei_annotations, plot_aoutline, squeeze_aoutline, patch_bioinformatics
# from dqcs.danalysis import show_annotations_txt, read_annotations_txt, plot_annotations, filter_nuclei_annotations, alter_anno, write_annotations, filter_membrane_annotations
# from skimage import exposure


# img = 'her2-0-score_train_8117.png'#'her2-2+-score_train_871.png'
# path = '../images/0/' #'../images/2/'

# #img = 'patch_l0_x4096_y21504.png'#'her2-2+-score_train_871.png'
# #path = '../images/spatches/' #'../images/2/'

# tpo = 'check/'
# img_n = path+img
# img_rgb    = io.imread(img_n)



# #Step1. Stain Deconvolution
# ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')

# ihc_hm2, _, ihc_dm2 = get_hed(img_n, h_file='h.png', d_file='d.png')

# print(img_n, np.min(ihc_hm), np.max(ihc_hm))


# ###Step2. Enhance nuclei intensity for proper identification
# #ihc_hm_enhanced = ihc_hm 
# ihc_hm_enhanced = exposure.adjust_gamma(ihc_hm , gamma=3.5)
# ihc_hm2_enhanced = exposure.adjust_gamma(ihc_hm2 , gamma=3.5)


 
# #Step3. Nuclei segmentation using cellpose
# outlines_h = cellpose_seg(ihc_hm_enhanced, diameter=80, anno_fig=tpo+img[:-4]+'_hanno.png', anno_file=tpo+img[:-4]+'_n.txt', anno_geojson=tpo+img[:-4]+'_n.geojson', center_file=tpo+img[:-4]+'_ncenter.txt')
# ##outlines_h is list of numpy arrays having coordinates of size Mx2

# #Step4. Membrane segmentation using cellpose
# outlines_hm2 = cellpose_seg(ihc_hm2_enhanced, diameter=100, anno_fig=tpo+img[:-4]+'_m_raw.png', anno_file=tpo+img[:-4]+'_m.txt', anno_geojson=tpo+img[:-4]+'_m.geojson', center_file=tpo+img[:-4]+'_mcenter.txt', is_membrane=True)
# ##outlines_h is list of numpy arrays having coordinates of size Mx2




# ##To clean nuclei outline/annotations based on shape and size of a proper nucleus
# outlines_h_clean = filter_nuclei_annotations(outlines_h,Cpx2um=0.25)
# outlines_hm2_clean = filter_nuclei_annotations(outlines_hm2,Cpx2um=0.25)

# outlines_h_clean_squeeze = squeeze_nuclei_annotations(outlines_h_clean, ihc_hm_enhanced, niter_max=10)

# outlines_hm2_clean = filter_membrane_annotations(outlines_hm2_clean, outlines_h_clean, cdist_threshold=10)


# write_annotations(outlines_h_clean, anno_file='new_nannotation.txt',anno_geojson='annotation.geojson', center_file='centers.txt')
# write_annotations(outlines_h_clean_squeeze, anno_file='sq_new_nannotation.txt',anno_geojson='sq_annotation.geojson', center_file='sq_centers.txt')
# write_annotations(outlines_hm2_clean, anno_file='m_nannotation.txt',anno_geojson='m_annotation.geojson', center_file='m_centers.txt')

 
# ###To only show nuclei annotations
# ###show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno=tpo+img[:-4]+'_n.txt', mem_anno='new_nannotation.txt')
# show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno = 'new_nannotation.txt')#, mem_anno='new_nannotation.txt')
# #show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno1.png', nuclei_anno = 'new_nannotation.txt', mem_anno='sq_new_nannotation.txt')
# show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_manno.png', nuclei_anno = 'sq_new_nannotation.txt', mem_anno='m_nannotation.txt')


# Nnuclei, mem_od_selected, mem_od_all, cyto_od, nuc_od, NMR = patch_bioinformatics(outlines_h_clean_squeeze, outlines_hm2_clean, img_rgb)

# print(f'Total nuclei in this patch = {Nnuclei} \nMem OD selected = {mem_od_selected}, Mem_OD_ALL = {mem_od_all}, Nuclei Mem OD = {nuc_od} Cyto_od = {cyto_od} \n ')
# print(f'  NMR = {round(NMR,3)},  % of Positive Tumor Cells {round(mem_od_selected/Nnuclei,1)}')

 
#Importing image analysis llibraries
import cv2
from skimage import data, io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/mnt/c/Users/abhishekd_dizzaroo/Desktop/qcs')
from dqcs.dseg import cellpose_seg, get_hd_clean, get_hed, membrane_seg, squeeze_nuclei_annotations, plot_aoutline, squeeze_aoutline, patch_bioinformatics, patch_bioinformatics_v2, get_OD_single_channel 
from dqcs.danalysis import (show_annotations_txt, read_annotations_txt, plot_annotations,
    filter_nuclei_annotations, alter_anno, write_annotations,
    filter_membrane_annotations, filter_membrane_by_shape, expand_outlines,
    filter_membrane_by_dab_intensity, refine_outlines_by_dab_points)
from skimage import exposure


img = 'her2-1+-score_test_202.png'#'her2-2+-score_train_871.png'
path = '../images/1/' #'../images/2/'

#img = 'patch_l0_x4096_y21504.png'#'her2-2+-score_tain_871.png'
#path = '../images/spatches/' #'../images/2/'

tpo = 'check_6/'
os.makedirs(tpo, exist_ok=True)
img_n = path+img
img_rgb    = io.imread(img_n)



#Step1. Stain Deconvolution
ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')

ihc_hm2, _, ihc_dm2 = get_hed(img_n, h_file='h.png', d_file='d.png')

# ── Compute DAB OD on RAW (pre-gamma) DAB channel FIRST ──────────────
# This is used for: (1) staining strength classification, (2) intensity filtering
dab_od, dab_od_avg = get_OD_single_channel(ihc_dm2)
print(f"  DAB OD stats (pre-gamma): avg={dab_od_avg:.1f}, "
      f"min={np.min(dab_od)}, max={np.max(dab_od)}")

# ── Classify staining strength using DAB OD (not H-channel min) ──────
# Strong DAB (3+): high average OD.  Weak DAB (1+/2+): low average OD.
h_min, h_max = int(np.min(ihc_hm)), int(np.max(ihc_hm))
is_strong = dab_od_avg > 10  # DAB OD avg > 10 indicates strong (3+) staining
print(f"{img_n}  H range: {h_min} – {h_max}, DAB OD avg: {dab_od_avg:.1f}")
print(f"  ➜ {'3+ (strong)' if is_strong else '1+/2+ (weak/moderate)'} staining detected")

## ── COMMON: Enhance & Segment ─────────────────────────────────
##   gamma=3.5 on H channel works for nuclei detection on ALL images.
##   For DAB: 3.5 on strong staining, 2.0 on weak staining.
gamma_h = 3.5                         # always — good for H-channel nuclei
gamma_d = 3.5 if is_strong else 2.0   # gentle on weak DAB

ihc_hm_enhanced  = exposure.adjust_gamma(ihc_hm,  gamma=gamma_h)
ihc_dm2_enhanced = exposure.adjust_gamma(ihc_dm2, gamma=gamma_d)

# Step 3: Nuclei segmentation (H channel) — same params for all
outlines_h = cellpose_seg(
    ihc_hm_enhanced,
    diameter=60,
    anno_fig=tpo+img[:-4]+'_hanno.png',
    anno_file=tpo+img[:-4]+'_n.txt',
    anno_geojson=tpo+img[:-4]+'_n.geojson',
    center_file=tpo+img[:-4]+'_ncenter.txt'
)

# Step 4: Membrane segmentation (DAB channel)
mem_diameter = 120 if is_strong else 80
outlines_hm2 = cellpose_seg(
    ihc_dm2_enhanced,
    diameter=mem_diameter,
    anno_fig=tpo+img[:-4]+'_m_raw.png',
    anno_file=tpo+img[:-4]+'_m.txt',
    anno_geojson=tpo+img[:-4]+'_m.geojson',
    center_file=tpo+img[:-4]+'_mcenter.txt',
    is_membrane=True
)

## ── ADAPTIVE FLOW ──────────────────────────────────────────────
if is_strong:
    ## ── 3+ : MEMBRANE-PRIMARY (DAB is reliable) ────────────────
    outlines_hm2_filtered = filter_membrane_by_shape(
        outlines_hm2, min_area=2000, min_height=30
    )
    outlines_hm2_clean, outlines_h_clean = filter_membrane_annotations(
        outlines_hm2_filtered, outlines_h,
        cdist_threshold=40, keep_hollow=True,
    )
    outlines_h_clean_squeeze = squeeze_nuclei_annotations(
        outlines_h_clean, ihc_hm_enhanced, niter_max=10
    )
else:
    ## ── 1+/2+ : NUCLEUS-PRIMARY (DAB is weak/fragmented) ───────
    ##   For weak DAB, Cellpose membrane detection is unreliable (broken,
    ##   overlapping). Instead, expand each nucleus outline × 1.3 to create
    ##   smooth continuous membrane rings. DAB OD within the ring is computed
    ##   during bioinformatics.
    outlines_h_clean = filter_nuclei_annotations(
        outlines_h, Cpx2um=0.25,
        min_area=1500, min_height=30   # relaxed for 1+/2+
    )

    # Expand all nuclei outlines to create smooth synthetic membrane contours
    outlines_hm2_clean = expand_outlines(outlines_h_clean, expand_factor=1.3)
    print(f"Nucleus-primary: {len(outlines_h_clean)} cells, "
          f"all membranes expanded ×1.3 (DAB too weak for Cellpose detection)")

    # Squeeze with fewer iterations (less aggressive on weaker staining)
    outlines_h_clean_squeeze = squeeze_nuclei_annotations(
        outlines_h_clean, ihc_hm_enhanced, niter_max=5
    )

write_annotations(
    outlines_h_clean,
    anno_file='new_nannotation.txt',
    anno_geojson='annotation.geojson',
    center_file='centers.txt'
)
write_annotations(
    outlines_h_clean_squeeze,
    anno_file='sq_new_nannotation.txt',
    anno_geojson='sq_annotation.geojson',
    center_file='sq_centers.txt'
)
write_annotations(
    outlines_hm2_clean,
    anno_file='m_nannotation.txt',
    anno_geojson='m_annotation.geojson',
    center_file='m_centers.txt'
)

 
###To only show nuclei annotations
###show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno=tpo+img[:-4]+'_n.txt', mem_anno='new_nannotation.txt')
show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno = 'new_nannotation.txt')#, mem_anno='new_nannotation.txt')
#show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno1.png', nuclei_anno = 'new_nannotation.txt', mem_anno='sq_new_nannotation.txt')
show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_manno.png', nuclei_anno = 'sq_new_nannotation.txt', mem_anno='m_nannotation.txt')


# ============================================================
# NEW BIOINFORMATICS ANALYSIS (Micron-based with CSV output)
# ============================================================
print('\n' + '='*60)
print('Running NEW bioinformatics analysis (micron-based)...')
print('='*60)

# Conversion factor: 1 pixel = 0.25 microns
cpx2um = 0.25

# Run new bioinformatics analysis
bioinformatics_results = patch_bioinformatics_v2(
    outlines_h_clean_squeeze, 
    outlines_hm2_clean, 
    img_rgb,
    cpx2um=cpx2um
)

# Extract results
cell_table = bioinformatics_results['cell_table']
metrics = bioinformatics_results['metrics']
tile_area_mm2 = bioinformatics_results['tile_area_mm2']
num_valid_cells = bioinformatics_results['num_valid_cells']
num_total_cells = bioinformatics_results['num_total_cells']

# Save cell table to CSV
csv_output_file = tpo + img[:-4] + '_cell_table.csv'
cell_table.to_csv(csv_output_file, index=False, float_format='%.3f')
print(f'\n✅ Cell table saved to: {csv_output_file}')
print(f'   Total cells: {num_total_cells}')
print(f'   Valid cells (20-250 µm²): {num_valid_cells}')
print(f'   Tile area: {tile_area_mm2:.6f} mm²')
print(f'   Conversion factor: {cpx2um} px/µm')

# Print metrics
print('\n' + '-'*60)
print('BIOINFORMATICS METRICS:')
print('-'*60)
print(f'  A. Membrane OD Q15 (Baseline):        {metrics["q15"]:.3f} OD')
print(f'  B. % Positive Cells (OD≥10):          {metrics["percent_positive_od10"]:.2f}%')
print(f'  C. Density High-Expressors (OD≥25):     {metrics["density_high_expressors"]:.2f} cells/mm²')
print(f'  D. bSPS Score (r=25µm, OD≥60):         {metrics["bsps_score"]:.2f}%')
print(f'  E. cSPS Mean (r=50µm):                {metrics["csps_mean"]:.3f} OD')
print('-'*60)

# ============================================================
# OLD BIOINFORMATICS ANALYSIS (for comparison)
# ============================================================
print('\n' + '='*60)
print('OLD bioinformatics analysis (for comparison)...')
print('='*60)

Nnuclei, mem_od_selected, mem_od_all, cyto_od, nuc_od, NMR = patch_bioinformatics(outlines_h_clean_squeeze, outlines_hm2_clean, img_rgb)

print(f'Total nuclei in this patch = {Nnuclei}')
print(f'Mem OD selected = {mem_od_selected:.3f}, Mem_OD_ALL = {mem_od_all:.3f}')
print(f'Nuclei Mem OD = {nuc_od:.3f}, Cyto_od = {cyto_od:.3f}')
print(f'NMR = {round(NMR,3)},  % of Positive Tumor Cells {round(mem_od_selected/Nnuclei,1) if Nnuclei > 0 else 0:.1f}%')
print('='*60 + '\n')

 