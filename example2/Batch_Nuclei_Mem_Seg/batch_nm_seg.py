
#Importing image analysis llibraries
import cv2
from skimage import data, io
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/mnt/c/Users/abhishekd_dizzaroo/Desktop/qcs')

#from dqcs import dpixel
from dqcs.dseg import (
    cellpose_seg,
    get_hd_clean,
    get_hed,
    membrane_seg,
    squeeze_nuclei_annotations,
    plot_aoutline,
    squeeze_aoutline,
    patch_bioinformatics,
    patch_bioinformatics_v2,
    get_OD_single_channel,
)
from dqcs.danalysis import (
    show_annotations_txt,
    read_annotations_txt,
    plot_annotations,
    filter_nuclei_annotations,
    alter_anno,
    write_annotations,
    filter_membrane_annotations,
    filter_membrane_by_shape,
    expand_outlines,
    filter_membrane_by_dab_intensity,
    refine_outlines_by_dab_points,
)
from skimage import exposure

path = '../images/'
tp = '3'
tpo = tp+'_nseg/'

if(not os.path.exists(tpo)):os.mkdir(tpo)
imn = os.listdir(path+tp)

# to store per-image summary metrics for the whole batch
batch_metrics = []

# Initialize Cellpose model 
for i in range(min(10, len(imn))):   
    #imn = f'{tps}/{imn[i][:-4]}_hm_contrast.png' #
    img_n = path+tp+'/'+imn[i] 
    img_rgb    = io.imread(img_n)
    
    #===================  
    #Step1. Stain Deconvolution
    ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')
    ihc_hm2, _, ihc_dm2 = get_hed(img_n, h_file='h.png', d_file='d.png')

    # ── Compute DAB OD on RAW (pre-gamma) DAB channel FIRST ──
    dab_od, dab_od_avg = get_OD_single_channel(ihc_dm2)
    print(f"  DAB OD stats (pre-gamma): avg={dab_od_avg:.1f}, "
          f"min={np.min(dab_od)}, max={np.max(dab_od)}")

    # ── Classify staining strength using DAB OD ──
    h_min, h_max = int(np.min(ihc_hm)), int(np.max(ihc_hm))
    is_strong = dab_od_avg > 10  # DAB OD avg > 10 indicates strong (3+) staining
    print(f"{img_n}  H range: {h_min} – {h_max}, DAB OD avg: {dab_od_avg:.1f}")
    print(f"  ➜ {'3+ (strong)' if is_strong else '1+/2+ (weak/moderate)'} staining detected")

    ## ── COMMON: Enhance & Segment ────────────────────────────
    gamma_h = 3.5
    gamma_d = 3.5 if is_strong else 2.0
    ihc_hm_enhanced  = exposure.adjust_gamma(ihc_hm,  gamma=gamma_h)
    ihc_dm2_enhanced = exposure.adjust_gamma(ihc_dm2, gamma=gamma_d)

    # Step 3: Nuclei segmentation (H channel)
    outlines_h = cellpose_seg(
        ihc_hm_enhanced,
        diameter=60,
        anno_fig=tpo+imn[i][:-4]+'_hanno.png',
        anno_file=tpo+imn[i][:-4]+'_n.txt',
        anno_geojson=tpo+imn[i][:-4]+'_n.geojson',
        center_file=tpo+imn[i][:-4]+'_ncenter.txt',
    )

    # Step 4: Membrane segmentation (DAB channel)
    mem_diameter = 120 if is_strong else 80
    outlines_hm2 = cellpose_seg(
        ihc_dm2_enhanced,
        diameter=mem_diameter,
        anno_fig=tpo+imn[i][:-4]+'_m_raw.png',
        anno_file=tpo+imn[i][:-4]+'_m.txt',
        anno_geojson=tpo+imn[i][:-4]+'_m.geojson',
        center_file=tpo+imn[i][:-4]+'_mcenter.txt',
        is_membrane=True,
    )

    ## ── ADAPTIVE FLOW ────────────────────────────────────────
    if is_strong:
        ## 3+: MEMBRANE-PRIMARY
        outlines_hm2_filtered = filter_membrane_by_shape(
            outlines_hm2, min_area=2000, min_height=30
        )
        outlines_hm2_clean, outlines_h_clean = filter_membrane_annotations(
            outlines_hm2_filtered, outlines_h,
            cdist_threshold=40, keep_hollow=True,
        )
        outlines_h_clean_squeeze = squeeze_nuclei_annotations(
            outlines_h_clean, ihc_hm_enhanced, niter_max=20,
        )
    else:
        ## 1+/2+: NUCLEUS-PRIMARY + DAB intensity filter + expand for membranes
        outlines_h_clean = filter_nuclei_annotations(
            outlines_h, Cpx2um=0.25, min_area=1500, min_height=30
        )

        # Step A: DAB intensity pre-filter — remove membrane outlines with low DAB OD
        outlines_hm2_dab_filtered = filter_membrane_by_dab_intensity(
            outlines_hm2, dab_od, min_mean_od=8, min_area=500
        )

        # Step B: Build paired membranes using only intensity-verified outlines
        outlines_hm2_clean = []
        n_real = 0
        n_synth = 0
        for nuc in outlines_h_clean:
            nuc_cx = float(np.mean(nuc[:, 0]))
            nuc_cy = float(np.mean(nuc[:, 1]))
            best_mem, best_area = None, 0
            for mem in outlines_hm2_dab_filtered:  # ← use intensity-filtered outlines
                mc = mem.reshape(-1, 1, 2).astype(np.float32)
                if cv2.pointPolygonTest(mc, (nuc_cx, nuc_cy), False) >= 0:
                    a = cv2.contourArea(mc)
                    if a > best_area:
                        best_area = a
                        best_mem = mem
            if best_mem is not None:
                outlines_hm2_clean.append(best_mem)
                n_real += 1
            else:
                outlines_hm2_clean.append(
                    expand_outlines([nuc], expand_factor=1.3)[0]
                )
                n_synth += 1
        print(f"Nucleus-primary: {len(outlines_h_clean)} cells "
              f"({n_real} DAB membrane, {n_synth} expanded)")

        # Step C: Refine membrane outlines — keep only contour points with high DAB OD
        outlines_hm2_clean = refine_outlines_by_dab_points(
            outlines_hm2_clean, dab_od, min_point_od=5, min_surviving_points=10
        )

        outlines_h_clean_squeeze = squeeze_nuclei_annotations(
            outlines_h_clean[:len(outlines_hm2_clean)], ihc_hm_enhanced, niter_max=5,
        )

    # Write cleaned/squeezed nuclei annotations and paired membranes
    write_annotations(
        outlines_h_clean,
        anno_file=tpo+imn[i][:-4]+'_nf.txt',
        anno_geojson=tpo+imn[i][:-4]+'_nf.geojson',
        center_file=tpo+imn[i][:-4]+'_nfcenter.txt'
    )
    write_annotations(
        outlines_h_clean_squeeze,
        anno_file=tpo+imn[i][:-4]+'_nsf.txt',
        anno_geojson=tpo+imn[i][:-4]+'_nsf.geojson',
        center_file=tpo+imn[i][:-4]+'_nsfcenter.txt'
    )

    #Step4. Membrane segmentation using DAB-based radial rays from squeezed nuclei
    outlines_hm2, nuc_refined, centers_refined = membrane_seg(
        img_rgb,
        nuclei_anno_txt=tpo+imn[i][:-4]+'_nsf.txt',
        anno_fig=tpo+imn[i][:-4]+'_manno_dab.png',
        anno_file=tpo+imn[i][:-4]+'_mf.txt',
        anno_geojson=tpo+imn[i][:-4]+'_mf.geojson',
        center_file=tpo+imn[i][:-4]+'_mfcenter.txt',
        rays_debug_fig=tpo+imn[i][:-4]+'_m_rays_debug.png',
    )
    ##outlines_hm2 is list of numpy arrays having coordinates of size Mx2
       
     
    ###To show annotations

    show_annotations_txt(in_img_file = img_n, out_img_file=tpo+imn[i][:-4]+'_nanno.png', nuclei_anno=tpo+imn[i][:-4]+'_nf.txt')#, mem_anno='new_nannotation.txt')
    show_annotations_txt(in_img_file = img_n, out_img_file=tpo+imn[i][:-4]+'_nsanno.png', nuclei_anno=tpo+imn[i][:-4]+'_nf.txt', mem_anno=tpo+imn[i][:-4]+'_nsf.txt')
    show_annotations_txt(in_img_file = img_n, out_img_file=tpo+imn[i][:-4]+'_manno.png', nuclei_anno=tpo+imn[i][:-4]+'_nsf.txt', mem_anno=tpo+imn[i][:-4]+'_mf.txt')

    # ============================================================
    # NEW BIOINFORMATICS ANALYSIS (Micron-based with CSV output)
    # ============================================================
    print('\n' + '='*60)
    print(f'Running NEW bioinformatics analysis (micron-based) for {imn[i]}...')
    print('='*60)

    # Conversion factor: 1 pixel = 0.25 microns
    cpx2um = 0.25

    # Run new bioinformatics analysis
    bioinformatics_results = patch_bioinformatics_v2(
        outlines_h_clean_squeeze, 
        outlines_hm2_clean, 
        img_rgb,
        cpx2um=cpx2um,
    )

    # Extract results
    cell_table = bioinformatics_results['cell_table']
    metrics = bioinformatics_results['metrics']
    tile_area_mm2 = bioinformatics_results['tile_area_mm2']
    num_valid_cells = bioinformatics_results['num_valid_cells']
    num_total_cells = bioinformatics_results['num_total_cells']

    # Save cell table to CSV (per image)
    csv_output_file = tpo + imn[i][:-4] + '_cell_table.csv'
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
    print(f'  C. Density High-Expressors (OD≥25):   {metrics["density_high_expressors"]:.2f} cells/mm²')
    print(f'  D. bSPS Score (r=25µm, OD≥60):        {metrics["bsps_score"]:.2f}%')
    print(f'  E. cSPS Mean (r=50µm):                {metrics["csps_mean"]:.3f} OD')
    print('-'*60)

    # ============================================================
    # OLD BIOINFORMATICS ANALYSIS (for comparison)
    # ============================================================
    print('\n' + '='*60)
    print(f'OLD bioinformatics analysis (for comparison) for {imn[i]}...')
    print('='*60)

    Nnuclei, mem_od_selected, mem_od_all, cyto_od, nuc_od, NMR = patch_bioinformatics(
        outlines_h_clean_squeeze,
        outlines_hm2_clean,
        img_rgb,
    )

    print(f'Total nuclei in this patch = {Nnuclei}')
    print(f'Mem OD selected = {mem_od_selected:.3f}, Mem_OD_ALL = {mem_od_all:.3f}')
    print(f'Nuclei Mem OD = {nuc_od:.3f}, Cyto_od = {cyto_od:.3f}')
    print(
        f'NMR = {round(NMR,3)},  % of Positive Tumor Cells '
        f'{round(mem_od_selected/Nnuclei,1) if Nnuclei > 0 else 0:.1f}%'
    )
    print('='*60 + '\n')

    # store combined metrics for this image for batch-level CSV
    batch_metrics.append({
        "image_name": imn[i],
        "total_cells": num_total_cells,
        "valid_cells": num_valid_cells,
        "tile_area_mm2": tile_area_mm2,
        "q15": metrics["q15"],
        "percent_positive_od10": metrics["percent_positive_od10"],
        "density_high_expressors": metrics["density_high_expressors"],
        "bsps_score": metrics["bsps_score"],
        "csps_mean": metrics["csps_mean"],
        "Nnuclei_old": Nnuclei,
        "mem_od_selected_old": mem_od_selected,
        "mem_od_all_old": mem_od_all,
        "cyto_od_old": cyto_od,
        "nuc_od_old": nuc_od,
        "NMR_old": NMR,
    })

# after processing the batch, save combined metrics table
if batch_metrics:
    batch_df = pd.DataFrame(batch_metrics)
    batch_csv_path = os.path.join(tpo, 'batch_bioinfo_summary.csv')
    batch_df.to_csv(batch_csv_path, index=False, float_format='%.3f')
    print(f'\nBatch bioinformatics summary saved to: {batch_csv_path}\n')