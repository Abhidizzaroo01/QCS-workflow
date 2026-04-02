
#Importing image analysis llibraries
import cv2
from skimage import data, io
import os
import sys
import numpy as np

sys.path.append('/home/aktsh/codes/')

#from dqcs import dpixel
from dqcs.dseg import cellpose_seg, get_hd_clean, get_hed, membrane_seg #, clean_nuclei_annotations
from dqcs.danalysis import show_annotations_txt, read_annotations_txt, plot_annotations, filter_nuclei_annotations, alter_anno, write_annotations, squeeze_nuclei_annotations
from skimage import exposure


path = '../images/'
tp = '0'
tpo = tp+'_nseg/'

if(not os.path.exists(tpo)):os.mkdir(tpo)
imn = os.listdir(path+tp)

# Initialize Cellpose model 
for i in range(len(imn)):   
    #imn = f'{tps}/{imn[i][:-4]}_hm_contrast.png' #
    img_n = path+tp+'/'+imn[i] 
    img_rgb    = io.imread(img_n)
    
    #Stain Deconvolution
    ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')
    
    print(img_n, np.min(ihc_hm), np.max(ihc_hm))
    
    ##Enhance nuclei intensity for proper identification
    #ihc_hm_enhanced = ihc_hm 
    ihc_hm_enhanced = exposure.adjust_gamma(ihc_hm , gamma=3.5)
    #ihc_hm_enhanced = histo_eq(ihc_hm) #gamma_correction(ihc_hm)
    
    #Nuclei segmentation using cellpose
    outlines_h = cellpose_seg(ihc_hm_enhanced, diameter=80, anno_fig=tpo+imn[i][:-4]+'_hanno.png', anno_file=tpo+imn[i][:-4]+'_n.txt', anno_geojson=tpo+imn[i][:-4]+'_n.geojson', center_file=tpo+imn[i][:-4]+'_ncenter.txt')
    ##outlines_h is list of numpy arrays having coordinates of size Mx2
    
    
    ##To clean nuclei outline/annotations based on shape and size of a proper nucleus
    outlines_h_clean = filter_nuclei_annotations(outlines_h,Cpx2um=0.25)
    
    write_annotations(outlines_h_clean, anno_file=tpo+imn[i][:-4]+'_nf.txt',anno_geojson=tpo+imn[i][:-4]+'_nf.geojson', center_file=tpo+imn[i][:-4]+'_nfcenter.txt')
    
    ##To only show nuclei annotations
    #show_annotations_txt(in_img_file = img_n, out_img_file=tpo+imn[i][:-4]+'_nanno.png', nuclei_anno=tpo+imn[i][:-4]+'_n.txt')#, mem_anno='out_cp_d.txt')
    show_annotations_txt(in_img_file = img_n, out_img_file=tpo+imn[i][:-4]+'_nanno.png', nuclei_anno=tpo+imn[i][:-4]+'_nf.txt')#, mem_anno='new_nannotation.txt')
    
     
    
    
    
    
    
    
    #_,_,_ = get_hed(img_n, h_file='h.png', d_file='d.png')
          
    
    #
    #outlines_d = cellpose_seg(ihc_dm, anno_fig='out_d.png', anno_file='out_cp_d.txt', anno_geojson='out_d.geojson', center_file='out_center_d.txt')
    #print(f'Number of nuclei are {len(outlines_h)}')
    #print(outlines_h, type(outlines_h),type(outlines_h[0]), outlines_h[0].shape)
        
    
    #------------------------------------------------------------------------
    
    
    ##To sho both nuclei and membrane annotations
    #show_annotations_txt(in_img_file = img_n, out_img_file='out_anno.png', nuclei_anno='out_cp_h.txt', mem_anno='out_cp_d.txt')
    #------------------------------------------------------------------------
    
    '''
    noutlines_coord, nc_coord = read_annotations_txt(file_txt = 'out_cp_h.txt', plot_anno=False, color='red', show_center=False, show_id=False)
    outlines_coord_new = alter_anno(noutlines_coord, nc_coord, alter_fac=1.3, plot_anno=False)
    
    plot_annotations(in_img_file = img_n, out_img_file='out_anno_alter.png', nuclei_anno=noutlines_coord, mem_anno=outlines_coord_new)
    
    
    membrane_seg(img_rgb, nuclei_anno_txt='out_cp_h.txt', anno_fig='mout.png', anno_file='mout.txt', anno_geojson='mout.geojson', center_file='mout_center.txt')
    '''