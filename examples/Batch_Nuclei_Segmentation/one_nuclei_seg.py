
#Importing image analysis llibraries
import cv2
from skimage import data, io
import os
import sys
import numpy as np

sys.path.append('/mnt/c/Users/admin/OneDrive/Desktop/Digital_Pathology/backend/qcs/')

#from dqcs import dpixel
from dqcs.dseg import cellpose_seg, get_hd_clean, get_hed, membrane_seg
from dqcs.danalysis import show_annotations_txt, read_annotations_txt, plot_annotations, filter_nuclei_annotations, alter_anno, write_annotations
from skimage import exposure


img = 'her2-2+-score_test_170.png'#'her2-2+-score_train_871.png'
path = '../images/2/' #'../images/2/'

#img = 'patch_l0_x4096_y21504.png'#'her2-2+-score_train_871.png'
#path = '../images/spatches/' #'../images/2/'

tpo = 'check/'
img_n = path+img
img_rgb    = io.imread(img_n)



#Step1. Stain Deconvolution
ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')
ihc_hm, _, ihc_dm = get_hed(img_n, h_file='h.png', d_file='d.png')

print(img_n, np.min(ihc_hm), np.max(ihc_hm))


###Step2. Enhance nuclei intensity for proper identification
#ihc_hm_enhanced = ihc_hm 
ihc_hm_enhanced = exposure.adjust_gamma(ihc_hm , gamma=1.1)
##ihc_hm_enhanced = histo_eq(ihc_hm) #gamma_correction(ihc_hm)
##ihc_hm_enhanced = exposure.equalize_adapthist(ihc_hm, clip_limit=0.03)
#ihc_hm_enhanced = exposure.equalize_hist(ihc_hm)


 
#Step3. Nuclei segmentation using cellpose
outlines_h = cellpose_seg(ihc_hm_enhanced, diameter=80, anno_fig=tpo+img[:-4]+'_hanno.png', anno_file=tpo+img[:-4]+'_n.txt', anno_geojson=tpo+img[:-4]+'_n.geojson', center_file=tpo+img[:-4]+'_ncenter.txt')
##outlines_h is list of numpy arrays having coordinates of size Mx2


#outlines_h_squeeze = squeeze_nuclei_annotations(outlines_h)

##To clean nuclei outline/annotations based on shape and size of a proper nucleus
outlines_h_clean = filter_nuclei_annotations(outlines_h,Cpx2um=0.25)


write_annotations(outlines_h_clean, anno_file='new_nannotation.txt',anno_geojson='annotation.geojson', center_file='centers.txt')
    
##To only show nuclei annotations
#show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno=tpo+img[:-4]+'_n.txt', mem_anno='new_nannotation.txt')
show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno = 'new_nannotation.txt')#, mem_anno='new_nannotation.txt')

