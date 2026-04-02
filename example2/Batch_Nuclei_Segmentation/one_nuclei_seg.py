
#Importing image analysis llibraries
import cv2
from skimage import data, io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/aktsh/codes/')

#from dqcs import dpixel
from dqcs.dseg import cellpose_seg, get_hd_clean, get_hed, membrane_seg, squeeze_nuclei_annotations, plot_aoutline, squeeze_aoutline
from dqcs.danalysis import show_annotations_txt, read_annotations_txt, plot_annotations, filter_nuclei_annotations, alter_anno, write_annotations
from skimage import exposure


img = 'her2-3+-score_test_448.png'#'her2-2+-score_train_871.png'
path = '../images/3/' #'../images/2/'

#img = 'patch_l0_x4096_y21504.png'#'her2-2+-score_train_871.png'
#path = '../images/spatches/' #'../images/2/'

tpo = 'check/'
img_n = path+img
img_rgb    = io.imread(img_n)



#Step1. Stain Deconvolution
ihc_hm, ihc_dm = get_hd_clean(img_n, h_file='h.png', d_file='d.png')
#ihc_hm, _, ihc_dm = get_hed(img_n, h_file='h.png', d_file='d.png')

print(img_n, np.min(ihc_hm), np.max(ihc_hm))


###Step2. Enhance nuclei intensity for proper identification
#ihc_hm_enhanced = ihc_hm 
ihc_hm_enhanced = exposure.adjust_gamma(ihc_hm , gamma=3.5)
##ihc_hm_enhanced = histo_eq(ihc_hm) #gamma_correction(ihc_hm)
##ihc_hm_enhanced = exposure.equalize_adapthist(ihc_hm, clip_limit=0.03)
#ihc_hm_enhanced = exposure.equalize_hist(ihc_hm)


 
#Step3. Nuclei segmentation using cellpose
outlines_h = cellpose_seg(ihc_hm_enhanced, diameter=80, anno_fig=tpo+img[:-4]+'_hanno.png', anno_file=tpo+img[:-4]+'_n.txt', anno_geojson=tpo+img[:-4]+'_n.geojson', center_file=tpo+img[:-4]+'_ncenter.txt')
##outlines_h is list of numpy arrays having coordinates of size Mx2


##outlines_h_squeeze = squeeze_nuclei_annotations(outlines_h, ihc_hm_enhanced)

##To clean nuclei outline/annotations based on shape and size of a proper nucleus
outlines_h_clean = filter_nuclei_annotations(outlines_h,Cpx2um=0.25)

##outlines_h_squeeze_clean = filter_nuclei_annotations(outlines_h_squeeze=0.25)
outlines_h_clean_squeeze = squeeze_nuclei_annotations(outlines_h_clean, ihc_hm_enhanced, niter_max=10)


write_annotations(outlines_h_clean, anno_file='new_nannotation.txt',anno_geojson='annotation.geojson', center_file='centers.txt')
write_annotations(outlines_h_clean_squeeze, anno_file='sq_new_nannotation.txt',anno_geojson='sq_annotation.geojson', center_file='sq_centers.txt')
 
##To only show nuclei annotations
#show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno=tpo+img[:-4]+'_n.txt', mem_anno='new_nannotation.txt')
show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno.png', nuclei_anno = 'new_nannotation.txt')#, mem_anno='new_nannotation.txt')
show_annotations_txt(in_img_file = img_n, out_img_file=tpo+img[:-4]+'_nanno1.png', nuclei_anno = 'new_nannotation.txt', mem_anno='sq_new_nannotation.txt')




'''
plt.figure()
plt.axis('off')
plt.imshow(ihc_hm_enhanced) 

plot_aoutline(outlines_h_clean[0], color='green')
oo = squeeze_aoutline(outlines_h_clean[0], ihc_hm_enhanced)
plot_aoutline(oo, color='orange')

#plt.xlim([870, 963])
#plt.ylim([13, 80])

plt.tight_layout()
plt.savefig('check_squeeze.png', dpi=300)
plt.show()
#plt.close()

'''


'''    
#####
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# ---- Load image ----

img_smooth = gaussian(ihc_hm_enhanced, 3.0)   # smooth to reduce noise

# ---- Your approximate polygon (Nx2 array) ----
# Example: replace with your real coordinates
approx_contour = outlines_h_clean[0]

# ---- Apply Active Contour (Snake) ----
snake = active_contour(
    img_smooth,
    approx_contour,
    alpha=0.015,   # elasticity (lower = more flexible)
    beta=100,       # smoothness (higher = smoother curve)
    gamma=0.001,   # step size
    #w_line=-1,      # intensity attraction
    #w_edge=-1,      # edge attraction
    #max_num_iter=2500
)

# ---- Visualize result ----
fig, ax = plt.subplots()
ax.imshow(img_smooth, cmap='gray')
ax.plot(approx_contour[:, 0], approx_contour[:, 1], '--r', lw=1.5, label='Initial')
ax.plot(snake[:, 0], snake[:, 1], '-g', lw=2, label='Refined')
ax.legend()
plt.show()
'''
 