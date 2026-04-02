import sys

sys.path.append('/home/aktsh/codes/')

#from dqcs import dpixel
from dqcs.dslide import open_slide_ndpi, Slide, filter_patches

fn = '58_HER2.ndpi'

s = Slide(fn)
##To create tumbnail for the WSI
s.get_thumbnail(fname = 't1.png', size=(400,300))

##To extract patches from ndpi slide
s.extract_patches_from_ndpi()

##To filter patches from the set of given patches based on OD
filter_patches(patch_dir='patches', filter_dir='filter_patches_new')

 
#open_slide_ndpi('58_HER2.ndpi') 
