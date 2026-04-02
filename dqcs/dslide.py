import numpy as np
import openslide
from PIL import Image # For handling image data
from skimage import io
import os
import shutil

from dqcs.dseg import get_OD, get_hed
#=============================================================================
class Slide:
    def __init__(self, slide_file):
        self.slide_file = slide_file
        # Open the slide file
        try:
            self.slide = openslide.OpenSlide(self.slide_file)
        except openslide.OpenSlideError as e:
            print(f"Error opening slide: {e}")
            return
        
        if(self.slide):self.print_slide_info()
    #
    
    def print_slide_info(self):
        if(self.slide_file[-4:]=='ndpi'):
            print(f"Details of histopathology slide {self.slide_file} in ndpi format are following:  ")
            print(f"Vendor: {self.slide.properties.get(openslide.PROPERTY_NAME_VENDOR)}")
            print(f"Objective Power: {self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)}")
            print(f"Dimensions at Level 0 (highest resolution): {self.slide.dimensions}")
            print(f"Number of Levels: {self.slide.level_count}")
            print(f"1 pixel = {self.slide.properties[openslide.PROPERTY_NAME_MPP_X]} micron in x direction")
            print(f"1 pixel = {self.slide.properties[openslide.PROPERTY_NAME_MPP_Y]} micron in y direction")
            print('\n')
            with self.slide as ss:
                for key, value in ss.properties.items():
                    print(f"  {key}: {value}")
                    
        if(self.slide_file[-4:]=='svs'):
            pass
        
        if(self.slide_file[-4:]=='tiff'):
            pass
        
    #
    
    def get_thumbnail(self, size=(300,300), fname="thumbnail.png"):
        thumbnail = self.slide.get_thumbnail(size) # size = (x,y) is maximum size of the thumbnai thumbnail, get_thumbnail method returns a PIL Image object
        thumbnail.save(fname)
        print(f"Thumbnail saved as {fname}")
    #
    
    def extract_patches_from_ndpi(self, output_dir='patches', patch_size=512, level=0):
        """
        Reads an NDPI file and extracts patches at a specified resolution level.
            output_dir (str): The directory to save the extracted patches.
            patch_size (int): The width and height of the square patches.
            level (int): The resolution level to read from (0 is the highest).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Get the dimensions of the selected level
        dimensions = self.slide.level_dimensions[level]
        width, height = dimensions
    
        print(f"Reading from resolution level {level} with dimensions: {width}x{height}")
    
        # Calculate the number of patches to extract
        num_patches_x = width // patch_size
        num_patches_y = height // patch_size
    
        # Extract and save patches
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # Calculate the top-left coordinate for the patch
                x_coord = x * patch_size
                y_coord = y * patch_size
                
                try:
                    # Read the region corresponding to the patch
                    patch = self.slide.read_region((x_coord, y_coord), level, (patch_size, patch_size))
                    
                    # Convert the patch to RGB (it's often RGBA)
                    rgb_patch = patch.convert("RGB")
                    
                    # Create a unique filename for each patch
                    filename = f"patch_l{level}_x{x_coord}_y{y_coord}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the patch as an image file
                    rgb_patch.save(filepath)
                    
                    print(f"Saved {filepath}")
    
                except openslide.OpenSlideError as e:
                    print(f"Error extracting patch at ({x_coord}, {y_coord}): {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
        print("Finished extracting patches.")
        # Close the slide object
        #self.slide.close()
    #
#=============================================================================  
   
def filter_patches(patch_dir, filter_dir):
    #This function will filter patches based on OD
    #patch_dir: Directory containing patches
    #filter_dir: Directory where filtered patches will be kept
    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)

    patches = os.listdir(patch_dir)
    print(f'Analyzing {len(patches)} patches')
    for p in patches:
        img_rgb = io.imread(patch_dir+'/'+p)
        h,e,d = get_hed(img_rgb, h_file=None, e_file=None, d_file=None)
        OD, av_OD = get_OD(h)
        if(av_OD > 1): #@@@ Here the threshold OD of 1 was chosen by analysing few patches, this might need attention in future @@@
            # Copy the file 
            shutil.copyfile(patch_dir+'/'+p, filter_dir+'/' + p)
            print(patches.index(p), p, ' patch with OD of ', round(av_OD,1), ' is selected')
         

#>>>To write
def get_patches_clam():
    #This function will make patches from slide using CLAM package
    pass
 
def stain_normalization_patch():
    #This function will normalize the color and stain of a patch
    pass
 
def normalize_wsi():
    #This function will normalize the color and stain of a patch
    pass
    


#@@@Remove following, it was just a trial function  
def open_slide_ndpi(slide_file):
    #Input:
        #slide_file: name of slide with path (e.g. "path/to/your/image.ndpi")
    try:
        # Open the NDPI slide
        slide = openslide.OpenSlide(slide_file)
    
        # Read a region at a specific level
        # level: zoom level (0 is highest resolution)
        # location: (x, y) coordinates of the top-left corner of the region
        # size: (width, height) of the region to read
        level = 0
        location = (0, 0)
        size = (1024, 768)
        region_image = slide.read_region(location, level, size)
        region_image.save("region_at_level_0.png")
        print("Region at level 0 saved as region_at_level_0.png")
    
        # Close the slide object when done
        slide.close()
    
    except openslide.OpenSlideError as e:
        print(f"Error opening or reading NDPI file: {e}")
    except FileNotFoundError:
        print(f"Error: NDPI file not found at {slide_file}")
