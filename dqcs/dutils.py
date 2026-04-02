import numpy as np
from skimage import io

def img_check(img):    
    if isinstance(img, str): #Checking if it is a string
        img_rgb = io.imread(img) #cv2.imread(img_file)
    else: #Else assuming it is a numpy array having image 
        img_rgb = img
        
    return img_rgb