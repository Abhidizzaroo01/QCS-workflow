##This file contains general library to select pixels as per need

#Importing general llibraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import sys
import pickle
import requests

#Importing image analysis llibraries
from skimage import data, io
from skimage.color import rgb2hed, hed2rgb, rgb2gray, rgba2rgb
from skimage.color import  separate_stains, combine_stains, hdx_from_rgb, rgb_from_hdx
from skimage.util import img_as_ubyte
import skimage.color

##Importing cellpose libraries
#from cellpose import models, io, core, plot, utils
#from cellpose.io import imread

def pixel_OD(img,i,j):
    """ This function computes optical density (OD) at a pixel point (i,j) of the image img
    img: numpy array of imgae read using skimage.io
    i,j: pixel coordinates, must be integer values """
    
    r,g,b = img[i,j,:]
            
    dr = -np.log((r/255.) + 0.0001) ###Scaling factor to increase maximum value of d to 255
    dg = -np.log((g/255.) + 0.0001)
    db = -np.log((b/255.) + 0.0001)
    
    d = (dr+dg+db)/3.
    d = d*255./5.75 ###Scaling factor to increase maximum value of d to 255
    return d

def interpolate_point(A, B, r):
    """
    Returns the coordinates of a point at ratio r between points A and B.
    Parameters:
        A: tuple (x1, y1)
        B: tuple (x2, y2)
        r: float, between 0 and 1 (0=A, 1=B)
    """
    x = A[0] + r * (B[0] - A[0])
    y = A[1] + r * (B[1] - A[1])
    return [x, y]

def points_inside_polygon(image_shape, polygon_points):
    """
    Returns coordinates of all pixels inside a polygon.
    Parameters:
        image_shape : tuple
            Shape of the image (height, width).
        polygon_points : np.ndarray of shape (N, 2)
            Polygon vertices as (x, y) coordinates.
    Returns:
        points : np.ndarray of shape (M, 2)
            Pixel coordinates (x, y) inside the polygon.
    """
    # Create empty mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Fill polygon on the mask
    polygon = np.array([polygon_points], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)

    # Get indices of non-zero mask (points inside polygon)
    ys, xs = np.where(mask == 255)
    return np.column_stack((xs, ys))  # shape: (M, 2)

def intensity_weighted_center(image, polygon_points):
    """
    Calculate intensity-weighted center of a polygonal region in an image.

    Parameters:
        image : np.ndarray
            Grayscale image (2D array).
        polygon_points : np.ndarray of shape (N, 2)
            Polygon vertices as (x, y) coordinates.

    Returns:
        (xc, yc) : tuple of floats
            Intensity-weighted center coordinates.
    """
    # Create mask for polygon
    mask = np.zeros(image.shape, dtype=np.uint8)
    polygon = np.array([polygon_points], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 1)

    # Get coordinates inside polygon
    ys, xs = np.where(mask == 1)
    intensities = image[ys, xs].astype(float)

    # Avoid division by zero
    total_intensity = np.sum(intensities)
    if total_intensity == 0:
        return None  # Or return simple geometric center instead
    
    #print('==>',np.sum(xs * intensities), np.sum(ys * intensities),total_intensity )
    # Compute weighted center
    xc = np.sum(xs * intensities) / total_intensity
    yc = np.sum(ys * intensities) / total_intensity

    return (xc, yc),total_intensity/len(intensities) #Return center coordinates and average intensity


def draw_tn(curve, length=5): #To get tangent and normals at points on the curve
    """
    Draw outward normal lines from each point of a closed 2D curve.

    Parameters:
        curve : ndarray shape (N,2)
            Closed curve (polygon).
        length : float
            Length of normal lines.
    """
    # Ensure curve is closed
    if not np.all(curve[0] == curve[-1]):
        curve = np.vstack([curve, curve[0]])

    ##plt.plot(curve[:,0], curve[:,1], 'b-', lw=2)  # draw curve
    tangent_pt = []
    normal_pt  = []
    N = len(curve)
    for i in range(N):
        p_prev = curve[i-1]
        p = curve[i]
        p_next = curve[(i+1)%N]

        # Approximate tangent
        #tangent = p_next - p_prev
        npav = 20;
        tangent = 0
        for j in range(npav,-npav,-1):
            tangent += curve[(i+j)%N] - curve[i-npav]
        #tangent  = (curve[(i+3)%N] - curve[i-3]) + (curve[(i+2)%N] - curve[i-3]) + (curve[(i+1)%N] - curve[i-3]) + (curve[(i)%N] - curve[i-3]) + (curve[(i-1)%N] - curve[i-3]) + (curve[(i-2)%N] - curve[i-3]) 
        tangent = tangent / np.linalg.norm(tangent)

        # Normal (perpendicular vector)
        normal = np.array([-tangent[1], tangent[0]])

        # Choose outward normal based on polygon orientation
        # (signed area >0 => CCW)
        area = 0.5*np.sum(curve[:-1,0]*curve[1:,1] - curve[1:,0]*curve[:-1,1])
        #if area < 0:  # clockwise polygon, flip normal
        #    normal = -normal

        # Draw normal line
        tangent_pt.append(p + length*tangent)
        normal_pt.append(p + length*normal)
        ##plt.plot([p[0], q[0]], [p[1], q[1]], 'r-',linewidth=1.0, alpha = 0.5)
        
        tt = p+7*tangent
        ##plt.plot([p[0], tt[0]], [p[1], tt[1]], 'g-',linewidth=1.0, alpha = 0.5)
        
    return np.array(tangent_pt),np.array(normal_pt)
        
        
def points_in_polygon(polygon, shape):
    """
    Return boolean mask of points inside a polygon.
    polygon: Nx2 array of polygon vertices
    shape: (H,W) shape of image
    """
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    points = np.vstack((x.ravel(), y.ravel())).T
    path = Path(polygon)
    mask = path.contains_points(points)
    return mask.reshape((ny, nx))

def pixels_between_curves(inner_curve, outer_curve, shape):
    """
    Get pixel coordinates between two curves.
    inner_curve, outer_curve: Nx2 arrays of polygon vertices
    shape: (H,W) image shape
    """
    mask_outer = points_inside_polygon(shape, outer_curve) #points_in_polygon(outer_curve, shape)
    mask_inner = points_inside_polygon(shape, inner_curve) # points_in_polygon(inner_curve, shape)
    mask = mask_outer & ~mask_inner   # region between
    coords = np.column_stack(np.nonzero(mask))
    return coords, mask
 

def line_pixels(p1, p2):
    """
    Return list of pixel coordinates (row,col) between two points p1 and p2.
    Uses Bresenham's line algorithm.
    
    p1, p2: (x, y) or (col, row)
    """
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)

    coords = []
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    while True:
        coords.append([x1, y1])
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy
    return coords

#

