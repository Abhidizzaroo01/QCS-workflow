
#Importing general llibraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import sys
import pickle
import requests
import json
import math

#Importing image analysis llibraries
from skimage import data, io
from skimage.color import rgb2hed, hed2rgb, rgb2gray, rgba2rgb
from skimage.color import  separate_stains, combine_stains, hdx_from_rgb, rgb_from_hdx
from skimage.util import img_as_ubyte
import skimage.color


#from dqcs import dpixel
from dqcs.dpixel import pixel_OD, interpolate_point, points_inside_polygon, intensity_weighted_center, draw_tn, points_in_polygon,  pixels_between_curves, line_pixels
from dqcs.danalysis import polyArea, calculate_perimeter_numpy, convert_ndarray, read_annotations_txt, plot_annotations, write_annotations
from dqcs.dutils import img_check

#==============================================================================
def cellpose_seg(img_rgb, diameter=100, anno_fig='out.png', anno_file='out_cp.txt', anno_geojson='out.geojson', center_file='out_center.txt', is_membrane=False):
    ####Taken from file step3b_nuclei_cellpose.py
    #img_rgb: image as numpy array
    #anno_fig: name of output figure to see annontations
    #is_membrane: if True, visualize outlines as membranes (yellow), otherwise as nuclei (red)
    
    #Importing cellpose libraries
    from cellpose import models, io, core, plot, utils
    from cellpose.io import imread
    #print('Is GPU Working: ',core.use_gpu())
    # Initialize Cellpose model 
    model_cyto = models.CellposeModel(gpu=True)      # For cell membrane detection
    # Define channels (0 for grayscale, [0, 0] for single-channel grayscale, [1, 2] for red/green channels)
    ## Perform nuclei segmentation on Hematoxylin channel
    masks_h, flows_h, styles_h = model_cyto.eval(img_rgb, flow_threshold=30, cellprob_threshold=-5, diameter= diameter)

    mask_RGB_h = plot.mask_overlay(img_rgb, masks_h)
    outlines_h = utils.outlines_list(masks_h)
    
    ##outlines_h: list of numpy arrays, where each np array is of size ~ 200-400 having pixel coordinates of nuclear annotations
    
    #with open(img[:-4]+'_nuclei_cellpose.pkl', 'wb') as f:
    #    pickle.dump(outlines_h, f)
    
    # Visualize as membranes (yellow) or nuclei (red) based on is_membrane flag
    if is_membrane:
        plot_annotations(img_rgb, anno_fig, nuclei_anno=None, mem_anno=outlines_h)
    else:
        plot_annotations(img_rgb, anno_fig, nuclei_anno=outlines_h, mem_anno=None)
    
    write_annotations(outlines_h,anno_file,anno_geojson, center_file)
                     
    return outlines_h
    
  
#==============================================================================
def membrane_seg(
    img_rgb,
    nuclei_anno_txt,
    anno_fig='mout.png',
    anno_file='mout.txt',
    anno_geojson='mout.geojson',
    center_file='mout_center.txt',
    rays_debug_fig='m_rays_debug.png',
):
    """
    Membrane segmentation driven by DAB optical density (OD) using
    radial rays from the nucleus center.

    - img_rgb: original RGB tile/patch or path to image
    - nuclei_anno_txt: nuclei annotation text file (QuPath-style)
    - anno_fig: final overlay PNG with nuclei (red) and membranes (yellow)
    - anno_file / anno_geojson / center_file: output annotation files for membranes
    - rays_debug_fig: debug PNG with all rays and chosen boundary points
    """
    # 1. Get H, E, D channels from HED deconvolution
    h, e, d = get_hed(img_rgb, h_file=None, e_file=None, d_file=None)

    # 2. Compute OD image from DAB channel (this is what we care about for membrane)
    dab_od, av_OD = get_OD_single_channel(d)  # 2D OD image, high = strong DAB
    img_d = dab_od                            # keep old variable name to reuse logic

    rx, ry = img_d.shape  # height, width

    # 3. Read nuclei annotations (polygon per nucleus + center from txt)
    noutlines_coord, nc_coord = read_annotations_txt(
        nuclei_anno_txt,
        plot_anno=False,
        color='red',
        show_center=False,
        show_id=False,
    )

    # 4. For each nucleus, compute intensity-weighted center IN DAB OD
    noutlines_coord_new = []   # refined nuclei boundary (optional, we keep it)
    nc_coord_new        = []   # refined centers (DAB-weighted)
    mem_outlines        = []   # final membrane outlines (what we will output)

    # Debug figure: show DAB OD, rays, and selected boundary points
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img_d, cmap='gray')

    # Parameters for ray casting
    max_steps = 100
    # NOTE: thresholds are on DAB OD; tune these empirically per dataset
    inner_to_outer_stop_below = 11  # stop when OD drops below this
    outer_to_inner_stop_above = 5   # stop when OD rises above this

    for point in range(len(nc_coord)):
        # Original geometric center (from txt)
        o = nc_coord[point]

        # Compute intensity-weighted center in DAB OD inside the original polygon
        inside_points = points_inside_polygon(img_d.shape, np.array(noutlines_coord[point]))
        xci = 0.0
        yci = 0.0
        si  = 0.0

        for cc in inside_points:
            od_val = float(img_d[int(cc[1]), int(cc[0])])
            xci  += cc[0] * od_val
            yci  += cc[1] * od_val
            si   += od_val

        if si > 0:
            xci /= si
            yci /= si
        else:
            # fallback: use original center if OD sum is zero
            xci, yci = o[0], o[1]

        plt.plot(xci, yci, 'd', color='purple', markersize=3)

        # (Re)use existing helper to get center + avg intensity
        center, avintensity = intensity_weighted_center(img_d, np.array(noutlines_coord[point]))
        nc_coord_new.append([center[0], center[1]])

        plt.plot(center[0], center[1], 'p', color='olive', markersize=3)

        # 5. Ray-based boundary detection from DAB OD
        no_coord_new = []         # refined nuclei boundary (if you still want it)
        mem_coord    = []         # membrane boundary points (what we really want)

        for cc in noutlines_coord[point]:
            # Cast a ray from center → cc (direction) and sample along it
            last_x1, last_y1 = center[0], center[1]  # for fallback

            for i in range(max_steps):
                t = i * 0.01  # step along the ray (0 → 1)

                # Two regimes depending on average intensity (legacy logic)
                if avintensity > 15:
                    # Move outwards: center + t*(cc - center)
                    x1, y1 = interpolate_point(center, cc, t)
                    if not (0 <= int(y1) < rx and 0 <= int(x1) < ry):
                        break
                    od_val = float(img_d[int(y1), int(x1)])
                    last_x1, last_y1 = x1, y1

                    # For membranes, you usually want to stop near a DAB peak or
                    # when DAB starts to drop; this condition is a simple proxy.
                    if od_val < inner_to_outer_stop_below:
                        plt.plot(x1, y1, '.', color='cyan', markersize=1.0)
                        no_coord_new.append([int(x1), int(y1)])
                        mem_coord.append([int(x1), int(y1)])
                        break
                else:
                    # Move inwards: cc → center
                    x1, y1 = interpolate_point(center, cc, 1 - t)
                    if not (0 <= int(y1) < rx and 0 <= int(x1) < ry):
                        break
                    od_val = float(img_d[int(y1), int(x1)])
                    last_x1, last_y1 = x1, y1

                    if od_val > outer_to_inner_stop_above:
                        plt.plot(x1, y1, '.', color='magenta', markersize=1.0)
                        no_coord_new.append([int(x1), int(y1)])
                        mem_coord.append([int(x1), int(y1)])
                        break

                # Fallback: last step if no threshold hit
                if i == max_steps - 1:
                    plt.plot(last_x1, last_y1, '.', color='yellow', markersize=1.0)
                    no_coord_new.append([int(last_x1), int(last_y1)])
                    mem_coord.append([int(last_x1), int(last_y1)])

        noutlines_coord_new.append(no_coord_new)   # refined nucleus (optional)
        mem_outlines.append(mem_coord)             # membrane outline for this nucleus

        # Draw rays from center to membrane boundary on debug fig
        for bx, by in mem_coord:
            plt.plot([center[0], bx], [center[1], by], color='lime', linewidth=0.3, alpha=0.3)

    # Save the debug rays figure (DAB OD + rays + boundary points)
    plt.tight_layout()
    plt.savefig(rays_debug_fig, dpi=300)
    plt.close()

    # 6. Save final overlay with original RGB, nuclei + membranes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img_rgb, cmap='gray')
    for point in range(len(nc_coord_new)):
        o = nc_coord_new[point]
        nuc = np.array(noutlines_coord_new[point])
        mem = np.array(mem_outlines[point])

        plt.plot(o[0], o[1], 'ro')
        if len(nuc) > 0:
            plt.plot(nuc[:, 0], nuc[:, 1], color='red', linewidth=1.0)
        if len(mem) > 0:
            plt.plot(mem[:, 0], mem[:, 1], color='yellow', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(anno_fig, dpi=300)
    plt.close()

    # 7. Write membrane annotations to txt / geojson / centers
    write_annotations(
        mem_outlines,
        anno_file=anno_file,
        anno_geojson=anno_geojson,
        center_file=center_file,
    )

    # Return outlines for downstream use
    return mem_outlines, noutlines_coord_new, nc_coord_new

#==============================================================================
def plot_OD(img_file,den_file):
    img_rgb    = io.imread(img_file)
    rx,ry,_    =  img_rgb.shape
    d1 = np.zeros_like(img_rgb[:, :, 0])
    for i in range(rx):
        for j in range(ry):
            d1[i,j] = pixel_OD(img_rgb,i,j)
    io.imsave(den_file, d1)
#==============================================================================

#==============================================================================

def get_OD(img_rgb):
    #This function is going to return OD (of RGB channels) of whole patch/image and average OD
    rx,ry,_    =  img_rgb.shape
    
    ##Following part is very slow
    '''
    d1 = np.zeros_like(img_rgb[:, :, 0])
    for i in range(rx):
        for j in range(ry):
            d1[i,j] = pixel_OD(img_rgb,i,j)
    #===================================================      
    '''
    
    ##Following is fast but consume more memory
    '''# Compute per-channel OD
    #img2 = (img_rgb / 255.0).astype(np.uint8)
    img2 = img_rgb.astype(np.float64) / 255.0
    dr = -np.log((img2[:,:,0] ) + 0.0001)
    dg = -np.log((img2[:,:,1] ) + 0.0001)
    db = -np.log((img2[:,:,2] ) + 0.0001)
    # Mean OD per pixel (average over RGB)
    d1 = (dr + dg + db) / 3.0
    # Scale as in your original code
    d1 = d2 * 255.0 / 5.75
    #===================================================
    '''
        
    ##Following is fast as well less memory consuming
    #img = img_rgb.astype(np.float32) / 255.0
    img1 = img_rgb.astype(np.float64) / 255.0
    #img1 = (img_rgb / 255.0).astype(np.uint8)
    img1 = -np.log(img1 + 0.0001)
    d1 = img1.mean(axis=2) * 255.0 / 5.75
    #===================================================
    
    d1 = d1.astype(np.uint8)
    d1_av = np.sum(d1)/(rx*ry)
    
    #print(np.sum(d1),d1_av)

    return d1, d1_av
#==============================================================================

#==============================================================================
def get_hed(img_file, h_file=None, e_file=None, d_file=None):  
    # Load an image and normalize it ---------------------------------------------
    if isinstance(img_file, str):
        img_rgb = io.imread(img_file) #cv2.imread(img_file)
    else:
        img_rgb = img_file
    if img_rgb.shape[2] == 4:
        print("Warning: Input image has an alpha channel. Removing it.")
        img_rgb = img_rgb[:,:,0:3]
    
    img_rgb[img_rgb>210]=255 #Making all offwhite background pixels to completely white   
    img_rgb_normalize = img_rgb / 255.0 #Normalizing pixel values to 0-1
    #----------------------------------------------------------------------------
    # Converting RGB to HED -----------------------------------------------------   
    img_hed = rgb2hed(img_rgb_normalize)
    # Creating an RGB image for each of the separated stains and convert them to ubyte for easy saving of image
    null = np.zeros_like(img_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, img_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, img_hed[:, :, 2]), axis=-1)))
    
    #ihc_h = hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1))
    #ihc_e = hed2rgb(np.stack((null, img_hed[:, :, 1], null), axis=-1))
    #ihc_d = hed2rgb(np.stack((null, null, img_hed[:, :, 2]), axis=-1))
    
    if(h_file): io.imsave(h_file, ihc_h)
    if(e_file): io.imsave(e_file, ihc_e)
    if(d_file): io.imsave(d_file, ihc_d)
    
    return ihc_h, ihc_e, ihc_d
#==============================================================================  

#==============================================================================
def get_hd_clean(img_file, h_file=None, d_file=None):  
    # Load an image and normalize it ---------------------------------------------
    if isinstance(img_file, str):
        img_rgb = io.imread(img_file) #cv2.imread(img_file)
    else:
        img_rgb = img_file
    if img_rgb.shape[2] == 4:
        print("Warning: Input image has an alpha channel. Removing it.")
        img_rgb = img_rgb[:,:,0:3]
    
    img_rgb[img_rgb>210]=255 #Making all offwhite background pixels to completely white   
    img_rgb_normalize = img_rgb / 255.0 #Normalizing pixel values to 0-1
    #----------------------------------------------------------------------------
    # Converting RGB to HED -----------------------------------------------------   
    img_hed = rgb2hed(img_rgb_normalize)
    # Creating an RGB image for each of the separated stains and convert them to ubyte for easy saving of image
    null = np.zeros_like(img_hed[:, :, 0])
    
    ihc_h = img_as_ubyte(hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, img_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, img_hed[:, :, 2]), axis=-1)))
    
    rx,ry,_ = img_hed.shape
    h_od,_    = get_OD(ihc_h)
    d_od,_    = get_OD(ihc_d)
    
    #Modifying pixel values in H and D channel to separate nuclei with membrane
    for xi in range(rx):
        for yj in range(ry):
            odh = h_od[xi, yj]
            odd = d_od[xi, yj]
            #odh = h_od[yj, xi]
            #odd = d_od[yj, xi]
            if( odh > odd): # If OD of hematoxylin is more than remove correspondint part from DAB channel
                img_hed[xi,yj,2] = 0
            if( odh < odd): # If OD of hematoxylin is more
                img_hed[xi,yj,0] = 0
    
    ihc_hm = img_as_ubyte(hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1)))  
    ihc_dm = img_as_ubyte(hed2rgb(np.stack((null, null, img_hed[:, :, 2]), axis=-1)))
    if(h_file): io.imsave(h_file, ihc_hm)
    if(d_file): io.imsave(d_file, ihc_dm)
    
    return ihc_hm, ihc_dm
#==============================================================================

def plot_aoutline(outline_coord, color='red'):
    plt.plot( np.mean(outline_coord[:, 0]), np.mean(outline_coord[:, 1]), marker='o', color=color )
    plt.plot(outline_coord[:,0], outline_coord[:,1], color=color, linewidth=1)
    #for cc in outline_coord: #Plotting the outline
    #   plt.plot(cc[0], cc[1], '*',color=color, markersize=1.0)
    #    #plt.plot(cc[0], cc[1], color=color)
        
#==============================================================================
def squeeze_nuclei_annotations(outlines, img, niter_max):
    img_rgb = img_check(img) #To read image or use np array
    h_od    = get_OD(img_rgb)
    rx,ry,_   = img_rgb.shape
    squeezed_outlines = []
    
    for i in range(len(outlines)): #Loop over each nuclei
        o = outlines[i]
        print(i,len(outlines))
        squeezed_outlines.append( squeeze_aoutline(o, img, niter_max = niter_max) )
    return squeezed_outlines

#==============================================================================

def squeeze_aoutline(outline_coord, img, niter_max = 5):
    img_rgb = img_check(img)  #To read image or use np array
    h_od,_  = get_OD(img_rgb) #To get OD of H channel
    #print('OD range: ',np.min(h_od), np.max(h_od) )
    rx,ry,_   = img_rgb.shape
    #filtered_outlines = []
    
    o = outline_coord
    conditions = np.zeros(len(o))
    
    #while(np.sum(conditions)==len(o)):
    #print(type(pip),pip[0],len(pip),pip.shape)
    #for pp in pip:
    #    plt.plot(pp[0],pp[1],'g.',markersize=5.5)
    
    for i in range(niter_max):
        if(np.sum(conditions)<len(o)): #If all points are not properly placed
            pip = points_inside_polygon((rx,ry), o )
            onew = [] #To get new approximation of squeezed outline
            otmp = [] #To get new points inside the cell
            cond_new = []
            
            for j in range(len(o)):#Loop over each point of annotation o
                if(conditions[j]==0):
                    x = o[j][0]
                    y = o[j][1]                    
                    if(h_od[y][x]>10): #If this pixel have certain OD than keep it as a outline point
                        conditions[j] = 1; #One condition for point near pixel has fulfilled
                    else:          
                        points = [[x+1,y], [x+1,y+1], [x,y+1], [x-1,y+1], [x-1,y], [x-1,y-1], [x, y-1], [x+1, y-1]]  
                        for pr in points: #Loop over generated nearby points
                            if(np.any(np.all(pip == pr, axis=1))): #If point is on or inside polygon
                                if(not np.any(np.all(o == pr, axis=1))): # To remove points on polygon
                                    #plt.plot(pr[0],pr[1],'b*',markersize=3.5)  
                                    otmp.append(pr)
                        
            for j in range(len(o)): #Loop over each point of annotation o
                if(conditions[j]==0):
                    if len(otmp) == 0:
                        # No interior candidate points — keep original point
                        onew.append([o[j][0], o[j][1]])
                        cond_new.append(0)
                        continue
                    dmin = 100;
                    newp = None
                    for k in range(len(otmp)): #Loop over each point of generated points
                        d = math.dist(o[j], otmp[k])
                        if(d<dmin):
                            dmin=d
                            newp = otmp[k]
                    
                    if newp is not None and (newp not in onew):
                        onew.append(newp)
                        cond_new.append(0)
                        #plt.plot(newp[0],newp[1],'r*',markersize=3.5)
                else:
                    x = o[j][0]
                    y = o[j][1]
                    onew.append([x,y])
                    cond_new.append(1)
                    ##print(j,h_od[y][x],h_od[891][44])
                    #plt.plot(x,y,'r*',markersize=5.5)

            #print(type(o), type(o[0]), o.shape, o[0].shape,  type(onew), type(onew[0]), len(o),len(onew))
            #plot_aoutline(np.array(onew), color='green')
            o = np.array(onew)
            conditions = np.array(cond_new)
            #print(niter_max, np.sum(conditions),h_od[891][44],h_od[44][891])  
    del img_rgb
    return o
#==============================================================================

def squeeze_aoutlinex(outline_coord, img, niter_max = 2):
    img_rgb = img_check(img) #To read image or use np array
    h_od    = get_OD(img_rgb)
    rx,ry,_   = img_rgb.shape
    #filtered_outlines = []
    
    o = outline_coord
    conditions = np.zeros(len(o))
    #while(np.sum(conditions)==len(o)):
    #print(type(pip),pip[0],len(pip),pip.shape)
    #for pp in pip:
    #    plt.plot(pp[0],pp[1],'g.',markersize=5.5)
    
    for i in range(niter_max):
        pip = points_inside_polygon((rx,ry), o )
        onew = []
        for j in range(len(o)):    #Loop over each point of annotation o
            x = o[j][0]
            y = o[j][1]
            points = [[x+1,y], [x+1,y+1], [x,y+1], [x-1,y+1], [x-1,y], [x-1,y-1], [x, y-1], [x+1, y-1]]  
            
            dcheck=100; #Initially low
            cnew = []
            for pr in points: #Loop over generated nearby points
                if(np.any(np.all(pip == pr, axis=1))): #If point is inside polygon
                    plt.plot(pr[0],pr[1],'b*',markersize=3.5)
                    
                    do = math.dist(o[j], pr)
                    dl = math.dist(o[(j-1)%len(o)], pr)
                    dr = math.dist(o[(j+1)%len(o)], pr)   # This logic is not working
                    #if(d>dcheck):
                    #if(d<dcheck):
                    if(do<=dl and do<=dr):
                        dcheck=do
                        cnew = [pr[0],pr[1]]
                        #print(i,j,[x,y],cnew,do,dl,dr)
                    
            #o[j][0]=cnew[0]
            #o[j][1]=cnew[1]
            if(len(cnew)>1): #Found a corresponding point inside
                onew.append(cnew)
                plt.plot(cnew[0],cnew[1],'k*',markersize=3.5)
                #else:
                #    plt.plot(pr[0],pr[1],'g.',markersize=0.5)
            
        print(type(o), type(o[0]), o.shape, o[0].shape,  type(onew), type(onew[0]),)
        plot_aoutline(np.array(onew), color='green')
        o = np.array(onew)
            
    return outline_coord
#==============================================================================

def squeeze_aoutliney(outline_coord, img, niter_max = 5):
    img_rgb = img_check(img) #To read image or use np array
    h_od    = get_OD(img_rgb)
    rx,ry,_   = img_rgb.shape
    #filtered_outlines = []
    
    o = outline_coord
    conditions = np.zeros(len(o))
    
    #while(np.sum(conditions)==len(o)):
    #print(type(pip),pip[0],len(pip),pip.shape)
    #for pp in pip:
    #    plt.plot(pp[0],pp[1],'g.',markersize=5.5)
    
    for i in range(niter_max):
        
        pip = points_inside_polygon((rx,ry), o )
        onew = [] #To get new approximation of squezed outline
        otmp = [] # To get new points inside the cell
        
        for j in range(len(o)):#Loop over each point of annotation o
            x = o[j][0]
            y = o[j][1]
            points = [[x,y], [x+1,y], [x+1,y+1], [x,y+1], [x-1,y+1], [x-1,y], [x-1,y-1], [x, y-1], [x+1, y-1]]  
            
            for pr in points: #Loop over generated nearby points
                if(np.any(np.all(pip == pr, axis=1))): #If point is on or inside polygon
                
                    #if(not np.any(np.all(o == pr, axis=1))
                    plt.plot(pr[0],pr[1],'b*',markersize=3.5) # To remove points on polygon
                    otmp.append(pr)
                    
        for j in range(len(o)): #Loop over each point of annotation o
            dmin = 100;
            for k in range(len(otmp)): #Loop over each point of annotation o
                d = math.dist(o[j], otmp[k])
                if(d<dmin):
                    dmin=d
                    newp = otmp[k]
            
            if(newp not in onew):
                onew.append(newp)
                plt.plot(newp[0],newp[1],'r*',markersize=3.5)
                    

        print(type(o), type(o[0]), o.shape, o[0].shape,  type(onew), type(onew[0]), len(o),len(onew))
        plot_aoutline(np.array(onew), color='green')
        o = np.array(onew)
            
    return outline_coord
#==============================================================================

# # def patch_bioinformatics(outline_n, outline_m, img):
#     #img: Raw image withe deconvolution
#     img_rgb = img_check(img) #To read image or use np array
#     OD,_    = get_OD(img_rgb)
    
#     nuc_od = 0.0
#     mem_od_selected = 0.0
#     mem_od_all = 0.0
#     cyto_od = 0.0
    
#     for i in range(len(outline_n)):  
#         mem = outline_n[i]
#         nuc = outline_m[i]

#         for pp in nuc:
#             nuc_od += OD[pp[0],pp[1]]
            
#         cellmod = 0.0
#         cellmod_selectecd = 0.0
#         for pp in mem:
#             cellmod += OD[pp[0],pp[1]]
#             if(OD[pp[0],pp[1]]>8):
#                 cellmod_selectecd += OD[pp[0],pp[1]]
        
#         #Implementing OD threshold
#         mem_od_all += cellmod
#         mem_od_selected += cellmod_selectecd
         
        
#         pair_count = min(len(mem), len(nuc))
#         if pair_count == 0:
#             continue

#         for pi in range(pair_count):
#             pixels = line_pixels(mem[pi], nuc[pi])
#             for pp in pixels:
#                 cyto_od += OD[pp[0], pp[1]]
    
#     NMR = mem_od_all/(mem_od_all+cyto_od)
#     Nnuclei = len(outline_n) 
#     #print(f'Mem OD = {mem_od}, Nuclei Mem OD = {nuc_od} Cyto_od = {cyto_od}, ')
#     #print(f'  NMR = {round(NMR,3)},  % of Positive Tumor Cells {round(mem_od/len(outline_n),1)}')
    
#     return Nnuclei, mem_od_selected, mem_od_all, cyto_od, nuc_od, NMR    
#==============================================================================
def patch_bioinformatics(outline_n, outline_m, img):
    #img: Raw image withe deconvolution
    img_rgb = img_check(img) #To read image or use np array
    OD,_    = get_OD(img_rgb)
    
    # Get image dimensions for bounds checking
    od_height, od_width = OD.shape
    
    # Additional safety: ensure OD dimensions match img_rgb
    img_height, img_width = img_rgb.shape[:2]
    if od_height != img_height or od_width != img_width:
        raise ValueError(f"OD shape {OD.shape} doesn't match image shape {img_rgb.shape[:2]}")
    
    nuc_od = 0.0
    mem_od_selected = 0.0
    mem_od_all = 0.0
    cyto_od = 0.0
    
    num_pairs = min(len(outline_n), len(outline_m))
    for i in range(num_pairs):
        mem = outline_n[i]
        nuc = outline_m[i]
        
        # Ensure mem and nuc are numpy arrays
        if not isinstance(mem, np.ndarray):
            mem = np.array(mem)
        if not isinstance(nuc, np.ndarray):
            nuc = np.array(nuc)

        for pp in nuc:
            # Check bounds and convert coordinates: pp is [x, y] but OD is indexed as [y, x]
            # Clamp coordinates to valid range to prevent index errors
            x = max(0, min(int(pp[0]), od_width - 1))
            y = max(0, min(int(pp[1]), od_height - 1))
            nuc_od += OD[y, x]
            
        cellmod = 0.0
        cellmod_selectecd = 0.0
        for pp in mem:
            # Check bounds and convert coordinates: pp is [x, y] but OD is indexed as [y, x]
            # Clamp coordinates to valid range to prevent index errors
            x = max(0, min(int(pp[0]), od_width - 1))
            y = max(0, min(int(pp[1]), od_height - 1))
            od_val = OD[y, x]
            cellmod += od_val
            if od_val > 8:
                cellmod_selectecd += od_val
        
        #Implementing OD threshold
        mem_od_all += cellmod
        mem_od_selected += cellmod_selectecd
         
        
        pair_count = min(len(mem), len(nuc))
        if pair_count == 0:
            continue

        for pi in range(pair_count):
            pixels = line_pixels(mem[pi], nuc[pi])
            for pp in pixels:
                # line_pixels returns [x, y], but OD is indexed as [y, x]
                # Clamp coordinates to valid range to prevent index errors
                x = max(0, min(int(pp[0]), od_width - 1))
                y = max(0, min(int(pp[1]), od_height - 1))
                cyto_od += OD[y, x]
    
    NMR = mem_od_all/(mem_od_all+cyto_od) if (mem_od_all+cyto_od) > 0 else 0.0
    Nnuclei = len(outline_n) 
    
    return Nnuclei, mem_od_selected, mem_od_all, cyto_od, nuc_od, NMR

#==============================================================================
def get_OD_single_channel(img_channel):
    """
    Calculate OD for a single channel image (e.g., DAB channel).
    
    Args:
        img_channel: Single channel image (grayscale, shape: H×W) or RGB image
    
    Returns:
        OD image and average OD
    """
    # Handle both grayscale and RGB input
    if len(img_channel.shape) == 3:
        # If RGB, convert to grayscale
        from skimage.color import rgb2gray
        img_gray = (rgb2gray(img_channel) * 255).astype(np.uint8)
    else:
        img_gray = img_channel.astype(np.uint8)
    
    img_normalized = img_gray.astype(np.float64) / 255.0
    od = -np.log(img_normalized + 0.0001)
    od_scaled = od * 255.0 / 5.75
    od_scaled = od_scaled.astype(np.uint8)
    od_av = np.sum(od_scaled) / (od_scaled.shape[0] * od_scaled.shape[1])
    return od_scaled, od_av

#==============================================================================
def patch_bioinformatics_v2(outline_n, outline_m, img, cpx2um=0.25):
    """
    Calculate bioinformatics metrics with micron-based calculations.
    Creates a cell-level table and calculates 5 key metrics.
    
    Args:
        outline_n: List of squeezed nuclei outlines (numpy arrays)
        outline_m: List of membrane outlines (numpy arrays)
        img: RGB image (numpy array)
        cpx2um: Conversion factor from pixels to microns (default: 0.25)
    
    Returns:
        Dictionary containing:
        - cell_table: DataFrame with per-cell metrics in microns
        - metrics: Dictionary with 5 key metrics (q15, percent_positive_od10, 
                   density_high_expressors, bsps_score, csps_mean)
        - tile_area_mm2: Tile area in square millimeters
        - num_valid_cells: Number of valid cells (20-150 µm²)
        - num_total_cells: Total number of cells detected
        - conversion_factor_px_to_um: Conversion factor used
    """
    try:
        import pandas as pd
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError(f"Required packages not installed: {e}. Install with: pip install pandas scipy")
    
    # Get image
    img_rgb = img_check(img)
    img_height, img_width = img_rgb.shape[:2]
    
    # Get DAB channel for membrane OD calculation
    _, _, ihc_d = get_hed(img_rgb, h_file=None, e_file=None, d_file=None)
    
    # Calculate OD for DAB channel (membrane staining)
    dab_od, _ = get_OD_single_channel(ihc_d)
    
    # Calculate tile area in mm²
    tile_width_um = img_width * cpx2um
    tile_height_um = img_height * cpx2um
    tile_area_um2 = tile_width_um * tile_height_um
    tile_area_mm2 = tile_area_um2 / 1_000_000
    
    # Initialize cell table
    cell_data = []
    
    num_pairs = min(len(outline_n), len(outline_m))
    
    for cell_id in range(num_pairs):
        # Get outlines
        nuc_outline = np.array(outline_n[cell_id]) if not isinstance(outline_n[cell_id], np.ndarray) else outline_n[cell_id]
        mem_outline = np.array(outline_m[cell_id]) if not isinstance(outline_m[cell_id], np.ndarray) else outline_m[cell_id]
        
        # Skip if empty
        if len(nuc_outline) == 0 or len(mem_outline) == 0:
            continue
        
        # ============================================================
        # 1. CALCULATE NUCLEUS AREA (in µm²)
        # ============================================================
        nuc_area_px2 = polyArea(nuc_outline[:, 0], nuc_outline[:, 1])
        nucleus_area_um2 = nuc_area_px2 * (cpx2um ** 2)
        
        # ============================================================
        # 2. CALCULATE CENTROID (in microns)
        # ============================================================
        centroid_x_px = np.mean(nuc_outline[:, 0])
        centroid_y_px = np.mean(nuc_outline[:, 1])
        centroid_x_micron = centroid_x_px * cpx2um
        centroid_y_micron = centroid_y_px * cpx2um
        
        # ============================================================
        # 3. CALCULATE MEMBRANE OD MEAN
        # ============================================================
        mem_pixels = points_inside_polygon((img_height, img_width), mem_outline)
        
        if len(mem_pixels) == 0:
            continue  # Skip if no membrane pixels found
        
        mem_od_values = []
        for px in mem_pixels:
            x, y = int(px[0]), int(px[1])
            if 0 <= y < img_height and 0 <= x < img_width:
                mem_od_values.append(float(dab_od[y, x]))
        
        if len(mem_od_values) == 0:
            continue
        
        membrane_od_mean = np.mean(mem_od_values)
        
        # ============================================================
        # 4. VALIDITY FILTER (20 µm² ≤ Area ≤ 250 µm²)
        # ============================================================
        is_valid = (20 <= nucleus_area_um2 <= 250)
        
        # Store cell data
        cell_data.append({
            'cell_id': cell_id,
            'centroid_x_micron': round(centroid_x_micron, 2),
            'centroid_y_micron': round(centroid_y_micron, 2),
            'membrane_od_mean': round(membrane_od_mean, 3),
            'nucleus_area_um2': round(nucleus_area_um2, 2),
            'is_valid': is_valid,
            # Additional helpful columns for reference
            'centroid_x_px': int(round(centroid_x_px)),
            'centroid_y_px': int(round(centroid_y_px)),
            'nucleus_area_px2': round(nuc_area_px2, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(cell_data)
    
    if len(df) == 0:
        return {
            'cell_table': df,
            'metrics': {
                'q15': 0.0,
                'percent_positive_od10': 0.0,
                'density_high_expressors': 0.0,
                'bsps_score': 0.0,
                'csps_mean': 0.0
            },
            'tile_area_mm2': tile_area_mm2,
            'num_valid_cells': 0,
            'num_total_cells': 0,
            'conversion_factor_px_to_um': cpx2um
        }
    
    # Filter to valid cells only
    valid_cells = df[df['is_valid'] == True].copy()
    
    if len(valid_cells) == 0:
        return {
            'cell_table': df,
            'metrics': {
                'q15': 0.0,
                'percent_positive_od10': 0.0,
                'density_high_expressors': 0.0,
                'bsps_score': 0.0,
                'csps_mean': 0.0
            },
            'tile_area_mm2': tile_area_mm2,
            'num_valid_cells': 0,
            'num_total_cells': len(df),
            'conversion_factor_px_to_um': cpx2um
        }
    
    # ============================================================
    # METRIC A: Membrane OD Quantile 15 (Q15)
    # ============================================================
    q15 = np.percentile(valid_cells['membrane_od_mean'], 15)
    
    # ============================================================
    # METRIC B: % Positive Cells (OD ≥ 10)
    # ============================================================
    n_positive = len(valid_cells[valid_cells['membrane_od_mean'] >= 10])
    percent_positive_od10 = (n_positive / len(valid_cells)) * 100 if len(valid_cells) > 0 else 0.0
    
    # ============================================================
    # METRIC C: Density of High-Expressors (OD ≥ 25) per mm²
    # ============================================================
    n_high_density = len(valid_cells[valid_cells['membrane_od_mean'] >= 25])
    density_high_expressors = n_high_density / tile_area_mm2 if tile_area_mm2 > 0 else 0.0
    
    # ============================================================
    # METRIC D: Binary Spatial Proximity Score (bSPS)
    # Parameters: r=25µm, OD≥60
    # ============================================================
    ref_cells = valid_cells[valid_cells['membrane_od_mean'] >= 60].copy()
    
    if len(ref_cells) > 1:
        coords = ref_cells[['centroid_x_micron', 'centroid_y_micron']].values
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=2, distance_upper_bound=25.0)
        has_neighbor = ~np.isinf(distances[:, 1])
        bsps_score = (np.sum(has_neighbor) / len(ref_cells)) * 100 if len(ref_cells) > 0 else 0.0
    else:
        bsps_score = 0.0
    
    # ============================================================
    # METRIC E: Continuous Spatial Proximity Score (cSPS)
    # Parameters: r=50µm, baseline=Q15
    # ============================================================
    neighbors = valid_cells[valid_cells['membrane_od_mean'] > q15].copy()
    
    if len(neighbors) > 0:
        all_coords = valid_cells[['centroid_x_micron', 'centroid_y_micron']].values
        neighbor_coords = neighbors[['centroid_x_micron', 'centroid_y_micron']].values
        neighbor_tree = cKDTree(neighbor_coords)
        
        csps_scores = []
        for idx, row in valid_cells.iterrows():
            query_point = [[row['centroid_x_micron'], row['centroid_y_micron']]]
            neighbor_indices = neighbor_tree.query_ball_point(query_point, r=50.0)
            
            if len(neighbor_indices[0]) > 0:
                neighbor_ods = neighbors.iloc[neighbor_indices[0]]['membrane_od_mean'].values
                csps_scores.append(np.sum(neighbor_ods))
            else:
                csps_scores.append(0.0)
        
        csps_mean = np.mean(csps_scores) if len(csps_scores) > 0 else 0.0
    else:
        csps_mean = 0.0
    
    # Compile metrics
    metrics = {
        'q15': float(q15),
        'percent_positive_od10': float(percent_positive_od10),
        'density_high_expressors': float(density_high_expressors),
        'bsps_score': float(bsps_score),
        'csps_mean': float(csps_mean)
    }
    
    return {
        'cell_table': df,
        'metrics': metrics,
        'tile_area_mm2': float(tile_area_mm2),
        'num_valid_cells': len(valid_cells),
        'num_total_cells': len(df),
        'conversion_factor_px_to_um': cpx2um
    }
#==============================================================================

'''    
    dabod = get_OD(ihc_dm)
    meandOD = np.sum(dabod)/(rx*ry)
    htod  = get_OD(ihc_hm)
    meanhOD = np.sum(htod)/(rx*ry)
    
    print(f'Max average DAB OD for {tp} is = {meandOD},{np.max(dabod)}; Max average Hematoxylin OD for {tp} is = {meanhOD},{np.max(htod)}')
    
    dod_analysis.append(meandOD)
    hod_analysis.append(meanhOD)

print(f'For {tp} max_average_dOD = {np.max(dod_analysis)}, min_average_dOD = {np.min(dod_analysis)}, average_dOD = {np.mean(dod_analysis)}, std = {np.std(dod_analysis)}')

print(f'For {tp} max_average_dOD = {np.max(hod_analysis)}, min_average_dOD = {np.min(hod_analysis)}, average_dOD = {np.mean(hod_analysis)}, std = {np.std(hod_analysis)}')


    
    if(h_file): io.imsave(h_file, ihc_h)
    if(e_file): io.imsave(e_file, ihc_e)
    if(d_file): io.imsave(d_file, ihc_d)
    
    return ihc_h, ihc_e, ihc_d

    
    print(np.min(img_hed[:, :, 0]), np.max(img_hed[:, :, 0]))

    # Modifying hed image to identify nuclei clearly
    img_hed[:, :, 1] = 0          #Nullifying E channel 
    img_rgb_n = hed2rgb(img_hed)  #Converting back to RGB
    img_rgb_clipped = np.clip(img_rgb_n, 0, 1) # Clipping values between 0 to 1
    img_rgb_final = (img_rgb_clipped* 255).astype(np.uint8)
     
    img_hed[:, :, 0] [img_hed[:, :, 2]>0.05] = 0 #Modifying (deleting significant DAB part) h channel for better nuclei segmentation
    ihc_hm = img_as_ubyte(hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1)))
    io.imsave(f'../{tpo}/'+fn[:-4]+'_hm.png', ihc_hm)
    plot_density(f'../{tpo}/'+fn[:-4]+'_hm.png', f'../{tpo}/'+fn[:-4]+'_hmdensity.png')
    
'''