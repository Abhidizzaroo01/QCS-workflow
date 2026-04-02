
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
from dqcs.dpixel import pixel_OD, interpolate_point, points_inside_polygon
from dqcs.dutils import img_check


#========================================================================
def polyArea(x,y):
    #Function to calculate area of a polygon from list of x and y coordinates
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#========================================================================
def calculate_perimeter_numpy(points):
    """
    Calculates the perimeter of a polygon given a list of coordinate points using NumPy.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) where N is the number of
                             vertices, and each row represents an (x, y) coordinate.

    Returns:
        float: The perimeter of the polygon.
    """
    if points.shape[0] < 2:
        return 0.0  # A perimeter requires at least two points to define a segment.

    # Calculate differences between consecutive points
    diffs = np.diff(points, axis=0)

    # Calculate squared Euclidean distances for each segment
    squared_distances = np.sum(diffs**2, axis=1)

    # Calculate Euclidean distances
    segment_lengths = np.sqrt(squared_distances)

    # Sum the lengths of all segments
    perimeter = np.sum(segment_lengths)

    # Add the distance between the last and the first point to close the polygon
    last_to_first_diff = points[0] - points[-1]
    perimeter += np.sqrt(np.sum(last_to_first_diff**2))

    return perimeter

#========================================================================
def read_annotations_txt(file_txt, plot_anno=False, color='red', show_center=False, show_id=False):
    #To read annotations from the text file in the given format and plot it
    outlines_coord = []  #Coordinates of  outline/annotation points
    c_coord        = []  #Coordinates of center 
    outline_points  = 100 #A counter
    #Reading outlines from the text file
    with open(file_txt) as fp:
        for line in fp:
            l = line.split()
            if(l[0][0]=='#'): ##Nuclei: Np x, y (# of boundary points, coordinates of center)
                c_coord.append([ int(l[2].strip(',')), int(l[3].strip(',')) ])
                outline_points = int(l[1])
                no_coord = []
            else:
                no_coord.append([ int(l[0]), int(l[1]) ])
                outline_points = outline_points -1
                
            if(outline_points==0):
                outlines_coord.append(no_coord)
                
    if(plot_anno):
        check = 0 
        for point in range(len(c_coord)): 
            o = c_coord[point]
            #print(check,len(c_coord), len(outlines_coord[point]), outlines_coord[point][0],o)
            check += 1
            if(show_center): plt.plot(o[0], o[1], 'bo')
            if(show_id): plt.text(o[0], o[1], f"{point}", fontsize=3)
            for cc in outlines_coord[point]: #Plotting the outline
                plt.plot(cc[0], cc[1], '.',color=color, markersize=1.0)
    
    return outlines_coord, c_coord
    
#========================================================================
def show_annotations_txt(in_img_file, out_img_file, nuclei_anno=None, mem_anno=None):
    #To show nuclei and membrane annotations on the image file
    img_rgb    = io.imread(in_img_file)
    plt.figure()
    plt.axis('off')
    plt.imshow(img_rgb)
    
    if(nuclei_anno):
        read_annotations_txt(file_txt = nuclei_anno, plot_anno=True, color='red', show_center=True, show_id=True)
    if(mem_anno):
        read_annotations_txt(file_txt = mem_anno, plot_anno=True, color='yellow')
        
    plt.tight_layout()
    plt.savefig(out_img_file, dpi=300)
    plt.close()

#==============================================================================
def write_annotations(outlines,anno_file='annotation.txt',anno_geojson='annotation.geojson', center_file='centers.txt'):
    fi = open(anno_file,'w')
    fic = open(center_file,'w')
    qupath_anno = []
    anno_id = 1;
    for o in outlines:
        # Skip only if truly empty
        if len(o) == 0:
            continue
        
        # Convert to numpy array if not already
        o = np.array(o)
        
        # Skip if empty after conversion
        if o.size == 0:
            continue
        
        # Try to calculate mean first - if this fails, skip this outline
        # This is the check that prevents the original NaN error
        try:
            mean_x = np.mean(o[:, 0])
            mean_y = np.mean(o[:, 1])
            
            # Skip if mean is NaN or Inf (this prevents the rounding error)
            if np.isnan(mean_x) or np.isnan(mean_y) or np.isinf(mean_x) or np.isinf(mean_y):
                continue
        except (IndexError, ValueError, TypeError):
            # Skip if we can't calculate mean (wrong shape, etc.)
            continue
        
        # Calculate area and perimeter (original logic)
        if len(o) > 20:
            anuclei = polyArea(o[:, 0], o[:, 1])
        else:
            anuclei = 0.0
        
        perimeter = calculate_perimeter_numpy(o)
        
        # Calculate sphericity (avoid division by zero)
        if perimeter > 0:
            sphericity = 4*3.141*anuclei/(perimeter**2)
        else:
            sphericity = 0.0
        
        fi.write(f'#Nuclei: {len(o)} {round(mean_x)}, {round(mean_y)} (# of boundary points, coordinates of center)\n')
        fic.write(f'{round(mean_x)} {round(mean_y)}  {round(anuclei)} {round(sphericity,2)} \n')
        for R in o:
            fi.write(f'{R[0]}  {R[1]}\n')
        
        feature = {"type":"Feature",
                   "id":f"{anno_id}",
                   "geometry":{"type":"LineString", #Polygon
                               "coordinates":convert_ndarray(o)},
                               #"coordinates":[convert_ndarray(o),convert_ndarray([o[0,0],o[0,1])]]},
                   "properties":{"objectType":"annotation",
                                 "classification":{"name":"Nucleus","color":[230, 77, 77]},
                                 "sphericity":round(sphericity, 4),
                                 "area":round(anuclei, 2),
                                 "perimeter":round(perimeter, 2),
                                 "center":{"x":round(mean_x), "y":round(mean_y)},
                                 "boundaryPoints":len(o)}}
        qupath_anno.append(feature)
        anno_id += 1
                    
    geojson_data = {"type": "FeatureCollection", "features":qupath_anno}
    # Write to GeoJSON file
    with open(anno_geojson, 'w') as f:
        json.dump(geojson_data, f) # indent for pretty-printing, indent=1
    
    #print(f"GeoJSON file '{anno_geojson}' created successfully.")
    fi.close()
    fic.close()

#==============================================================================
def filter_nuclei_annotations(outlines, Cpx2um=0.25, min_area=2500, min_height=45):
    """Filter nuclei annotations/outlines based on shape and size.

    Args:
        outlines: list of numpy arrays having coordinates of size Mx2
        Cpx2um: Conversion factor to convert 1px to micron
        min_area: minimum area in px² to keep a nucleus (default 2500)
        min_height: minimum bounding-rect height in px (default 45)

    Returns:
        filtered_outlines: list of nuclei outlines that pass the filter
    """
    filtered_outlines = []
    for o in outlines:
        anuclei = polyArea(o[:, 0], o[:, 1])  # Area in px^2
        if anuclei > min_area:
            rect = cv2.minAreaRect(o.reshape(-1, 1, 2))
            (center_x, center_y), (width, height), angle = rect
            if height > min_height:
                filtered_outlines.append(o)

    print(f"Nuclei filter: {len(filtered_outlines)} kept from {len(outlines)}"
          f"  (min_area={min_area}, min_height={min_height})")

    return filtered_outlines

#==============================================================================
def expand_outlines(outlines, expand_factor=1.3):
    """Expand each outline radially from its centroid.

    Used to create synthetic membrane boundaries from nuclei outlines
    when DAB membrane staining is weak/incomplete (e.g. 2+ images).

    Args:
        outlines: list of numpy arrays (Mx2 coordinates)
        expand_factor: >1 to expand, <1 to contract (default 1.3 = 30% expansion)

    Returns:
        expanded: list of numpy arrays with expanded coordinates
    """
    expanded = []
    for o in outlines:
        cx = np.mean(o[:, 0])
        cy = np.mean(o[:, 1])
        new_o = np.zeros_like(o, dtype=np.float64)
        new_o[:, 0] = cx + (o[:, 0] - cx) * expand_factor
        new_o[:, 1] = cy + (o[:, 1] - cy) * expand_factor
        expanded.append(new_o.astype(np.int32))
    return expanded
    
#==============================================================================
def filter_membrane_by_shape(outlines, min_area=2000, min_height=30):
    """Filter membrane outlines by shape and size criteria appropriate for membranes.

    Membranes are larger structures than nuclei, so thresholds are relaxed
    compared to filter_nuclei_annotations (area>2500, height>45).

    Args:
        outlines: list of numpy arrays having coordinates of size Mx2
        min_area: minimum area in px² to keep a membrane (default 2000)
        min_height: minimum bounding-rect height in px (default 30)

    Returns:
        filtered_outlines: list of membrane outlines that pass the filter
    """
    filtered_outlines = []
    for o in outlines:
        area = polyArea(o[:, 0], o[:, 1])
        if area > min_area:
            rect = cv2.minAreaRect(o.reshape(-1, 1, 2))
            (center_x, center_y), (width, height), angle = rect
            if height > min_height:
                filtered_outlines.append(o)
    print(f"Membrane shape filter: {len(filtered_outlines)} kept from {len(outlines)}")
    return filtered_outlines

#==============================================================================
def filter_membrane_by_dab_intensity(outlines, dab_od, min_mean_od=8, min_area=500):
    """Filter membrane outlines by mean DAB optical density inside the contour.

    Only keep outlines where the average DAB OD inside the polygon exceeds
    min_mean_od. This prevents false membranes on regions with little/no
    DAB staining (common in 1+/2+ HER2 images).

    IMPORTANT: dab_od should be computed on the RAW DAB channel (before gamma
    enhancement) so the intensity values are unmodified/true.

    Args:
        outlines: list of numpy arrays (Mx2 coordinates)
        dab_od: 2D numpy array — DAB optical density image
                (from get_OD_single_channel on raw/pre-gamma DAB channel)
        min_mean_od: minimum mean DAB OD to keep a membrane (default 8)
        min_area: minimum contour area in px² (default 500)

    Returns:
        filtered: list of outlines that pass the DAB intensity gate
    """
    img_h, img_w = dab_od.shape[:2]
    filtered = []
    rejected_low_od = 0
    rejected_small = 0

    for o in outlines:
        # Area check first (cheap)
        area = polyArea(o[:, 0], o[:, 1])
        if area < min_area:
            rejected_small += 1
            continue

        # Create mask for pixels inside the contour
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        contour = o.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)

        # Sample DAB OD values inside the contour
        od_values = dab_od[mask == 255]

        if len(od_values) == 0:
            rejected_small += 1
            continue

        mean_od = float(np.mean(od_values))

        if mean_od >= min_mean_od:
            filtered.append(o)
        else:
            rejected_low_od += 1

    print(f"DAB intensity filter: {len(filtered)} kept from {len(outlines)} "
          f"(rejected: {rejected_low_od} low OD, {rejected_small} too small, "
          f"threshold={min_mean_od})")
    return filtered

#==============================================================================
def refine_outlines_by_dab_points(outlines, dab_od, min_point_od=5, min_surviving_points=10):
    """Refine membrane outlines by keeping only contour points with high DAB OD.

    For each outline, every boundary point (x, y) is checked against the
    DAB OD image.  Only points where dab_od[y, x] >= min_point_od are kept.
    If fewer than min_surviving_points survive, the entire outline is dropped.

    IMPORTANT: dab_od should be computed on the RAW DAB channel (before gamma).

    Args:
        outlines: list of numpy arrays (Mx2 coordinates, each row = [x, y])
        dab_od: 2D numpy array — DAB optical density image
        min_point_od: minimum DAB OD at a contour point to keep it (default 5)
        min_surviving_points: discard outline if fewer points survive (default 10)

    Returns:
        refined: list of numpy arrays with only high-DAB-OD contour points
    """
    img_h, img_w = dab_od.shape[:2]
    refined = []
    n_dropped = 0

    for o in outlines:
        kept = []
        for pt in o:
            x, y = int(pt[0]), int(pt[1])
            # Bounds check
            if 0 <= y < img_h and 0 <= x < img_w:
                if dab_od[y, x] >= min_point_od:
                    kept.append([x, y])

        if len(kept) >= min_surviving_points:
            refined.append(np.array(kept, dtype=np.int32))
        else:
            n_dropped += 1

    print(f"DAB point-level filter: {len(refined)} outlines kept, "
          f"{n_dropped} dropped (min_point_od={min_point_od}, "
          f"min_points={min_surviving_points})")
    return refined

#==============================================================================
def _create_synthetic_nucleus(membrane_outline, radius=15, n_points=50):
    """Create a small circular nucleus outline at the membrane centroid.

    Used for 'hollow' membranes where no detected nucleus exists but the
    membrane structure implies a cell is present (white space inside).
    """
    cx = np.mean(membrane_outline[:, 0])
    cy = np.mean(membrane_outline[:, 1])
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y]).astype(np.int32)

#==============================================================================
def filter_membrane_annotations(outlines_m, outlines_n, cdist_threshold=40, keep_hollow=True):
    """
    Membrane-primary pairing with polygon containment check.

    For each membrane (primary object) we look for a real nucleus in two steps:
      1. **Containment check** – is any nucleus centroid geometrically INSIDE
         the membrane polygon?  Pick the closest one that is inside.
      2. **Fallback distance** – if nothing is inside, is a nucleus centroid
         within *cdist_threshold* px of the membrane centroid?
      3. If neither → hollow membrane → synthetic small-circle nucleus at the
         membrane centroid (only when *keep_hollow=True*).

    Args:
        outlines_m: list of membrane outlines (PRIMARY — iterate these)
        outlines_n: list of ALL nuclei outlines (raw, for pairing)
        cdist_threshold: fallback max centroid distance (only used when no
                         nucleus is found inside the polygon)
        keep_hollow: if True, keep membranes with no nucleus at all and
                     assign a synthetic nucleus

    Returns:
        filtered_moutlines: list of accepted membrane outlines
        paired_noutlines:   list of nucleus outlines paired with each membrane
                            (synthetic circle for truly hollow membranes)
    """
    filtered_moutlines = []
    paired_noutlines = []
    n_inside = 0
    n_fallback = 0
    n_hollow = 0

    for om in outlines_m:  # membrane is PRIMARY
        rmc = [np.mean(om[:, 0]), np.mean(om[:, 1])]  # membrane centroid
        # Build cv2 contour for containment test
        mem_contour = om.reshape(-1, 1, 2).astype(np.float32)

        # ── Step 1: find nuclei whose centroid is INSIDE the membrane ──
        inside_candidates = []
        for on in outlines_n:
            rnc_x = float(np.mean(on[:, 0]))
            rnc_y = float(np.mean(on[:, 1]))
            # pointPolygonTest: >0 inside, ==0 on edge, <0 outside
            if cv2.pointPolygonTest(mem_contour, (rnc_x, rnc_y), False) >= 0:
                d = math.dist([rnc_x, rnc_y], rmc)
                inside_candidates.append((d, on))

        if inside_candidates:
            # pick the closest nucleus that sits inside the membrane
            inside_candidates.sort(key=lambda x: x[0])
            filtered_moutlines.append(om)
            paired_noutlines.append(inside_candidates[0][1])
            n_inside += 1
            continue

        # ── Step 2: fallback – closest nucleus by centroid distance ──
        dmin = 1e9
        best_nuc = None
        for on in outlines_n:
            rnc = [float(np.mean(on[:, 0])), float(np.mean(on[:, 1]))]
            d = math.dist(rnc, rmc)
            if d < dmin:
                dmin = d
                best_nuc = on

        if dmin <= cdist_threshold:
            filtered_moutlines.append(om)
            paired_noutlines.append(best_nuc)
            n_fallback += 1
        elif keep_hollow:
            # ── Step 3: truly hollow membrane ──
            filtered_moutlines.append(om)
            paired_noutlines.append(_create_synthetic_nucleus(om))
            n_hollow += 1

    print(f"Membrane pairing: {len(filtered_moutlines)} total "
          f"({n_inside} inside, {n_fallback} fallback, {n_hollow} hollow)")

    return filtered_moutlines, paired_noutlines

#==============================================================================
def alter_anno(outlines_coord, c_coord, alter_fac=1.0, plot_anno=False):
    #alter_fac of > 1 represents expansion and < 1 represents contraction
    
    outlines_coord_new = []
    for point in range(len(outlines_coord)): 
        o = c_coord[point]
        new_coord = []
        for cc in outlines_coord[point]: #Plotting the outline
            [x1,y1]=interpolate_point(o, cc, alter_fac)
            new_coord.append([x1,y1])
            if(plot_anno):
                plt.plot(cc[0], cc[1], '.',color='red',markersize=1.0)
                plt.plot(x1, y1, '.',color='yellow',markersize=1.0)
                
        outlines_coord_new.append(new_coord)
    
    return outlines_coord_new
  
#========================================================================
def plot_annotations(in_img_file, out_img_file, nuclei_anno=None, mem_anno=None):
    #To show nuclei and membrane annotations on the same image file
    #nuclei_anno and mem_anno are list variables having coordinates of annotations
    img_rgb    = img_check(in_img_file) #To read image or use np array
    plt.figure()
    plt.axis('off')
    plt.imshow(img_rgb)
    if(nuclei_anno): 
        for point in range(len( nuclei_anno)): 
            for cc in nuclei_anno[point]: #Plotting the outline
                plt.plot(cc[0], cc[1], '.',color='red', markersize=1.0)
    if(mem_anno):
        for point in range(len( mem_anno)): 
            for cc in mem_anno[point]: #Plotting the outline
                plt.plot(cc[0], cc[1], '.',color='yellow', markersize=1.0)
    plt.tight_layout()
    plt.savefig(out_img_file, dpi=300)
    plt.close()

#========================================================================
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(elem) for elem in obj]
    else:
        return obj

#========================================================================


        
#========================================================================