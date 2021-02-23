from PIL import Image
import numpy as np
import sys
import cv2
import os
from pathlib import Path
nums = set()
	
def remove_colormap(srcpath, filename, destpath):
    """Konvertiert ein farbiges Ground-Truth Pascal-VOC Bild mit Colormap zu einem gt-Bild mit indizierten Pixel_Werten (0,1,2,3,4,...,20)"""
    global nums
    # Anscheinend konvertiert PIL.Image beim Laden der Bilder die RGB-Werte in Farbindizes automatisch um
    # s. https://stackoverflow.com/questions/46289423/how-to-create-the-labeled-images-in-pascal-voc-12-as-the-ground-truth
    cm_removed = np.array(Image.open(srcpath / filename))
    cm_removed = np.where(cm_removed == 255, 0, cm_removed)
    for row in cm_removed:
        for pix in row:
            nums.add(pix)
           
    cm_removed_img = Image.fromarray(cm_removed)
    cm_removed_img.save(destpath / filename)
    
	
		
	
def remove_colormap_all():
    """Ruft remove_colormap() mit entsprechenden Parametern fuer alle Dateien im ./VOC2012/SegmentationClass/ Ordner auf"""
    basepath = Path("VOC2012")
    colormapped_path = basepath / Path("SegmentationClass")
    dest_path = basepath / Path("SegmentationClass-Colormap-removed")
    gt_files = os.listdir(colormapped_path)
    for gt_file in gt_files:
        remove_colormap(colormapped_path, gt_file, dest_path)

		
remove_colormap_all()
		
print(nums)
