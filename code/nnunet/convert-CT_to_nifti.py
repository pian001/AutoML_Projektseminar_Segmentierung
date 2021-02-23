import nrrd
from PIL import Image
import numpy as np
import sys
import cv2
import os
from pathlib import Path
import nibabel as nib
import imageio

np.set_printoptions(threshold=sys.maxsize)
# Nummerierung startet bei 0002 ? 001 fehlt

def convert_gt_to_nifti(path, filename):
	"""Konvertiert ein gt-Bild (nrrd-Format) zu nifti"""
	data, header = nrrd.read(path)
	data = data.astype(np.uint8)
	#https://stackoverflow.com/questions/55090463/how-to-convert-3d-numpy-array-to-nifti-image-in-nibabel
	ni_img = nib.Nifti1Image(data, None)  # None?
	# Datentyp in ni_img ist normalerweise np.uint16 nicht np.uint8, ABER: oben umkonvertiert zu uint8
	nib.save(ni_img, "nifti_gt/" + filename + '.nii.gz')
	
def convert_ct_to_nifti(path, filename):
	"""Konvertiert eine CT-Aufnahme (Ordner voll mit dicom .dcom Bildern) zu nifti"""
	path = Path(path)
	print(filename)
	ims = imageio.get_reader(path, mode="v") # Modus=Volumen?
	volume = ims.get_data(0)  # ein einzelnes Volumen wird gelesen (aus x vielen Slices)

	volume = volume.astype(np.uint8)
	# volume ist 3D-Array mit Tiefe x Breite x Hoehe
	# Wir brauchen (wie bei gt) Breite x Hoehe x Tiefe
	volume_right_order = np.einsum('ijk->jki', volume)
	print(np.shape(volume_right_order))  # Sollte Shape 512 x 512 x XXX sein
	
	ni_img = nib.Nifti1Image(volume_right_order, None)  # None?
	nib.save(ni_img, "nifti_ct/" + filename + '.nii.gz')
		
	
def convert_gt_all():
	"""Ruft convert_gt_to_nifti() mit entsprechenden Parametern fuer alle Dateien im Groud-truth/ Ordner auf"""
	dateien = os.listdir("Ground-truth/")
	for datei in sorted(dateien):
		if not str(datei).endswith(".nrrd"):  # Falls Datei keine .nrrd Datei (z.B. Ordner)
			continue
		# Erstelle den richtigen Dateinamen, unter dem gespeichert werden soll
		nifti_name = "CT_" + "".join(datei.split("_")[1][1:])
		convert_gt_to_nifti("Ground-truth/" +  datei, nifti_name)

		
def convert_ct_all():
	"""Ruft convert_ct_to_nifti() mit entsprechenden Parametern fuer alle Dateien im ct_files/ Ordner auf"""
	path = Path("ct_files")
	folders =dirs = [e for e in path.iterdir() if e.is_dir()]
	for folder in folders:
		filename = "CT_" + "".join(str(folder).split("_")[2])[1:] + "_0000"
		convert_ct_to_nifti(folder, filename)
		


		
convert_ct_all()
convert_gt_all()

