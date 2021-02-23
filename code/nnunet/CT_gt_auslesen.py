import nrrd
from PIL import Image
import numpy as np
import sys
import cv2
import os
from pathlib import Path

breite = set()
hoehe= set()
tiefe = set()
def extract_masks(filename):
    """Extrahiert alle slices als .png aus dem .nrrd Format um die Masken aus .nrrd Dateien zu sichten und einen Ueberblick ueber den Datensatz zu bekommen"""
	global breite, hoehe, tiefe
	data, header = nrrd.read(filename)
	breite.add(header["sizes"][0])
	hoehe.add(header["sizes"][1])
	tiefe.add(header["sizes"][2])
	print(header)
	print(np.shape(data))
	for i in range(0,header["sizes"][2]):
		#Schreibe Ergebnis-Bild in uebergebene Datei
		img = data[:, :, i]
		img *= 255
		img = np.array(img,dtype=np.uint8)
		slice_pfad = filename.split(".")[0] + "/slice" + "{:03d}".format(i) + ".png"
		# Schreibe slice als png
		#cv2.imwrite(slice_pfad, img)
def extract_all():
	dateien = os.listdir("Ground-truth/")
	for datei in dateien:
		if not str(datei).endswith(".nrrd"):
			continue
		result_ordner = Path("Ground-truth/sliced") / Path(datei.split(".")[0])
		result_ordner.mkdir(parents=True, exist_ok=True)
		extract_masks("Ground-truth/" + datei)

extract_all()

print("Breiten:")
print(breite)
print("Hoehen:")
print(hoehe)
print("Tiefen:")
print(tiefe)
	