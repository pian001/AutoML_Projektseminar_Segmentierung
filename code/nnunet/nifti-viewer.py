import nrrd
from PIL import Image
import numpy as np
import sys
import cv2
import os
from pathlib import Path
import nibabel as nib
import imageio
import math

seen_numbers = dict()
# Extrahiert aus einem .nii.gz Nifti Format alle Slices
def extract_nifti(folder, filename):
    """Extrahiert aus einem Nifti in 'folder' mit Namen 'filename' alle Slices und speichert sie durchnummeriert im selben Ordner"""
    global seen_numbers
    # nifti laden
    nifti_volume=nib.load(folder + filename).get_data()
    # je nach Datensatz: Datentyp zu uint8 aendern, damit cv2 daraus ein png schreiben kann
    #nifti_volume = nifti_volume.astype(np.uint8)
    # Anzahl an Slices herausfinden
    num_slices = np.shape(nifti_volume)[2]
    
    
    # Um 90 Grad im Uhrzeigersinn drehen und horizontal spiegeln, da NNUnet die 2D (!) Bilder intern dreht und spiegelt und man dann schlecht vergleichen kann
    #nifti_volume = np.rot90(nifti_volume)
    #nifti_volume = np.flip(nifti_volume, axis=0)
       
    # Hochskalieren auf 0...255
    skaliere_hoch(nifti_volume)

    # Pascal Label zu Pascal Color Codes
    #nifti_volume = convert_to_pascal_colors(nifti_volume)
    
    print(np.shape(nifti_volume))
    
    
    print(type(nifti_volume[0,0,0]))
    i = 0
    # durch alle Slices durchgehen
    for slice_index in range(0, num_slices):
        # slice aus Volumen holen
        slice = nifti_volume[...,slice_index]
        # Slice pixelweise durchgehen
        counter = 0
        #for row in slice:
            #for pixel in row:
                #if type(pixel) is not np.ndarray:
                    # Pixelwert (falls Graustufen-Bild) in seen_numbers packen
                    #if pixel in seen_numbers:
                        #seen_numbers[pixel] += 1
                    #else:
                        #seen_numbers[pixel] = 1
                    #if pixel != 0:
                        #counter += 1
        #print(str(counter) + " Pixel in " + str(i))
        # Wir muessen bei farbigen Bildern die RGB Werte im Array zu BGR Werten "tauschen", damit opencv die Farben richtig schreibt... Warum auch immer das so komisch definiert wurde... S/W-Bilder bleiben dadurch unveraendert (?)
        #slice = cv2.cvtColor(slice, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(folder) + str(filename) + str(i) + ".png", slice)
        i+=1
def extract_all():
    """Extrahiert alle Nifti-Dateien in dem unten angegebenen Ordner 'nifti-viewer/'"""
    dateien = os.listdir("nifti-viewer/")
    i = 0
    # alle Dateien durchgehen
    for datei in dateien:
        # wenn es kein Nifti ist, ueberspringe
        if not str(datei).endswith(".nii.gz"):
            continue
        # ansonsten Nifti auspacken
        i+=1
        extract_nifti("nifti-viewer/",datei)
        print(i)  # Nummer printen, um Fortschritt besser verfolgen zu koennen


def skaliere_hoch(array):
    """Skaliert das uebergebene Array auf einen Wertebereich 0...255 hoch, sodass dieser voll ausgenutzt wird. Uebergebenes Array sollte nur Werte zwischen 0 und 255 enthalten"""
    faktor = math.floor(255 / np.amax(array))
    array *= faktor  # zum Hochskalieren/Mappen auf 0...255
    
def convert_to_pascal_colors(array):
    """Wandelt ein S/W-Bild (Pixelwert = Label = Klassennummer) in ein farbiges Bild um entsprechend der Pascal-Color-Codes"""
    # Maximum und Shape herausfinden
    max_ = np.amax(array)
    shape = np.shape(array)
    # nur wenn vorher ausschliesslich Label 0..20 vorhanden waren handelt es sich um GT Bild
    # ("Sicherheitscheck" ob es sich um Maske von Pascal handelt, da in diese Methode evtl. auch Original Bilder aus dem Pascal Datensatz gegeben werden koennen)
    if max_ <= 20:
        cmap = pascal_color_map()  # Hole Array mit den Color-Mappings zu Pascal (Label bzw. Klasse X -> RGB mit cmap[x] = RGB)
        # erstelle neues, erweitertes Array (RGB) mit alter Graustufen-Array Shape
        rgb_array = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
        # Fuelle rgb_array mit entsprechenden RGB Werten passend zu den Klassennummern in 'array'
        for zeile in range(0, shape[0]):
            for spalte in range(0, shape[1]):
                label = array[zeile, spalte, 0]
                rgb_array[zeile, spalte, :] = cmap[label]
        rgb_array = rgb_array[..., np.newaxis]  # Kompaibilitaet mit der 'extract_nifti()' Methode
        return rgb_array
    return array
    

def pascal_color_map(N=256, normalized=False):
    """gibt Array zurueck, das index (Pixel Label) zu RGB-Farbcode mappt entsprechend den Pascal-VOC-Farbcodes
    von https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae kopiert
    """
    
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


extract_all()

# Printe alle (Graustufen-) Werte aus die in allen konvertierten Niftis gefunden wurden
if seen_numbers.__len__() <= 500:  # Falls es sich um float-Pixelwerte handelt werden es evtl. sehr viele, dann wird nicht geprintet
    print(seen_numbers)
    
