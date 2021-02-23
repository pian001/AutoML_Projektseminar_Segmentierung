from PIL import Image
import numpy as np
import sys
import cv2
from os import listdir
#Printout von Matrizen nicht abkuerzen sondern komplett alle Pixel auf Konsole ausgeben
np.set_printoptions(threshold=sys.maxsize)


def convert_image(rel_path_in, rel_path_out):
    """Wandelt uebergebenes Bild 'rel_path_in' von Graustufen zu RGB-Farbe um. Pixel: (X) -> (X, X, X)  Kopiere Graustufen-Wert in alle Farbkanaele"""
    #Lade Bild mit Originalmaske im Grayscale-Modus
    img = cv2.imread(rel_path_in, cv2.IMREAD_GRAYSCALE)
    color_img = []
    #Jetzt steht in img ein 2D-Array/Matrix mit jedem Graufstufen-Wert der Pixel
    #Skaliere Pixelwerte runter
    for zeilen_index in range(0,img.__len__()):
        color_img.append([])
        for spalten_index in range(0, img[zeilen_index].__len__()):
                #Mache Pixel "farbig§
                value = img[zeilen_index][spalten_index]
                color_img[zeilen_index].append([value,value,value])
    #Schreibe Ergebnis-Bild in uebergebene Datei
    color_img = np.array(color_img,dtype=np.uint8)
    cv2.imwrite(rel_path_out, color_img)


def convert_all(input: str, out: str):
    """Konvertiert alle Bilder in 'input' nach 'out'"""
    dateien = listdir(input)
    for datei in dateien:
        out_datei = datei.replace(" ", "_")  # Leertasten durch Unterstriche ersetzen
        convert_image("input/" + datei, "output/" + out_datei)

convert_all("input","output")