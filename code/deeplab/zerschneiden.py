from PIL import Image
import numpy as np
import sys
import cv2
from os import listdir
# Regionen in den Originalmasken (180x420) die ausgeschnitten werden sollen
X_CUTS = [(0, 64), (58, 122), (116, 180)]  # 3 Regionen in X-Richtung
Y_CUTS = [(0, 64), (60, 124), (120, 184),(179, 243),(238, 302),(297, 361),(356, 420)]  # 7 Regionen in Y-Richtung

def zerschneide_image(input_path, output_path, datei):
    """Zerschneidet ein Larvenmasken-Bild (Graustufen) 'input_path'/'datei' in oben definierte Regionen und legt sie nummeriert in 'output_path' ab"""
    # Dateiendung und Dateiname aus 'datei' extrahieren
    dateiendung = datei[-4:]
    dateiname = datei[:-4]
    # Bild lesen (Graustufenmodus)
    an_image = cv2.imread(input_path + dateiname + dateiendung, cv2.IMREAD_GRAYSCALE)  # , cv2.IMREAD_GRAYSCALE
    boxen = an_image.copy()  # Kopie des Originalbildes, in das die Regionen eingezeichnet werden (Visualisierung)
    i = 0
    # Gehe oben definierte Regionen durch
    for x_min, x_max in X_CUTS:
        for y_min, y_max in Y_CUTS:
            # Schneid das Bild aus
            cropped = an_image[y_min : y_max, x_min : x_max]
            # Erstelle Dateinamen an uebergebenem 'output_path'
            out_full_path = output_path + dateiname + "_Schnitt" + str(i) + dateiendung
            # Schreibe die ausgeschniten Region an entsprechenden Pfad
            cv2.imwrite(out_full_path, cropped)
            i += 1
            # Zeichne ein Rechteck entsprechend der aktuellen Region in 'boxen'
            cv2.rectangle(boxen, (x_min, y_min), (x_max-1, y_max-1), 255)
    # nachdem alle Regionen ausgeschnitten und in 'boxen' markiert wurden, schreibe 'boxen' in extra-Ordner "vis"
    cv2.imwrite(output_path + "vis/" + dateiname + dateiendung, boxen)

def convert_all(input: str, out: str):
    """Zerschneidet alle Bilder in 'input' und schreibt sie nach 'out'"""
    dateien = listdir(input)
    for datei in dateien:
        out_datei = datei.replace(" ", "_")
        zerschneide_image(input, out, out_datei)

convert_all("input/","output/")

















