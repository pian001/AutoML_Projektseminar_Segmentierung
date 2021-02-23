from PIL import Image
import numpy as np
import sys
import cv2
from os import listdir
#Printout von Matrizen nicht abkuerzen sondern komplett alle Pixel auf Konsole ausgeben
np.set_printoptions(threshold=sys.maxsize)

def umrandung_pixel(img, zeile, spalte):
    """Nimmt ein 2D-Array img und eine Zeile und Spalte darin entgegen und
    markiert alle direkt benachbarten Pixel
    (oben drueber, unten drunter, links, rechts) als Umrandung, 
    falls dort Hintergrund waere"""
    #Oben drueber
    #Index existiert und oben drueber ist Hintergrund
    if zeile - 1 > 0 and img[zeile-1][spalte]==0:
        img[zeile-1][spalte]=255#markiere dortigen Pixel als Umrandung (=255)
    #Links daneben
    #Index existiert und links daneben ist Hintergrund
    if spalte - 1 > 0 and img[zeile][spalte-1]==0:
        img[zeile][spalte-1]=255#markiere dortigen Pixel als Umrandung (=255)
    #Rechts daneben
    #Index existiert und rechts daneben ist Hintergrund
    if spalte + 1 < img[zeile].__len__() and img[zeile][spalte+1]==0:
        img[zeile][spalte+1]=255#markiere dortigen Pixel als Umrandung (=255)
    #Unten drunter
    #Index existiert und unten drunter ist Hintergrund
    if zeile + 1 < img.__len__() and img[zeile+1][spalte]==0:
        img[zeile+1][spalte]=255#markiere dortigen Pixel als Umrandung (=255)

def umrandung_zeichnen(img):
    """Zeichnet eine Umrandung um Pixel, die != 0 sind (also die nicht zum Hintergrund gehoeren)"""
    for zeilen_index in range(0,img.__len__()):
        for spalten_index in range(0, img[zeilen_index].__len__()):
            if img[zeilen_index][spalten_index] == 1:
                #Markiere benachbarte Pixel
                umrandung_pixel(img, zeilen_index, spalten_index)
def convert_image(rel_path_in, rel_path_out):
    """Konvertiert ein Bild (rel_path_in) mit Hintergrund=0 und Objekt=255 zu einem Bild (rel_path_out)
    mit Hintergrund=0, Objekt=1 und zeichnet eine Umrandung UM das Objekt herum
    (eigentlich zum Hintergrund gehoerende Pixel
    werden zur Umrandung; Die Pixel des Objekts bleiben erhalten)"""
    #Lade Bild mit Originalmaske im Grayscale-Modus
    img = cv2.imread(rel_path_in, cv2.IMREAD_GRAYSCALE)
    #Jetzt steht in img ein 2D-Array/Matrix mit jedem Graufstufen-Wert der Pixel
    #Skaliere Pixelwerte runter
    for zeilen_index in range(0,img.__len__()):
        for spalten_index in range(0, img[zeilen_index].__len__()):
            #Hole Pixel-Wert an aktueller Stelle
            wert = img[zeilen_index][spalten_index]
            #Falls Wert != 0 (also Pixel gehoert nicht zum Hintergrund)
            if wert != 0:  # != 0 statt == 255, da auch z.B. 253er Werte in den Masken existieren... (vielleicht durch Konvertierung in anderes Format?)
                #Markiere den Pixel mit 1 statt 255
                img[zeilen_index][spalten_index]=1
    #print(img)
    #*NACHDEM* alle Pixel skaliert wurden, zeichne Umrandung der Objekte
    umrandung_zeichnen(img)
    #change_color(img, 0, 255)
    #change_color(img, 1, 0)
    #print(img)
    #Schreibe Ergebnis-Bild in uebergebene Datei
    cv2.imwrite(rel_path_out, img)

def change_color(img, vorher, nachher):
    """Ersetzt alle Werte 'vorher' im Array 'img' durch Wert 'nachher'"""
    for zeile in range(0, img.__len__()):
        for spalte in range(0, img[zeile].__len__()):
            if img[zeile][spalte] == vorher:
                img[zeile][spalte] = nachher

def convert_all(input: str, out: str):
    """Konvertiert alle Dateien in 'input' nach 'out'"""
    dateien = listdir(input)
    for datei in dateien:
        out_datei = datei.replace(" ", "_")  # Leertasten durch Unterstriche ersetzen
        convert_image(input + datei, out + out_datei)

convert_all("input/","output/")
















