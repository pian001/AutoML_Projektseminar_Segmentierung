import scipy.io as sio
import numpy as np
from pathlib import Path
import nibabel as nib
import os
import matplotlib.pyplot as plt

def map_down(x):
    """Mappe von {0,255} auf {0,1}"""
    if x == 255:
        return 1
    else:
        return 0
def scatter3d(arr_3d):
    """Zeigt die Label als Punkte-Cloud an um ein Gefuehl fuer den Datensatz zu bekommen"""
    fig = plt.figure()
    # 3D Subplot erstellen
    ax = fig.add_subplot(111, projection='3d')
    # Shape extrahieren
    x_range,y_range,z_range = np.shape(arr_3d)
    # xs, ys und zs erst sammeln dann am Ende gebuendelt plotten (geht viel schneller)
    xs = []
    ys = []
    zs = []
    # gehe durch alle Dimensionen durch
    for x in range(x_range):
        print(str(x) + "/" + str(x_range))  # Printout zum Verfolgen des Fortschritts
        for y in range(y_range):
            for z in range(z_range):
                if arr_3d[x,y,z] == 1:  # Falls Pixel zur Ader gehoert
                    # Fuege dessen Koordinaten (X,Y,Z) in xs, ys und zs ein
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
    # Plotte alle Pixel, die Ader sind, als Point-Cloud
    ax.scatter(xs,ys,zs, marker=".", s=1)
    plt.show()
    

def convert_to_nifti(src, dst):
    """Extrahiert aus der .mat Datei 'src' alle Bestandteile (OCTA, Mask, ROI; OCTA_flat, Mask_flat, ROI_flat) und speichert sie in Ordner gruppiert in 'dst'"""
    print(src)
    # Ausgabe-Pfade erstellen
    mask_path = dst / "labels"
    octa_path = dst / "images"
    roi_path = dst / "roi"
    mask_path.mkdir(parents=True, exist_ok=True)
    octa_path.mkdir(parents=True, exist_ok=True)
    roi_path.mkdir(parents=True, exist_ok=True)
    
    # Dateinamen aus der 'src' bestimmen
    filename = src.name
    # Dateiendung ".mat" abschneiden
    filename = str(filename)[:-4]
    
    # Daten laden
    data = sio.loadmat(src)
    # Octa, ROI und Maske rausziehen
    if "OCTA" in data:  # normale Version (nicht flat)
        print(type(data["OCTA"][0,0,0]))
        octa = data["OCTA"]
        roi = data["ROI"]
        mask = data["Mask"]
        # evtl unnoetig? :
        #map(lambda x:1 if x == 255 else 0, mask)
    else:  # flat Version
        print(type(data["OCTA_flat"][0,0,0]))
        octa = data["OCTA_flat"]
        roi = data["ROI_flat"]
        mask = data["Mask_flat"]
        # evtl unnoetig? :
        #map(lambda x:1 if x == 255 else 0, mask)
    # Aus 3D-Arrays ein Nifti erstellen
    ni_octa = nib.Nifti1Image(octa, None)  # None?
    ni_roi = nib.Nifti1Image(roi, None)  # None?
    ni_mask = nib.Nifti1Image(mask, None)  # None?
    # eventuell als Point-Cloud anzeigen um ein gefuehl fuer den Datensatz zu bekommen
    #scatter3d(mask)
    #return  # dann return, da ein Sample zu visualisieren reicht
    
    # Niftis abspeichern in entsprechenden Pfaden
    nib.save(ni_octa, octa_path / (filename + "_0000.nii.gz"))  # _0000 am Ende hinzugefuegt fuer nnUNet
    nib.save(ni_roi, roi_path / (filename + ".nii.gz"))
    nib.save(ni_mask, mask_path / (filename + ".nii.gz"))
    


def convert_all(src, dst):
    """Konvertiert alle .mat Dateien  zu Niftis"""
    dateien = os.listdir(src)
    for datei in sorted(dateien):
        if not str(datei).endswith(".mat"):# Falls Datei keine .mat Datei (z.B. Ordner, andere Dateien, ...)
            continue
        convert_to_nifti(src / datei, dst)
        #return

    
convert_all(Path("retina-3d"), Path("converted_to_nifti"))
