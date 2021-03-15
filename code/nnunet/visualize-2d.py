from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
import json
import math
from PIL import Image
import numpy as np
import random
import nibabel as nib

NNUNET_ROOT_PATH = Path("G:/AutoML/NNUnet/")  # PC
# NNUNET_ROOT_PATH = Path("D:/AutoML/NNUnet/")  # Laptop
# NNUNET_ROOT_PATH = Path("/home/uni/AutoML-Share/NNUnet/")  # Linux

ORIG_IMGS_DATEIENDUNG = ".jpg"  # Dateiendung der Original-Bilder
ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/VOC2012/JPEGImages")  # Pfad zu den Original-Bildern

ORIG_MASKS_DATEIENDUNG = ".png"  # Dateiendung der Original-Masken
ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/VOC2012/SegmentationClass/")  # Pfad zu den Original-Masken

PRED_MASKS_DATEIENDUNG = ".nii.gz0.png"  # Dateiendung zu den vorhergesagten Masken (der Prediction)
PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path(
    "nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTrTs/")  # Pfad zu den vorhergesagten Masken (der Prediction)

NUM_SAMPLES = 2  # Anzahl an Samples (Bilder) die visualisiert werden sollen (jeweils fuer "Beste", "Schlechteste" und "Median")
NUM_SPALTEN = 3  # Anzahl an Spalten *IM* Subplot ("Original", "Original-Maske", "Prediction")
TITLE_SPACING = 115  # Hard-gecodeter Abstand im Titel um Sub-Titel fuer jede Spalte zu "erzeugen"

# Verwende *immer* die Pascal Farben, andere Datensaetze kriegen dann halt farbige Punkte je nach Klassennummer ¯\_(^.^)_/¯
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
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

cmap = pascal_color_map()

def visualize(path_to_json: Path, title: str, rgb: bool):
    """Visualisiert (zeigt die zugehoerigen Bilder, Masken und Predictions an) die schlechtesten, besten und mittleren (median) Bilder zu einer summary.json ('path_to_json') aus nnUNet
    'title' wird der Diagrammtitel (z.B. Task100 CT)
    'rgb' gibt an, ob die Originalbilder farbig (True) oder schwarzweiss (False) sind
    """
    # Lese die .json Datei
    with open(path_to_json) as json_file:
        json_data = json.load(json_file)
    # Hauptplot (enthaelt 3 Spalten "Beste", "Schlechteste", "Median" als GridSpec)
    hauptplot = plt.figure(figsize=(20, 10))
    gs_haupt = gridspec.GridSpec(1, 3, figure=hauptplot)
    # Sub-Plot mit besten Predictions
    gs_sub_left = gs_haupt[0].subgridspec(NUM_SAMPLES, NUM_SPALTEN, wspace=0.05)
    # Sub-Plot mit schlechtesten Predictions
    gs_sub_middle = gs_haupt[1].subgridspec(NUM_SAMPLES, NUM_SPALTEN, wspace=0.05)
    # Sub-Plot mit median Predictions
    gs_sub_right = gs_haupt[2].subgridspec(NUM_SAMPLES, NUM_SPALTEN, wspace=0.05)
    # Setze den Diagrammtitel aus 'title' und pfusche Sub-Titel fuer jeden Sub-Graph dazu mit dem hardgecodeten TITLE_SPACING
    plt.title(
        title + " (Original, Maske, Prediction)\n Schlechteste" + TITLE_SPACING * " " + "Beste" + TITLE_SPACING * " " + "Median")
    # Achsen im Hauptplot ausschalten
    plt.axis('off')
    # Offset der Subplots im ganzen Fenster (soll das Fenster fast voll ausfuellen, nur schmale Raender aussen)
    plt.subplots_adjust(left=0, top=0.9, bottom=0.1, right=1, hspace=0.1, wspace=0.2)

    # Gehe alle Subplots durch
    for aktiver_subplot in [gs_sub_left, gs_sub_middle, gs_sub_right]:

        offset = 0
        if aktiver_subplot is gs_sub_left:  # links=schlechteste Predictions
            # Sortiere nach "Dice" aufsteigend (schlechteste zuerst); Ueber *ALLE* Klassen gemittelt
            json_data["results"]["all"].sort(key=lambda x: avg_wert_ueber_alle_klassen(x, "Dice"),
                                             reverse=False)  # reverse = False sortiert ascending
        elif aktiver_subplot is gs_sub_middle:  # mitte=beste Predictions
            # Sortiere nach "Dice" absteigend (beste zuerst); Ueber *ALLE* Klassen gemittelt
            json_data["results"]["all"].sort(key=lambda x: avg_wert_ueber_alle_klassen(x, "Dice"),
                                             reverse=True)  # reverse = True sortiert descending
        else:  # ansonsten (median): lege offset so, dass mittlerer Bereich im Sample-Array abgedeckt wird
            offset = math.floor(len(json_data["results"]["all"]) / 2 - NUM_SAMPLES / 2)

        # Schleife durch die Anzahl der gewuenschten Samples pro Subplot
        subplot_index = 0
        for sample_index in range(0, NUM_SAMPLES):
            # Hole aktuelles Sample (json Infos darueber)
            current_sample = json_data["results"]["all"][sample_index + offset]

            # Original-Bilder plotten
            dateiname = extract_filename(current_sample[
                                             "reference"]) + ORIG_IMGS_DATEIENDUNG  # Hole Dateinamen aus dem json Abschnitt fuer aktuelles Sample
            orig_img = mpimg.imread(ORIG_IMGS_PFAD / dateiname)
            # Fuege Subplot hinzu in aktuelles GridSpec (links, mitte, rechts)
            hauptplot.add_subplot(aktiver_subplot[math.floor(subplot_index / NUM_SPALTEN), subplot_index % NUM_SPALTEN])
            # Achsen ausschalten
            plt.axis('off')
            # Falls Original-Bild farbig:
            if rgb:
                # Plotte mit default cmap
                plt.imshow(orig_img, cmap=None)
            else:
                # Fuer Graustufen: Plotte mit cmap="gray"
                plt.imshow(orig_img, cmap="gray")
            subplot_index += 1

            # Original-Masken plotten
            dateiname = extract_filename(current_sample[
                                             "reference"]) + ORIG_MASKS_DATEIENDUNG  # Hole Dateinamen aus dem json Abschnitt fuer aktuelles Sample
            orig_mask = mpimg.imread(ORIG_MASKS_PFAD / dateiname)
            # Fuege Subplot hinzu in aktuelles GridSpec (links, mitte, rechts)
            hauptplot.add_subplot(aktiver_subplot[math.floor(subplot_index / NUM_SPALTEN), subplot_index % NUM_SPALTEN])
            # Achsen ausschalten
            plt.axis('off')
            plt.imshow(orig_mask, cmap='gray')
            subplot_index += 1

            # Predictions plotten
            dateiname = extract_filename(current_sample[
                                             "reference"]) + PRED_MASKS_DATEIENDUNG  # Hole Dateinamen aus dem json Abschnitt fuer aktuelles Sample
            pred_mask = mpimg.imread(PRED_MASKS_PFAD / dateiname)
            # Fuege Subplot hinzu in aktuelles GridSpec (links, mitte, rechts)
            hauptplot.add_subplot(aktiver_subplot[math.floor(subplot_index / NUM_SPALTEN), subplot_index % NUM_SPALTEN])
            # Achsen ausschalten
            plt.axis('off')
            plt.imshow(pred_mask, cmap='gray')
            subplot_index += 1

            # Abbruch, wenn genuegend Subplots erstellt (je Sample werden NUM_SPALTEN-viele Subplots erstellt)
            if subplot_index >= (NUM_SAMPLES * NUM_SPALTEN):
                break
    plt.show()


def extract_filename(path) -> str:
    """Extrahiert den Dateinamen eines Bildes aus dem Pfad"""
    # Pfad abschneiden, nur alles nach dem letzten Schraegstrich (/) bleibt ueber
    filename = str(path).split("/")[-1]  # -1 = letztes Element in der Liste
    # Dateiendung .nii.gz abschneiden
    filename = filename.replace(".nii.gz", "")
    return filename

def get_color(klasse):
    """Berechne (Pascal-) Color zu Klassennummer"""
    return '#%02x%02x%02x' % (cmap[int(klasse)][0], cmap[int(klasse)][1], cmap[int(klasse)][2])
def avg_wert_ueber_alle_klassen(klassendict, feld_name):
    """Hole aus dem json Abschnitt zu einem Sample den durchschnittlichen Dice-Wert ueber *alle* vorhandenen Klassen (korrekt gewichtet nach Pixeln) raus"""
    summe = 0
    menge = 0  # Anzahl gefundener Pixel
    for key in klassendict:
        # Gehe durch alle Klassen durch ("reference" und "test" ueberspringen) und pruefe, ob die Klasse ueberhaupt im Bild vorhanden ist (sonst ist Dice NaN)
        if key not in ["reference", "test"] and klassendict[key]["Total Positives Reference"] != 0:
            summe += klassendict[key][feld_name] * klassendict[key]["Total Positives Reference"]  # Wert fuer aktuelle Klasse auf Summe draufrechnen (multipliziert mit Anzahl an Pixeln zu der Klasse in Original-Maske)
            menge += klassendict[key]["Total Positives Reference"]  # Menge um gefundene Pixel in Original-Maske hochzaehlen
    if menge == 0:  # Falls keine Klasse gefunden (sollte eigl. nie vorkommen, ausser Maske ist komplett schwarz / nur Background)
        return 0
    return summe / menge  # Gebe Durchschnitt zurueck (korrekt gewichtet nach Pixeln)

def scatterplot_haeufigkeiten(jsonpath, nifti_ordner, diagrammtitel, label_names, stichproben_anteil = 1):
    """Scatterplot fuer anteilige Haeufigkeitsverteilung im Split (zur summary.json), Jedes Sample kriegt eine x-Stelle, alle Klassen mit zugehoeriger Auftrittswahrscheinlichkeit werden vertikal darueber geplottet
    Es kann eine Begrenzung der Stichprobe angegeben werden 'stichproben_anteil' wenn der Graph sonst zu unuebersichtlich wird.
    Durchschnittslinien sind immer auf den *ganzen* Split zur summary.json bezogen, unabhaengig von der Stichprobengroesse
    """
    # Oeffne Json-Datei summary.json
    with open(jsonpath) as jsonfile:
        summary_json = json.load(jsonfile)
    total_pixels = 0  # Pixel im gesamten Split, alle Samples zusammen
    i = 0
    # Gehe durch jedes Sampel durch
    for sample in summary_json["results"]["all"]:
        print(i)  # Sample Nummer printen um Fortschritt zu verfolgen
        i+=1
        # Lade Prediction als Nifti und extrahiere Shape, um Gesamtzahl der Pixel zu bestimmen um Haeufigkeiten zu gewichten
        sample_nifti = nib.load(nifti_ordner / (extract_filename(sample["test"]) + ".nii.gz"))
        #print(sample_nifti.shape)
        total_pixels_per_sample = 1
        for dim in sample_nifti.shape:
            total_pixels_per_sample *= dim  # Berechne Anzahl an Pixeln im Volumen des aktuellen Samples
        total_pixels += total_pixels_per_sample  # Addiere aktuelle Pixelanzahl auf Gesamtanzahl aller Samples drauf
        x_value = random.uniform(0, 1)  # x_Stelle fuer aktuelles Sample auswaehlen
        if x_value > stichproben_anteil:  # Nehme nur 'stichproben_anteil' aller Samples in Plot mit auf, wird sonst unuebersichtlich
            continue
        x_value /= stichproben_anteil  # Wertebereich X wieder auf 0...1 ausdehnen


        
        # Schleife ueber alle Klassen je Sample (auch wenn nicht tatsaechlich im Sample vorhanden)
        for klasse in sample:
            if klasse in ["reference","test"]:  # Meta-Daten ueberspringen
                continue
            #print(klasse)
            plt.scatter(x_value, sample[klasse]["Total Positives Reference"] / total_pixels_per_sample * 100, marker="x", s=10, color=get_color(klasse))
    
    # Mittelwerte plotten (sind immer genau unabhaengig von 'stichproben_anteil')
    for klasse in summary_json["results"]["mean"]:
        # Jedes Sample hat im Durchschnitt mean_pixels_per_sample
        mean_pixels_per_sample = total_pixels / len(summary_json["results"]["all"])
        # Durchschnitts-Auftritts-Wahrscheinlichkeit je Klasse ueber alle Sample gemittelt in Prozent
        mean = summary_json["results"]["mean"][klasse]["Total Positives Reference"] / mean_pixels_per_sample * 100
        plt.hlines(mean, 0, 1, color=get_color(klasse))  # Linie zeichnen
    plt.ylabel('Anteil je Objekt in %')
    plt.title(diagrammtitel)
    # Legende bauen
    legend_labels = []
    for klasse in summary_json["results"]["mean"]:
        legend_labels.append(mpatches.Patch(color=get_color(klasse), label=label_names[int(klasse)]))
    plt.legend(handles=legend_labels)
    plt.show()

def scatterplot_dice(train, test, label_names, test_exists = True):
    """Scatterplot mit einem Punkt je Klasse je Sample fuer Train- und Testsplit (falls vorhanden) nebeneinander mit Linie als Durchschnitt"""
    # Oeffne summary.json fuer Train-Split
    with open(train) as train_evaluation_json:
        json_train = json.load(train_evaluation_json)
    # Oeffne summary.json fuer Test-Split, falls vorhanden
    if test_exists:
        with open(test) as test_evaluation_json:
            json_test = json.load(test_evaluation_json)
    # Teile Hauptplot in der Mitte (1 Zeile, 2 Spalten)
    fig, axs = plt.subplots(1, 2 if test_exists else 1)  # Spaltenaufteilung abhaengig davon, ob Testsplit ueberhaupt existiert
    # Fuer jeden der beiden Scatterplots (Train und Test)
    if test_exists:
        axs_train = axs[0]
        axs_test = axs[1]
    else:
        axs_train = axs
        axs_test = None
    splits = [("Train", json_train["results"], axs_train)]
    if test_exists:
        splits.append(("Test", json_test["results"], axs_test))
    for (title, sample_liste, ax) in splits:
        # ax.set_yscale('log')  # Falls die Skalierung durch sehr gute Werte komisch wird, stelle auf logarithmisch um
        # Setze den Titel entsprechend
        ax.set_title(title)
        i = 0
        # Gehe alle Samples durch
        for sample in sample_liste["all"]:
            print(i)  # printout um Fortschritt zu verfolgen
            i += 1
            # Fuer jede Klasse im Sample
            for key in sample:
                # Muss Klassen-Eintrag sein, keine Meta-Infos in json; Klasse mus tatsaechlich vorkommen
                if key in ["reference", "test"] or sample[key]["Total Positives Reference"] == 0:
                    continue
                # Plotte den Punkt an zufaellige x-Stelle damit sich nicht alles stackt
                ax.scatter(random.uniform(0, 1), sample[key]["Dice"], marker="x", c=get_color(key), s=10)
        # Erstelle Legende
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        if not test_exists:  # Wenn es keinen Test-Split gibt, ueberspringe das Zeichnen des Graphs dafuer
            break

    # Hole alle existierenden keys
    existinglabels = [key for key in json_train["results"]["all"][
        0]]  # Hier sind immer alle Klassen drin, nicht vorhandene Klassen haben NaN bzw 0 Werte aber trotzdem einen Eintrag
    # Die Keys fuer Metadaten ueber die Samples koennen ignoriert werden, wir brauchen nur die Klassennummern die vorkommen
    existinglabels.remove("reference")
    existinglabels.remove("test")
    labels = []
    for label in existinglabels:  # gehe durch alle existierenden Label durch
        # Hole Color-Code fuer das Label
        # Fuege mpatches.Patch fuer die aktuelle Klassennummer mit richtiger (Pascal VOC2012-) Farbe und Namen aus 'label_names' in 'labels' ein
        labels.append(mpatches.Patch(color=get_color(label), label=label_names[int(label)]))

        # Ziehe Durchschnitts-Linie fuer aktuelle Klassennummer in Train
        mean = json_train["results"]["mean"][label][
            "Dice"]  # Durchschnitt kommt aus dem von nnUNet berechneten Durchschnitt
        axs_train.hlines(mean, 0, 1, color=get_color(label))
        if test_exists:
            # und in Test
            mean = json_test["results"]["mean"][label]["Dice"]
            axs_test.hlines(mean, 0, 1, color=get_color(label))

    # Setze die Legende entsprechend 'labels', das vorher zusammengebaut wurde (Farbe + Klassenname fuer jede Klassennummer)
    axs_train.legend(handles=labels)
    if test_exists:  # Falls es ueberhaupt einen Testsplit gibt
        axs_test.legend(handles=labels)
    
    
    avg_train = avg_wert_ueber_alle_klassen(json_train["results"]["mean"], "Dice")
    print("Durchschnitt-Dice auf Train ueber alle Klassen korrekt gewichtet nach Pixeln: " + str(avg_train))
    if test_exists:  # Falls es ueberhaupt einen Testsplit gibt
        avg_test = avg_wert_ueber_alle_klassen(json_test["results"]["mean"], "Dice")
        print("Durchschnitt-Dice auf Test ueber alle Klassen korrekt gewichtet nach Pixeln: " + str(avg_test))
    
    
    
    plt.show()


def vis_task_nummer(task_nummer: int):
    """Vereinfachter Aufruf einer Visualisierung zu einer Tasknummer durch vorgefertigte Variablen und Befehle"""
    global ORIG_IMGS_PFAD, ORIG_MASKS_PFAD, PRED_MASKS_PFAD
    global ORIG_IMGS_DATEIENDUNG, ORIG_MASKS_DATEIENDUNG, PRED_MASKS_DATEIENDUNG
    # Retina 3D Datensatz
    if task_nummer == 108:
        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_fullres/imagesTr")
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 108 Retina 3D - Train", label_names)
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_fullres/imagesTs")
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 108 Retina 3D - Test", label_names)
        
        # 3d Fullres
        print("3D_FULLRES")
        scatterplot_dice(NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task108_Retina-3d_fullres/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path(
                        "nnUNet_predictions/Task108_Retina-3d_fullres/imagesTs/") / "summary.json",
                    label_names)
        # 2D
        print("2D")
        scatterplot_dice(NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task108_Retina-3d_2dNet/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path(
                        "nnUNet_predictions/Task108_Retina-3d_2dNet/imagesTs/") / "summary.json",
                    label_names)
        # Ensemble 2D und 3D_fullres
        print("ENSEMBLE")
        scatterplot_dice(NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task108_ensemble-2d_3d_fullres/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path(
                        "nnUNet_predictions/Task108_ensemble-2d_3d_fullres/imagesTs/") / "summary.json",
                    label_names)
    # CT Datensatz in INT16 Originalform
    elif task_nummer == 109:
        label_names = ["Background",
                       "Kalziumablagerung"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        # Haeufigkeiten plotten (sind fuer 2d, 3d_fullres und cascade gleich)
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task109_CT_int16_3d_fullresNet_2000Epochs/imagesTr")
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Häufigkeitsverteilung 109 CT nur Trainsplit", label_names)
        
        # 3D_FULLRES 2000 Epochen
        print("109-3D_FULLRES mit 2000 Epochen")
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", None, label_names, False)  # Scatterplot ohne Testsplit (None bzw False)
        
        # 3D_CASCADE
        print("109-3D_CASCADE")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task109_CT_int16_3d_cascade/imagesTr")
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", None, label_names, False)  # Scatterplot ohne Testsplit (None bzw False)
        
        # 2D
        print("109-2D")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task109_CT_int16_2dNet/imagesTr")
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", None, label_names, False)  # Scatterplot ohne Testsplit (None bzw False)
        
    # Larven ohne split (Train=Test)
    elif task_nummer == 200:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".png"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/imgs")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/masks/")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task200_Larven_train_eq_test/imagesTr/")
        label_names = ["Background", "Larve"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        # Hauefigkeitsverteilung im Trainsplit plotten
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 200 Larven nur Trainsplit - Train", label_names)
        # Dice-Koeffizientenverteilung plotten
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", None, label_names, False)  # Scatterplot ohne Testsplit
        
        # Visualisieren
        visualize(PRED_MASKS_PFAD / "summary.json", "Task200_Larven_train_eq_test", False)


    # Larven drittel split
    elif task_nummer == 201:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".png"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/imgs")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/masks/")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task201_Larven_split_drittel-ist-test/imagesTr/")

        label_names = ["Background",
                       "Larve"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        # Hauefigkeitsverteilung im Trainsplit plotten
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 201 Larven drittel split - Train", label_names)
        # Dice-Koeffizientenverteilung plotten
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task201_Larven_split_drittel-ist-test/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task201_Larven_split_drittel-ist-test__TRAIN", False)

        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task201_Larven_split_drittel-ist-test/imagesTs/")
        # Hauefigkeitsverteilung im Testsplit plotten
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 201 Larven drittel split - Test", label_names)
        # Visualisieren der Test-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task201_Larven_split_drittel-ist-test__TEST", False)



    # Augen (Retina 2D) drittel split
    elif task_nummer == 203:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".jpg"
        ORIG_MASKS_DATEIENDUNG = ".tif"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Augen-Adern-Datensatz/images/")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Augen-Adern-Datensatz/manual1/")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task203_Augenadern-drittel-test/imagesTr/")

        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        # Dice-Koeffizientenverteilung plotten
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task203_Augenadern-drittel-test/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train- und Testsamples ausgelassen, da kaum Unterschied zu "Task 205 minimal Trainign Samples", Hauefigkeitsverteilung ebenfalls





    # Pascal VOC2012 achtel Split
    elif task_nummer == 204:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".png"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/VOC2012/labeled-PNGImages")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/VOC2012/SegmentationClass")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTr/")

        # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        label_names = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
                       'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa',
                       'Train', 'TVmonitor']
        # Hauefigkeitsverteilung im Trainsplit plotten (10% Stichprobe)
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 204 PascalVOC Train (10% Stichprobe)", label_names, 0.1)
        # Dice-Koeffizientenverteilung plotten
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task204_PascalVOC2012-achtel-split__TRAIN", True)  # Original-Bilder sind RGB => True
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTs/")
        # Hauefigkeitsverteilung im Testsplit plotten (50% Stichprobe)
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 204 PascalVOC Test (50% Stichprobe)", label_names,0.5)
        # Visualisieren der Test-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task204_PascalVOC2012-achtel-split__TEST", True)  # Original-Bilder sind RGB => True





    # Augen (Retina 2D) minimal training samples (13 Training, Rest Test)
    elif task_nummer == 205:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".jpg"
        ORIG_MASKS_DATEIENDUNG = ".tif"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Augen-Adern-Datensatz/images/")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Augen-Adern-Datensatz/manual1/")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task205_Augen_weniger/imagesTr/")

        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        # Hauefigkeitsverteilung im Trainsplit plotten 
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 205 Retina-Train 2D minimal", label_names)
        # Dice-Koeffizientenverteilung plotten
        scatterplot_dice(PRED_MASKS_PFAD / "summary.json",
                    NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task205_Augen_weniger/imagesTs/") / "summary.json",
                    label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task205_Augen_weniger__TRAIN", False)
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task205_Augen_weniger/imagesTs/")
        # Hauefigkeitsverteilung im Testsplit plotten 
        scatterplot_haeufigkeiten(PRED_MASKS_PFAD / "summary.json", PRED_MASKS_PFAD, "Haeufigkeitsverteilung 205 Retina-Test 2D minimal", label_names)
        # Visualisieren der Test-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task205_Augen_weniger__TEST", False)


# 2D Datensaetze (Nummer 2XX)
#vis_task_nummer(200)
vis_task_nummer(201)
#vis_task_nummer(203)
#vis_task_nummer(204)
#vis_task_nummer(205)


# 3D Datensaetze (Nummer 1XX)
#vis_task_nummer(108)
#vis_task_nummer(109)


