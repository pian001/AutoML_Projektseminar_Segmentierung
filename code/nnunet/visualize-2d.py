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

NUM_SAMPLES = 8  # Anzahl an Samples (Bilder) die visualisiert werden sollen (jeweils fuer "Beste", "Schlechteste" und "Median")
NUM_SPALTEN = 3  # Anzahl an Spalten *IM* Subplot ("Original", "Original-Maske", "Prediction")
TITLE_SPACING = 115  # Hard-gecodeter Abstand im Titel um Sub-Titel fuer jede Spalte zu "erzeugen"


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
    plt.subplots_adjust(left=0, top=0.9, bottom=0.1, right=1, hspace=0.3, wspace=0.4)

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


def avg_wert_ueber_alle_klassen(klassendict, feld_name):
    """Hole aus dem json Abschnitt zu einem Sample den durchschnittlichen Dice-Wert ueber *alle* vorhandenen Klassen raus"""
    summe = 0
    menge = 0  # Anzahl gefundener Klassen
    for key in klassendict:
        # Gehe durch alle Klassen durch ("reference" und "test" ueberspringen) und pruefe, ob die Klasse ueberhaupt im Bild vorhanden ist (sonst ist Dice NaN)
        if key not in ["reference", "test"] and klassendict[key]["Total Positives Test"] != 0:
            summe += klassendict[key][feld_name]  # Wert fuer aktuelle Klasse auf Summe draufrechnen
            menge += 1  # Menge einen hoch zaehlen
    if menge == 0:  # Falls keine Klasse gefunden (sollte eigl. nie vorkommen, ausser Maske ist komplett schwarz / nur Background)
        return 0
    return summe / menge  # Gebe Durchschnitt zurueck


def scatterplot(train, test, label_names):
    """Scatterplot mit einem Punkt je Klasse je Sample fuer Train- und Testsplit nebeneinander mit Linie als Durchschnitt"""
    # Oeffne summary.json fuer Train-Split
    with open(train) as train_evaluation_json:
        json_train = json.load(train_evaluation_json)
    # Oeffne summary.json fuer Test-Split
    with open(test) as test_evaluation_json:
        json_test = json.load(test_evaluation_json)
    # Verwende *immer* die Pascal Farben, andere Datensaetze kriegen dann halt farbige Punkte je nach Klassennummer ¯\_(^.^)_/¯
    cmap = pascal_color_map()
    # Teile Hauptplot in der Mitte (1 Zeile, 2 Spalten)
    fig, axs = plt.subplots(1, 2)
    # Fuer jeden der beiden Scatterplots (Train und Test)
    for (title, sample_liste, ax) in [("Train", json_train["results"], axs[0]), ("Test", json_test["results"], axs[1])]:
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
                if key in ["reference", "test"] or sample[key]["Total Positives Test"] == 0:
                    continue
                # Konvertiere das 1d-Array (3 Eintraege RGB) in hexadezimalen ColorCode fuer pyplot
                color = '#%02x%02x%02x' % (cmap[int(key)][0], cmap[int(key)][1], cmap[int(key)][2])
                # Plotte den Punkt an zufaellige x-Stelle damit sich nicht alles stackt
                ax.scatter(random.uniform(0, 1), sample[key]["Dice"], marker="x", c=color, s=10)
        # Erstelle Legende
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    # Hole alle existierenden keys
    existinglabels = [key for key in json_train["results"]["all"][
        0]]  # Hier sind immer alle Klassen drin, nicht vorhandene Klassen haben NaN bzw 0 Werte aber trotzdem einen Eintrag
    # Die Keys fuer Metadaten ueber die Samples koennen ignoriert werden, wir brauchen nur die Klassennummern die vorkommen
    existinglabels.remove("reference")
    existinglabels.remove("test")
    labels = []
    for label in existinglabels:  # gehe durch alle existierenden Label durch
        # Hole Color-Code fuer das Label
        color = '#%02x%02x%02x' % (cmap[int(label)][0], cmap[int(label)][1], cmap[int(label)][2])
        # Fuege mpatches.Patch fuer die aktuelle Klassennummer mit richtiger (Pascal VOC2012-) Farbe und Namen aus 'label_names' in 'labels' ein
        labels.append(mpatches.Patch(color=color, label=label_names[int(label)]))

        # Ziehe Durchschnitts-Linie fuer aktuelle Klassennummer in Train
        mean = json_train["results"]["mean"][label][
            "Dice"]  # Durchschnitt kommt aus dem von nnUNet berechneten Durchschnitt
        axs[0].hlines(mean, 0, 1, color=color)
        # und in Test
        mean = json_test["results"]["mean"][label]["Dice"]
        axs[1].hlines(mean, 0, 1, color=color)

    # Setze die Legende entsprechend 'labels', das vorher zusammengebaut wurde (Farbe + Klassenname fuer jede Klassennummer)
    axs[0].legend(handles=labels)
    axs[1].legend(handles=labels)

    plt.show()


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


def vis_task_nummer(task_nummer):
    """Vereinfachter Aufruf einer Visualisierung zu einer Tasknummer durch vorgefertigte Variablen und Befehle"""
    global ORIG_IMGS_PFAD, ORIG_MASKS_PFAD, PRED_MASKS_PFAD
    global ORIG_IMGS_DATEIENDUNG, ORIG_MASKS_DATEIENDUNG, PRED_MASKS_DATEIENDUNG

    # Larven ohne split (Train=Test)
    if task_nummer == 200:
        # Variablen setzen
        ORIG_IMGS_DATEIENDUNG = ".png"
        ORIG_IMGS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/imgs")
        ORIG_MASKS_PFAD = NNUNET_ROOT_PATH / Path("raw_datasets/Larven/masks/")
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task200_Larven_train_eq_test/imagesTr/")
        # Visualisieren (Scatterplot hier unnoetig / nicht anwendbar da alles Train ist und kein Testsplit existiert)
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
        scatterplot(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task201_Larven_split_drittel-ist-test/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task201_Larven_split_drittel-ist-test__TRAIN", False)

        # Visualisieren der Test-Samples
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task201_Larven_split_drittel-ist-test/imagesTs/")
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
        scatterplot(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task203_Augenadern-drittel-test/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task203_Augenadern-drittel-test__TRAIN", False)
        # Visualisieren der Test-Samples
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task203_Augenadern-drittel-test/imagesTs/")
        visualize(PRED_MASKS_PFAD / "summary.json", "Task203_Augenadern-drittel-test__TEST", False)





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
        scatterplot(PRED_MASKS_PFAD / "summary.json", NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTs/") / "summary.json", label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task204_PascalVOC2012-achtel-split__TRAIN",
                  True)  # Original-Bilder sind RGB => True
        # Visualisieren der Test-Samples
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task204_PascalVOC2012-achtel-split/imagesTs/")
        visualize(PRED_MASKS_PFAD / "summary.json", "Task204_PascalVOC2012-achtel-split__TEST",
                  True)  # Original-Bilder sind RGB => True





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
        scatterplot(PRED_MASKS_PFAD / "summary.json",
                    NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task205_Augen_weniger/imagesTs/") / "summary.json",
                    label_names)

        # Visualisieren der Train-Samples
        visualize(PRED_MASKS_PFAD / "summary.json", "Task205_Augen_weniger__TRAIN", False)
        # Visualisieren der Test-Samples
        PRED_MASKS_PFAD = NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task205_Augen_weniger/imagesTs/")
        visualize(PRED_MASKS_PFAD / "summary.json", "Task205_Augen_weniger__TEST", False)

    # Retina 3D Scatterplot, Visualisieren von 3D Slices wird von Hand gemacht
    elif task_nummer == "108-3d_fullres-ohneNPZ":  # 3d_fullres-Netz mit NPZ (zweiter Anlauf) 
        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        scatterplot(NNUNET_ROOT_PATH / Path(
            "nnUNet_predictions/Task108_Retina-3d-3d_fullres_ohneNPZ/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path(
                        "nnUNet_predictions/Task108_Retina-3d-3d_fullres_ohneNPZ/imagesTs/") / "summary.json",
                    label_names)



    # Retina 3D Scatterplot, Visualisieren von 3D Slices wird von Hand gemacht
    elif task_nummer == "108-3d_fullres":  # 3d_fullres-Netz ohne NPZ (erster Anlauf) 
        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        scatterplot(NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_fullres/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_fullres/imagesTs/") / "summary.json",
                    label_names)


    # Retina 3D Scatterplot, Visualisieren von 3D Slices wird von Hand gemacht
    elif task_nummer == "108-2d":  # 2d-Netz
        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        scatterplot(NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_2dNet/imagesTr/") / "summary.json",
                    NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_Retina-3d_2dNet/imagesTs/") / "summary.json",
                    label_names)


    # Retina 3D Scatterplot, Visualisieren von 3D Slices wird von Hand gemacht
    elif task_nummer == "108-2d-3d_fullres-Ensemble":  # 3d_fullres und 2d Ensemble
        label_names = ["Background",
                       "Augenader"]  # Labelnamen muessen Background am Anfang zusaetzlich enthalten, auch wenn Label 0 nicht in der summary.json auftaucht
        scatterplot(
            NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_ensemble-2d_3d_fullres/imagesTr/") / "summary.json",
            NNUNET_ROOT_PATH / Path("nnUNet_predictions/Task108_ensemble-2d_3d_fullres/imagesTs/") / "summary.json",
            label_names)


vis_task_nummer("108-2d-3d_fullres-Ensemble")
