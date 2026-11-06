import os
import shutil

import cv2
import numpy as np

from utils import citeste_splituri, contururi_din_poza, gaseste_poze_mari

FOLDER_YOLO = "data/yolo_sipakmed"

split_grup = citeste_splituri()

for split in ["train", "val", "test"]:
    os.makedirs(FOLDER_YOLO + "/images/" + split, exist_ok=True)
    os.makedirs(FOLDER_YOLO + "/labels/" + split, exist_ok=True)
os.makedirs(FOLDER_YOLO + "/preview", exist_ok=True)

nr_poze = {"train": 0, "val": 0, "test": 0}
nr_celule = {"train": 0, "val": 0, "test": 0}
margini = []  #

for cale, clasa, nume, grup in gaseste_poze_mari():
    if grup not in split_grup:
        continue
    split = split_grup[grup]
    contururi = contururi_din_poza(cale)
    if len(contururi) == 0:
        continue

    poza = cv2.imread(cale)
    inaltime, latime = poza.shape[:2]
    nume_nou = clasa + "_" + nume 

    destinatie = FOLDER_YOLO + "/images/" + split + "/" + nume_nou + ".bmp"
    if not os.path.exists(destinatie):
        try:
            os.link(cale, destinatie)
        except OSError:
            shutil.copy(cale, destinatie)

    randuri = []
    for numar, contur in contururi:
        rand = "0"
        for x, y in contur:
            x = min(max(x / latime, 0), 1)
            y = min(max(y / inaltime, 0), 1)
            rand += " " + str(round(x, 6)) + " " + str(round(y, 6))
        randuri.append(rand)

        crop = cv2.imread(os.path.dirname(cale) + "/CROPPED/" + nume + "_" + str(numar).zfill(2) + ".bmp")
        if crop is not None:
            latime_contur = contur[:, 0].max() - contur[:, 0].min()
            if latime_contur > 0:
                margini.append((crop.shape[1] / latime_contur - 1) / 2)

    with open(FOLDER_YOLO + "/labels/" + split + "/" + nume_nou + ".txt", "w") as f:
        f.write("\n".join(randuri) + "\n")

    if nr_poze["train"] + nr_poze["val"] + nr_poze["test"] < 10:
        for numar, contur in contururi:
            cv2.polylines(poza, [contur.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.imwrite(FOLDER_YOLO + "/preview/" + nume_nou + ".jpg", cv2.resize(poza, (1024, 768)))

    nr_poze[split] += 1
    nr_celule[split] += len(contururi)

if nr_poze["train"] == 0:
    print("Nu am gasit poze cu contururi in data/sipakmed. Ai rulat src/get_dataset.py?")
    exit()

with open(FOLDER_YOLO + "/data.yaml", "w") as f:
    f.write("path: " + os.path.abspath(FOLDER_YOLO) + "\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    f.write("  0: cell\n")

print("Gata, setul YOLO e in", FOLDER_YOLO)
for split in ["train", "val", "test"]:
    print(" ", split, ":", nr_poze[split], "poze,", nr_celule[split], "celule marcate")
print("Poze de verificare in", FOLDER_YOLO + "/preview")

if margini:
    print("\nDecupajele originale au o margine de aproximativ", round(float(np.median(margini)), 2))
    print("Pune valoarea asta la MARGINE in crop_cells.py si make_crop_splits.py")
