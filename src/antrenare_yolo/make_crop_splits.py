#decupam cu YOLO toate pozele mari din SIPaKMeD si scriem splits_yolo/*.csv,
import os

import cv2
import pandas as pd
from ultralytics import YOLO

from utils import (CLASE, citeste_splituri, contururi_din_poza, cutie_din_contur, decupeaza,
                   gaseste_celule, gaseste_poze_mari, iou)

MODEL = "results/yolo/yolov8s-seg/weights/best.pt"
INCREDERE = 0.25
MARGINE = 0.1          
IOU_MINIM = 0.5

split_grup = citeste_splituri()
model = YOLO(MODEL)
os.makedirs("splits_yolo", exist_ok=True)

randuri = []
gasite = {"train": 0, "val": 0, "test": 0}
marcate = {"train": 0, "val": 0, "test": 0}

for cale, clasa, nume, grup in gaseste_poze_mari():
    if grup not in split_grup:
        continue
    split = split_grup[grup]
    poza = cv2.imread(cale)

    cutii_medici = [cutie_din_contur(contur) for numar, contur in contururi_din_poza(cale)]
    deja_folosite = []  # o celula marcata se potriveste cu o singura detectie
    os.makedirs("data/yolo_crops/" + clasa, exist_ok=True)

    for i, (cutie, scor) in enumerate(gaseste_celule(model, poza, INCREDERE)):
        # cautam celula marcata care se suprapune cel mai mult cu detectia
        cel_mai_bun_iou = 0
        potrivire = -1
        for j in range(len(cutii_medici)):
            if j in deja_folosite:
                continue
            valoare = iou(cutie, cutii_medici[j])
            if valoare > cel_mai_bun_iou:
                cel_mai_bun_iou = valoare
                potrivire = j

        if cel_mai_bun_iou < IOU_MINIM:
            continue
        deja_folosite.append(potrivire)

        cale_celula = "data/yolo_crops/" + clasa + "/" + nume + "_" + str(i).zfill(2) + ".png"
        cv2.imwrite(cale_celula, decupeaza(poza, cutie, MARGINE))

        randuri.append({
            "path": cale_celula,
            "dataset": "sipakmed",
            "class_name": clasa,
            "label": CLASE[clasa],
            "group": grup,
            "split": split,
            "conf": round(scor, 3),
            "iou": round(cel_mai_bun_iou, 3),
        })

    gasite[split] += len(deja_folosite)
    marcate[split] += len(cutii_medici)

if len(randuri) == 0:
    print("YOLO nu a gasit nicio celula marcata. Verifica MODEL si INCREDERE.")
    exit()

tabel = pd.DataFrame(randuri)
tabel.to_csv("splits_yolo/all.csv", index=False)
for split in ["train", "val", "test"]:
    tabel[tabel["split"] == split].to_csv("splits_yolo/" + split + ".csv", index=False)

print("Cate din celulele marcate de medici a gasit YOLO:")
for split in ["train", "val", "test"]:
    if marcate[split] > 0:
        procent = round(100 * gasite[split] / marcate[split], 1)
        print(" ", split, ":", gasite[split], "din", marcate[split], "(" + str(procent) + "%)")

print("\nCelule pe split (0 = normal, 1 = anormal):")
print(pd.crosstab(tabel["split"], tabel["label"], margins=True))
print("\nSalvat in splits_yolo/")
