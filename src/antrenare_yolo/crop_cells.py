# folosim YOLO antrenat ca sa decupam celulele din orice poze.
# fiecare celula gasita e salvata separat, gata de trimis la ResNet / DenseNet.

import glob
import os

import cv2
import pandas as pd
from ultralytics import YOLO

from utils import decupeaza, gaseste_celule

MODEL = "results/yolo/yolov8s-seg/weights/best.pt"
SURSA = "data/yolo_sipakmed/images/test"   
IESIRE = "results/crops"
INCREDERE = 0.25   # detectiile cu scor mai mic sunt ignorate
MARGINE = 0.1     

os.makedirs(IESIRE + "/cells", exist_ok=True)
os.makedirs(IESIRE + "/vis", exist_ok=True)

model = YOLO(MODEL)
poze = sorted(glob.glob(SURSA + "/*.bmp") + glob.glob(SURSA + "/*.png") + glob.glob(SURSA + "/*.jpg"))
tabel = []

for cale in poze:
    poza = cv2.imread(cale)
    nume = os.path.splitext(os.path.basename(cale))[0]
    desen = poza.copy()
    celule = gaseste_celule(model, poza, INCREDERE)

    for i, (cutie, scor) in enumerate(celule):
        celula = decupeaza(poza, cutie, MARGINE)
        cale_celula = IESIRE + "/cells/" + nume + "_" + str(i).zfill(2) + ".png"
        cv2.imwrite(cale_celula, celula)

        x1, y1, x2, y2 = [int(v) for v in cutie]
        tabel.append({"path": cale_celula, "image": os.path.basename(cale), "conf": round(scor, 3),
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # desenam un dreptunghi verde cu numarul celulei si scorul
        cv2.rectangle(desen, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(desen, str(i) + " " + str(round(scor, 2)), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(IESIRE + "/vis/" + nume + ".jpg", desen)
    print(os.path.basename(cale), ":", len(celule), "celule")

pd.DataFrame(tabel).to_csv(IESIRE + "/cells.csv", index=False)
print("\nGata,", len(tabel), "celule salvate in", IESIRE + "/cells")
