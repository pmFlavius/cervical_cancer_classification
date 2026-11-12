import json
import os

import torch
from ultralytics import YOLO

from utils import MARIME_YOLO

MODEL = "yolov8n.pt"   
EPOCI = 50
BATCH = 8              

DATE = "data/yolo_sipakmed/data.yaml"
FOLDER_RULARE = os.path.abspath("results/yolo/celule")

if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"


def antreneaza():
    if DEVICE == 0:
        print("Antrenez pe placa video:", torch.cuda.get_device_name(0))
    else:
        print("Nu am gasit placa video NVIDIA, antrenez pe procesor (va dura mai mult)")

    model = YOLO(MODEL)  

    model.train(
        data=DATE,
        imgsz=MARIME_YOLO,
        epochs=EPOCI,
        batch=BATCH,
        device=DEVICE,
        workers=2,
        patience=15,         
        seed=42,
        project=os.path.dirname(FOLDER_RULARE),
        name=os.path.basename(FOLDER_RULARE),
        exist_ok=True,
        fliplr=0.5,          # oglindire stanga-dreapta
        flipud=0.5,          # oglindire sus-jos
        degrees=90,          # rotatie
        scale=0.3,           # marire / micsorare
        hsv_h=0.02,          # nuanta
        hsv_s=0.5,           # saturatie
        hsv_v=0.3,           # luminozitate
    )

    # evaluam cel mai bun model pe val si pe test
    cel_mai_bun = YOLO(FOLDER_RULARE + "/weights/best.pt")
    scoruri = {}
    for split in ["val", "test"]:
        r = cel_mai_bun.val(data=DATE, split=split, imgsz=MARIME_YOLO, batch=BATCH, device=DEVICE,
                            project=FOLDER_RULARE, name="evaluare_" + split, exist_ok=True)
        scoruri[split] = {
            "precision": round(float(r.box.mp), 4),   # din ce a gasit, cat sunt celule reale
            "recall": round(float(r.box.mr), 4),      # din celulele marcate, cate a gasit
            "mAP50": round(float(r.box.map50), 4),    # scor general
            "mAP50-95": round(float(r.box.map), 4),   # cat de exact incadreaza celula
        }
        print(split, scoruri[split])

    with open(FOLDER_RULARE + "/metrics.json", "w") as f:
        json.dump(scoruri, f, indent=2)
    print("\nGata! Modelul e in", FOLDER_RULARE + "/weights/best.pt")

if __name__ == "__main__":
    antreneaza()
