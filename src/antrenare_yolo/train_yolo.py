# antrenam YOLOv8 sa gaseasca celulele in pozele mari.
import json
import os

from ultralytics import YOLO

MODEL = "yolov8s-seg.pt"  
MARIME = 1024             
EPOCI = 100
BATCH = 8                 
DEVICE = 0                

DATE = "data/yolo_sipakmed/data.yaml"
NUME_RULARE = MODEL.replace(".pt", "")
FOLDER_RULARE = os.path.abspath("results/yolo/" + NUME_RULARE)

model = YOLO(MODEL)  

model.train(
    data=DATE,
    imgsz=MARIME,
    epochs=EPOCI,
    batch=BATCH,
    device=DEVICE,
    patience=20,         
    seed=42,
    project=os.path.dirname(FOLDER_RULARE),
    name=NUME_RULARE,
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.001,          
    # augmentare
    fliplr=0.5,          
    flipud=0.5,          
    degrees=90,         
    scale=0.3,           
    hsv_h=0.02,          
    hsv_s=0.5,           
    hsv_v=0.3,           
)

cel_mai_bun = YOLO(FOLDER_RULARE + "/weights/best.pt")
scoruri = {}
for split in ["val", "test"]:
    r = cel_mai_bun.val(data=DATE, split=split, imgsz=MARIME, batch=BATCH, device=DEVICE,
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
