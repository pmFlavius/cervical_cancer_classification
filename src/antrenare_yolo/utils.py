import glob
import os

import numpy as np

# 0 = normal, 1 = anormal 
CLASE = {
    "Superficial-Intermediate": 0,
    "Parabasal": 0,
    "Metaplastic": 0,
    "Koilocytotic": 1,
    "Dyskeratotic": 1,
}


def citeste_splituri():
    import pandas as pd
    tabel = pd.read_csv("splits/all.csv")
    split_grup = {}
    for _, rand in tabel.iterrows():
        if rand["dataset"] == "sipakmed":
            split_grup[rand["group"]] = rand["split"]
    return split_grup


def gaseste_poze_mari():
    poze = []
    for folder_cropped in glob.glob("data/sipakmed/**/CROPPED", recursive=True):
        folder = os.path.dirname(folder_cropped)
        clasa = os.path.basename(folder).replace("im_", "")
        if clasa not in CLASE:
            continue
        for cale in glob.glob(folder + "/*.bmp"):
            nume = os.path.basename(cale).replace(".bmp", "")
            if nume.isdigit():  
                grup = "sipakmed_" + clasa + "_" + nume  
                poze.append((cale, clasa, nume, grup))
    poze.sort()
    return poze


def citeste_contur(cale_dat):
    with open(cale_dat) as f:
        text = f.read()
    numere = [float(x) for x in text.replace(",", " ").split()]
    return np.array(numere).reshape(-1, 2)


def contururi_din_poza(cale_poza):
    baza = cale_poza.replace(".bmp", "")
    contururi = []
    for cale_dat in sorted(glob.glob(baza + "_cyt*.dat")):
        numar = int(cale_dat.split("_cyt")[-1].replace(".dat", ""))
        contururi.append((numar, citeste_contur(cale_dat)))
    return contururi


def cutie_din_contur(contur):
    # cel mai mic dreptunghi care cuprinde tot conturul
    x1 = contur[:, 0].min()
    y1 = contur[:, 1].min()
    x2 = contur[:, 0].max()
    y2 = contur[:, 1].max()
    return [x1, y1, x2, y2]


def iou(a, b):
    # cat de mult se suprapun doua dreptunghiuri: 1 = identice, 0 = deloc
    latime = min(a[2], b[2]) - max(a[0], b[0])
    inaltime = min(a[3], b[3]) - max(a[1], b[1])
    if latime <= 0 or inaltime <= 0:
        return 0
    comun = latime * inaltime
    aria_a = (a[2] - a[0]) * (a[3] - a[1])
    aria_b = (b[2] - b[0]) * (b[3] - b[1])
    return comun / (aria_a + aria_b - comun)


def decupeaza(poza, cutie, margine):
    # taiem celula din poza mare, cu putin spatiu in jur 
    inaltime, latime = poza.shape[:2]
    x1, y1, x2, y2 = cutie
    plus_x = (x2 - x1) * margine
    plus_y = (y2 - y1) * margine

    x1 = int(max(0, x1 - plus_x))
    y1 = int(max(0, y1 - plus_y))
    x2 = int(min(latime, x2 + plus_x))
    y2 = int(min(inaltime, y2 + plus_y))

    return poza[y1:y2, x1:x2]


def gaseste_celule(model, poza, incredere=0.25, marime=1024):
    rezultat = model.predict(poza, conf=incredere, imgsz=marime, verbose=False)[0]
    cutii = rezultat.boxes.xyxy.tolist()
    scoruri = rezultat.boxes.conf.tolist()

    celule = []
    for i in range(len(cutii)):
        x1, y1, x2, y2 = cutii[i]
        # aruncam detectiile foarte mici, sunt aproape sigur greseli
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        celule.append((cutii[i], scoruri[i]))
    return celule
