import os
import sys

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision

sys.path.append("src/yolo")
from utils import MODEL_YOLO, decupeaza, gaseste_celule 

MODEL_CLASIFICARE = "results/densenet121_sipakmed-herlev_seed42/best.pt"
ARHITECTURA = "densenet121"  
MARGINE = 0.1                
NUME_CLASE = ["Normal", "Anormal"]

MEDIE = np.array([0.485, 0.456, 0.406])
DEVIATIE = np.array([0.229, 0.224, 0.225])


def incarca_clasificator():
    if not os.path.exists(MODEL_CLASIFICARE):
        print("Nu am gasit", MODEL_CLASIFICARE, "- arat doar celulele gasite, fara clasificare")
        return None

    salvat = torch.load(MODEL_CLASIFICARE, map_location="cpu", weights_only=False)

    if isinstance(salvat, torch.nn.Module):
        model = salvat
    else:
        for cheie in ["model_state_dict", "state_dict", "model"]:
            if isinstance(salvat, dict) and cheie in salvat:
                salvat = salvat[cheie]
        ponderi = {}
        for nume, valoare in salvat.items():
            ponderi[nume.replace("module.", "")] = valoare

        if ARHITECTURA == "densenet121":
            model = torchvision.models.densenet121(weights=None)
            nr_clase = ponderi["classifier.weight"].shape[0]
            model.classifier = torch.nn.Linear(1024, nr_clase)
        else:
            model = torchvision.models.resnet50(weights=None)
            nr_clase = ponderi["fc.weight"].shape[0]
            model.fc = torch.nn.Linear(2048, nr_clase)
        model.load_state_dict(ponderi)

    model.eval()
    print("Clasificator incarcat:", MODEL_CLASIFICARE)
    return model


def pregateste(celula):
    rgb = cv2.cvtColor(celula, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    x = rgb / 255.0
    x = (x - MEDIE) / DEVIATIE
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  
    return x.unsqueeze(0)  


def clasifica_cu_gradcam(model, celula):
    x = pregateste(celula)

    if ARHITECTURA == "densenet121":
        harti = model.features(x)                     
        y = torch.relu(harti)
        y = torch.nn.functional.adaptive_avg_pool2d(y, 1).flatten(1)
        scoruri = model.classifier(y)
    else:
        y = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        harti = model.layer4(model.layer3(model.layer2(model.layer1(y))))  
        scoruri = model.fc(model.avgpool(harti).flatten(1))

    harti.retain_grad()  # vrem sa aflam si gradientul acestor harti
    probabilitati = torch.softmax(scoruri, dim=1)[0].detach()
    clasa = int(probabilitati.argmax())

    # cat de mult ar creste scorul clasei alese daca s-ar schimba fiecare harta
    model.zero_grad()
    scoruri[0, clasa].backward()

    importanta = harti.grad[0].mean(dim=(1, 2))   # cat conteaza fiecare harta
    harti = harti[0].detach()
    cam = torch.relu((importanta[:, None, None] * harti).sum(dim=0)).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # marim harta la marimea celulei si o coloram (albastru = putin, rosu = mult)
    cam = cv2.resize(cam, (celula.shape[1], celula.shape[0]))
    culori = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    suprapus = cv2.addWeighted(celula, 0.6, culori, 0.4, 0)

    return clasa, float(probabilitati[clasa]), suprapus


# modelele se incarca o singura data, la pornire
clasificator = incarca_clasificator()
detector = None
if os.path.exists(MODEL_YOLO):
    from ultralytics import YOLO
    detector = YOLO(MODEL_YOLO)
    print("YOLO incarcat:", MODEL_YOLO)
else:
    print("Nu am gasit", MODEL_YOLO, "- pozele vor fi tratate ca o singura celula")


def analizeaza(poza, mod, incredere):
    if poza is None:
        return None, "Incarca o poza.", []

    poza = cv2.cvtColor(poza, cv2.COLOR_RGB2BGR)  
    inaltime, latime = poza.shape[:2]

    # decidem daca e o poza mare sau o singura celula
    if mod == "O singura celula":
        o_celula = True
    elif mod == "Poza cu mai multe celule":
        o_celula = False
    else:
        o_celula = max(inaltime, latime) < 1000

    if o_celula or detector is None:
        cutii = [([0, 0, latime, inaltime], 1.0)]
    else:
        cutii = gaseste_celule(detector, poza, incredere)

    desen = poza.copy()
    galerie = []
    normale = 0
    anormale = 0

    for i, (cutie, scor) in enumerate(cutii):
        if o_celula:
            celula = poza
        else:
            celula = decupeaza(poza, cutie, MARGINE)

        if clasificator is not None:
            clasa, probabilitate, cam = clasifica_cu_gradcam(clasificator, celula)
            text = NUME_CLASE[clasa] if clasa < len(NUME_CLASE) else "Clasa " + str(clasa)
            text = text + " " + str(round(100 * probabilitate)) + "%"
            if clasa == 0:
                culoare = (0, 200, 0)    # verde = normal
                normale += 1
            else:
                culoare = (0, 0, 255)    # rosu = anormal
                anormale += 1
            galerie.append((cv2.cvtColor(cam, cv2.COLOR_BGR2RGB), str(i) + ": " + text))
        else:
            text = "celula"
            culoare = (255, 0, 0)
            galerie.append((cv2.cvtColor(celula, cv2.COLOR_BGR2RGB), str(i)))

        if not o_celula:
            x1, y1, x2, y2 = [int(v) for v in cutie]
            cv2.rectangle(desen, (x1, y1), (x2, y2), culoare, 4)
            cv2.putText(desen, str(i) + " " + text, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, culoare, 3)

    if o_celula:
        rezumat = "O singura celula: **" + galerie[0][1].split(": ")[-1] + "**"
    else:
        rezumat = "Am gasit **" + str(len(cutii)) + " celule**"
        if clasificator is not None:
            rezumat += ": " + str(normale) + " normale, " + str(anormale) + " anormale"
    if clasificator is None:
        rezumat += "\n\n(Nu am gasit modelul de clasificare, vezi MODEL_CLASIFICARE in src/app.py)"

    return cv2.cvtColor(desen, cv2.COLOR_BGR2RGB), rezumat, galerie


with gr.Blocks(title="Clasificarea celulelor cervicale") as aplicatie:
    gr.Markdown("# Clasificarea celulelor cervicale\n"
                "Incarca o poza de la microscop (cu mai multe celule) sau o singura celula.")
    with gr.Row():
        with gr.Column():
            intrare = gr.Image(label="Poza", type="numpy")
            mod = gr.Radio(["Automat", "Poza cu mai multe celule", "O singura celula"],
                           value="Automat", label="Tipul pozei")
            incredere = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Incredere minima YOLO")
            buton = gr.Button("Analizeaza", variant="primary")
        with gr.Column():
            iesire = gr.Image(label="Rezultat (verde = normal, rosu = anormal)")
            rezumat = gr.Markdown()
    galerie = gr.Gallery(label="Celulele decupate, cu Grad-CAM (rosu = zona care a contat cel mai mult)",
                         columns=6, height="auto")

    if os.path.isdir("demo_images"):
        gr.Examples(examples=sorted(os.path.join("demo_images", f) for f in os.listdir("demo_images")),
                    inputs=intrare)

    buton.click(analizeaza, inputs=[intrare, mod, incredere], outputs=[iesire, rezumat, galerie])

if __name__ == "__main__":
    aplicatie.launch()
