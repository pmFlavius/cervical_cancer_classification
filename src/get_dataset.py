from pathlib import Path
import kagglehub

DATASETS = {
    "sipakmed": "marinaeplissiti/sipakmed",
    "herlev": "yuvrajsinhachowdhury/herlev-dataset",
}

data_dir = Path(__file__).resolve().parent.parent / "data"

for name, handle in DATASETS.items():
    path = kagglehub.dataset_download(handle, output_dir=str(data_dir / name))
    print(f"{name}: downloaded in {path}")