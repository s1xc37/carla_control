"""
train.py — быстрое обучение YOLO на синтетическом датасете из make_dataset.py.

    python train.py                                   # yolo26n, 960 px, до 30 эпох
    YOLO_EPOCHS=10 YOLO_IMGSZ=640 python train.py      # совсем быстро — проверить, что всё работает
    YOLO_MODEL=yolo26s.pt python train.py              # модель побольше: точнее, но дольше

Перед обучением лучше выключить CARLA: она держит ~4 ГБ видеопамяти из 8, и обучению остаётся мало.
Предобученные веса (yolo26n.pt и т.п., несколько МБ) ultralytics сам скачает при первом запуске.

Результат — runs/detect/<YOLO_NAME>/:
    weights/best.pt  — обученная модель (её и отдавать: YOLO("best.pt").predict(...));
    results.png, confusion_matrix.png и др. — графики обучения;
    predict/         — примеры распознавания на кадрах из val, посмотреть глазами.
"""
import subprocess
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from envconf import env

# ---- конфиг ----
DATA = env("YOLO_DATA", "dataset/data.yaml")
MODEL = env("YOLO_MODEL", "yolo26n.pt")      # n — самая быстрая; s, m — точнее и медленнее
EPOCHS = env("YOLO_EPOCHS", 30)
PATIENCE = env("YOLO_PATIENCE", 10)          # остановиться, если столько эпох нет улучшения
IMGSZ = env("YOLO_IMGSZ", 960)               # машины вдали мелкие: при 640 заметно хуже
BATCH = env("YOLO_BATCH", -1)                # -1 — подобрать под свободную видеопамять
CACHE = env("YOLO_CACHE", "")                # ram — кадры в память (~3 ГБ), быстрее; при работающей CARLA не надо
FRACTION = env("YOLO_FRACTION", 1.0)         # доля train, на которой учиться (0.1 — быстрая проба)
WORKERS = env("YOLO_WORKERS", 8)             # процессов загрузки данных
DEVICE = env("YOLO_DEVICE", "0")             # "cpu" — без видеокарты (очень медленно)
PROJECT = env("YOLO_PROJECT", "runs/detect")
NAME = env("YOLO_NAME", "crosswalk")
PREDICT_N = env("YOLO_PREDICT_N", 8)         # кадров из val с нарисованным распознаванием


def main():
    data = Path(DATA)
    if not data.exists():
        raise SystemExit(f"Нет {data} — сначала make_dataset.py")
    counts = {s: len(list((data.parent / "images" / s).glob("*.jpg"))) for s in ("train", "val")}
    print(f"Датасет {data}: train {counts['train']}, val {counts['val']} кадров")
    if not counts["train"] or not counts["val"]:
        raise SystemExit("В train или val нет кадров — val набирается с 200-го кадра, нужен датасет побольше")

    if DEVICE != "cpu":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA не видна — запусти с YOLO_DEVICE=cpu (очень медленно)")
        free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
        print(f"Видеокарта {torch.cuda.get_device_name(0)}: свободно {free:.1f} из {total:.1f} ГБ")
        if subprocess.run(["pgrep", "-x", "CarlaUE4-Linux-"], capture_output=True).returncode == 0:
            print("ВНИМАНИЕ: CARLA работает и занимает видеопамять — обучение будет медленнее, "
                  "при нехватке памяти выключи её или поставь YOLO_BATCH=4")

    t0 = time.time()
    model = YOLO(MODEL)
    model.train(data=str(data), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH,
                cache=CACHE or False, fraction=FRACTION, workers=WORKERS, device=DEVICE,
                project=PROJECT, name=NAME, plots=True)
    best = Path(model.trainer.best)
    print(f"\nОбучение заняло {(time.time() - t0) / 60:.1f} мин, лучшая модель: {best}")

    # итог на val — по классам
    metrics = YOLO(best).val(data=str(data), imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False)
    print(f"\nval: mAP50 {metrics.box.map50:.3f}, mAP50-95 {metrics.box.map:.3f}")
    for i, name in metrics.names.items():
        print(f"  {name:8} mAP50-95 {metrics.box.maps[i]:.3f}")

    # примеры распознавания — посмотреть глазами
    frames = sorted((data.parent / "images" / "val").glob("*.jpg"))[::max(1, counts["val"] // PREDICT_N)][:PREDICT_N]
    YOLO(best).predict(source=[str(f) for f in frames], imgsz=IMGSZ, device=DEVICE, conf=0.25,
                       save=True, project=str(best.parent.parent), name="predict", exist_ok=True, verbose=False)
    print(f"\nПримеры распознавания: {best.parent.parent / 'predict'}")
    print(f"Использовать: YOLO('{best}').predict(source='http://192.168.0.16:8080/peds.mjpg', stream=True)")


if __name__ == "__main__":
    main()
