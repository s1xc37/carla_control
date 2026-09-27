"""
make_dataset.py — синтетический датасет для YOLO с камер нашего перекрёстка.

Отдельный процесс рядом со сценой: run_scene.py крутит мир с машинами и пешеходами
(запускать её с DRAW_DEBUG=0, иначе debug-отрисовка попадёт в кадр).
В точку каждой камеры (peds, cars — позы из camera.py) ставится пара сенсоров: RGB и
instance segmentation. Instance segmentation кодирует в пикселе semantic-тег (R) и id актора
(G + B<<8) — так в примере CARLA bounding_boxes.py. Отсюда точные рамки ВИДИМОЙ части каждого
человека и машины, без ручной разметки. Тип машины — как в JSON (ws_sender.vehicle_type).
Раз в DS_EVERY с симуляции сохраняет кадр и разметку; каждые DS_WEATHER_EVERY кадров меняет
погоду по кругу из DS_WEATHERS (в конце возвращает прежнюю).

    python make_dataset.py                 # DS_IMAGES кадров в dataset/
    DS_IMAGES=300 python make_dataset.py

Результат — формат ultralytics YOLO:
    dataset/data.yaml
    dataset/images/{train,val}/<камера>_<запуск>_<кадр>.jpg
    dataset/labels/{train,val}/<камера>_<запуск>_<кадр>.txt — «класс x_центр y_центр ширина высота», доли кадра
Повторный запуск дописывает в ту же папку, ничего не затирая; начать заново — удалить dataset/.
    dataset/preview/*.jpg                           — первые кадры с рамками, проверить глазами
Обучение у друга:  yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=1280
"""
import os
import threading
import time
from collections import Counter

import carla
import numpy as np
from PIL import Image, ImageDraw

import camera
from envconf import CARLA_HOST, CARLA_PORT, env
from ws_sender import Observer, vehicle_type

# ---- конфиг ----
DS_OUT = env("DS_OUT", "dataset")
DS_IMAGES = env("DS_IMAGES", 2000)          # сколько кадров сохранить (с обеих камер вместе)
DS_EVERY = env("DS_EVERY", 1.0)             # кадр с каждой камеры раз в столько с симуляции
DS_WEATHER_EVERY = env("DS_WEATHER_EVERY", 100)  # смена погоды каждые столько кадров
DS_WEATHERS = env("DS_WEATHERS", "ClearNoon,CloudyNoon,WetNoon,SoftRainNoon,MidRainyNoon,"
                                 "ClearSunset,WetCloudySunset,HardRainNoon,ClearNight,WetNight")
DS_VAL_EVERY = env("DS_VAL_EVERY", 5)       # каждый 5-й кусок по 50 кадров — в val (≈20 %)
DS_MIN_BOX = env("DS_MIN_BOX", 6)           # рамки меньше стольких пикселей по стороне — выкидываем
DS_MIN_PIXELS = env("DS_MIN_PIXELS", 25)    # ... и видимых пикселей меньше стольких
DS_EMPTY_SHARE = env("DS_EMPTY_SHARE", 0.1)  # доля кадров без объектов (фон), не больше
DS_PREVIEW = env("DS_PREVIEW", 10)          # кадров с рамками на камеру в preview/
CHUNK = 50                                  # кусок для разбивки train/val

CLASSES = ["person", "car", "van", "truck", "bus", "special"]
# semantic-теги CARLA (как в bounding_boxes.py): 12 pedestrian, 14 car, 15 truck, 16 bus
TAG_CLASS = {12: "person", 14: "car", 15: "truck", 16: "bus"}
COLORS = {"person": (255, 60, 200), "car": (60, 160, 255), "van": (60, 255, 200),
          "truck": (255, 170, 40), "bus": (255, 230, 0), "special": (255, 40, 40)}


def boxes_from_instance(inst, vehicle_classes):
    """Рамки видимой части объектов из кадра instance segmentation: [(класс, x0, y0, x1, y1)]."""
    raw = np.frombuffer(inst.raw_data, dtype=np.uint8).reshape(inst.height, inst.width, 4)
    tags = raw[..., 2]                                            # R — semantic-тег
    ids = raw[..., 1].astype(np.uint32) + (raw[..., 0].astype(np.uint32) << 8)   # G + B<<8 — id актора
    mask = np.isin(tags, list(TAG_CLASS))
    if not mask.any():
        return []
    ys, xs = np.nonzero(mask)
    keys = ids[ys, xs] * 256 + tags[ys, xs]      # объект = (id, тег): у декораций id может совпасть
    out = []
    for key in np.unique(keys):
        sel = keys == key
        if sel.sum() < DS_MIN_PIXELS:
            continue
        x0, x1, y0, y1 = xs[sel].min(), xs[sel].max(), ys[sel].min(), ys[sel].max()
        if x1 - x0 + 1 < DS_MIN_BOX or y1 - y0 + 1 < DS_MIN_BOX:
            continue
        actor_id, tag = int(key) // 256, int(key) % 256
        cls = "person" if tag == 12 else vehicle_classes.get(actor_id, TAG_CLASS[tag])
        out.append((cls, int(x0), int(y0), int(x1), int(y1)))
    return out


def rss_mb():
    """Память этого процесса, МБ — чтобы утечку было видно в выводе."""
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) // 1024 for line in f if line.startswith("VmRSS"))


def yolo_lines(boxes, w, h):
    return [f"{CLASSES.index(c)} {(x0 + x1 + 1) / 2 / w:.6f} {(y0 + y1 + 1) / 2 / h:.6f} "
            f"{(x1 - x0 + 1) / w:.6f} {(y1 - y0 + 1) / h:.6f}" for c, x0, y0, x1, y1 in boxes]


def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(DS_OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DS_OUT, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(DS_OUT, "preview"), exist_ok=True)
    with open(os.path.join(DS_OUT, "data.yaml"), "w") as f:
        # без path: ultralytics берёт папки относительно самого data.yaml — датасет можно переносить
        f.write("train: images/train\nval: images/val\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    try:
        world = client.get_world()
    except RuntimeError:
        raise SystemExit(f"Сервер CARLA на {CARLA_HOST}:{CARLA_PORT} не отвечает — запусти ~/simulator/CarlaUE4.sh "
                         f"и сцену (DRAW_DEBUG=0 python run_scene.py), потом этот скрипт.")
    if not world.get_settings().synchronous_mode:
        print("ВНИМАНИЕ: мир не в синхронном режиме — run_scene.py не запущена? Машин и людей будет мало.")
    obs = Observer(world)
    poses = [("peds", camera.parse_pose(camera.CAM_PEDS)),
             ("cars", camera.auto_pose(obs) if camera.CAM_CARS == "auto" else camera.parse_pose(camera.CAM_CARS))]
    weathers = [w.strip() for w in DS_WEATHERS.split(",") if w.strip()]
    for w in weathers:
        if not hasattr(carla.WeatherParameters, w):
            raise SystemExit(f"Нет такой погоды: {w}")
    old_weather = world.get_weather()
    run = time.strftime("%m%d%H%M")       # метка запуска: номера кадров после перезапуска сервера повторяются

    lib = world.get_blueprint_library()
    sensors, pending, lock = [], {}, threading.Lock()
    # Кадры идут быстрее, чем мы их сохраняем (сцена быстрее реального времени). Очередь без предела
    # съела память, и OOM убил сервер. Поэтому храним только самую свежую пару на камеру, лишние выбрасываем.
    latest, fresh, dropped = {}, threading.Event(), [0]

    def on_image(img, cam, kind):
        """RGB и instance одной камеры приходят отдельно — склеиваем по номеру кадра."""
        with lock:
            pair = pending.setdefault((cam, img.frame), {})
            pair[kind] = img
            if len(pair) == 2:
                del pending[(cam, img.frame)]
                dropped[0] += cam in latest          # прошлую пару не успели сохранить
                latest[cam] = pair
                fresh.set()
            if len(pending) > 8:                     # непарные хвосты не копим
                for key in sorted(pending, key=lambda k: k[1])[:-4]:
                    del pending[key]

    saved, empty, boxes_total, per_split, weather_i, last_cam = 0, 0, Counter(), Counter(), -1, None
    previews = Counter()
    try:
        for cam, pose in poses:
            if camera.risky(pose, obs):
                raise SystemExit(f"Поза камеры {cam} опасна (горизонт вдоль нашей дороги роняет сервер)")
            for kind, bp_id in (("rgb", "sensor.camera.rgb"), ("inst", "sensor.camera.instance_segmentation")):
                bp = lib.find(bp_id)
                bp.set_attribute("image_size_x", str(camera.CAM_W))
                bp.set_attribute("image_size_y", str(camera.CAM_H))
                bp.set_attribute("fov", str(camera.CAM_FOV))
                bp.set_attribute("sensor_tick", str(DS_EVERY))
                s = world.spawn_actor(bp, pose)
                s.listen(lambda img, cam=cam, kind=kind: on_image(img, cam, kind))
                sensors.append(s)
            print(f"Камера {cam}: RGB + instance segmentation, CAM_POSE={camera.pose_str(pose)}")
        print(f"Цель: {DS_IMAGES} кадров в {os.path.abspath(DS_OUT)}, погода: {', '.join(weathers)}")

        t0 = time.time()
        while saved < DS_IMAGES:
            if saved // DS_WEATHER_EVERY != weather_i:        # новый блок — новая погода
                weather_i = saved // DS_WEATHER_EVERY
                name = weathers[weather_i % len(weathers)]
                world.set_weather(getattr(carla.WeatherParameters, name))
                print(f"  погода: {name}")
                with lock:                                     # кадры старой погоды выкидываем
                    pending.clear()
                    latest.clear()
                    fresh.clear()
            if not fresh.wait(timeout=30):
                raise SystemExit("30 с нет кадров — сервер или сцена остановились?")
            with lock:                                         # камеры по очереди
                cam = next((c for c in latest if c != last_cam), next(iter(latest)))
                pair = latest.pop(cam)
                if not latest:
                    fresh.clear()
            last_cam = cam
            rgb, inst = pair["rgb"], pair["inst"]
            classes = {v.id: vehicle_type(v.attributes) for v in world.get_actors().filter("vehicle.*")}
            boxes = boxes_from_instance(inst, classes)
            if not boxes:
                if empty >= DS_EMPTY_SHARE * (saved + 1):
                    continue                                   # фона уже достаточно
                empty += 1
            split = "val" if (saved // CHUNK) % DS_VAL_EVERY == DS_VAL_EVERY - 1 else "train"
            stem = f"{cam}_{run}_{rgb.frame:08d}"
            picture = Image.fromarray(camera.to_rgb(rgb))
            picture.save(os.path.join(DS_OUT, "images", split, stem + ".jpg"), quality=95)
            with open(os.path.join(DS_OUT, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines(boxes, rgb.width, rgb.height)) + ("\n" if boxes else ""))
            if boxes and previews[cam] < DS_PREVIEW:
                draw = ImageDraw.Draw(picture)
                for c, x0, y0, x1, y1 in boxes:
                    draw.rectangle((x0, y0, x1, y1), outline=COLORS[c], width=2)
                    draw.text((x0 + 2, max(0, y0 - 12)), c, fill=COLORS[c])
                picture.save(os.path.join(DS_OUT, "preview", stem + ".jpg"), quality=90)
                previews[cam] += 1
            saved += 1
            per_split[split] += 1
            boxes_total.update(c for c, *_ in boxes)
            if saved % 50 == 0:
                print(f"  {saved}/{DS_IMAGES} кадров, {time.time() - t0:.0f} с, память {rss_mb()} МБ, "
                      f"рамок: {dict(boxes_total)}")
    finally:
        removed = 0
        for s in sensors:
            try:
                s.stop()
                removed += bool(s.destroy())         # при недоступном сервере destroy() не бросает, а даёт False
            except RuntimeError as e:
                print(f"Сенсор {s.id} убрать не удалось: {e}")
        weather_ok = False
        try:
            world.set_weather(old_weather)
            client.get_server_version()               # сервер ответил — значит, погода дошла
            weather_ok = True
        except RuntimeError as e:
            print(f"Погоду вернуть не удалось: {e}")
        print(f"Сенсоров убрано: {removed} из {len(sensors)}"
              + (", погода возвращена." if weather_ok else ", погода НЕ возвращена (сервер не отвечает)."))
        if dropped[0]:
            print(f"Пропущено пар кадров (не успевали сохранять, это нормально): {dropped[0]}")
        print(f"Сохранено {saved} кадров (train {per_split['train']}, val {per_split['val']}, без объектов {empty}); "
              f"рамок: {dict(boxes_total)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass                                  # уборка уже прошла в finally
