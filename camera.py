"""
camera.py — камеры на нашем перекрёстке: ставит RGB-камеры, показывает картинку, может сохранять кадры.

Отдельный процесс, как ws_sender.py: сцену не трогает, ставит только свои камеры и убирает их на выходе.
Запускать при работающей сцене (run_scene.py крутит мир — без неё в синхронном режиме кадров нет).

Камеры по умолчанию (две, картинки в окне рядом):
  peds — на зебру и пешеходов: CAM_PEDS=x,y,z,yaw,pitch (точка снята свободной камерой в окне CARLA);
  cars — на очередь у стоп-линии: CAM_CARS=auto — над перекрёстком со стороны главной дороги,
         CAM_BACK м от центра зебры, высота CAM_HEIGHT (15 м), вдоль нашей дороги с наклоном CAM_PITCH (−35°).
  CAM_POSE=x,y,z,yaw,pitch[;x,y,z,yaw,pitch...] — вместо них поставить камеры ровно сюда;
  --from-spectator — одна камера туда, куда смотрит свободная камера в окне CARLA. Печатает её CAM_POSE;
  --print-pose — только напечатать позу свободной камеры и сказать, безопасна ли она; камеру не ставит.

Взгляд вдоль нашей дороги с горизонтом в кадре роняет сервер (Signal 11, см. CLAUDE.md) —
скрипт предупреждает о такой позе до того, как поставить камеру.

    python camera.py                        # окно с картинками, Esc — выход
    python camera.py --from-spectator
    python camera.py --print-pose
    python camera.py --save frames/         # плюс кадры (jpg) в frames/peds/, frames/cars/
    python camera.py --screenshot shot.jpg  # по одному кадру: shot_peds.jpg, shot_cars.jpg — и выйти
    python camera.py --no-window            # без окна: только раздача по HTTP

Раздача по HTTP (как у IP-камеры), порт CAM_HTTP_PORT (8080, 0 — выключить):
  http://<ip>:8080/            — страничка с обеими камерами, проверить в браузере;
  http://<ip>:8080/peds.mjpg   — MJPEG-поток: cv2.VideoCapture(url) или source=url в YOLO;
  http://<ip>:8080/peds.jpg    — один последний кадр.

Для кадров под YOLO в сцене нужен DRAW_DEBUG=0 — иначе debug-отрисовка попадёт в кадр.
"""
import argparse
import io
import math
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import carla
import numpy as np
import pygame
from PIL import Image

from envconf import CARLA_HOST, CARLA_PORT, env
from ws_sender import Observer

# ---- конфиг ----
CAM_W, CAM_H = env("CAM_W", 1280), env("CAM_H", 720)   # разрешение, px
CAM_FOV = env("CAM_FOV", 90.0)                         # горизонтальный угол обзора, °
CAM_FPS = env("CAM_FPS", 10.0)                         # кадров в секунду симуляции
CAM_PEDS = env("CAM_PEDS", "21.94,-169.66,8.40,-50.7,-29.2")   # на зебру: северо-западный угол, 8 м
CAM_CARS = env("CAM_CARS", "auto")     # на очередь: auto — от геометрии перекрёстка, или x,y,z,yaw,pitch
CAM_BACK = env("CAM_BACK", 12.0)       # auto: столько м от центра зебры в сторону главной дороги
CAM_SHIFT = env("CAM_SHIFT", 0.0)      # ... и столько м вбок (+ — к стороне E)
CAM_HEIGHT = env("CAM_HEIGHT", 15.0)   # ... высота над дорогой, м
CAM_PITCH = env("CAM_PITCH", -35.0)    # ... наклон вниз, °
CAM_POSE = env("CAM_POSE", "")         # "x,y,z,yaw,pitch[;...]" — вместо камер выше
WIN_W = 1280                           # ширина окна предпросмотра, px (камеры делят её поровну)
CAM_HTTP_HOST = env("CAM_HTTP_HOST", "0.0.0.0")   # раздача кадров: 0.0.0.0 — доступно из сети
CAM_HTTP_PORT = env("CAM_HTTP_PORT", 8080)        # 0 — не раздавать
CAM_STREAM_FPS = env("CAM_STREAM_FPS", 10.0)      # потолок кадров/с в MJPEG-потоке, по часам
CAM_JPEG_QUALITY = env("CAM_JPEG_QUALITY", 85)

HALF_VFOV = math.degrees(math.atan(math.tan(math.radians(CAM_FOV / 2)) * CAM_H / CAM_W))


def auto_pose(obs):
    """Над перекрёстком перед зеброй, смотрит вдоль нашей дороги (к стоп-линии)."""
    c, (ux, uy), (nx, ny) = obs.center, obs.u, obs.n
    loc = carla.Location(c.x - nx * CAM_BACK + ux * CAM_SHIFT,
                         c.y - ny * CAM_BACK + uy * CAM_SHIFT, c.z + CAM_HEIGHT)
    return carla.Transform(loc, carla.Rotation(yaw=math.degrees(math.atan2(ny, nx)), pitch=CAM_PITCH))


def risky(pose, obs):
    """Смотрит вдоль нашей дороги (±45°) и горизонт в кадре — такой вид роняет сервер."""
    yaw = math.radians(pose.rotation.yaw)
    along = math.cos(yaw) * obs.n[0] + math.sin(yaw) * obs.n[1] > math.cos(math.radians(45))
    return along and pose.rotation.pitch + HALF_VFOV > -3.0


def parse_pose(text):
    x, y, z, yaw, pitch = (float(v) for v in text.split(","))
    return carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw, pitch=pitch))


def pose_str(tr):
    l, r = tr.location, tr.rotation
    return f"{l.x:.2f},{l.y:.2f},{l.z:.2f},{r.yaw:.1f},{r.pitch:.1f}"


def to_rgb(img):
    """carla.Image (BGRA) -> массив RGB."""
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, 2::-1]


class Frames:
    """Последний кадр каждой камеры и его JPEG. Кадры пишут потоки CARLA, читают окно и HTTP."""

    def __init__(self):
        self.raw, self._jpeg, self._lock = {}, {}, threading.Lock()

    def jpeg(self, name):
        """(номер кадра, JPEG) последнего кадра или (None, None). Сжимаем один раз на кадр."""
        img = self.raw.get(name)
        if img is None:
            return None, None
        with self._lock:
            cached = self._jpeg.get(name)
            if cached is None or cached[0] != img.frame:
                buf = io.BytesIO()
                Image.fromarray(to_rgb(img)).save(buf, "JPEG", quality=CAM_JPEG_QUALITY)
                cached = self._jpeg[name] = (img.frame, buf.getvalue())
            return cached


def serve_http(frames, names):
    """HTTP как у IP-камеры: /<камера>.mjpg — поток, /<камера>.jpg — кадр, / — страничка."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # не засоряем консоль
            pass

        def do_GET(self):
            path = self.path.split("?")[0].strip("/")
            if path == "":
                page = "".join(f'<h3>{n}</h3><p><a href="/{n}.mjpg">/{n}.mjpg</a> — поток, '
                               f'<a href="/{n}.jpg">/{n}.jpg</a> — кадр</p><img src="/{n}.mjpg" width="640">'
                               for n in names)
                body = f"<html><meta charset='utf-8'><body>{page}</body></html>".encode()
                return self._send(200, "text/html; charset=utf-8", body)
            name, ext = os.path.splitext(path)
            if name not in names or ext not in (".jpg", ".mjpg"):
                return self._send(404, "text/plain; charset=utf-8", "нет такой камеры\n".encode())
            if ext == ".jpg":
                _, data = frames.jpeg(name)
                if data is None:
                    return self._send(503, "text/plain; charset=utf-8", "кадров ещё нет\n".encode())
                return self._send(200, "image/jpeg", data)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    fid, data = frames.jpeg(name)
                    if data is not None and fid != last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n")
                        last = fid
                    time.sleep(1.0 / CAM_STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass                      # клиент отключился

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((CAM_HTTP_HOST, CAM_HTTP_PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def local_ips():
    """Адреса этой машины (hostname -I): первый — обычно локальная сеть, дальше Docker, VPN и т.п.
    Маршрут «по умолчанию» не годится: при VPN он ведёт в туннель, а не в локальную сеть."""
    try:
        ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        ips = []
    return [ip for ip in ips if ":" not in ip] or ["127.0.0.1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-spectator", action="store_true", help="одна камера туда, куда смотрит камера в окне CARLA")
    ap.add_argument("--print-pose", action="store_true", help="напечатать позу свободной камеры и выйти")
    ap.add_argument("--save", metavar="DIR", help="сохранять кадры в DIR/<камера>/")
    ap.add_argument("--screenshot", metavar="FILE", help="сохранить по кадру с каждой камеры и выйти")
    ap.add_argument("--no-window", action="store_true", help="без окна: только раздача по HTTP")
    args = ap.parse_args()

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    obs = Observer(world)
    if args.print_pose:
        pose = world.get_spectator().get_transform()
        print(f"CAM_POSE={pose_str(pose)}")
        print("ОПАСНО: смотрит вдоль нашей дороги с горизонтом в кадре — наклони круче вниз"
              if risky(pose, obs) else "ок: такая камера сервер не уронит")
        return
    if args.from_spectator:
        poses = [("spectator", world.get_spectator().get_transform())]
    elif CAM_POSE:
        poses = [(f"cam{i + 1}", parse_pose(p)) for i, p in enumerate(CAM_POSE.split(";"))]
    else:
        poses = [("peds", parse_pose(CAM_PEDS)),
                 ("cars", auto_pose(obs) if CAM_CARS == "auto" else parse_pose(CAM_CARS))]

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(CAM_W))
    bp.set_attribute("image_size_y", str(CAM_H))
    bp.set_attribute("fov", str(CAM_FOV))
    bp.set_attribute("sensor_tick", str(1.0 / CAM_FPS))

    cams, frames, server = {}, Frames(), None
    latest = frames.raw                  # кадры приходят в потоке клиента CARLA — храним последний
    try:
        for name, pose in poses:
            if risky(pose, obs):
                print(f"ВНИМАНИЕ [{name}]: смотрит вдоль нашей дороги с горизонтом в кадре — такой вид "
                      f"роняет сервер (Signal 11). Нужен наклон круче {-(HALF_VFOV + 3):.0f}°.")
            cams[name] = world.spawn_actor(bp, pose)
            print(f"Камера {name} ({cams[name].id}): CAM_POSE={pose_str(pose)}")
            if args.save:
                os.makedirs(os.path.join(args.save, name), exist_ok=True)

            def on_image(img, name=name):
                latest[name] = img
                if args.save:
                    img.save_to_disk(os.path.join(args.save, name, f"{img.frame:08d}.jpg"))

            cams[name].listen(on_image)
        print(f"  {CAM_W}×{CAM_H}, fov {CAM_FOV:g}°, {CAM_FPS:g} кадров/с симуляции")
        if CAM_HTTP_PORT:
            server = serve_http(frames, list(cams))
            ip, *other = local_ips()
            print(f"Раздача по HTTP: http://{ip}:{CAM_HTTP_PORT}/  (страничка с камерами)"
                  + (f"  другие адреса машины: {', '.join(other)}" if other else ""))
            for name in cams:
                print(f"  {name}: http://{ip}:{CAM_HTTP_PORT}/{name}.mjpg — поток, /{name}.jpg — кадр")
        if args.no_window:
            print("Без окна. Остановка — Ctrl+C.")
            while True:
                time.sleep(1)

        pygame.init()
        pane = (WIN_W // len(cams), round(WIN_W // len(cams) * CAM_H / CAM_W))
        scr = pygame.display.set_mode((pane[0] * len(cams), pane[1]))
        pygame.display.set_caption("Камеры перекрёстка")
        font = pygame.font.Font(pygame.font.match_font("dejavusansmono,liberationmono,monospace"), 14)
        clock, t0 = pygame.time.Clock(), time.monotonic()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
            for i, name in enumerate(cams):
                x0, img = i * pane[0], latest.get(name)
                if img is None:
                    pygame.draw.rect(scr, (20, 20, 25), (x0, 0, *pane))
                    msg = "жду кадр..." if time.monotonic() - t0 < 5 else "кадров нет — run_scene.py запущен?"
                    scr.blit(font.render(f"{name}: {msg}", True, (230, 230, 230)), (x0 + 10, 10))
                    continue
                # BGRA -> RGB, в размер своей части окна
                rgb = to_rgb(img)
                scr.blit(pygame.transform.smoothscale(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), pane), (x0, 0))
                scr.blit(font.render(f"{name}  кадр {img.frame}  t={img.timestamp:.1f} с", True, (255, 255, 0)),
                         (x0 + 8, 6))
            pygame.display.flip()
            if args.screenshot and all(latest.get(n) for n in cams):
                root, ext = os.path.splitext(args.screenshot)
                for name in cams:
                    latest[name].save_to_disk(f"{root}_{name}{ext or '.jpg'}")
                    print(f"Сохранено: {root}_{name}{ext or '.jpg'}")
                return
            clock.tick(30)
    finally:
        if server:
            server.shutdown()
        pygame.quit()
        for cam in cams.values():
            try:
                cam.stop()
                cam.destroy()
            except RuntimeError as e:
                print(f"Камеру {cam.id} убрать не удалось: {e}")
        print(f"Камер убрано: {len(cams)}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass                             # уборка уже прошла в finally
