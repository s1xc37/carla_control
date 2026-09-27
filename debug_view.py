"""
debug_view.py — отладочное окно: вид сверху на переход, машины с координатами.

Отдельный процесс, мир только читает (как ws_sender.py). Запускать при работающей сцене:
    python debug_view.py
    python debug_view.py --screenshot shot.png   # сохранить картинку окна и выйти

Слева — карта: полосы (серые точки), зебра (белая), зоны ожидания W (зелёная) и E (синяя),
жёлтая рамка — коридор нашей дороги (машины в нём попадают в JSON), машины по габаритам
(цвет — тип), пешеходы (розовые точки).
Справа — таблица машин: id, тип, x, y, полоса, км/ч; ★ — машина попадает в JSON.
Внизу — координаты мира под курсором. Клик — печатает их (и машину под курсором) в консоль.
Колесо мыши — масштаб, Esc — выход.
Ориентация как у камеры, которую run_scene.py ставит над перекрёстком: +x вверх, +y вправо.
"""
import argparse
import math
import time

import carla
import pygame

import control
from envconf import CARLA_HOST, CARLA_PORT
from ws_sender import Observer

# ---- конфиг ----
MAP_W, PANEL_W, H = 720, 470, 720   # размеры окна, px
SCALE = 5.0                         # px на метр: весь RADIUS влезает в карту (колесо мыши меняет)
RADIUS = 70.0                       # машины и пешеходы в таком радиусе от зебры, м
FPS = 15

COLOR = {"car": (170, 170, 170), "van": (110, 170, 255), "truck": (255, 160, 60),
         "bus": (255, 220, 0), "special": (255, 60, 60)}
PHASE_COLOR = {"MAIN": (90, 200, 90), "SIDE": (90, 160, 255), "PED": (255, 120, 255)}
BG, ROAD, ZEBRA, CORRIDOR = (25, 25, 30), (75, 75, 85), (240, 240, 240), (230, 200, 40)
ZONE = {"W": (0, 200, 0), "E": (0, 120, 255)}
WALKER, TEXT, DIM = (255, 120, 255), (230, 230, 230), (140, 140, 150)

# ползунки: ключ команды сцене (control.py), подпись, мин, макс, шаг
SLIDERS = [("veh_per_hour", "машин в час", 0, 3600, 100),
           ("max_vehicles", "машин максимум", 0, 120, 5),
           ("ped_per_hour", "пешеходов в час", 0, 1800, 30),
           ("max_per_side", "пешеходов на сторону (живых)", 0, 8, 1)]
ROW = 40                                          # высота строки ползунка, px
CONTROLS_TOP = H - 26 - 34 - ROW * len(SLIDERS)   # где начинается блок управления


class View:
    def __init__(self, world, obs):
        self.world, self.obs = world, obs
        self.c = obs.center                       # центр карты — центр зебры
        self.scale = SCALE
        # полосы вокруг — один раз, это статичная карта
        self.road = [(w.transform.location.x, w.transform.location.y)
                     for w in world.get_map().generate_waypoints(2.0)
                     if math.hypot(w.transform.location.x - self.c.x,
                                   w.transform.location.y - self.c.y) < RADIUS + 30]

    # мир <-> экран: +x вверх, +y вправо
    def to_screen(self, x, y):
        return (MAP_W / 2 + (y - self.c.y) * self.scale, H / 2 - (x - self.c.x) * self.scale)

    def to_world(self, sx, sy):
        return self.c.x + (H / 2 - sy) / self.scale, self.c.y + (sx - MAP_W / 2) / self.scale

    def read(self):
        """Снимок мира: машины и пешеходы в радиусе, фаза, время."""
        snap = self.world.get_snapshot()
        cars = []
        for v in self.world.get_actors().filter("vehicle.*"):
            tr = v.get_transform()
            p = tr.location
            d = math.hypot(p.x - self.c.x, p.y - self.c.y)
            if d > RADIUS:
                continue
            vel = v.get_velocity()
            cars.append(dict(id=v.id, type=self.obs.vtype(v), x=p.x, y=p.y, z=p.z,
                             yaw=tr.rotation.yaw, ext=v.bounding_box.extent, dist=d,
                             kmh=math.hypot(vel.x, vel.y) * 3.6, lane=self.obs.lane_of(p)))
        peds = [w.get_location() for w in self.world.get_actors().filter("walker.pedestrian.*")]
        peds = [p for p in peds if math.hypot(p.x - self.c.x, p.y - self.c.y) < RADIUS]
        return dict(t=snap.timestamp.elapsed_seconds, frame=snap.frame, phase=self.obs.phase(),
                    cars=sorted(cars, key=lambda c: c["dist"]), peds=peds,
                    counts=self.obs.pedestrians())

    def car_at(self, s, wx, wy):
        near = [c for c in s["cars"] if math.hypot(c["x"] - wx, c["y"] - wy) < max(c["ext"].x, 2.0)]
        return min(near, key=lambda c: math.hypot(c["x"] - wx, c["y"] - wy), default=None)

    def draw(self, scr, font, small, s, mouse):
        scr.fill(BG)
        for x, y in self.road:
            scr.fill(ROAD, (*self.to_screen(x, y), 2, 2))
        pts = lambda ps: [self.to_screen(x, y) for x, y in ps]
        pygame.draw.polygon(scr, CORRIDOR, pts(self.obs.corridor()), 1)
        pygame.draw.polygon(scr, ZEBRA, pts([(q.x, q.y) for q in self.obs.g.zebra]), 2)
        for side, w in self.obs.g.wait.items():
            (ux, uy), (nx, ny), h = self.obs.u, self.obs.n, 1.5
            sq = [(w.x + ux * a + nx * b, w.y + uy * a + ny * b) for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))]
            pygame.draw.polygon(scr, ZONE[side], pts(sq), 2)
            scr.blit(small.render(side, True, ZONE[side]), self.to_screen(w.x + 2.5, w.y))

        hover = self.car_at(s, *self.to_world(*mouse)) if mouse[0] < MAP_W else None
        for c in s["cars"]:
            yaw = math.radians(c["yaw"])
            fx, fy, rx, ry = math.cos(yaw), math.sin(yaw), -math.sin(yaw), math.cos(yaw)
            ex, ey = c["ext"].x, c["ext"].y
            box = [(c["x"] + fx * a * ex + rx * b * ey, c["y"] + fy * a * ex + ry * b * ey)
                   for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            pygame.draw.polygon(scr, COLOR[c["type"]], pts(box))
            if c["lane"]:
                pygame.draw.polygon(scr, CORRIDOR, pts(box), 2)
            if c is hover:
                pygame.draw.polygon(scr, (255, 255, 255), pts(box), 3)
            nose = self.to_screen(c["x"] + fx * ex, c["y"] + fy * ex)
            pygame.draw.circle(scr, (0, 0, 0), nose, 2)             # перёд машины
            label = f"{c['id']}" + (f" п{c['lane'][0]}" if c["lane"] else "")
            sx, sy = self.to_screen(c["x"], c["y"])
            scr.blit(small.render(label, True, TEXT), (sx + 6, sy - 6))
        for p in s["peds"]:
            pygame.draw.circle(scr, WALKER, self.to_screen(p.x, p.y), 3)

        # ---- панель справа ----
        pygame.draw.rect(scr, (15, 15, 18), (MAP_W, 0, PANEL_W, H))
        x0, y = MAP_W + 12, 10
        scr.blit(font.render(f"Фаза {s['phase']}", True, PHASE_COLOR[s["phase"]]), (x0, y))
        scr.blit(small.render(f"t={s['t']:.1f} с   кадр {s['frame']}", True, DIM), (x0 + 170, y + 3))
        y += 26
        ped = s["counts"]
        in_json = sum(1 for c in s["cars"] if c["lane"])
        scr.blit(small.render(f"машин в {RADIUS:.0f} м: {len(s['cars'])}, в JSON ★: {in_json}   "
                              f"пешеходы: W={ped['waiting']['W']} E={ped['waiting']['E']} "
                              f"на зебре={ped['crossing']}", True, TEXT), (x0, y))
        y += 26
        scr.blit(font.render("    id  тип        x        y  полоса км/ч", True, DIM), (x0, y))
        y += 20
        for c in s["cars"]:
            if y > CONTROLS_TOP - 20:
                break
            lane = f"{c['lane'][0]} {c['lane'][1]}" if c["lane"] else "-"
            row = (f"{'★' if c['lane'] else ' '}{c['id']:>5} {c['type']:<7} "
                   f"{c['x']:>8.1f} {c['y']:>8.1f}  {lane:<6} {c['kmh']:>4.0f}")
            color = (255, 255, 255) if c is hover else COLOR[c["type"]]
            scr.blit(font.render(row, True, color), (x0, y))
            y += 18

        # ---- низ: курсор ----
        wx, wy = self.to_world(*mouse)
        info = f"курсор: x={wx:.2f}  y={wy:.2f}" if mouse[0] < MAP_W else "курсор вне карты"
        if hover:
            info += f"   | машина {hover['id']} ({hover['type']}): x={hover['x']:.2f} y={hover['y']:.2f} " \
                    f"z={hover['z']:.2f} курс={hover['yaw']:.0f}°"
        pygame.draw.rect(scr, (15, 15, 18), (0, H - 26, MAP_W + PANEL_W, 26))
        scr.blit(small.render(info + "   (клик — в консоль, колесо — масштаб / ползунок)", True, TEXT), (10, H - 20))


class Controls:
    """Ползунки справа внизу: шлют сцене потоки и лимиты (control.py) и показывают, что она приняла."""

    def __init__(self):
        self.ctl = control.Client()
        self.scene = None                     # последние значения, подтверждённые сценой
        self.values = {}                      # что показываем (пока тянем — своё)
        self.t_reply = self.t_get = self.hold = -math.inf
        self.drag = None
        self.spec = {key: (label, lo, hi, step) for key, label, lo, hi, step in SLIDERS}
        self.tracks = {key: pygame.Rect(MAP_W + 12, CONTROLS_TOP + 54 + i * ROW, PANEL_W - 24, 6)
                       for i, (key, *_) in enumerate(SLIDERS)}

    def connected(self, now):
        return now - self.t_reply < 3.0

    def update(self, now):
        """Раз в секунду спрашиваем сцену, ответы забираем каждый кадр."""
        if now - self.t_get > 1.0:
            self.ctl.send(cmd="get")
            self.t_get = now
        reply = self.ctl.latest()
        if reply:
            self.scene, self.t_reply = reply, now
            if self.drag is None and (now > self.hold or reply == self.values):
                self.values = dict(reply)

    def _value_at(self, key, mx):
        _, lo, hi, step = self.spec[key]
        r = self.tracks[key]
        f = min(1.0, max(0.0, (mx - r.x) / r.w))
        return int(round((lo + f * (hi - lo)) / step) * step)

    def _hit(self, pos):
        return next((k for k, r in self.tracks.items() if r.inflate(0, 24).collidepoint(pos)), None)

    def _send(self, now):
        self.ctl.send(cmd="set", **self.values)
        self.hold = now + 2.0                 # пока сцена не подтвердила — показываем своё

    def event(self, e, now):
        """True, если событие забрали ползунки."""
        if not self.connected(now):
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and (k := self._hit(e.pos)):
            self.drag = k
            self.values[k] = self._value_at(k, e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.drag:
            self.values[self.drag] = self._value_at(self.drag, e.pos[0])
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
            self.drag = None
            self._send(now)                   # шлём, когда отпустили
            return True
        if e.type == pygame.MOUSEWHEEL and (k := self._hit(pygame.mouse.get_pos())):
            _, lo, hi, step = self.spec[k]
            self.values[k] = min(hi, max(lo, self.values[k] + e.y * step))
            self._send(now)
            return True
        return False

    def draw(self, scr, font, small, now):
        x0, ok = MAP_W + 12, self.connected(now)
        pygame.draw.line(scr, DIM, (x0, CONTROLS_TOP), (MAP_W + PANEL_W - 12, CONTROLS_TOP))
        head = "Управление сценой" if ok else "Управление: сцена не отвечает (run_scene.py запущен?)"
        scr.blit(font.render(head, True, TEXT if ok else (255, 120, 120)), (x0, CONTROLS_TOP + 8))
        for key, (label, lo, hi, step) in self.spec.items():
            r, v = self.tracks[key], self.values.get(key)
            wait = ok and v != self.scene.get(key)
            txt = f"{label}: {v if ok else '—'}" + (f"   (сейчас в сцене {self.scene[key]})" if wait else "")
            scr.blit(small.render(txt, True, TEXT if ok else DIM), (r.x, r.y - 18))
            pygame.draw.rect(scr, (60, 60, 70), r, border_radius=3)
            if ok:
                fx = r.x + r.w * (v - lo) / (hi - lo)
                pygame.draw.rect(scr, CORRIDOR, (r.x, r.y, fx - r.x, r.h), border_radius=3)
                pygame.draw.circle(scr, (255, 255, 255) if key == self.drag else CORRIDOR, (fx, r.centery), 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", help="сохранить картинку окна в файл и выйти")
    args = ap.parse_args()

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    view = View(world, Observer(world))

    pygame.init()
    scr = pygame.display.set_mode((MAP_W + PANEL_W, H))
    pygame.display.set_caption("Переход: отладка")
    mono = pygame.font.match_font("dejavusansmono,liberationmono,monospace")
    font, small = pygame.font.Font(mono, 14), pygame.font.Font(mono, 12)
    clock = pygame.time.Clock()
    controls = Controls()
    t_start = time.monotonic()
    try:
        while True:
            s = view.read()
            now = time.monotonic()
            controls.update(now)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if controls.event(e, now):
                    continue
                if e.type == pygame.MOUSEWHEEL:
                    view.scale = min(40.0, max(2.0, view.scale * (1.15 ** e.y)))
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and e.pos[0] < MAP_W:
                    wx, wy = view.to_world(*e.pos)
                    car = view.car_at(s, wx, wy)
                    print(f"x={wx:.2f}, y={wy:.2f}" + (
                        f"   машина {car['id']} {car['type']}: x={car['x']:.2f}, y={car['y']:.2f}, "
                        f"z={car['z']:.2f}, курс {car['yaw']:.0f}°, полоса {car['lane']}, "
                        f"{car['kmh']:.0f} км/ч" if car else ""))
            view.draw(scr, font, small, s, pygame.mouse.get_pos())
            controls.draw(scr, font, small, now)
            pygame.display.flip()
            if args.screenshot and (controls.connected(now) or now - t_start > 3):
                pygame.image.save(scr, args.screenshot)
                print(f"Сохранено: {args.screenshot}")
                return
            clock.tick(FPS)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
