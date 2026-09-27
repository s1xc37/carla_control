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

import carla
import pygame

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
        scr.blit(font.render("   id  тип        x        y  полоса км/ч", True, DIM), (x0, y))
        y += 20
        for c in s["cars"]:
            if y > H - 50:
                break
            lane = f"{c['lane'][0]} {c['lane'][1]}" if c["lane"] else "-"
            row = (f"{'★' if c['lane'] else ' '}{c['id']:>4} {c['type']:<7} "
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
        scr.blit(small.render(info + "   (клик — в консоль, колесо — масштаб)", True, TEXT), (10, H - 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", help="сохранить картинку окна в файл и выйти")
    args = ap.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    view = View(world, Observer(world))

    pygame.init()
    scr = pygame.display.set_mode((MAP_W + PANEL_W, H))
    pygame.display.set_caption("Переход: отладка")
    mono = pygame.font.match_font("dejavusansmono,liberationmono,monospace")
    font, small = pygame.font.Font(mono, 14), pygame.font.Font(mono, 12)
    clock = pygame.time.Clock()
    try:
        while True:
            s = view.read()
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
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
            pygame.display.flip()
            if args.screenshot:
                pygame.image.save(scr, args.screenshot)
                print(f"Сохранено: {args.screenshot}")
                return
            clock.tick(FPS)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
