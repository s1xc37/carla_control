"""
ws_sender.py — читает состояние перехода из CARLA и рассылает его по WebSocket.

Отдельный процесс: подключается к тому же серверу CARLA вторым клиентом и только читает мир
(машины, светофоры, пешеходов), ничего в нём не меняя. Сцену крутит run_scene.py.
Формат сообщения — JSON_FORMAT.md.

Запуск: сначала run_scene.py, потом в другом терминале
    python ws_sender.py
Клиенты подключаются к ws://<ip этой машины>:8765 и получают JSON раз в SEND_PERIOD с симуляции.
Нужна библиотека websockets:  uv pip install websockets
"""
import asyncio
import json
import math

import carla
from websockets.asyncio.server import broadcast, serve

from envconf import CARLA_HOST, CARLA_MAP, CARLA_PORT, env
from pedestrians import d2, find_geometry

# ---- конфиг ----
CW_CENTER = carla.Location(x=32.3, y=-178.5, z=0.0)  # центр зебры, как в run_scene.py
HOST, PORT = env("WS_HOST", "0.0.0.0"), env("WS_PORT", 8765)
SEND_PERIOD = env("SEND_PERIOD", 0.5)     # раз в столько секунд симуляции
ROAD_LENGTH = env("ROAD_LENGTH", 50.0)    # машины на нашей дороге — до столько м от зебры
ZEBRA_MARGIN = 2.0            # ... и на самой зебре: до столько м за её дальним краем
ZONE_HALF = 1.5               # зона ожидания пешеходов — квадрат 3×3 м вокруг точки ожидания
TICK_TIMEOUT = 10.0           # столько ждём тик мира, с

GO = (carla.TrafficLightState.Green, carla.TrafficLightState.Yellow)


def inside(poly, p):
    """Точка внутри многоугольника (по земле)."""
    res = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if (a.y > p.y) != (b.y > p.y) and p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x:
            res = not res
    return res


def vehicle_type(attrs):
    """Тип машины по атрибутам чертежа CARLA: car / van / truck / bus / special (как в JSON_FORMAT.md)."""
    base = attrs.get("base_type", "").lower()
    if attrs.get("special_type") == "emergency":
        return "special"
    return base if base in ("van", "truck", "bus") else "car"


class Observer:
    """Собирает одно сообщение. Геометрия ищется один раз, дальше мир только читается."""

    def __init__(self, world):
        self.world = world
        cmap = world.get_map()
        g = self.g = find_geometry(cmap, CW_CENTER, log=lambda *a: None)
        self.center = carla.Location(sum(q.x for q in g.zebra) / 4, sum(q.y for q in g.zebra) / 4, 0)
        self.u = ((g.ends["E"].x - g.ends["W"].x) / g.length,        # поперёк дороги, с W на E
                  (g.ends["E"].y - g.ends["W"].y) / g.length)

        # наш светофор и главная дорога — так же, как в run_scene.py
        lights = list(world.get_actors().filter("traffic.traffic_light"))
        self.our = min(lights, key=lambda tl: min(
            (d2(w.transform.location, CW_CENTER) for w in tl.get_stop_waypoints()), default=1e9))
        self.main = [tl for tl in self.our.get_group_traffic_lights() if tl.id != self.our.id]

        # вдоль нашей дороги, от зебры: нормаль к зебре, смотрящая на стоп-линию
        stop = self.our.get_stop_waypoints()[0]
        nx, ny = g.normal
        if self._dot(stop.transform.location, self.center, (nx, ny)) < 0:
            nx, ny = -nx, -ny
        self.n = (nx, ny)
        self.half = max(abs(self._dot(q, self.center, self.n)) for q in g.zebra)  # полширины зебры

        # полосы нашей дороги на уровне стоп-линии: положение поперёк, ширина, направление
        lanes = []
        for lid in range(-6, 7):
            wp = cmap.get_waypoint_xodr(stop.road_id, lid, stop.s) if lid else None
            if wp is not None and wp.lane_type == carla.LaneType.Driving:
                f = wp.transform.get_forward_vector()
                lanes.append((self._dot(wp.transform.location, g.ends["W"], self.u), wp.lane_width,
                              "in" if f.x * nx + f.y * ny < 0 else "out"))
        self.lanes = sorted(lanes)     # номер полосы = индекс + 1, с W на E
        self.types = {}                # id машины -> type (атрибуты не меняются)

        print(f"Наш светофор {self.our.id}, дорога road={stop.road_id}; полосы с W на E: "
              + ", ".join(f"{i + 1}:{d}" for i, (_, _, d) in enumerate(self.lanes)))
        print("Зоны ожидания: " + ", ".join(
            f"{s} ({w.x:.2f}, {w.y:.2f})" for s, w in g.wait.items()) + f", {2 * ZONE_HALF:.0f}×{2 * ZONE_HALF:.0f} м")

    @staticmethod
    def _dot(p, origin, axis):
        return (p.x - origin.x) * axis[0] + (p.y - origin.y) * axis[1]

    def phase(self):
        if self.our.state in GO:
            return "SIDE"
        if any(tl.state in GO for tl in self.main):
            return "MAIN"
        return "PED"

    def vtype(self, v):
        t = self.types.get(v.id)
        if t is None:
            t = self.types[v.id] = vehicle_type(v.attributes)
        return t

    def _bounds(self):
        """Коридор нашей дороги: поперёк (от торца W) и вдоль (от центра зебры), м."""
        lo = self.lanes[0][0] - self.lanes[0][1] / 2
        hi = self.lanes[-1][0] + self.lanes[-1][1] / 2
        return lo, hi, -(self.half + ZEBRA_MARGIN), ROAD_LENGTH

    def lane_of(self, p):
        """(номер полосы 1–4, направление), если точка в коридоре нашей дороги, иначе None."""
        lo, hi, b_lo, b_hi = self._bounds()
        a = self._dot(p, self.g.ends["W"], self.u)              # поперёк дороги
        b = self._dot(p, self.center, self.n)                   # вдоль дороги от зебры
        if not (lo <= a <= hi and b_lo <= b <= b_hi):
            return None
        i = min(range(len(self.lanes)), key=lambda k: abs(self.lanes[k][0] - a))
        return i + 1, self.lanes[i][2]

    def corridor(self):
        """Углы коридора в координатах мира — для отрисовки."""
        lo, hi, b_lo, b_hi = self._bounds()
        w, (ux, uy), (nx, ny) = self.g.ends["W"], self.u, self.n
        b0 = self._dot(w, self.center, self.n)                  # торец W относительно центра вдоль дороги
        return [(w.x + ux * a + nx * (b - b0), w.y + uy * a + ny * (b - b0))
                for a, b in ((lo, b_lo), (hi, b_lo), (hi, b_hi), (lo, b_hi))]

    def vehicles(self):
        out = []
        for v in self.world.get_actors().filter("vehicle.*"):
            lane = self.lane_of(v.get_location())
            if lane is None:
                continue
            vel = v.get_velocity()
            out.append({"id": v.id, "type": self.vtype(v), "lane": lane[0],
                        "direction": lane[1],
                        "speed_kmh": round(math.hypot(vel.x, vel.y) * 3.6, 1)})
        return out

    def pedestrians(self):
        waiting, crossing = {"W": 0, "E": 0}, 0
        for w in self.world.get_actors().filter("walker.pedestrian.*"):
            p = w.get_location()
            for s, c in self.g.wait.items():
                if abs(self._dot(p, c, self.u)) <= ZONE_HALF and abs(self._dot(p, c, self.n)) <= ZONE_HALF:
                    waiting[s] += 1
            crossing += inside(self.g.zebra, p)
        return {"waiting": waiting, "crossing": crossing}

    def message(self, snap):
        return {"t": round(snap.timestamp.elapsed_seconds, 2), "frame": snap.frame,
                "phase": self.phase(), "vehicles": self.vehicles(),
                "pedestrians": self.pedestrians()}


async def main():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    # сцена могла ещё не загрузить нужную карту — ждём, иначе геометрия будет не та
    while CARLA_MAP and CARLA_MAP not in world.get_map().name:
        print(f"Открыта {world.get_map().name}, жду {CARLA_MAP} (её грузит run_scene.py)...")
        await asyncio.sleep(3)
        world = client.get_world()
    obs = Observer(world)

    async def handler(ws):
        print(f"+ клиент {ws.remote_address}")
        await ws.wait_closed()
        print(f"- клиент {ws.remote_address}")

    async with serve(handler, HOST, PORT) as server:
        print(f"WebSocket: ws://{HOST}:{PORT}, сообщение раз в {SEND_PERIOD} с симуляции")
        last, sent = -math.inf, 0
        while True:
            try:
                snap = await asyncio.to_thread(world.wait_for_tick, TICK_TIMEOUT)
            except RuntimeError:
                print(f"Нет тиков мира {TICK_TIMEOUT:.0f} с — CARLA и run_scene.py запущены?")
                continue
            if snap.timestamp.elapsed_seconds - last < SEND_PERIOD:
                continue
            last = snap.timestamp.elapsed_seconds
            msg = obs.message(snap)
            broadcast(server.connections, json.dumps(msg, ensure_ascii=False))
            sent += 1
            if sent % 20 == 0:
                print(f"[{msg['phase']:4}] отправлено {sent}, клиентов {len(server.connections)}, "
                      f"машин {len(msg['vehicles'])}, пешеходы {msg['pedestrians']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nОстановлен.")
