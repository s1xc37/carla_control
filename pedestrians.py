"""
pedestrians.py — пешеходы на нашем переходе.

Пешеходы ходят вручную: каждый тик задаём телу направление и скорость (WalkerControl)
по своей ломаной. ИИ-контроллер CARLA (controller.ai.walker) не используем: на этой зебре
он не может перейти с E на W, а дойдя до цели, сам уходит в случайное место.

Машина состояний на каждого пешехода:
    APPROACH (к бордюру) -> WAIT -> CROSS (только в фазе PED) -> LEAVE (по тротуару прочь) -> сток
Наружу отдаём PedDemand: те же числа потом будет давать камера/YOLO.

Запуск отдельно (только геометрия и debug-отрисовка, никого не спавнит):
    python pedestrians.py
"""
import math
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import carla

from envconf import CARLA_HOST, CARLA_PORT, env

# ---- конфиг ----
PED_PER_HOUR = env("PED_PER_HOUR", 360)  # появлений в час на обе стороны (группа = одно появление)
GROUP_PROB = 0.0              # вероятность группы; группы 5–10 не влезают в лимит MAX_PER_SIDE
GROUP_SIZE = (5, 10)
GROUP_RADIUS = 2.0            # разброс группы вокруг точки появления, м
SPEED = (env("PED_SPEED_MIN", 1.2), env("PED_SPEED_MAX", 1.6))  # м/с
SPAWN_RING = (20.0, 40.0)     # где появляются: столько м от точки ожидания
SPAWN_MIN_FALLBACK = 10.0     # если у стороны тротуара на 20 м нет — ближняя граница
WAIT_OFFSET = 1.0             # точка ожидания: столько м от торца зебры наружу
WAIT_SNAP_MAX = 4.0           # если там не тротуар — ищем тротуар не дальше
SLOT_SPREAD = 1.2             # места у бордюра: разброс вдоль бордюра, м
SLOT_BACK = 0.8               # ... и от бордюра назад, м
LANE_OFFSET = (0.2, 1.0)      # на зебре: встречные идут по разным половинам, м от оси
ROAD_CLEAR = 0.3              # переход закончен: на столько м за крайней полосой
CROSS_MARGIN = env("CROSS_MARGIN", 2.0)  # запас по времени на переход, с
REACH = 0.5                   # точка ломаной пройдена, м
LEAVE_REACH = 1.2             # при уходе точку ожидания проходим с запасом — там могут стоять
AVOID_DIST = 0.8              # не наступаем на соседа: ближе этого впереди — обходим или ждём, м
SINK_REACH = 2.5              # до точки ухода точно доходить не нужно, м
NEAR_WAIT = 2.5               # толпа не пускает к месту: стоит ближе этого к точке ожидания ...
STILL_TIME = 2.0              # ... столько секунд — значит, пришёл
RESLOT = 1.5                  # ждущего оттолкнули дальше этого — возвращается на место, м
CURB_ZONE = 3.0               # у торцов зебры бордюр: упёршегося подсаживаем на него, м
STUCK_TIME = 10.0             # не сдвинулся на 0.5 м за столько секунд — застрял
GROUP_MIN = 5                 # столько ждущих с одной стороны = группа
MAX_PER_SIDE = env("MAX_PER_SIDE", 3)    # живых пешеходов, пришедших с одной стороны
NAV_SAMPLES = 10000           # выборок navmesh для поиска точек появления
DRAW_DEBUG = env("DRAW_DEBUG", True)     # выключить при записи кадров для YOLO
DEBUG_LIFE = 60.0

OTHER = {"W": "E", "E": "W"}
COLOR = {"W": carla.Color(0, 255, 0), "E": carla.Color(0, 120, 255)}


def d2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def path_len(loc, path):
    pts = [loc] + path
    return sum(d2(a, b) for a, b in zip(pts, pts[1:]))


def find_zebra(cmap, center):
    """Ближайший к center контур из get_crosswalks() (без повтора первой точки)."""
    polys, cur = [], []
    for p in cmap.get_crosswalks():
        cur.append(p)
        if len(cur) > 2 and p.distance(cur[0]) < 0.01:
            polys.append(cur[:-1])
            cur = []

    def centroid(pl):
        return carla.Location(x=sum(q.x for q in pl) / len(pl),
                              y=sum(q.y for q in pl) / len(pl), z=0)
    return min(polys, key=lambda pl: d2(centroid(pl), center))


def on_sidewalk(cmap, loc):
    return cmap.get_waypoint(loc, project_to_road=False,
                             lane_type=carla.LaneType.Sidewalk) is not None


def on_road(cmap, loc):
    return cmap.get_waypoint(loc, project_to_road=False,
                             lane_type=carla.LaneType.Driving) is not None


def sidewalk_line(cmap, a, b):
    """Весь отрезок a-b лежит на тротуаре (проверка каждый метр) — по нему можно идти по прямой."""
    n = max(1, int(d2(a, b)))
    for k in range(n + 1):
        t = k / n
        if not on_sidewalk(cmap, carla.Location(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z)):
            return False
    return True


def find_geometry(cmap, center, log=print):
    """Зебра у center: контур, торцы W/E, точки ожидания, оси, края полос.
    Общая для pedestrians.py и ws_sender.py — зоны и там и там одни и те же."""
    g = SimpleNamespace()
    g.zebra = find_zebra(cmap, center)
    pl = g.zebra
    if len(pl) != 4:
        raise RuntimeError(f"Ожидал 4-угольную зебру, а в ней {len(pl)} точек")
    # торцы = две самые короткие стороны, берём их середины
    edges = sorted(((pl[i], pl[(i + 1) % 4]) for i in range(4)), key=lambda e: d2(*e))[:2]
    ends = [carla.Location((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2) for a, b in edges]
    ends.sort(key=lambda e: e.x)   # условные имена: W — торец с меньшим x
    g.ends = {"W": ends[0], "E": ends[1]}
    L = d2(*ends)
    ux, uy = (ends[1].x - ends[0].x) / L, (ends[1].y - ends[0].y) / L
    g.normal = (-uy, ux)        # поперёк зебры
    log(f"Зебра: длина {L:.1f} м, торцы W=({ends[0].x:.2f}, {ends[0].y:.2f}) "
          f"E=({ends[1].x:.2f}, {ends[1].y:.2f})")

    g.wait, g.out = {}, {}
    for s in ("W", "E"):
        e, o = g.ends[s], g.ends[OTHER[s]]
        g.out[s] = ((e.x - o.x) / L, (e.y - o.y) / L)       # от зебры наружу
        p = carla.Location(e.x + g.out[s][0] * WAIT_OFFSET,
                           e.y + g.out[s][1] * WAIT_OFFSET, e.z)
        if on_sidewalk(cmap, p):
            g.wait[s] = p
            log(f"  {s}: ожидание ({p.x:.2f}, {p.y:.2f}) — на тротуаре")
            continue
        wp = cmap.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
        if wp is None or d2(wp.transform.location, p) > WAIT_SNAP_MAX:
            raise RuntimeError(f"{s}: у торца нет тротуара ближе {WAIT_SNAP_MAX} м")
        # от торца к центру полосы тротуара: первая точка на тротуаре + 0.5 м вглубь
        c = wp.transform.location
        n = d2(e, c)
        k = next(k for k in range(int(n / 0.1) + 1) if on_sidewalk(cmap, carla.Location(
            e.x + (c.x - e.x) * k * 0.1 / n, e.y + (c.y - e.y) * k * 0.1 / n, e.z)))
        f = min(1.0, (k * 0.1 + 0.5) / n)
        q = carla.Location(e.x + (c.x - e.x) * f, e.y + (c.y - e.y) * f, c.z)
        g.wait[s] = q
        log(f"  {s}: ожидание ({q.x:.2f}, {q.y:.2f}) — ПРИВЯЗАНА к краю тротуара "
              f"road={wp.road_id} lane={wp.lane_id}: расчётная точка ({p.x:.2f}, {p.y:.2f}) "
              f"не на тротуаре, до торца {d2(q, e):.1f} м")

    # край проезжей части вдоль оси зебры: идём от торца внутрь, пока не попадём на полосу
    g.stop = {}
    for s in ("W", "E"):
        e, (ox, oy) = g.ends[s], g.out[s]
        k = next(k for k in range(int(L / 0.1)) if on_road(cmap, carla.Location(
            e.x - ox * k * 0.1, e.y - oy * k * 0.1, e.z)))
        inset = max(0.0, k * 0.1 - ROAD_CLEAR)
        g.stop[s] = carla.Location(e.x - ox * inset, e.y - oy * inset, e.z)
        log(f"  {s}: край полос в {k * 0.1:.1f} м от торца, переход сюда заканчивается "
              f"в {inset:.1f} м от торца")
    g.length = L
    return g


@dataclass
class PedDemand:
    """Спрос на пешеходную фазу — только числа, без акторов CARLA."""
    waiting: dict           # {"W": n, "E": n} — стоят у бордюра
    max_wait: dict          # {"W": с, "E": с} — дольше всех ждёт
    mean_wait: float        # среднее текущее ожидание, с
    group_side: str | None  # сторона, где ждут >= GROUP_MIN, иначе None
    crossed_total: int


class Ped:
    def __init__(self, walker, side, speed, path, t, born):
        self.walker = walker
        self.side = side        # откуда пришёл
        self.speed = speed
        self.path = path        # оставшиеся точки ломаной
        self.born = born        # где появился
        self.state, self.t_state = "APPROACH", t
        self.t_wait = None
        self.need = 0.0         # сколько нужно на переход, с
        self.anchor, self.t_anchor = born, t   # для проверки «стоит на месте»
        self.t_nudge = -1.0
        self.slot = path[0]     # своё место у бордюра

    def set(self, state, t, path=None):
        self.state, self.t_state = state, t
        self.path = path or []
        self.anchor, self.t_anchor = self.walker.get_location(), t


class Pedestrians:
    def __init__(self, world, client, center, seed=0):
        self.world, self.client = world, client
        self.cmap = world.get_map()
        self.center = center
        self.rng = random.Random(seed + 1)   # свой генератор, чтобы не сбить поток машин
        world.set_pedestrians_seed(seed)     # повторяемые точки navmesh

        self._find_geometry()
        self._find_spawn_points()
        if DRAW_DEBUG:
            self._draw()

        self.bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
        self.peds = []
        self.pos = {}           # id тела -> позиция на этом тике
        self.rate = PED_PER_HOUR / 3600.0
        self.next_spawn = self.rng.expovariate(self.rate)
        self.spawned = self.blocked = self.failed = 0
        self.crossed = 0
        self.stuck = {"APPROACH": 0, "CROSS": 0, "LEAVE": 0}
        self.done_waits = []
        self.cross_times = []   # фактическое время перехода, с
        self.slow = 0           # переходов дольше расчёта заметно
        self.nudges = 0         # подсадок на бордюр

    # ---- геометрия ----
    def _find_geometry(self):
        g = find_geometry(self.cmap, self.center)
        self.zebra, self.ends, self.wait = g.zebra, g.ends, g.wait
        self.out, self.normal, self.stop = g.out, g.normal, g.stop

        for s in ("W", "E"):
            n = path_len(self.wait[s], [self.ends[s], self.stop[OTHER[s]]])
            print(f"  переход {s}->{OTHER[s]}: {n:.1f} м, нужно "
                  f"{n / SPEED[1] + CROSS_MARGIN:.1f}–{n / SPEED[0] + CROSS_MARGIN:.1f} с "
                  f"(скорость {SPEED[0]}–{SPEED[1]} м/с + запас {CROSS_MARGIN} с)")

    def _find_spawn_points(self):
        """Точки navmesh в кольце SPAWN_RING от точки ожидания, до которой по прямой — только тротуар."""
        t0 = time.time()
        near = {"W": [], "E": []}
        for _ in range(NAV_SAMPLES):
            p = self.world.get_random_location_from_navigation()
            if p is None:
                continue
            for s, w in self.wait.items():
                if SPAWN_MIN_FALLBACK <= d2(p, w) <= SPAWN_RING[1] and sidewalk_line(self.cmap, p, w):
                    near[s].append(p)
        print(f"  точки появления ({NAV_SAMPLES} выборок navmesh, {time.time() - t0:.1f} с):")
        self.cands = {}
        for s, c in near.items():
            far = [p for p in c if d2(p, self.wait[s]) >= SPAWN_RING[0]]
            if far:
                self.cands[s] = far
                print(f"    {s}: {len(far)} в {SPAWN_RING[0]:.0f}–{SPAWN_RING[1]:.0f} м")
            elif c:
                self.cands[s] = c
                top = max(d2(p, self.wait[s]) for p in c)
                print(f"    {s}: {len(c)} — ВНИМАНИЕ: тротуар по прямой кончается через {top:.0f} м, "
                      f"появляются ближе: {SPAWN_MIN_FALLBACK:.0f}–{top:.0f} м")
            else:
                raise RuntimeError(f"{s}: нет точек появления — увеличь NAV_SAMPLES")

    def _draw(self):
        dbg, up = self.world.debug, carla.Location(z=0.3)
        pl = self.zebra
        for a, b in zip(pl, pl[1:] + pl[:1]):
            dbg.draw_line(a + up, b + up, thickness=0.08,
                          color=carla.Color(255, 255, 255), life_time=DEBUG_LIFE)
        for s in ("W", "E"):
            dbg.draw_point(self.ends[s] + up, size=0.15, color=carla.Color(255, 0, 0),
                           life_time=DEBUG_LIFE)
            dbg.draw_point(self.wait[s] + up, size=0.3, color=COLOR[s], life_time=DEBUG_LIFE)
            dbg.draw_string(self.wait[s] + carla.Location(z=2), f"WAIT {s}",
                            color=COLOR[s], life_time=DEBUG_LIFE)
            for c in self.cands[s]:
                dbg.draw_point(c + up, size=0.1, color=COLOR[s], life_time=DEBUG_LIFE)

    # ---- маршруты ----
    def _slot(self, side):
        """Своё место у бордюра, чтобы толпа не толкалась в одной точке."""
        a = self.rng.uniform(-SLOT_SPREAD, SLOT_SPREAD)
        b = self.rng.uniform(0.0, SLOT_BACK)
        (nx, ny), (ox, oy), w = self.normal, self.out[side], self.wait[side]
        p = carla.Location(w.x + nx * a + ox * b, w.y + ny * a + oy * b, w.z)
        return p if on_sidewalk(self.cmap, p) else w

    def _cross_path(self, side):
        """Свой торец -> сразу за крайней полосой того края; встречные идут по разным
        половинам зебры. Бордюр на том конце — уже LEAVE, в CROSS считаем только дорогу."""
        k = self.rng.uniform(*LANE_OFFSET) * (1 if side == "E" else -1)
        nx, ny = self.normal
        return [carla.Location(e.x + nx * k, e.y + ny * k, e.z)
                for e in (self.ends[side], self.stop[OTHER[side]])]

    def _near(self, base):
        for _ in range(5):
            r = GROUP_RADIUS * math.sqrt(self.rng.random())
            a = self.rng.uniform(0, 2 * math.pi)
            p = carla.Location(base.x + r * math.cos(a), base.y + r * math.sin(a), base.z)
            if on_sidewalk(self.cmap, p):
                return p
        return base

    # ---- поток ----
    def _spawn_event(self, t):
        n = self.rng.randint(*GROUP_SIZE) if self.rng.random() < GROUP_PROB else 1
        room = [s for s in ("W", "E") if sum(p.side == s for p in self.peds) + n <= MAX_PER_SIDE]
        if not room:                     # обе стороны заполнены
            self.blocked += 1
            return
        side = self.rng.choice(room)
        base = self.rng.choice(self.cands[side])
        speed = self.rng.uniform(*SPEED)   # у группы одна скорость — идут вместе
        for i in range(n):
            loc = base if i == 0 else self._near(base)
            slot = self._slot(side)
            yaw = math.degrees(math.atan2(slot.y - loc.y, slot.x - loc.x))
            w = self.world.try_spawn_actor(
                self.rng.choice(self.bps),
                carla.Transform(loc + carla.Location(z=1.0), carla.Rotation(yaw=yaw)))
            if w is None:
                self.failed += 1
                continue
            self.peds.append(Ped(w, side, speed, [slot], t, loc))
            self.spawned += 1

    # ---- машина состояний, вызывать каждый тик после world.tick() ----
    def update(self, t, phase, phase_left):
        cmds, gone = [], []
        self.pos = {p.walker.id: p.walker.get_location() for p in self.peds}
        for p in self.peds:
            loc = self.pos[p.walker.id]
            if d2(loc, p.anchor) > 0.5:          # сдвинулся — не застрял
                p.anchor, p.t_anchor = loc, t
            still = t - p.t_anchor

            if p.state == "WAIT":
                path = self._cross_path(p.side)
                p.need = path_len(loc, path) / p.speed + CROSS_MARGIN
                # начинаем, только если успеем до конца фазы PED
                if phase == "PED" and phase_left >= p.need:
                    self.done_waits.append(t - p.t_wait)
                    p.set("CROSS", t, path)
                    cmds.append(self._control(p, loc))
                elif p.path:                     # возвращается на своё место
                    if d2(loc, p.path[0]) < REACH or still > STILL_TIME:
                        p.slot, p.path = loc, []  # дошёл или толпа не пускает — стоит здесь
                    cmds.append(self._control(p, loc))
                elif d2(loc, p.slot) > RESLOT:   # оттолкнули — идёт обратно
                    p.path = [p.slot]
                    p.anchor, p.t_anchor = loc, t
                    cmds.append(self._control(p, loc))
                continue

            # идём по ломаной
            reach = LEAVE_REACH if p.state == "LEAVE" else REACH
            while p.path and d2(loc, p.path[0]) < reach:
                p.path.pop(0)
            done = not p.path or (p.state == "LEAVE" and len(p.path) == 1
                                  and d2(loc, p.path[0]) < SINK_REACH)
            if p.state == "APPROACH" and not done and still > STILL_TIME \
                    and d2(loc, self.wait[p.side]) < NEAR_WAIT:
                done = True                      # толпа у бордюра не пускает ближе

            if done:
                if p.state == "APPROACH":
                    if p.path:                   # не дошёл до места из-за толпы — место здесь
                        p.slot = loc
                    p.set("WAIT", t)
                    p.t_wait = t
                    cmds.append(self._control(p))
                    continue
                if p.state == "CROSS":
                    self.crossed += 1
                    took, need = t - p.t_state, p.need - CROSS_MARGIN
                    self.cross_times.append(took)
                    if took > need + 3:
                        self.slow += 1
                        print(f"   ! долгий переход {p.side}->{OTHER[p.side]}: {took:.0f}с "
                              f"при расчёте {need:.0f}с, ушёл с ({p.anchor.x:.1f}, {p.anchor.y:.1f})")
                    # сначала на тротуар через точку ожидания, потом прочь
                    other = OTHER[p.side]
                    p.set("LEAVE", t, [self.wait[other], self.rng.choice(self.cands[other])])
                else:                            # LEAVE дошёл — сток
                    gone.append(p)
                    continue
            elif still > STUCK_TIME:
                self._stuck(p, loc)
                gone.append(p)
                continue
            self._curb(p, loc, t)
            cmds.append(self._control(p, loc))

        if cmds:
            self.client.apply_batch(cmds)
        if gone:
            self._remove(gone)
        if t >= self.next_spawn:
            self._spawn_event(t)
            self.next_spawn = t + self.rng.expovariate(self.rate)

    def _curb(self, p, loc, t):
        """Ручной пешеход не забирается на бордюр у торцов зебры — подсаживаем на 30 см."""
        v = p.walker.get_velocity()
        if math.hypot(v.x, v.y) > 0.2 or t - p.t_state < 0.5 or t - p.t_nudge < 0.5 \
                or min(d2(loc, e) for e in self.ends.values()) > CURB_ZONE:
            return
        tg = p.path[0]
        n = d2(loc, tg)
        p.walker.set_location(carla.Location(loc.x + (tg.x - loc.x) / n * 0.3,
                                             loc.y + (tg.y - loc.y) / n * 0.3, loc.z + 0.3))
        p.t_nudge = t
        self.nudges += 1

    def _control(self, p, loc=None):
        """Команда телу: идти к очередной точке ломаной, а без ломаной — стоять лицом к дороге."""
        if not p.path:
            ox, oy = self.out[p.side]
            ctrl = carla.WalkerControl(direction=carla.Vector3D(-ox, -oy, 0), speed=0.0)
        else:
            tg = p.path[0]
            dx, dy = tg.x - loc.x, tg.y - loc.y
            n = math.hypot(dx, dy)
            d = self._free_dir(p, loc, dx / n, dy / n)
            if d is None:                        # впереди везде люди — ждём на месте
                ctrl = carla.WalkerControl(direction=carla.Vector3D(dx / n, dy / n, 0), speed=0.0)
            else:
                ctrl = carla.WalkerControl(direction=carla.Vector3D(d[0], d[1], 0), speed=p.speed)
        return carla.command.ApplyWalkerControl(p.walker.id, ctrl)

    def _free_dir(self, p, loc, ux, uy):
        """Не наступаем на соседа (иначе физика выталкивает, бывает — на дорогу):
        прямо, иначе обход правее/левее на 60°; None — занято везде."""
        near = [q for wid, q in self.pos.items() if wid != p.walker.id and d2(q, loc) < AVOID_DIST]
        for a in (0.0, -60.0, 60.0):
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            rx, ry = ux * c - uy * s, ux * s + uy * c
            # сосед «впереди» — в секторе ±60° от направления
            if all((q.x - loc.x) * rx + (q.y - loc.y) * ry < 0.5 * d2(q, loc) for q in near):
                return rx, ry
        return None

    def _stuck(self, p, loc):
        self.stuck[p.state] += 1
        tg = p.path[0] if p.path else loc
        print(f"   ! застрял {p.state} {p.side}: ({loc.x:.1f}, {loc.y:.1f}) до цели "
              f"{d2(loc, tg):.1f} м, появился ({p.born.x:.1f}, {p.born.y:.1f})")

    def _remove(self, peds):
        self.client.apply_batch([carla.command.DestroyActor(p.walker.id) for p in peds])
        ids = {p.walker.id for p in peds}
        self.peds = [p for p in self.peds if p.walker.id not in ids]

    # ---- данные наружу ----
    def demand(self, t):
        waits = {"W": [], "E": []}
        for p in self.peds:
            if p.state == "WAIT":
                waits[p.side].append(t - p.t_wait)
        allw = waits["W"] + waits["E"]
        big = [s for s in ("W", "E") if len(waits[s]) >= GROUP_MIN]
        return PedDemand(
            waiting={s: len(w) for s, w in waits.items()},
            max_wait={s: max(w, default=0.0) for s, w in waits.items()},
            mean_wait=sum(allw) / len(allw) if allw else 0.0,
            group_side=max(big, key=lambda s: len(waits[s])) if big else None,
            crossed_total=self.crossed)

    def stats_line(self, d):
        n = {s: sum(p.state == s for p in self.peds) for s in ("APPROACH", "CROSS", "LEAVE")}
        hist = sum(self.done_waits) / len(self.done_waits) if self.done_waits else 0.0
        ct = (f"{sum(self.cross_times) / len(self.cross_times):.0f}/{max(self.cross_times):.0f}с"
              if self.cross_times else "-")
        group = f"  ГРУППА {d.group_side}" if d.group_side else ""
        stuck = sum(self.stuck.values())
        return (f"   пеш: ждут W={d.waiting['W']:2} (макс {d.max_wait['W']:3.0f}с) "
                f"E={d.waiting['E']:2} (макс {d.max_wait['E']:3.0f}с){group}  "
                f"ср.ожид {d.mean_wait:3.0f}с (у перешедших {hist:.0f}с) | "
                f"идут {n['APPROACH']} переходят {n['CROSS']} уходят {n['LEAVE']} | "
                f"перешло {d.crossed_total} (переход ср/макс {ct}, долгих {self.slow}, подсадок {self.nudges})  +{self.spawned} "
                f"(лимит {MAX_PER_SIDE} на сторону — не появились {self.blocked}, спавн не удался {self.failed})"
                + (f"  ЗАСТРЯЛИ {self.stuck}" if stuck else ""))

    def destroy(self):
        self.client.apply_batch([carla.command.DestroyActor(p.walker.id) for p in self.peds])
        print(f"Пешеходов убрано: {len(self.peds)}")
        self.peds = []


if __name__ == "__main__":
    # только геометрия и отрисовка, никого не спавним
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    center = carla.Location(x=32.3, y=-178.5, z=0.0)
    Pedestrians(world, client, center)
    world.get_spectator().set_transform(carla.Transform(
        center + carla.Location(z=45), carla.Rotation(pitch=-89)))
    print(f"Нарисовано на {DEBUG_LIFE:.0f} с. W — зелёный, E — синий, красные — торцы.")
