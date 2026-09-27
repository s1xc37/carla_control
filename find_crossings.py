"""
find_crossings.py — поиск регулируемых переходов на карте CARLA.

Перебирает все светофоры карты, для каждого считает полосы на подходе
(попутные / встречные) через OpenDRIVE и ищет рядом пешеходный переход.
Выводит отсортированный список кандидатов и подписывает их прямо в мире.

Примеры:
    python find_crossings.py                     # текущая карта
    python find_crossings.py --map Town10HD_Opt  # загрузить карту (сбросит мир!)
    python find_crossings.py --goto 133          # камера над светофором 133
"""
import argparse
import math

import carla

p = argparse.ArgumentParser()
p.add_argument("--map", help="загрузить карту перед поиском, например Town05 (сбросит мир!)")
p.add_argument("--lanes", type=int, default=2, help="нужно полос в каждую сторону")
p.add_argument("--cw-dist", type=float, default=15.0, help="макс. расстояние стоп-линия -> переход, м")
p.add_argument("--goto", type=int, help="id светофора: перенести камеру к нему")
p.add_argument("--life", type=float, default=120.0, help="сколько секунд висят подписи")
args = p.parse_args()

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.load_world(args.map) if args.map else client.get_world()
m = world.get_map()
print("Карта:", m.name)


def dist2d(a, b):
    """Расстояние по земле, без учёта высоты."""
    return math.hypot(a.x - b.x, a.y - b.y)


# ---- режим перелёта ----
if args.goto is not None:
    tl = world.get_actor(args.goto)
    if tl is None:
        raise SystemExit(f"Светофор {args.goto} не найден (id могли поменяться после перезагрузки)")
    loc = tl.get_location()
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=45), carla.Rotation(pitch=-89)))
    raise SystemExit(f"Камера над TL {args.goto}: ({loc.x:.1f}, {loc.y:.1f})")

# ---- переходы: плоский список точек -> контуры -> центры ----
# get_crosswalks() отдаёт все переходы одним списком; каждый контур
# замыкается повтором своей первой точки
polys, cur = [], []
for pt in m.get_crosswalks():
    cur.append(pt)
    if len(cur) > 2 and pt.distance(cur[0]) < 0.01:
        polys.append(cur)
        cur = []
cw_centers = [carla.Location(x=sum(q.x for q in pl) / len(pl),
                             y=sum(q.y for q in pl) / len(pl), z=0) for pl in polys]


# ---- подсчёт полос через OpenDRIVE ----
# в OpenDRIVE lane_id < 0 — полосы по направлению дороги, > 0 — встречные.
# спрашиваем у карты полосы с id ±1..±8 в той же точке дороги (road_id, s)
def count_lanes(wp):
    neg = pos = 0
    for lid in range(1, 9):
        for sign in (-1, 1):
            w = m.get_waypoint_xodr(wp.road_id, sign * lid, wp.s)
            if w is not None and w.lane_type == carla.LaneType.Driving:
                if sign < 0:
                    neg += 1
                else:
                    pos += 1
    # «попутные» — те, что на стороне самой стоп-линии
    return (neg, pos) if wp.lane_id < 0 else (pos, neg)


# ---- перебор светофоров ----
rows = []
for tl in world.get_actors().filter("traffic.traffic_light"):
    stops = tl.get_stop_waypoints()
    if not stops:
        continue
    wp = stops[0]
    towards, against = count_lanes(wp)
    sl = wp.transform.location
    cw_d = min((dist2d(sl, c) for c in cw_centers), default=float("inf"))
    rows.append(dict(tl=tl, loc=tl.get_location(), stops=len(stops),
                     towards=towards, against=against, cw=cw_d,
                     group=len(tl.get_group_traffic_lights())))

cands = [r for r in rows
         if r["towards"] >= args.lanes and r["against"] >= args.lanes
         and r["cw"] <= args.cw_dist]
# сначала точные совпадения по полосам, внутри — по близости перехода
cands.sort(key=lambda r: (r["towards"] != args.lanes or r["against"] != args.lanes, r["cw"]))

print(f"\nВсего светофоров: {len(rows)}, кандидатов: {len(cands)}\n")
print(f"{'TL':>5} {'x':>8} {'y':>8}  полосы(к/от)  стоп-линий  переход,м  в группе")
for r in cands:
    print(f"{r['tl'].id:>5} {r['loc'].x:>8.1f} {r['loc'].y:>8.1f}  "
          f"{r['towards']:>5}/{r['against']:<6} {r['stops']:>10}  {r['cw']:>9.1f}  {r['group']:>8}")
    exact = r["towards"] == args.lanes and r["against"] == args.lanes
    world.debug.draw_string(r["loc"] + carla.Location(z=8),
                            f"CAND {r['tl'].id} {r['towards']}/{r['against']}",
                            color=carla.Color(0, 255, 0) if exact else carla.Color(255, 160, 0),
                            life_time=args.life)

print("\nЛететь к кандидату: python find_crossings.py --goto <TL>")
print("Группа 3 = T-перекрёсток, 4 = обычный крестовой.")