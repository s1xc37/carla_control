import math
import carla

# ---- конфиг участка ----
CENTER = carla.Location(x=32.3, y=-178.5, z=0.0)
RADIUS_TL = 50.0          # где ищем светофоры
SPAWN_RING = (50.0, 150.0) # кольцо для спавна машин
LIFE = 60.0                # сколько секунд висят отрисовки

def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
m = world.get_map()
dbg = world.debug
print("Карта:", m.name)

# центр участка
dbg.draw_point(CENTER + carla.Location(z=1), size=0.3,
               color=carla.Color(255, 0, 255), life_time=LIFE)

# ---- 1. светофоры рядом ----
lights = [tl for tl in world.get_actors().filter("traffic.traffic_light")
          if dist2d(tl.get_location(), CENTER) < RADIUS_TL]
print(f"\nСветофоров в радиусе {RADIUS_TL} м: {len(lights)}")
for tl in sorted(lights, key=lambda t: dist2d(t.get_location(), CENTER)):
    loc = tl.get_location()
    stops = tl.get_stop_waypoints()
    print(f"  id={tl.id}  dist={dist2d(loc, CENTER):.1f} м  "
          f"state={tl.state}  стоп-линий={len(stops)}")
    dbg.draw_string(loc + carla.Location(z=6), f"TL {tl.id}",
                    color=carla.Color(255, 255, 0), life_time=LIFE)
    for wp in stops:  # стоп-линии, которыми управляет этот светофор
        dbg.draw_point(wp.transform.location + carla.Location(z=0.5),
                       size=0.2, color=carla.Color(255, 0, 0), life_time=LIFE)

# ---- 2. ближайший переход ----
# get_crosswalks() отдаёт плоский список точек; каждый контур
# замыкается повтором своей первой точки
pts = m.get_crosswalks()
polys, cur = [], []
for p in pts:
    cur.append(p)
    if len(cur) > 2 and p.distance(cur[0]) < 0.01:
        polys.append(cur)
        cur = []

def centroid(poly):
    return carla.Location(x=sum(p.x for p in poly) / len(poly),
                          y=sum(p.y for p in poly) / len(poly), z=0)

polys.sort(key=lambda poly: dist2d(centroid(poly), CENTER))
print(f"\nПереходов на карте: {len(polys)}")
for i, poly in enumerate(polys[:3]):  # три ближайших
    c = centroid(poly)
    print(f"  переход #{i}: центр=({c.x:.1f}, {c.y:.1f})  "
          f"dist={dist2d(c, CENTER):.1f} м  точек={len(poly)}")
    color = carla.Color(0, 255, 0) if i == 0 else carla.Color(0, 120, 255)
    for a, b in zip(poly, poly[1:]):
        dbg.draw_line(a + carla.Location(z=0.3), b + carla.Location(z=0.3),
                      thickness=0.15, color=color, life_time=LIFE)
    dbg.draw_string(c + carla.Location(z=2), f"CW {i}",
                    color=color, life_time=LIFE)

# ---- 3. точки спавна машин, едущих К переходу ----
good = []
for sp in m.get_spawn_points():
    d = dist2d(sp.location, CENTER)
    if not (SPAWN_RING[0] < d < SPAWN_RING[1]):
        continue
    fwd = sp.get_forward_vector()
    to_c = CENTER - sp.location
    # скалярное произведение > 0 — машина смотрит в сторону центра
    if fwd.x * to_c.x + fwd.y * to_c.y > 0:
        good.append(sp)
        dbg.draw_arrow(sp.location + carla.Location(z=1),
                       sp.location + fwd * 4 + carla.Location(z=1),
                       thickness=0.2, arrow_size=0.5,
                       color=carla.Color(0, 255, 255), life_time=LIFE)
print(f"\nТочек спавна в кольце {SPAWN_RING}, смотрящих к центру: {len(good)}")