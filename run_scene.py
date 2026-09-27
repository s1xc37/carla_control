import math
import random
import signal
import carla
import control
from envconf import CARLA_HOST, CARLA_MAP, CARLA_PORT, env
from pedestrians import Pedestrians

# ---- конфиг ----
CW_CENTER = carla.Location(x=32.3, y=-178.5, z=0.0)  # центр зебры
SOURCE_RING = (40.0, 150.0)   # где появляются машины
SINK_RADIUS = 170.0           # дальше — удаляем
VEH_PER_HOUR = env("VEH_PER_HOUR", 1200)   # интенсивность суммарно по всем подходам
MAX_VEHICLES = env("MAX_VEHICLES", 60)     # потолок, чтобы не задушить сервер
SEED = env("SEED", 42)
TM_PORT = env("TM_PORT", 8000)            # порт Traffic Manager
DT = 0.05
# длительности фаз, с; PED >= время перехода (~14–18 с)
PHASES = [("MAIN", env("PHASE_MAIN", 20.0)), ("SIDE", env("PHASE_SIDE", 15.0)),
          ("PED", env("PHASE_PED", 20.0))]

G, R = carla.TrafficLightState.Green, carla.TrafficLightState.Red
random.seed(SEED)

def d2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def on_sigterm(*_):  # docker stop шлёт SIGTERM — уходим через ту же уборку, что и Ctrl+C
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, on_sigterm)

client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(20.0)
world = client.get_world()
if CARLA_MAP and CARLA_MAP not in world.get_map().name:
    print(f"Открыта {world.get_map().name}, загружаю {CARLA_MAP}...")
    client.set_timeout(120.0)                 # загрузка карты долгая
    world = client.load_world(CARLA_MAP)
    client.set_timeout(20.0)
print(f"Сервер {CARLA_HOST}:{CARLA_PORT}, карта {world.get_map().name}; "
      f"фазы {', '.join(f'{n} {d:g}с' for n, d in PHASES)}; машин/ч {VEH_PER_HOUR}, SEED {SEED}")
tm = client.get_trafficmanager(TM_PORT)
tm.set_random_device_seed(SEED)
vehicles = []
peds = None
ctl = None

try:
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = DT
    world.apply_settings(s)
    tm.set_synchronous_mode(True)

    # ---- светофоры ----
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    our = min(lights, key=lambda tl: min(
        (d2(w.transform.location, CW_CENTER) for w in tl.get_stop_waypoints()),
        default=1e9))
    group = our.get_group_traffic_lights()
    main_road = [tl for tl in group if tl.id != our.id]
    print(f"Наш: {our.id}, главная дорога: {[tl.id for tl in main_road]}")

    world.freeze_all_traffic_lights(True)
    group_ids = {tl.id for tl in group}
    for tl in lights:
        if tl.id not in group_ids:
            tl.set_state(G)

    # ---- пешеходы (до спавна: cross_factor и точки ожидания) ----
    peds = Pedestrians(world, client, CW_CENTER, SEED)
    try:  # ползунки debug_view.py
        ctl = control.Server()
        print(f"Управление для debug_view.py: udp {control.CONTROL_HOST}:{control.CONTROL_PORT}")
    except OSError as e:
        print(f"Управление недоступно ({e}) — сцена работает без ползунков")

    world.get_spectator().set_transform(carla.Transform(
        CW_CENTER + carla.Location(z=60), carla.Rotation(pitch=-89)))

    # ---- источники: точки спавна в кольце, смотрящие к перекрёстку ----
    sources = []
    for sp in world.get_map().get_spawn_points():
        if not (SOURCE_RING[0] < d2(sp.location, CW_CENTER) < SOURCE_RING[1]):
            continue
        f, to_c = sp.get_forward_vector(), CW_CENTER - sp.location
        if f.x * to_c.x + f.y * to_c.y > 0:
            sources.append(sp)
    print(f"Источников: {len(sources)}")

    bps = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
           if bp.has_attribute("number_of_wheels")
           and bp.get_attribute("number_of_wheels").as_int() == 4]

    def try_spawn():
        alive = world.get_actors(vehicles)
        random.shuffle(sources)
        for sp in sources:
            # точка свободна? иначе машина заспавнится в другую
            if any(d2(v.get_location(), sp.location) < 10 for v in alive):
                continue
            v = world.try_spawn_actor(random.choice(bps), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                tm.update_vehicle_lights(v, True)   # фары по погоде и времени суток — ночью в кадре не тёмные машины
                vehicles.append(v.id)
                return True
        return False  # все источники заняты — очередь доползла до спавна

    veh_per_hour, max_vehicles = VEH_PER_HOUR, MAX_VEHICLES   # меняются ползунками debug_view.py

    def next_car(now):
        """Время следующей машины: пуассоновский поток; 0 машин/ч — никогда."""
        return now + random.expovariate(veh_per_hour / 3600.0) if veh_per_hour > 0 else math.inf

    next_spawn = next_car(0.0)

    # ---- фазы ----
    def apply_phase(name):
        our.set_state(G if name == "SIDE" else R)
        for tl in main_road:
            tl.set_state(G if name == "MAIN" else R)
        print(f"\n=== фаза {name} ===")

    our_stops = [w.transform.location for w in our.get_stop_waypoints()]
    phase_i, t_phase, t, tick = 0, 0.0, 0.0, 0
    spawned = removed = blocked = 0
    apply_phase(PHASES[0][0])

    while True:
        world.tick()
        tick += 1
        t += DT
        t_phase += DT

        if t_phase >= PHASES[phase_i][1]:
            phase_i = (phase_i + 1) % len(PHASES)
            t_phase = 0.0
            apply_phase(PHASES[phase_i][0])

        peds.update(t, PHASES[phase_i][0], PHASES[phase_i][1] - t_phase)

        # источник
        if t >= next_spawn:
            if len(vehicles) < max_vehicles and try_spawn():
                spawned += 1
            else:
                blocked += 1
            next_spawn = next_car(t)

        # раз в секунду: сток + статистика
        if tick % int(1 / DT) == 0:
            alive = {v.id: v for v in world.get_actors(vehicles)}
            far = [vid for vid, v in alive.items()
                   if d2(v.get_location(), CW_CENTER) > SINK_RADIUS]
            if far:
                client.apply_batch([carla.command.DestroyActor(x) for x in far])
                removed += len(far)
            vehicles[:] = [vid for vid in vehicles if vid in alive and vid not in far]

            waiting = 0
            for v in alive.values():
                if v.id in far:
                    continue
                vel = v.get_velocity()
                if math.hypot(vel.x, vel.y) < 0.5 and any(
                        d2(v.get_location(), s) < 25 for s in our_stops):
                    waiting += 1
            print(f"[{PHASES[phase_i][0]:4} {t_phase:4.0f}s] наш={our.state}  "
                  f"в зоне={len(vehicles):3}  ждут у нас={waiting:2}  "
                  f"+{spawned} -{removed} (не влезло {blocked})")
            # спрос пешеходов — данные для контроллера (пока цикл фиксированный)
            demand = peds.demand(t)
            print(peds.stats_line(demand))

            # ползунки debug_view.py: потоки и лимиты меняются на лету. Потоки пуассоновские
            # (без памяти), поэтому пересчитать время следующего появления можно в любой момент
            for msg, addr in (ctl.poll() if ctl else []):
                if msg.get("cmd") == "set":
                    try:
                        num = lambda key, cur: max(0, int(msg.get(key, cur)))
                        veh_per_hour = num("veh_per_hour", veh_per_hour)
                        max_vehicles = num("max_vehicles", max_vehicles)
                        peds.set_flow(t, num("ped_per_hour", peds.per_hour),
                                      num("max_per_side", peds.max_per_side))
                        next_spawn = next_car(t)
                        print(f"   > управление: машин/ч {veh_per_hour}, макс {max_vehicles}; "
                              f"пешеходов/ч {peds.per_hour}, на сторону {peds.max_per_side}")
                    except (TypeError, ValueError):
                        print(f"   > управление: непонятная команда {msg}")
                ctl.reply(addr, {"veh_per_hour": veh_per_hour, "max_vehicles": max_vehicles,
                                 "ped_per_hour": peds.per_hour, "max_per_side": peds.max_per_side})

except KeyboardInterrupt:
    print("\nОстановка...")
finally:
    if ctl:
        ctl.close()
    if peds:
        try:
            peds.destroy()
        except RuntimeError as e:  # уборка мира ниже должна пройти в любом случае
            print(f"Уборка пешеходов упала: {e}")
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
    world.freeze_all_traffic_lights(False)
    tm.set_synchronous_mode(False)
    s = world.get_settings()
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    world.apply_settings(s)
    print("Убрано, мир отпущен.")