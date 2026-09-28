import math
import random
import signal
import carla
import control
import controller
from envconf import CARLA_HOST, CARLA_MAP, CARLA_PORT, env
from pedestrians import Pedestrians
from ws_sender import Observer

# ---- конфиг ----
CW_CENTER = carla.Location(x=32.3, y=-178.5, z=0.0)  # центр зебры
SOURCE_RING = (40.0, 150.0)   # где появляются машины
SINK_RADIUS = 170.0           # дальше — удаляем
VEH_PER_HOUR = env("VEH_PER_HOUR", 1200)   # интенсивность суммарно по всем подходам
MAX_VEHICLES = env("MAX_VEHICLES", 60)     # потолок, чтобы не задушить сервер
SEED = env("SEED", 42)
TM_PORT = env("TM_PORT", 8000)            # порт Traffic Manager
DT = 0.05
# фазы светофора (длительности, режим) — в controller.py
TRAFFIC_PERIOD = 0.5          # контроллер получает данные «с камеры» так часто, как уходит JSON, с
INTENSITY_WINDOW = 300.0      # интенсивность нашей дороги — по машинам за столько с

G, Y, R = carla.TrafficLightState.Green, carla.TrafficLightState.Yellow, carla.TrafficLightState.Red
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
      f"машин/ч {VEH_PER_HOUR}, SEED {SEED}")
print(f"Светофор: режим {controller.SIGNAL_MODE}; зелёный MAIN {controller.STAGE_S['MAIN']:g}с, "
      f"SIDE {controller.STAGE_S['SIDE']:g}с, PED {controller.PED_S:g}с, жёлтый {controller.YELLOW_S:g}с, "
      f"очистка {controller.ALL_RED_AFTER_CARS_S:g}/{controller.ALL_RED_AFTER_PEDS_S:g}с")
try:
    signal_ctl = controller.Controller()   # до того, как трогать мир: кривой SIGNAL_MODE — сразу выход
except ValueError as e:
    raise SystemExit(f"SIGNAL_MODE: {e}")
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
                vehicles.append(v.id)
                return True
        return False  # все источники заняты — очередь доползла до спавна

    veh_per_hour, max_vehicles = VEH_PER_HOUR, MAX_VEHICLES   # меняются ползунками debug_view.py

    def next_car(now):
        """Время следующей машины: пуассоновский поток; 0 машин/ч — никогда."""
        return now + random.expovariate(veh_per_hour / 3600.0) if veh_per_hour > 0 else math.inf

    next_spawn = next_car(0.0)

    # ---- фазы: решает controller.py, здесь только данные «с камеры» и цвета светофоров ----
    obs = Observer(world)     # тот же коридор нашей дороги, что уходит в JSON
    seen = {}                 # id машины в коридоре -> когда увидели впервые (для интенсивности)

    def observe():
        ids, queue = obs.queue()
        for vid in ids:
            seen.setdefault(vid, t)
        for vid in [v for v, s in seen.items() if t - s > INTENSITY_WINDOW]:
            del seen[vid]
        vph = len(seen) * 3600.0 / max(60.0, min(t, INTENSITY_WINDOW))
        d = peds.demand(t)
        return controller.Traffic(vehicles=queue, peds_waiting=d.waiting, peds_max_wait=d.max_wait,
                                  peds_crossing=d.crossing, vph_per_lane=vph / max(1, len(obs.lanes)))

    def phase_name(sig):
        return f"{sig.phase} {sig.stage}" if sig.phase == "YELLOW" else sig.phase

    def apply_phase(sig):
        """MAIN/SIDE — едет эта дорога, YELLOW — ей жёлтый, ALL_RED и PED — всем машинам красный."""
        road = sig.stage if sig.phase == "YELLOW" else sig.phase
        color = Y if sig.phase == "YELLOW" else G
        our.set_state(color if road == "SIDE" else R)
        for tl in main_road:
            tl.set_state(color if road == "MAIN" else R)
        why = f"  ({signal_ctl.last_change})" if sig.phase == "YELLOW" else ""
        print(f"\n=== фаза {phase_name(sig)} ==={why}")

    our_stops = [w.transform.location for w in our.get_stop_waypoints()]
    t_phase, t, tick = 0.0, 0.0, 0
    spawned = removed = blocked = 0
    traffic = controller.Traffic()
    sig = signal_ctl.step(0.0, traffic)
    name = phase_name(sig)
    apply_phase(sig)

    while True:
        world.tick()
        tick += 1
        t += DT
        t_phase += DT

        if tick % round(TRAFFIC_PERIOD / DT) == 0:
            traffic = observe()
        sig = signal_ctl.step(DT, traffic)
        if phase_name(sig) != name:
            name, t_phase = phase_name(sig), 0.0
            apply_phase(sig)

        peds.update(t, sig.phase, sig.ped_left)

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
            print(f"[{name:11} {t_phase:4.0f}s] наш={our.state}  "
                  f"в зоне={len(vehicles):3}  ждут у нас={waiting:2}  "
                  f"+{spawned} -{removed} (не влезло {blocked})")
            demand = peds.demand(t)
            print(peds.stats_line(demand))
            print(f"   светофор [{signal_ctl.mode}]: {signal_ctl.reason}  | камера: машин до зебры "
                  f"{len(traffic.vehicles)}, {traffic.vph_per_lane:.0f} авт/ч на полосу")

            # debug_view.py: потоки, лимиты и режим светофора меняются на лету. Потоки пуассоновские
            # (без памяти), поэтому пересчитать время следующего появления можно в любой момент
            for msg, addr in (ctl.poll() if ctl else []):
                if msg.get("cmd") == "set":
                    try:
                        if any(k in msg for k in control.FLOW_KEYS):
                            num = lambda key, cur: max(0, int(msg.get(key, cur)))
                            veh_per_hour = num("veh_per_hour", veh_per_hour)
                            max_vehicles = num("max_vehicles", max_vehicles)
                            peds.set_flow(t, num("ped_per_hour", peds.per_hour),
                                          num("max_per_side", peds.max_per_side))
                            next_spawn = next_car(t)
                            print(f"   > управление: машин/ч {veh_per_hour}, макс {max_vehicles}; "
                                  f"пешеходов/ч {peds.per_hour}, на сторону {peds.max_per_side}")
                        if "signal_mode" in msg:
                            signal_ctl.set_mode(msg["signal_mode"])
                        if "manual_phase" in msg:
                            signal_ctl.set_manual(msg["manual_phase"])
                        if any(k in msg for k in control.SIGNAL_KEYS):
                            print(f"   > управление: режим светофора {signal_ctl.mode}"
                                  + (f", фаза {signal_ctl.manual}" if signal_ctl.mode == "manual" else ""))
                    except (TypeError, ValueError) as e:
                        print(f"   > управление: непонятная команда {msg} ({e})")
                ctl.reply(addr, {"veh_per_hour": veh_per_hour, "max_vehicles": max_vehicles,
                                 "ped_per_hour": peds.per_hour, "max_per_side": peds.max_per_side,
                                 "signal_mode": signal_ctl.mode, "manual_phase": signal_ctl.manual,
                                 "phase": name, "reason": signal_ctl.reason})

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