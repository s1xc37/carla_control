"""
controller.py — контроллер фаз светофора нашего перехода. Про CARLA ничего не знает:
на входе числа «с камеры» (Traffic), на выходе — какая фаза должна гореть (Signal).

Перенесён из backend сокомандника (backend/app/simulation/signal.py и adaptive.py). Режимы те же:
    cycle    — фиксированный цикл: MAIN → SIDE → жёлтый → очистка → PED → очистка → MAIN ...
    button   — машинам зелёный, пока пешеход не вызвал (стоит у бордюра — «нажал кнопку»);
               вызов обслуживается после минимального зелёного
    adaptive — когда пускать пешеходов, решает взвешенное сравнение спроса машин и пешеходов;
               пешеход ждёт не дольше 60/90 с (ГОСТ Р 70716-2023), спецтехника держит зелёный
    manual   — фазу выбирает оператор (debug_view.py), сразу, без жёлтого и очистки

Автоматические режимы — один автомат, как в backend:
    car_green → car_yellow → clear_to_ped → ped_green → clear_to_car → car_green ...
и отличаются только тем, когда уходить из car_green. Поэтому смена режима автомат не сбрасывает
(в backend — с чистого листа) и не рвёт идущую пешеходную фазу.

Отличия от backend (там прямая дорога и один сигнал машинам, у нас T-перекрёсток):
- «зелёный машинам» — две фазы перекрёстка по очереди: MAIN (главная) и SIDE (наша дорога);
  пешеходам — всем машинам красный. После пешеходов едет дорога, что стояла перед ними;
  только что включившуюся дорогу не обрываем раньше STAGE_MIN_S;
- пешеходная фаза во всех режимах PHASE_PED: наши пешеходы начинают переход, только если
  успевают (pedestrians.py), им нужен запас на старт сверх ширины перехода / скорости;
- «ждём, пока переход освободится» перед зелёным машинам — во всех режимах, не только в adaptive;
- трамваев и автобусов с остановками и пассажирами в CARLA нет — синхронизации с ОТ нет,
  автобус весит как ОТ без данных (2). Горизонт спроса из backend (~30 с × 8 м/с) длиннее
  коридора камеры (50 м), поэтому в спросе все машины коридора, ещё не доехавшие до зебры.
"""
import math
from dataclasses import dataclass, field

from envconf import env

# ---- конфиг ----
SIGNAL_MODE = env("SIGNAL_MODE", "cycle")    # режим при запуске: cycle / button / adaptive / manual
STAGE_S = {"MAIN": env("PHASE_MAIN", 20.0),  # зелёный главной дороге, с
           "SIDE": env("PHASE_SIDE", 15.0)}  # ... нашей
PED_S = env("PHASE_PED", 20.0)               # пешеходная фаза, с; не меньше времени перехода (~14–18 с)
YELLOW_S = env("PHASE_YELLOW", 3.0)          # жёлтый машинам перед пешеходами
ALL_RED_AFTER_CARS_S = env("ALL_RED_AFTER_CARS", 3.0)  # всем красный: машины съезжают с зебры
ALL_RED_AFTER_PEDS_S = env("ALL_RED_AFTER_PEDS", 3.0)  # ... пешеходы доходят
STAGE_MIN_S = 7.0             # только что включившуюся дорогу (MAIN/SIDE) не обрываем раньше, с
BUTTON_MIN_GREEN_S = 15.0     # button: минимальный зелёный машинам перед вызовом (backend)
# adaptive — значения из backend/app/simulation/adaptive.py
MIN_GREEN_CAR_S = 15.0        # минимальный / максимальный зелёный машинам (ГОСТ Р 71096)
MAX_GREEN_CAR_S = 90.0
MIN_GREEN_PED_S = 7.0         # минимальная пешеходная фаза (MUTCD)
PED_WAIT_CRITICAL_S = (60.0, 90.0)  # ГОСТ Р 70716-2023: пешеход ждёт не дольше, при ≤ / > ...
PED_WAIT_INTENSITY_VPH = 700.0      # ... стольких машин в час на полосу
PED_WAIT_WARNING = 0.6        # с этой доли критического ожидания вес пешеходов ×3
PED_WEIGHT = 1.0              # один ждущий пешеход весит как машина
VEHICLE_WEIGHT = {"bus": 2.0,       # ОТ без данных о пассажирах и опоздании
                  "special": 15.0}  # спецтехника с запросом приоритета; остальные — 1

MODES = ("cycle", "button", "adaptive", "manual")
MANUAL_PHASES = ("MAIN", "SIDE", "PED", "ALL_RED")
OTHER = {"MAIN": "SIDE", "SIDE": "MAIN"}
PHASE_OF = {"car_yellow": "YELLOW", "clear_to_ped": "ALL_RED", "ped_green": "PED",
            "clear_to_car": "ALL_RED"}


@dataclass
class Traffic:
    """Что видит камера у перехода — вход контроллера. Сейчас собирается из мира CARLA
    (run_scene.py), потом те же числа даст YOLO."""
    vehicles: list = field(default_factory=list)  # типы машин нашей дороги, ещё не доехавших до зебры
    peds_waiting: dict = field(default_factory=lambda: {"W": 0, "E": 0})      # ждут у бордюра
    peds_max_wait: dict = field(default_factory=lambda: {"W": 0.0, "E": 0.0})  # дольше всех, с
    peds_crossing: int = 0        # идут по проезжей части
    vph_per_lane: float = 0.0     # измеренная интенсивность нашей дороги, машин в час на полосу


@dataclass
class Signal:
    """Что должно гореть на этом тике."""
    phase: str        # MAIN / SIDE (едет эта дорога) / YELLOW / ALL_RED / PED (всем машинам красный)
    stage: str        # MAIN или SIDE: чей жёлтый при YELLOW, иначе — кто ехал последним
    ped_left: float   # сколько осталось пешеходной фазы, с: 0 — не PED, inf — ручной PED
    reason: str       # почему так — для консоли и debug_view.py


class Controller:
    def __init__(self, mode=SIGNAL_MODE):
        self.ped_s = max(MIN_GREEN_PED_S, PED_S)
        self.state, self.elapsed = "car_green", 0.0  # автомат автоматических режимов
        self.stage, self.stage_t = "MAIN", 0.0       # какая дорога едет в car_green и сколько уже
        self.call = False                            # button: вызов ждёт обслуживания
        self.manual = "MAIN"
        self.reason = ""
        self.last_change = ""   # почему последний раз ушли из car_green или вернулись в него
        self.car_w = self.ped_w = 0.0                # adaptive: веса, которые сравниваем
        self.mode = None
        self.set_mode(mode)

    def phase(self):
        if self.mode == "manual":
            return self.manual
        return self.stage if self.state == "car_green" else PHASE_OF[self.state]

    # ---- управление ----
    def set_mode(self, mode):
        """Смена режима на лету. Между автоматическими режимами автомат общий — ничего не рвётся."""
        if mode not in MODES:
            raise ValueError(f"режим {mode!r}: есть {', '.join(MODES)}")
        if mode == self.mode:
            return
        if mode == "manual":          # оператор начинает с того, что горит; с жёлтого — всем красный
            self.manual = self.phase() if self.phase() in MANUAL_PHASES else "ALL_RED"
        elif self.mode == "manual":   # из ручного: едет дорога — продолжаем, иначе очистка перехода
            if self.manual in OTHER:
                self.state, self.stage = "car_green", self.manual
            else:
                self.state = "clear_to_car"
            self.elapsed = self.stage_t = 0.0
        self.mode, self.call = mode, False

    def set_manual(self, phase):
        """Ручная фаза; заодно переводит в ручной режим."""
        if phase not in MANUAL_PHASES:
            raise ValueError(f"ручная фаза {phase!r}: есть {', '.join(MANUAL_PHASES)}")
        self.set_mode("manual")
        self.manual = phase
        if phase in OTHER:
            self.stage = phase

    # ---- тик ----
    def step(self, dt, traffic):
        """Один тик: продвинуть автомат на dt и вернуть, что должно гореть."""
        if self.mode == "manual":
            self.reason = f"Ручной режим: {self.manual}"
            return Signal(self.manual, self.stage, math.inf if self.manual == "PED" else 0.0, self.reason)
        if self.mode == "button" and any(traffic.peds_waiting.values()):
            self.call = True             # пешеход ждёт у бордюра — автонажатие кнопки, как в backend
        self.elapsed += dt
        if self.state == "car_green":
            self.stage_t += dt
        for _ in range(5):               # нулевые жёлтый и очистку проходим за один тик
            if not self._next(traffic):
                break
        if self.state == "car_green" and self.stage_t >= STAGE_S[self.stage]:
            self.stage, self.stage_t = OTHER[self.stage], 0.0   # MAIN ⇄ SIDE
        ped_left = self.ped_s - self.elapsed if self.state == "ped_green" else 0.0
        return Signal(self.phase(), self.stage, ped_left, self.reason)

    def _next(self, tr):
        """Переход автомата, если пора; True — состояние сменилось."""
        s = self.state
        if s == "car_green":
            why = self._leave_cars(tr)
            if why is None:
                return False
            self.reason = self.last_change = why
            self.call = False
            return self._go("car_yellow")
        if s == "car_yellow":
            self.reason = f"Жёлтый {self.stage} — переход к пешеходной фазе"
            return self.elapsed >= YELLOW_S and self._go("clear_to_ped")
        if s == "clear_to_ped":
            self.reason = "Очистка перехода перед пешеходами (всем красный)"
            return self.elapsed >= ALL_RED_AFTER_CARS_S and self._go("ped_green")
        if s == "ped_green":
            self.reason = f"Пешеходная фаза {self.ped_s:.0f} с (не прерывается)"
            return self.elapsed >= self.ped_s and self._go("clear_to_car")
        # clear_to_car: зелёный машинам — только когда переход свободен (безопасность)
        if self.elapsed < ALL_RED_AFTER_PEDS_S:
            self.reason = "Очистка перехода перед машинами (всем красный)"
            return False
        if tr.peds_crossing > 0:
            self.reason = f"Ждём, пока переход освободится: на нём ещё {tr.peds_crossing} чел."
            return False
        self.stage = OTHER[self.stage]   # едет дорога, которая стояла перед пешеходами
        self.reason = self.last_change = f"Переход свободен — зелёный {self.stage}"
        return self._go("car_green")

    def _go(self, state):
        self.state, self.elapsed = state, 0.0
        if state == "car_green":
            self.stage_t = 0.0
        return True

    # ---- когда уходить из car_green: причина или None (тогда в self.reason — почему нет) ----
    def _leave_cars(self, tr):
        if self.mode == "cycle":
            if self.stage == "SIDE" and self.stage_t >= STAGE_S["SIDE"]:
                return "Цикл: после SIDE — пешеходная фаза"
            self.reason = f"Цикл: {self.stage} {self.stage_t:.0f}/{STAGE_S[self.stage]:.0f} с"
            return None
        why = self._button() if self.mode == "button" else self._adaptive(tr)
        if why is not None and self.stage_t < STAGE_MIN_S:
            self.reason = (f"{why}; ждём: {self.stage} только что получила зелёный "
                           f"({self.stage_t:.0f}/{STAGE_MIN_S:.0f} с)")
            return None
        return why

    def _button(self):
        if not self.call:
            self.reason = "Кнопка: зелёный машинам, пока пешеход не вызвал"
            return None
        if self.elapsed < BUTTON_MIN_GREEN_S:
            self.reason = (f"Вызов принят, идёт минимальный зелёный машинам "
                           f"({self.elapsed:.0f}/{BUTTON_MIN_GREEN_S:.0f} с)")
            return None
        return "Кнопка вызова: пешеходы ждут"

    def _adaptive(self, tr):
        """Перенос adaptive.decide(), ветка car_green: тот же порядок проверок."""
        low, high = PED_WAIT_CRITICAL_S
        critical = high if tr.vph_per_lane > PED_WAIT_INTENSITY_VPH else low
        n = sum(tr.peds_waiting.values())
        wait = max(tr.peds_max_wait.values())
        self.ped_w = n * PED_WEIGHT * (3.0 if wait >= critical * PED_WAIT_WARNING else 1.0)
        self.car_w = sum(VEHICLE_WEIGHT.get(v, 1.0) for v in tr.vehicles)
        if self.elapsed < MIN_GREEN_CAR_S:
            self.reason = (f"Минимальный зелёный машинам ещё не истёк "
                           f"({self.elapsed:.0f}/{MIN_GREEN_CAR_S:.0f} с)")
            return None
        if "special" in tr.vehicles:
            self.reason = "В очереди спецтранспорт с приоритетом — зелёный машинам удерживается"
            return None
        if n == 0:
            self.reason = "Нет ожидающих пешеходов — зелёный машинам продолжается"
            return None
        # дорога сейчас сменится, а новую раньше STAGE_MIN_S не оборвём — не дотягиваем до критического
        ending = self.stage_t >= STAGE_S[self.stage]
        if wait >= critical or (ending and wait + STAGE_MIN_S > critical):
            return (f"Критическое ожидание пешеходов: {wait:.0f} с из {critical:.0f} с "
                    f"(ГОСТ Р 70716-2023) — переключение обязательно")
        if self.car_w == 0:
            return "В очереди нет машин, видимых камерой — уступаем пешеходам"
        if self.ped_w > self.car_w:
            return f"Спрос пешеходов важнее: вес {self.ped_w:.1f} > вес машин {self.car_w:.1f}"
        if self.elapsed >= MAX_GREEN_CAR_S:
            return f"Достигнут максимальный зелёный машинам ({MAX_GREEN_CAR_S:.0f} с)"
        self.reason = f"Машины важнее: вес {self.car_w:.1f} >= вес пешеходов {self.ped_w:.1f}"
        return None
