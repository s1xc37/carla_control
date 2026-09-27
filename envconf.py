"""
envconf.py — параметры из переменных окружения (для Docker и не только).

env("VEH_PER_HOUR", 1200) вернёт значение переменной VEH_PER_HOUR, приведённое к типу значения
по умолчанию, или само значение по умолчанию, если переменная не задана или пустая.
"""
import os

TRUE = ("1", "true", "yes", "on", "да")


def env(name, default):
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    if isinstance(default, bool):
        return v.lower() in TRUE
    try:
        return type(default)(v)
    except ValueError:
        raise SystemExit(f"Переменная {name}={v!r}: ожидал {type(default).__name__}")


CARLA_HOST = env("CARLA_HOST", "localhost")   # где сервер CARLA
CARLA_PORT = env("CARLA_PORT", 2000)
CARLA_MAP = env("CARLA_MAP", "Town05")        # run_scene.py загрузит, если открыта другая карта
