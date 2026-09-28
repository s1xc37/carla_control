"""
control.py — управление сценой на лету по UDP: debug_view.py крутит ползунки, run_scene.py слушает.

Сообщения — JSON: {"cmd": "get"}, {"cmd": "set", "veh_per_hour": 1500, ...},
{"cmd": "set", "signal_mode": "adaptive"}, {"cmd": "set", "manual_phase": "PED"}.
На все сцена отвечает текущими значениями FLOW_KEYS и SIGNAL_KEYS, плюс phase и reason —
что горит сейчас и почему (только чтение).
"""
import json
import socket

from envconf import env

CONTROL_HOST = env("CONTROL_HOST", "127.0.0.1")   # 127.0.0.1 — управлять можно только с этой машины
CONTROL_PORT = env("CONTROL_PORT", 8770)
FLOW_KEYS = ("veh_per_hour", "max_vehicles", "ped_per_hour", "max_per_side")   # потоки, целые числа
SIGNAL_KEYS = ("signal_mode", "manual_phase")   # режим светофора (controller.MODES) и ручная фаза


class Server:
    """Сторона сцены: неблокирующий приём команд."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((CONTROL_HOST, CONTROL_PORT))
        self.sock.setblocking(False)

    def poll(self):
        """Все пришедшие команды: [(dict, адрес)]. Мусор пропускаем."""
        out = []
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return out
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if isinstance(msg, dict):
                out.append((msg, addr))

    def reply(self, addr, state):
        self.sock.sendto(json.dumps(state).encode(), addr)

    def close(self):
        self.sock.close()


class Client:
    """Сторона окна: отправить команду, забрать последний ответ сцены."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.addr = ("127.0.0.1" if CONTROL_HOST == "0.0.0.0" else CONTROL_HOST, CONTROL_PORT)

    def send(self, **msg):
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError:
            pass                         # сцена не запущена — узнаем по отсутствию ответа

    def latest(self):
        """Последний ответ сцены или None, если новых нет."""
        state = None
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, ConnectionRefusedError):   # Refused — порт сцены закрыт
                return state
            try:
                state = json.loads(data)
            except ValueError:
                pass
