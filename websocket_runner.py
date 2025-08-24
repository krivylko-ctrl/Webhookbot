#!/usr/bin/env python3
"""
Bybit v5 WebSocket Runner для KWIN Trading Bot
- Подписка на kline.{tf}.{symbol} и publicTrade.{symbol}
- Обработка ТОЛЬКО confirm=True (закрытые свечи) для честного бэктеста
- Автоподдержка соединения (heartbeat + авто-реконнект)
- Нормализация числовых полей к float
- (Опционально) сохранение закрытых свечей в CSV для офлайн-бэктеста
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import websockets

# -------------------- ЛОГИРОВАНИЕ --------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("WS_LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("WS_LOG_FILE", "websocket.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("BybitWS")

# -------------------- НАСТРОЙКИ --------------------
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() not in ("0", "false", "no")
# Список символов/ТФ из ENV, по умолчанию ETH/BTC на 15м + трейды
SYMBOLS = [s.strip().upper() for s in os.getenv("WS_SYMBOLS", "ETHUSDT,BTCUSDT").split(",") if s.strip()]
KL_TFS  = [tf.strip() for tf in os.getenv("WS_TFS", "15").split(",") if tf.strip()]  # в минутах, напр. "1,15"
SAVE_CSV = os.getenv("WS_SAVE_CSV", "true").lower() not in ("0", "false", "no")
DATA_DIR = os.getenv("WS_DATA_DIR", "data")

URL = "wss://stream-testnet.bybit.com/v5/public/linear" if BYBIT_TESTNET \
      else "wss://stream.bybit.com/v5/public/linear"

PING_INTERVAL_S = 20
READ_TIMEOUT_S  = 60
RECONNECT_MAX_DELAY_S = 60


@dataclass
class KlineBar:
    symbol: str
    interval: str  # строка минут ("1","3","5","15","60",...)
    start_ms: int
    end_ms: Optional[int]
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirm: bool


def _to_float(x, default=0.0) -> float:
    try:
        # bybit шлёт числа строками
        return float(x)
    except Exception:
        return float(default)


def _topic_parts(topic: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    kline.15.ETHUSDT -> ("kline", "15", "ETHUSDT")
    publicTrade.ETHUSDT -> ("publicTrade", None, "ETHUSDT")
    """
    parts = (topic or "").split(".")
    if not parts:
        return "", None, None
    if parts[0] == "kline" and len(parts) >= 3:
        return "kline", parts[1], parts[2]
    if parts[0] == "publicTrade" and len(parts) >= 2:
        return "publicTrade", None, parts[1]
    return parts[0], None, parts[-1] if parts else None


def _dt_utc_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _csv_path(symbol: str, interval: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{symbol}_{interval}m.csv")


async def _append_bar_to_csv(bar: KlineBar):
    """Пишем только закрытые бары (confirm=True) — формат: ts,open,high,low,close,volume"""
    if not (SAVE_CSV and bar.confirm):
        return
    path = _csv_path(bar.symbol, bar.interval)
    line = f"{bar.start_ms},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}\n"
    loop = asyncio.get_running_loop()
    # неблокирующая запись
    await loop.run_in_executor(None, _append_line_sync, path, line)


def _append_line_sync(path: str, line: str):
    first = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if first:
            f.write("timestamp,open,high,low,close,volume\n")
        f.write(line)


class BybitWebSocket:
    def __init__(self, url: str, symbols: List[str], tfs: List[str]):
        self.url = url
        self.symbols = symbols
        self.tfs = tfs
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscribed_args: List[str] = []
        self._ping_task: Optional[asyncio.Task] = None

    # --------- Public API ---------
    async def run_forever(self):
        """Основной цикл: подключение → подписка → чтение → реконнект при обрыве."""
        self._running = True
        backoff = 1
        while self._running:
            try:
                log.info(f"Подключение к {self.url}...")
                async with websockets.connect(self.url, ping_interval=None, close_timeout=5) as ws:
                    self.ws = ws
                    log.info("WS подключён")
                    await self._subscribe_all()
                    self._start_heartbeat()
                    await self._listen_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"WS ошибка/обрыв: {e!r}")

            # реконнект
            self._stop_heartbeat()
            self.ws = None
            if not self._running:
                break
            delay = min(backoff, RECONNECT_MAX_DELAY_S)
            log.info(f"Реконнект через {delay}с...")
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, RECONNECT_MAX_DELAY_S)

    async def close(self):
        self._running = False
        self._stop_heartbeat()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        log.info("WS закрыт")

    # --------- Internal ---------
    async def _subscribe_all(self):
        args = []
        for s in self.symbols:
            for tf in self.tfs:
                args.append(f"kline.{tf}.{s}")
            # трейды необязательны — полезны для отладки
            args.append(f"publicTrade.{s}")
        self._subscribed_args = args
        await self._send_json({"op": "subscribe", "args": args})
        log.info(f"Подписка: {args}")

    def _start_heartbeat(self):
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._heartbeat_loop())

    def _stop_heartbeat(self):
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        self._ping_task = None

    async def _heartbeat_loop(self):
        while self._running and self.ws:
            try:
                await self._send_json({"op": "ping"})
            except Exception:
                # пусть упадёт в listen и реконнектнется
                return
            await asyncio.sleep(PING_INTERVAL_S)

    async def _send_json(self, obj: Dict):
        if not self.ws:
            return
        await self.ws.send(json.dumps(obj))

    async def _listen_loop(self):
        """Чтение сообщений + разбор; реконнект при таймауте."""
        assert self.ws
        while self._running:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=READ_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.warning("Таймаут чтения — пингуем заново")
                # дёрнем пинг сразу, пусть heartbeat продолжит
                try:
                    await self._send_json({"op": "ping"})
                except Exception:
                    raise
                continue
            except websockets.exceptions.ConnectionClosed:
                log.warning("Соединение закрыто")
                break

            try:
                data = json.loads(msg)
            except Exception:
                log.debug(f"Не-JSON: {msg[:200]}")
                continue

            await self._dispatch(data)

    async def _dispatch(self, data: Dict):
        # ответы на ping/pong/subscribe
        if data.get("op") in ("pong",) or "success" in data:
            log.debug(f"Svc msg: {data}")
            return

        topic = data.get("topic", "")
        if not topic:
            log.debug(f"Без topic: {data}")
            return

        kind, tf, sym = _topic_parts(topic)

        if kind == "kline":
            await self._on_kline(tf, sym, data)
        elif kind == "publicTrade":
            await self._on_trade(sym, data)
        else:
            log.debug(f"Неизвестный topic: {topic}")

    async def _on_kline(self, tf: Optional[str], symbol: Optional[str], data: Dict):
        try:
            arr = data.get("data") or []
            if not arr:
                return
            k = arr[0]  # bybit шлёт по одному объекту

            # confirm=True → бар ЗАКРЫТ
            confirm = bool(k.get("confirm", False))
            # поля по спецификации v5 (строки → float)
            bar = KlineBar(
                symbol=symbol or str(k.get("symbol") or ""),
                interval=str(tf or k.get("interval") or "15"),
                start_ms=int(k.get("start", 0)),
                end_ms=int(k["end"]) if k.get("end") not in (None, "") else None,
                open=_to_float(k.get("open")),
                high=_to_float(k.get("high")),
                low=_to_float(k.get("low")),
                close=_to_float(k.get("close")),
                volume=_to_float(k.get("volume")),
                turnover=_to_float(k.get("turnover")),
                confirm=confirm,
            )

            # Только закрытые бары пишем/логируем — это важно для бэктеста
            if bar.confirm:
                ts = _dt_utc_from_ms(bar.start_ms).isoformat()
                log.info(
                    f"[{bar.symbol} {bar.interval}m] close @ {ts}  O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}"
                )
                await _append_bar_to_csv(bar)
            else:
                log.debug(f"[{bar.symbol} {bar.interval}m] промежуточный бар (confirm=False)")

        except Exception as e:
            log.error(f"Ошибка kline: {e}")
            log.debug(f"payload: {data}")

    async def _on_trade(self, symbol: Optional[str], data: Dict):
        try:
            for t in (data.get("data") or []):
                side = t.get("S")  # "Buy"/"Sell"
                price = _to_float(t.get("p"))
                size = _to_float(t.get("v"))
                ts = int(t.get("T") or 0)
                log.debug(f"Trade {symbol or t.get('s')}: {side} {size} @ {price} ({_dt_utc_from_ms(ts)})")
        except Exception as e:
            log.error(f"Ошибка trade: {e}")


# -------------------- MAIN --------------------
async def main():
    # Список подписок
    subs = []
    for s in SYMBOLS:
        for tf in KL_TFS:
            subs.append(f"kline.{tf}.{s}")
        subs.append(f"publicTrade.{s}")
    log.info(f"Запуск WS: {URL}")
    log.info(f"Подписки: {subs}")
    if SAVE_CSV:
        os.makedirs(DATA_DIR, exist_ok=True)
        log.info(f"Сохранение закрытых свечей в CSV: {DATA_DIR}")

    ws = BybitWebSocket(URL, SYMBOLS, KL_TFS)
    try:
        await ws.run_forever()
    except KeyboardInterrupt:
        log.info("Ctrl+C")
    finally:
        await ws.close()


if __name__ == "__main__":
    # Пробуем проверить доступность REST (косвенный индикатор гео-блокировок)
    try:
        import requests
        r = requests.get("https://api.bybit.com/v5/market/time", timeout=5)
        if r.status_code == 200:
            log.info("✅ Bybit REST доступен")
        elif r.status_code == 403:
            log.warning("⚠️ Возможны гео-ограничения Bybit. WS может не работать.")
        else:
            log.warning(f"⚠️ REST ответ: {r.status_code}")
    except Exception:
        log.warning("⚠️ Нет доступа к REST; продолжаем с WS")

    asyncio.run(main())
