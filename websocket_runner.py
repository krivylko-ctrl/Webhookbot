#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit v5 WebSocket Runner для KWIN Trading Bot
— Подписка на kline.{tf}.{symbol} + publicTrade.{symbol} (для диагностики)
— В стратегию отправляются ТОЛЬКО закрытые бары (confirm=true) → паритет Pine/бэктест
— Автосид истории через REST (опционально) для 1m/15m/60m, строго по закрытым барам и в хронологическом порядке
— Автоподдержка соединения (heartbeat + авто-реконнект, экспоненциальный бэкофф)
— Нормализация типов, безопасные фолы
— (Опционально) сохранение закрытых свечей в CSV
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

try:
    import requests  # для REST-сида
except Exception:
    requests = None

# ---------- ЛОГИ ---------
logging.basicConfig(
    level=getattr(logging, os.getenv("WS_LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("WS_LOG_FILE", "websocket.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("BybitWS")

# ---------- ENV / НАСТРОЙКИ ----------
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() not in ("0", "false", "no")
WS_CATEGORY   = os.getenv("WS_CATEGORY", "linear").strip().lower()  # linear|inverse|spot|option
SYMBOLS       = [s.strip().upper() for s in os.getenv("WS_SYMBOLS", "ETHUSDT").split(",") if s.strip()]

# Таймфреймы для WS (подписка) и для сидинга истории (можно разделять)
KL_TFS        = [tf.strip() for tf in os.getenv("WS_TFS", "1,15").split(",") if tf.strip()]        # для лайва
SEED_TFS      = [tf.strip() for tf in os.getenv("WS_SEED_TFS", ",".join(KL_TFS)).split(",") if tf.strip()]  # для сида

# Исторический сид
WS_SEED_HISTORY        = os.getenv("WS_SEED_HISTORY", "true").lower() not in ("0", "false", "no")
WS_SEED_LIMIT_1M       = int(os.getenv("WS_SEED_LIMIT_1M", "1000"))
WS_SEED_LIMIT_15M      = int(os.getenv("WS_SEED_LIMIT_15M", "1500"))
WS_SEED_LIMIT_60M      = int(os.getenv("WS_SEED_LIMIT_60M", "1000"))

SAVE_CSV      = os.getenv("WS_SAVE_CSV", "true").lower() not in ("0", "false", "no")
DATA_DIR      = os.getenv("WS_DATA_DIR", "data")

BRIDGE_TO_KWIN = os.getenv("BRIDGE_TO_KWIN", "true").lower() not in ("0", "false", "no")
EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "false").lower() not in ("0", "false", "no")

if WS_CATEGORY not in ("linear", "inverse", "spot", "option"):
    WS_CATEGORY = "linear"

URL = (f"wss://stream-testnet.bybit.com/v5/public/{WS_CATEGORY}"
       if BYBIT_TESTNET else f"wss://stream.bybit.com/v5/public/{WS_CATEGORY}")

REST_BASE = ("https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com")

PING_INTERVAL_S = 20
READ_TIMEOUT_S  = 60
RECONNECT_MAX_DELAY_S = 60

# ---------- KWIN BRIDGE (опционально) ----------
KWIN_ENABLED = False
KWIN = None
KWIN_SYMBOL = None
KWIN_INTRABAR_TF = None  # "1", "3", "5"...

def _try_init_kwin():
    """Создаёт экземпляр KWINStrategy из локального проекта.
       Безопасно: если модулей нет — просто не включаем мост.
    """
    global KWIN_ENABLED, KWIN, KWIN_SYMBOL, KWIN_INTRABAR_TF
    if not BRIDGE_TO_KWIN:
        log.info("KWIN bridge disabled by env (BRIDGE_TO_KWIN=false)")
        return

    try:
        from types import SimpleNamespace
        try:
            from config import Config  # полноценный класс, если есть
            cfg = Config(
                symbol=os.getenv("SYMBOL", "ETHUSDT"),
                risk_pct=float(os.getenv("RISK_PCT", "3.0")),
                risk_reward=float(os.getenv("RISK_REWARD", "3.0")),
                use_intrabar=True,
                intrabar_tf=os.getenv("INTRABAR_TF", "1"),
                period_choice=os.getenv("PERIOD_CHOICE", "30"),
                days_back=int(os.getenv("DAYS_BACK", "30")),
                taker_fee_rate=float(os.getenv("TAKER_FEE_RATE", "0.00055")),
                price_for_logic=os.getenv("PRICE_FOR_LOGIC", "close"),
                limit_qty_enabled=True,
            )
        except Exception:
            # Фоллбек — простая конфигурация
            cfg = SimpleNamespace(
                symbol=os.getenv("SYMBOL", "ETHUSDT"),
                risk_pct=float(os.getenv("RISK_PCT", "3.0")),
                risk_reward=float(os.getenv("RISK_REWARD", "3.0")),
                use_intrabar=True,
                intrabar_tf=os.getenv("INTRABAR_TF", "1"),
                period_choice=os.getenv("PERIOD_CHOICE", "30"),
                days_back=int(os.getenv("DAYS_BACK", "30")),
                taker_fee_rate=float(os.getenv("TAKER_FEE_RATE", "0.00055")),
                price_for_logic=os.getenv("PRICE_FOR_LOGIC", "close"),
                limit_qty_enabled=True,
            )

        from state_manager import StateManager
        from database import Database
        from kwin_strategy import KWINStrategy

        api = None
        if EXECUTE_ORDERS:
            try:
                from bybit_api import BybitAPI
                api = BybitAPI(
                    api_key=os.getenv("BYBIT_API_KEY"),
                    api_secret=os.getenv("BYBIT_API_SECRET"),
                    account_type=os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED"),
                )
                log.info("KWIN bridge: real BybitAPI attached (EXECUTE_ORDERS=true)")
            except Exception as e:
                log.warning(f"Не удалось инициализировать BybitAPI: {e!r}. Переходим в локальный режим без исполнения.")
                api = None

        st = StateManager()
        db = Database(os.getenv("DB_DSN", ":memory:"))

        KWIN = KWINStrategy(cfg, api=api, state_manager=st, db=db)
        KWIN_SYMBOL = str(getattr(cfg, "symbol", "ETHUSDT")).upper()
        KWIN_INTRABAR_TF = str(getattr(cfg, "intrabar_tf", "1")).strip()
        KWIN_ENABLED = True
        log.info(f"KWIN bridge enabled → symbol={KWIN_SYMBOL}, intrabar_tf={KWIN_INTRABAR_TF}")
    except Exception as e:
        log.warning(f"KWIN bridge init failed: {e!r}. Работаем без стратегии.")
        KWIN_ENABLED = False

_try_init_kwin()

# ---------- Структуры ----------
@dataclass
class KlineBar:
    symbol: str
    interval: str  # "1","3","5","15","60",...
    start_ms: int
    end_ms: Optional[int]
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirm: bool

_csv_lock = asyncio.Lock()

def _to_float(x, default=0.0) -> float:
    try:
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
    if not (SAVE_CSV and bar.confirm):
        return
    path = _csv_path(bar.symbol, bar.interval)
    line = f"{bar.start_ms},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}\n"
    async with _csv_lock:
        first = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if first:
                f.write("timestamp,open,high,low,close,volume\n")
            f.write(line)

# ---------- REST утилиты для сида ----------
_INTERVAL_NORMALIZE = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
    "1h": "60",
    "2h": "120",
}

def _norm_tf(tf: str) -> str:
    t = tf.strip().lower()
    if t.endswith("m") or t.endswith("h"):
        return _INTERVAL_NORMALIZE.get(t, t.rstrip("mh"))
    return t  # "1","3","5","15","60",...

def _tf_limit(tf: str) -> int:
    t = _norm_tf(tf)
    if t in ("1", "3", "5"):
        return WS_SEED_LIMIT_1M
    if t in ("15", "30"):
        return WS_SEED_LIMIT_15M
    return WS_SEED_LIMIT_60M

def _rest_kline_url(category: str, symbol: str, interval: str, limit: int) -> str:
    return f"{REST_BASE}/v5/market/kline?category={category}&symbol={symbol}&interval={interval}&limit={limit}"

def _rest_fetch_klines(category: str, symbol: str, tf: str) -> List[Dict]:
    if requests is None:
        log.warning("requests не доступен — пропускаю сид истории")
        return []
    interval = _norm_tf(tf)
    limit = _tf_limit(interval)
    url = _rest_kline_url(category, symbol, interval, limit)
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log.warning(f"REST kline HTTP {r.status_code} {symbol} {interval}")
            return []
        j = r.json()
        rows = (j.get("result") or {}).get("list") or []
        out: List[Dict] = []
        # Bybit v5 возвращает list с newest first → переведём в oldest→newest
        for it in reversed(rows):
            # Форматы могут различаться: иногда list, иногда dict; нормализуем
            if isinstance(it, dict):
                start_ms = int(it.get("start") or it.get("t") or it.get("open_time") or 0)
                o = it.get("open") or it.get("o")
                h = it.get("high") or it.get("h")
                l = it.get("low") or it.get("l")
                c = it.get("close") or it.get("c")
                v = it.get("volume") or it.get("v")
                tv = it.get("turnover") or it.get("tv")
            else:
                # спецификация v5/list обычно: [start, open, high, low, close, volume, turnover]
                start_ms = int(it[0])
                o, h, l, c, v, tv = it[1], it[2], it[3], it[4], it[5], (it[6] if len(it) > 6 else 0)

            if start_ms < 1_000_000_000_000:
                start_ms *= 1000

            out.append({
                "timestamp": start_ms,
                "open":  _to_float(o),
                "high":  _to_float(h),
                "low":   _to_float(l),
                "close": _to_float(c),
                "volume": _to_float(v),
                "turnover": _to_float(tv),
                "confirm": True,   # сидаем только закрытые бары
                "interval": interval,
                "symbol": symbol,
            })
        return out
    except Exception as e:
        log.warning(f"REST fetch error {symbol} {interval}: {e!r}")
        return []

# ---------- Анти-дубликаты закрытых баров ----------
# Ключ: (symbol, interval) → последний start_ms закрытого бара, уже отфиденный в KWIN/CSV
_last_closed_start_ms: Dict[Tuple[str, str], int] = {}

def _is_new_closed_bar(bar: KlineBar) -> bool:
    key = (bar.symbol, bar.interval)
    prev = _last_closed_start_ms.get(key)
    if prev is None or int(bar.start_ms) > int(prev):
        _last_closed_start_ms[key] = int(bar.start_ms)
        return True
    return False

async def _seed_history(symbols: List[str], tfs: List[str]):
    """Исторический сид закрытых баров (строго confirm=true) в стратегию.
       Порядок: для каждого 15m окна предварительно прокидываем M1 окна (если есть такой tf),
       затем закрытие 15m, и т.д. Это даёт идентичное поведение Pine-реплею.
    """
    if not (WS_SEED_HISTORY and KWIN_ENABLED and KWIN is not None):
        return
    category = WS_CATEGORY
    for s in symbols:
        if KWIN_SYMBOL and s.upper() != KWIN_SYMBOL:
            log.info(f"Seed skip {s}: KWIN ожидает {KWIN_SYMBOL}")
            continue

        # Раздельно тянем TF
        tf_set = {_norm_tf(tf) for tf in tfs}
        have_1m = ("1" in tf_set) or (KWIN_INTRABAR_TF and _norm_tf(KWIN_INTRABAR_TF) == "1")
        have_15 = "15" in tf_set
        have_60 = "60" in tf_set

        bars_1m: List[Dict] = _rest_fetch_klines(category, s, "1") if have_1m else []
        bars_15: List[Dict] = _rest_fetch_klines(category, s, "15") if have_15 else []
        bars_60: List[Dict] = _rest_fetch_klines(category, s, "60") if have_60 else []

        # Индексы по времени для быстрых срезов
        def _bars_in_window(bars: List[Dict], start_ms: int, end_ms: int) -> List[Dict]:
            return [b for b in bars if start_ms <= int(b["timestamp"]) <= end_ms]

        # Вспомогательное: длительность TF
        def _tf_to_ms(interval: str) -> int:
            i = str(interval)
            if i == "1":
                return 60_000
            if i == "3":
                return 180_000
            if i == "5":
                return 300_000
            if i == "15":
                return 900_000
            if i == "30":
                return 1_800_000
            if i == "60":
                return 3_600_000
            return 60_000

        async def _feed_bar(b: Dict):
            bar = KlineBar(
                symbol=s.upper(),
                interval=b["interval"],
                start_ms=int(b["timestamp"]),
                end_ms=int(b["timestamp"]) + _tf_to_ms(b["interval"]) - 1,
                open=_to_float(b["open"]),
                high=_to_float(b["high"]),
                low=_to_float(b["low"]),
                close=_to_float(b["close"]),
                volume=_to_float(b["volume"]),
                turnover=_to_float(b.get("turnover", 0.0)),
                confirm=True,
            )
            if _is_new_closed_bar(bar):
                await _append_bar_to_csv(bar)
                await _feed_kwin(bar)

        # 1) H1 — просто по порядку
        for b60 in bars_60:
            await _feed_bar(b60)

        # 2) Для каждого 15m окна: сначала M1 внутри, затем 15m
        if bars_15:
            for b15 in bars_15:
                start_15 = int(b15["timestamp"])
                end_15 = start_15 + _tf_to_ms("15") - 1
                if bars_1m:
                    for m1 in _bars_in_window(bars_1m, start_15, end_15):
                        await _feed_bar(m1)
                await _feed_bar(b15)
        # 3) Если 15m нет, но есть 1m — просто прогрев
        elif bars_1m:
            for m1 in bars_1m:
                await _feed_bar(m1)

        log.info(f"Seeded history for {s}: 1m={len(bars_1m)} 15m={len(bars_15)} 60m={len(bars_60)}")

# ---------- WebSocket клиент ----------
class BybitWebSocket:
    def __init__(self, url: str, symbols: List[str], tfs: List[str]):
        self.url = url
        self.symbols = symbols
        self.tfs = tfs
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscribed_args: List[str] = []
        self._ping_task: Optional[asyncio.Task] = None

    async def run_forever(self):
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

    async def _subscribe_all(self):
        args = []
        for s in self.symbols:
            for tf in self.tfs:
                args.append(f"kline.{_norm_tf(tf)}.{s}")
            args.append(f"publicTrade.{s}")  # полезно для диагностики
        if not args:
            return
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
                return
            await asyncio.sleep(PING_INTERVAL_S)

    async def _send_json(self, obj: Dict):
        if not self.ws:
            return
        await self.ws.send(json.dumps(obj))

    async def _listen_loop(self):
        assert self.ws
        while self._running:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=READ_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.warning("Таймаут чтения — отправляем ping")
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
        # сервисные
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

            for k in arr:
                confirm = bool(k.get("confirm", False))
                if not confirm:
                    # Стратегию кормим только закрытыми барами
                    continue

                tf_eff  = str(_norm_tf(tf or k.get("interval") or "15"))
                sym_eff = (symbol or str(k.get("symbol") or "")).upper()

                start_ms = int(k.get("start") or k.get("t") or 0)
                if start_ms and start_ms < 1_000_000_000_000:
                    start_ms *= 1000
                end_ms = k.get("end")
                if end_ms not in (None, ""):
                    end_ms = int(end_ms)
                    if end_ms < 1_000_000_000_000:
                        end_ms *= 1000
                else:
                    end_ms = None

                bar = KlineBar(
                    symbol=sym_eff,
                    interval=tf_eff,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    open=_to_float(k.get("open") or k.get("o")),
                    high=_to_float(k.get("high") or k.get("h")),
                    low=_to_float(k.get("low") or k.get("l")),
                    close=_to_float(k.get("close") or k.get("c")),
                    volume=_to_float(k.get("volume") or k.get("v")),
                    turnover=_to_float(k.get("turnover") or k.get("tv")),
                    confirm=True,
                )

                # Только закрытые бары → Pine-паритет, плюс анти-дубликат
                if bar.start_ms and _is_new_closed_bar(bar):
                    ts = _dt_utc_from_ms(bar.start_ms).isoformat()
                    log.info(f"[{bar.symbol} {bar.interval}m] close @{ts} O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")
                    await _append_bar_to_csv(bar)
                    await _feed_kwin(bar)
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

# ---------- KWIN FEED ----------
async def _feed_kwin(bar: KlineBar):
    """Прокидываем закрытый бар в стратегию (если включен мост)."""
    if not KWIN_ENABLED or KWIN is None:
        return
    if KWIN_SYMBOL and bar.symbol != KWIN_SYMBOL:
        # Не смешиваем разные символы в одном экземпляре стратегии
        log.debug(f"KWIN ignore {bar.symbol}, ожидался {KWIN_SYMBOL}")
        return

    candle = {
        "timestamp": bar.start_ms,  # стратегия ждёт 'timestamp' в ms начала бара
        "open":  bar.open,
        "high":  bar.high,
        "low":   bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }

    try:
        intrabar_1m_aliases = ("1", "1m")
        if bar.interval in intrabar_1m_aliases:
            KWIN.on_bar_close_1m(candle)
        elif bar.interval in ("15", "15m"):
            KWIN.on_bar_close_15m(candle)
        elif bar.interval in ("60", "60m", "1h"):
            KWIN.on_bar_close_60m(candle)
        else:
            # Добавь при необходимости: "3","5","30","120" и их хэндлеры
            pass
    except Exception as e:
        log.error(f"KWIN feed error ({bar.interval}m): {e!r}")

# ---------- MAIN ----------
async def main():
    subs = []
    for s in SYMBOLS:
        for tf in KL_TFS:
            subs.append(f"kline.{_norm_tf(tf)}.{s}")
        subs.append(f"publicTrade.{s}")

    log.info(f"Запуск WS: {URL}")
    log.info(f"Подписки: {subs}")
    if SAVE_CSV:
        os.makedirs(DATA_DIR, exist_ok=True)
        log.info(f"Сохранение закрытых свечей в CSV: {DATA_DIR}")
    if KWIN_ENABLED:
        log.info("KWIN bridge: активен (в стратегию идут только закрытые бары).")

    # --- Исторический сид (опционально) ---
    if WS_SEED_HISTORY:
        await _seed_history(SYMBOLS, SEED_TFS)

    ws = BybitWebSocket(URL, SYMBOLS, KL_TFS)
    try:
        await ws.run_forever()
    except KeyboardInterrupt:
        log.info("Ctrl+C")
    finally:
        await ws.close()

if __name__ == "__main__":
    # Лёгкая проверка REST (гео-блоки)
    try:
        if requests is None:
            raise RuntimeError("requests not available")
        r = requests.get(f"{REST_BASE}/v5/market/time", timeout=5)
        if r.status_code == 200:
            log.info("✅ Bybit REST доступен")
        elif r.status_code == 403:
            log.warning("⚠️ Возможны гео-ограничения Bybit. WS/REST могут не работать.")
        else:
            log.warning(f"⚠️ REST ответ: {r.status_code}")
    except Exception:
        log.warning("⚠️ Нет доступа к REST; продолжаем с WS")
    asyncio.run(main())
