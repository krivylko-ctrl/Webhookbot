import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import requests   # ← нужен для прямого запроса v5
import time       # ← анти-рейткэп в загрузчике
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype  # ← для стабильной нормализации времени
import sys
import os

# путь к корню проекта (чтобы импортировать локальные модули)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kwin_strategy import KWINStrategy
from database import Database
from config import Config
from bybit_api import BybitAPI  # используется только для совместимости импортов
from state_manager import StateManager

# -------------------- Глобальные заглушки для бэктеста --------------------
# одна БД и один StateManager на сессию
api = None
db = Database(memory=True)            # или Database("kwin_bot.db") — если хочешь файл
state = StateManager(db)

# ===================== прямой загрузчик Bybit v5 =====================
BYBIT_V5_URL = "https://api.bybit.com/v5/market/kline"

def fetch_bybit_v5_window(symbol: str, days: int, interval: str = "15", category: str = "linear") -> list[dict]:
    """
    Реальные 15m свечи Bybit v5 за окно [UTC-now - days, UTC-now] c устойчивыми ретраями.
    Возвращает список {timestamp, open, high, low, close, volume} (timestamp в мс, по возрастанию).
    """
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    end_ms = now_ms

    limit = 1000
    tf_ms = 15 * 60 * 1000
    chunk_ms = limit * tf_ms  # ~10.4 дня на запрос

    out = []
    cursor_start = start_ms
    req_id = 0

    while cursor_start <= end_ms:
        req_id += 1
        cursor_end = min(end_ms, cursor_start + chunk_ms - 1)

        params = {
            "category": category,
            "symbol": symbol.upper(),
            "interval": str(interval),
            "start": int(cursor_start),
            "end": int(cursor_end),
            "limit": int(limit),
        }
        st.caption(
            f"▸ Bybit v5 запрос #{req_id}: "
            f"{datetime.utcfromtimestamp(params['start']/1000):%Y-%m-%d %H:%M} → "
            f"{datetime.utcfromtimestamp(params['end']/1000):%Y-%m-%d %H:%M} UTC"
        )

        # -------- РЕТРАИ НА ОДИН И ТОТ ЖЕ ЧАНК --------
        max_retries = 6
        backoff = 1.5  # сек стартовая пауза
        attempt = 0
        got_chunk = False

        while attempt < max_retries and not got_chunk:
            attempt += 1
            try:
                r = requests.get(BYBIT_V5_URL, params=params, timeout=25)
            except Exception as e:
                st.error(f"[#{req_id}/try{attempt}] Сетевой сбой: {e}")
                time.sleep(backoff)
                backoff *= 1.6
                continue

            if r.status_code == 403:
                st.error(f"HTTP 403 (rate limit) на чанк #{req_id}, попытка {attempt}/{max_retries}. Жду {backoff:.1f}s…")
                time.sleep(backoff)
                backoff *= 1.6
                continue

            if r.status_code != 200:
                st.error(f"HTTP {r.status_code} на чанк #{req_id}: {r.text[:200]}")
                time.sleep(backoff)
                backoff *= 1.6
                continue

            try:
                data = r.json()
            except Exception:
                st.error(f"Не удалось распарсить JSON на чанк #{req_id}: {r.text[:200]}")
                time.sleep(backoff)
                backoff *= 1.6
                continue

            if str(data.get("retCode")) != "0":
                st.error(f"retCode={data.get('retCode')} retMsg={data.get('retMsg')} на чанк #{req_id}")
                time.sleep(backoff)
                backoff *= 1.6
                continue

            rows = ((data.get("result") or {}).get("list") or [])
            if not rows:
                # Пустой ответ: такое бывает на краях окна — считаем чанк обработанным
                got_chunk = True
                break

            # Успех
            for row in rows:
                ts = int(row[0])
                if start_ms <= ts <= end_ms:
                    out.append({
                        "timestamp": ts,
                        "open":  float(row[1]),
                        "high":  float(row[2]),
                        "low":   float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]) if row[5] is not None else 0.0,
                    })
            got_chunk = True

        # если не удалось взять чанк после всех попыток — двигаемся дальше, чтобы не застревать
        if not got_chunk:
            st.error(f"Чанк #{req_id} не получен после {max_retries} попыток — пропускаю.")

        # следующий отрезок окна + мягкая пауза между удачными чанками
        cursor_start = cursor_end + 1
        time.sleep(0.6)

    # дедуп и сортировка
    out = sorted({b["timestamp"]: b for b in out}.values(), key=lambda x: x["timestamp"])

    if out:
        first_dt = datetime.utcfromtimestamp(out[0]["timestamp"]/1000)
        last_dt  = datetime.utcfromtimestamp(out[-1]["timestamp"]/1000)
        st.success(f"✅ Свечи Bybit v5 загружены: {len(out)} шт • {first_dt:%Y-%m-%d %H:%М} — {last_dt:%Y-%m-%d %H:%М} UTC")
    else:
        st.warning("Bybit v5 вернул пустой набор за выбранный период.")

    return out

# ===================== paper-API для стратегии =====================
class PaperBybitAPI:
    """Мини-эмулятор методов, которые читает стратегия (без реальных ордеров)."""
    def __init__(self):
        self._price = None
    def set_price(self, price: float):
        self._price = float(price)
    def get_ticker(self, symbol: str):
        return {"mark_price": self._price, "last_price": self._price}
    def place_order(self, **kwargs):
        return {"status": "Filled", "orderId": "paper"}
    def modify_order(self, **kwargs):
        return {"status": "OK"}
    def get_wallet_balance(self):
        return {"list": []}

# ========================================================================
def main():
    st.set_page_config(page_title="KWIN Backtest", page_icon="📈", layout="wide")

    st.title("📊 KWIN Strategy Backtest")
    st.markdown("Тестирование стратегии на исторических данных.")

    # выбор источника свечей
    data_src = st.radio(
        "Источник данных",
        ["Bybit v5 (реальные 15m)", "Синтетика (демо)"],
        horizontal=True,
        index=0
    )

    # Параметры бэктеста
    col1, col2 = st.columns(2)
    with col1:
        start_capital = st.number_input("Начальный капитал ($)", min_value=100, value=10_000, step=100)
        period_days   = st.selectbox("Период тестирования", [7, 14, 30, 60, 90], index=2)
    with col2:
        symbol   = st.selectbox("Торговая пара", ["ETHUSDT", "BTCUSDT"], index=0)
        fee_rate = st.number_input("Комиссия (%)", min_value=0.01, max_value=1.0, value=0.055, step=0.005)

    # Настройки стратегии
    st.subheader("⚙️ Параметры стратегии")
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_reward = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.3, step=0.1)
        sfp_len     = st.number_input("SFP Length", min_value=1, max_value=10, value=2, step=1)
        risk_pct    = st.number_input("Risk %", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    with c2:
        enable_smart_trail = st.checkbox("Smart Trailing", value=True)
        trailing_perc      = st.number_input("Trailing % (of entry)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        trailing_offset    = st.number_input("Trailing Offset %",   min_value=0.0, max_value=2.0, value=0.4, step=0.1)
        # +++ ДОБАВЛЕНО UI ДЛЯ ARM +++
        arm_after_rr       = st.checkbox("Arm after RR", value=True)
        arm_rr             = st.number_input("ARM RR (R)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    with c3:
        use_sfp_quality = st.checkbox("SFP Quality Filter", value=True)
        wick_min_ticks  = st.number_input("Min Wick Ticks", min_value=1, max_value=20, value=7, step=1)
        close_back_pct  = st.number_input("Close Back (0..1)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    if st.button("🚀 Запустить бэктест", type="primary"):
        with st.spinner("Выполняется бэктест..."):
            try:
                # Конфигурация стратегии
                config = Config()
                config.symbol = symbol
                config.risk_reward = float(risk_reward)
                config.sfp_len = int(sfp_len)
                config.risk_pct = float(risk_pct)

                config.enable_smart_trail = bool(enable_smart_trail)
                config.trailing_perc = float(trailing_perc) # проценты → доля
                config.trailing_offset_perc = float(trailing_offset)
                config.trailing_offset = float(trailing_offset)

                # +++ ПРОКИНУЛИ ARM В CONFIG +++
                config.use_arm_after_rr = bool(arm_after_rr)
                config.arm_rr = float(arm_rr)

                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct if close_back_pct <= 1 else close_back_pct / 100.0)
                config.taker_fee_rate = float(fee_rate) / 100.0  # 0.055% -> 0.00055
                config.days_back = int(period_days)  # чтобы _is_in_backtest_window_utc совпадало с окном

                # Инициализируем стратегию с существующими db/state
                strategy = KWINStrategy(config, api, state, db)
                # --- Биржевые фильтры для paper-режима (после создания strategy) ---
                if config.symbol.upper() == "ETHUSDT":
                    strategy.tick_size      = 0.01
                    strategy.qty_step       = 0.001
                    strategy.min_order_qty  = 0.001
                else:
                    # дефолт на всякий случай
                    strategy.tick_size      = 0.01
                    strategy.qty_step       = 0.001
                    strategy.min_order_qty  = 0.001

                # синхронизируем в config – часть проверок читает оттуда
                config.tick_size = strategy.tick_size
                config.qty_step = strategy.qty_step
                config.min_order_qty = strategy.min_order_qty

                # ===== Выбор источника =====
                if data_src.startswith("Bybit"):
                    candles = fetch_bybit_v5_window(symbol, period_days, interval="15", category="linear")
                    if not candles:
                        st.warning("Не удалось получить исторические свечи Bybit за выбранный период.")
                        return
                    results = run_backtest_real(strategy, candles, start_capital)
                else:
                    results = run_backtest(strategy, period_days, start_capital)  # твой демо на синтетике

                # Вывод
                display_backtest_results(results)

            except Exception as e:
                st.error(f"Ошибка выполнения бэктеста: {e}")
                st.exception(e)

# ========================================================================
def run_backtest(strategy: KWINStrategy, period_days: int, start_capital: float):
    """
    Синтетические 15m свечи -> прогон через KWINStrategy (paper).
    Все параметры стратегии из UI реально влияют на вход/SL/TP.
    """

    # ===== 1) Сгенерим синтетические свечи (UTC, 15m), timestamp в МИЛЛИСЕКУНДАХ =====
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=period_days)

    bars = period_days * 24 * 4  # 15m
    dates = pd.date_range(start=start_date, periods=bars, freq="15T", tz="UTC")

    base_price = 4500 if strategy.config.symbol.upper() == "ETHUSDT" else 118000
    price_changes = np.random.randn(len(dates)) * 0.002  # ~0.2% вола
    prices = base_price * np.exp(np.cumsum(price_changes))

    candles: list[dict] = []
    for i, (dt, p) in enumerate(zip(dates, prices)):
        vol = abs(np.random.randn() * 0.001)
        high = p * (1 + vol)
        low  = p * (1 - vol)
        open_p  = prices[i-1] if i > 0 else p
        close_p = p
        candles.append({
            "timestamp": int(pd.Timestamp(dt).timestamp() * 1000),  # ms
            "open": float(open_p),
            "high": float(high),
            "low":  float(low),
            "close": float(close_p),
            "volume": float(np.random.uniform(1_000, 10_000)),
        })

    # ===== 2) Подготовим paper-API и стартовые состояния =====
    state.set_equity(float(start_capital))
    class _Paper:
        def __init__(self): self._p = None
        def set_price(self, p): self._p = float(p)
        def get_ticker(self, symbol): return {"mark_price": self._p, "last_price": self._p}
        def place_order(self, **kw): return {"status": "Filled"}
        def modify_order(self, **kw): return {"status": "OK"}
        def get_wallet_balance(self): return {"list": []}

    paper = _Paper()
    strategy.api = paper

    bt_trades: list[dict] = []
    equity_points: list[dict] = []

    # Вспомогательное закрытие позиции с PnL и комиссиями
    def _close(exit_price: float, ts_ms: int):
        pos = state.get_current_position()
        if not pos or pos.get("status") != "open":
            return
        direction   = pos["direction"]
        entry_price = float(pos["entry_price"])
        qty         = float(pos["size"])
        fee_rate    = float(getattr(strategy.config, "taker_fee_rate", 0.00055))
        gross = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
        fees  = (entry_price + exit_price) * qty * fee_rate
        pnl   = gross - fees
        state.set_equity(float(state.get_equity() or 0) + pnl)

        bt_trades.append({
            "symbol": strategy.config.symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "stop_loss": float(pos.get("stop_loss") or 0),
            "take_profit": float(pos.get("take_profit") or 0),
            "quantity": qty,
            "pnl": float(pnl),
            "rr": None,
            "entry_time": datetime.utcfromtimestamp(int(pos.get("entry_time_ts", ts_ms))/1000),
            "exit_time":  datetime.utcfromtimestamp(int(ts_ms)/1000),
            "status": "closed",
        })
        pos["status"] = "closed"
        pos["exit_price"] = float(exit_price)
        pos["exit_time"]  = datetime.utcfromtimestamp(int(ts_ms)/1000)
        state.set_position(pos)

    # ===== 2.5) Pine-подобный Smart Trailing для синтетики (как в real) =====
    def apply_smart_trail(pos: dict, bar_high: float, bar_low: float) -> None:
        cfg = strategy.config
        if not getattr(cfg, "enable_smart_trail", True):
            return
        if not pos or pos.get("status") != "open":
            return

        entry = float(pos["entry_price"])
        sl    = float(pos.get("stop_loss") or 0.0)
        if entry <= 0 or sl <= 0:
            return

        # anchor экстремум с момента входа
        if pos["direction"] == "long":
            anchor = float(pos.get("trail_anchor", entry))
            anchor = max(anchor, float(bar_high))
            pos["trail_anchor"] = anchor
        else:
            anchor = float(pos.get("trail_anchor", entry))
            anchor = min(anchor, float(bar_low))
            pos["trail_anchor"] = anchor
        state.set_position(pos)

        # ARM по RR
        armed = bool(pos.get("armed", not getattr(cfg, "use_arm_after_rr", True)))
        if not armed and getattr(cfg, "use_arm_after_rr", True):
            risk = abs(entry - sl)
            if risk > 0:
                if pos["direction"] == "long":
                    rr = (float(bar_high) - entry) / risk
                else:
                    rr = (entry - float(bar_low)) / risk
                if rr >= float(getattr(cfg, "arm_rr", 0.5)):
                    armed = True
                    pos["armed"] = True
                    state.set_position(pos)
        if not armed:
            return

        # процентный стоп от entry с отступом
        trail_dist  = entry * (float(getattr(cfg, "trailing_perc", 0.5)) / 100.0)
        offset_dist = entry * (float(getattr(cfg, "trailing_offset_perc", 0.4)) / 100.0)

        if pos["direction"] == "long":
            candidate = anchor - trail_dist - offset_dist
            if candidate > sl:
                pos["stop_loss"] = candidate
                state.set_position(pos)
        else:
            candidate = anchor + trail_dist + offset_dist
            if candidate < sl:
                pos["stop_loss"] = candidate
                state.set_position(pos)

    # 3) Прогон от старых к новым =====
    for bar in candles:
        # гарантируем типы
        ts_ms = int(bar["timestamp"])
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        # 1) сначала обновим "текущую цену" для стратегии
        paper.set_price(c)  # <-- БЫЛО paper_api.set_price(c)

        # 2) СНАЧАЛА даём Pine-трейлу подвигать SL на этом баре
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            apply_smart_trail(pos, bar_high=h, bar_low=l)

        # 3) теперь проверяем SL/TP на текущем баре с уже обновлённым стопом
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            sl = float(pos.get("stop_loss") or 0)
            tp = pos.get("take_profit")
            if pos["direction"] == "long":
                if sl > 0 and l <= sl:
                    _close(sl, ts_ms)  # <-- БЫЛО close_position
                elif tp is not None and h >= float(tp):
                    _close(float(tp), ts_ms)
            else:  # short
                if sl > 0 and h >= sl:
                    _close(sl, ts_ms)
                elif tp is not None and l <= float(tp):
                    _close(float(tp), ts_ms)

        # 4) подаём закрытие 15m бара в стратегию (возможен вход)
        before_pos = state.get_current_position()
        strategy.on_bar_close_15m({"timestamp": ts_ms, "open": o, "high": h, "low": l, "close": c})
        after_pos = state.get_current_position()

        # если на этом баре открылась позиция — проставим время входа, ARM и trail_anchor от entry
        if after_pos and after_pos is not before_pos and after_pos.get("status") == "open":
            if "entry_time_ts" not in after_pos:
                after_pos["entry_time_ts"] = ts_ms
            after_pos["armed"] = not getattr(strategy.config, "use_arm_after_rr", True)
            after_pos["trail_anchor"] = float(after_pos["entry_price"])
            state.set_position(after_pos)

        # 5) снимем equity на конец бара
        equity_points.append({"timestamp": int(ts_ms), "equity": float(state.get_equity() or start_capital)})

    # Закроем хвост, если осталось открыто
    pos = state.get_current_position()
    if pos and pos.get("status") == "open":
        last = candles[-1]
        _close(float(last["close"]), int(last["timestamp"]))

    # Результаты
    trades_df = pd.DataFrame(bt_trades)
    equity_df = pd.DataFrame(equity_points)
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "final_equity": float(state.get_equity() or start_capital),
        "initial_equity": float(start_capital),
    }

# ===================== реальный прогон через стратегию =====================
def run_backtest_real(strategy: KWINStrategy, candles: list[dict], start_capital: float):
    """
    Прогон реальных 15m свечей через KWINStrategy (paper) c корректным порядком:
    1) Обновить Smart Trailing на этом баре
    2) Затем проверить SL/TP по high/low текущего бара
    3) Только потом отдать бар стратегии (входы считаются на закрытии)
    """
    import pandas as pd
    from datetime import datetime

    # --- нормализация входа (мс, float-ы) ---
    norm = []
    for b in candles or []:
        try:
            ts = b.get("timestamp")
            if isinstance(ts, str):
                ts = int(pd.to_datetime(ts, utc=True).value // 10**6)
            elif isinstance(ts, (pd.Timestamp, np.datetime64)):
                ts = int(pd.to_datetime(ts, utc=True).value // 10**6)
            else:
                ts = int(ts)
            norm.append({
                "timestamp": ts,
                "open":  float(b["open"]),
                "high":  float(b["high"]),
                "low":   float(b["low"]),
                "close": float(b["close"]),
                "volume": float(b.get("volume", 0.0)),
            })
        except Exception:
            continue
    candles = sorted(norm, key=lambda x: x["timestamp"])
    if not candles:
        return {
            "trades_df": pd.DataFrame([]),
            "equity_df": pd.DataFrame([]),
            "final_equity": float(start_capital),
            "initial_equity": float(start_capital),
        }

    # --- подготовка окружения ---
    state = strategy.state
    db = strategy.db
    state.set_equity(float(start_capital))

    class _Paper:
        def __init__(self): self._p = None
        def set_price(self, p): self._p = float(p)
        def get_ticker(self, symbol): return {"mark_price": self._p, "last_price": self._p}
        def place_order(self, **kw): return {"status": "Filled"}
        def modify_order(self, **kw): return {"status": "OK"}
        def get_wallet_balance(self): return {"list": []}

    paper_api = _Paper()
    strategy.api = paper_api

    bt_trades = []
    equity_points = []

    # --- утилиты ---
    def close_position(exit_price: float, ts_ms: int, reason: str):
        pos = state.get_current_position()
        if not pos or pos.get("status") != "open":
            return
        direction   = pos["direction"]
        entry_price = float(pos["entry_price"])
        qty         = float(pos["size"])
        fee_rate    = float(getattr(strategy.config, "taker_fee_rate", 0.00055))
        gross = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
        fees  = (entry_price + exit_price) * qty * fee_rate
        pnl   = gross - fees
        new_eq = float(state.get_equity() or start_capital) + pnl
        state.set_equity(new_eq)

        bt_trades.append({
            "symbol": strategy.config.symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "stop_loss": float(pos.get("stop_loss") or 0),
            "take_profit": float(pos.get("take_profit") or 0),
            "quantity": qty,
            "pnl": float(pnl),
            "rr": None,
            "entry_time": datetime.utcfromtimestamp(int(pos.get("entry_time_ts", ts_ms))/1000),
            "exit_time":  datetime.utcfromtimestamp(int(ts_ms)/1000),
            "exit_reason": reason,
            "status": "closed",
        })
        pos["status"] = "closed"
        pos["exit_price"] = float(exit_price)
        pos["exit_time"]  = datetime.utcfromtimestamp(int(ts_ms)/1000)
        state.set_position(pos)

    # ----------- Pine-подобный Smart Trailing с якорем экстремума -----------
    def apply_smart_trail(pos: dict, bar_high: float, bar_low: float) -> None:
        """
        Pine-эквивалент процентного трейла:
        - якорь = экстремум с момента входа (high для long, low для short)
        - активация по RR считается по барному high/low
        - стоп двигаем только в сторону сокращения риска
        """
        cfg = strategy.config
        if not getattr(cfg, "enable_smart_trail", True):
            return
        if not pos or pos.get("status") != "open":
            return

        entry = float(pos["entry_price"])
        sl    = float(pos.get("stop_loss") or 0.0)
        if entry <= 0 or sl <= 0:
            return

        # 1) поддерживаем якорь
        if pos["direction"] == "long":
            anchor = float(pos.get("trail_anchor", entry))
            anchor = max(anchor, float(bar_high))
            pos["trail_anchor"] = anchor
        else:
            anchor = float(pos.get("trail_anchor", entry))
            anchor = min(anchor, float(bar_low))
            pos["trail_anchor"] = anchor
        state.set_position(pos)

        # 2) ARM по RR (если включено)
        armed = bool(pos.get("armed", not getattr(cfg, "use_arm_after_rr", True)))
        if not armed and getattr(cfg, "use_arm_after_rr", True):
            risk = abs(entry - sl)
            if risk > 0:
                if pos["direction"] == "long":
                    rr = (float(bar_high) - entry) / risk
                else:
                    rr = (entry - float(bar_low)) / risk
                if rr >= float(getattr(cfg, "arm_rr", 0.5)):
                    armed = True
                    pos["armed"] = True
                    state.set_position(pos)
        if not armed:
            return

        # 3) расчёт процентного стопа от цены входа
        trail_dist  = entry * (float(getattr(cfg, "trailing_perc", 0.5)) / 100.0)
        offset_dist = entry * (float(getattr(cfg, "trailing_offset_perc", 0.4)) / 100.0)

        if pos["direction"] == "long":
            candidate = anchor - trail_dist - offset_dist
            if candidate > sl:
                pos["stop_loss"] = candidate
                state.set_position(pos)
        else:
            candidate = anchor + trail_dist + offset_dist
            if candidate < sl:
                pos["stop_loss"] = candidate
                state.set_position(pos)
    # ---------------------------------------------------------------------

    # --- цикл по барам ---
    for bar in candles:
        ts_ms = int(bar["timestamp"])
        o = float(bar["open"]); h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
        paper_api.set_price(c)

        # ===== (A) Сначала применяем Smart Trailing на этом баре =====
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            apply_smart_trail(pos, bar_high=h, bar_low=l)

        # ===== (B) Затем проверяем SL/TP по high/low этого бара =====
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            sl = float(pos.get("stop_loss") or 0)
            tp = pos.get("take_profit")
            if pos["direction"] == "long":
                if sl > 0 and l <= sl:
                    close_position(sl, ts_ms, reason="SL")
                elif tp is not None and h >= float(tp):
                    close_position(float(tp), ts_ms, reason="TP")
            else:
                if sl > 0 and h >= sl:
                    close_position(sl, ts_ms, reason="SL")
                elif tp is not None and l <= float(tp):
                    close_position(float(tp), ts_ms, reason="TP")

        # ===== (C) Теперь передаём закрытую 15m свечу в стратегию (возможен вход) =====
        before_pos = state.get_current_position()
        strategy.on_bar_close_15m({"timestamp": ts_ms, "open": o, "high": h, "low": l, "close": c})
        after_pos = state.get_current_position()

        # если позиция открылась — зафиксируем время входа, ARM и trail_anchor от entry
        if after_pos and after_pos is not before_pos and after_pos.get("status") == "open":
            if "entry_time_ts" not in after_pos:
                after_pos["entry_time_ts"] = ts_ms
            after_pos["armed"] = not getattr(strategy.config, "use_arm_after_rr", True)
            after_pos["trail_anchor"] = float(after_pos["entry_price"])
            state.set_position(after_pos)

        # equity-снимок
        equity_points.append({"timestamp": ts_ms, "equity": float(state.get_equity() or start_capital)})

    # хвост: если что-то осталось открытым — закрываем по последнему close
    pos = state.get_current_position()
    if pos and pos.get("status") == "open":
        last = candles[-1]
        close_position(float(last["close"]), int(last["timestamp"]), reason="EOD")

    trades_df = pd.DataFrame(bt_trades)
    equity_df = pd.DataFrame(equity_points)
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "final_equity": float(state.get_equity() or start_capital),
        "initial_equity": float(start_capital),
    }

# ========================================================================
def display_backtest_results(results):
    trades_df = results["trades_df"]
    equity_df = results["equity_df"]
    final_equity = results["final_equity"]
    initial_equity = results["initial_equity"]

    # Метрики
    if trades_df.empty:
        total_trades = winning_trades = losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        max_dd = 0.0
    else:
        total_trades = len(trades_df)
        winning_trades = int((trades_df["pnl"] > 0).sum())
        losing_trades  = int((trades_df["pnl"] < 0).sum())
        win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

        gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss   = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        if not equity_df.empty and len(equity_df) > 1:
            eq = equity_df.copy()

            # устойчиво нормализуем время
            if "timestamp" in eq.columns:
                if is_numeric_dtype(eq["timestamp"]):
                    ts = pd.to_datetime(eq["timestamp"], unit="ms", utc=True)
                elif is_datetime64_any_dtype(eq["timestamp"]):
                    ts = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
                eq["timestamp"] = ts.dt.tz_localize(None)

            eq["cummax"]  = eq["equity"].cummax()
            eq["drawdown"] = (eq["equity"] - eq["cummax"]) / eq["cummax"] * 100.0
            max_dd = float(eq["drawdown"].min())
        else:
            max_dd = 0.0

    total_return = ((final_equity - initial_equity) / initial_equity) * 100.0

    st.subheader("📈 Результаты бэктеста")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Общие сделки", total_trades)
    c2.metric("Винрейт", f"{win_rate:.1f}%")
    c3.metric("Profit Factor", "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}")
    c4.metric("Max DD", f"{max_dd:.2f}%")
    c5.metric("Доходность", f"{total_return:.2f}%")

    c1, c2 = st.columns(2)
    c1.metric("Начальный капитал", f"${initial_equity:,.2f}")
    profit_loss = final_equity - initial_equity
    c2.metric("Итоговый капитал", f"${final_equity:,.2f}", delta=f"${profit_loss:,.2f}")

    # График Equity
    if not equity_df.empty and len(equity_df) > 1:
        st.subheader("📊 Кривая Equity")
        eq = equity_df.copy()
        if "timestamp" in eq.columns:
            if is_numeric_dtype(eq["timestamp"]):
                ts = pd.to_datetime(eq["timestamp"], unit="ms", utc=True)
            else:
                ts = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
            eq["timestamp"] = ts.dt.tz_localize(None)

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity", "Drawdown"),
                            shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["equity"], mode="lines",
                                 name="Equity", line=dict(color="green", width=2)), row=1, col=1)

        eq["cummax"]  = eq["equity"].cummax()
        eq["drawdown"] = (eq["equity"] - eq["cummax"]) / eq["cummax"] * 100.0
        fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["drawdown"], mode="lines",
                                 name="Drawdown", line=dict(color="red", width=1),
                                 fill="tozeroy", fillcolor="rgba(255,0,0,0.2)"), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, title_text="Анализ производительности")
        fig.update_xaxes(title_text="Время", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Таблица сделок
    if not trades_df.empty:
        st.subheader("📋 История сделок")
        disp = trades_df.copy()
        for col in ("entry_time", "exit_time"):
            if col in disp.columns:
                disp[col] = pd.to_datetime(disp[col], errors="coerce").dt.tz_localize(None)
        for col in ("pnl", "rr", "entry_price", "exit_price"):
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)
        if "quantity" in disp.columns:
            disp["quantity"] = pd.to_numeric(disp["quantity"], errors="coerce").round(4)
        st.dataframe(disp.tail(20), use_container_width=True)

    st.markdown("---")
    st.info(
        "Выбери источник: **Bybit v5 (реальные 15m)** — прогон через стратегию; "
        "**Синтетика (демо)** — старый случайный симулятор."
    )

# ========================================================================
if __name__ == "__main__":
    main()
