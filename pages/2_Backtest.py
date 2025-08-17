# 2_Backtest.py
# Реальный бэктест: Bybit Futures 15m OHLC -> KWINStrategy (paper) -> сделки/статистика

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# путь к корню проекта (чтобы импортировать локальные модули)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kwin_strategy import KWINStrategy
from database import Database
from config import Config
from bybit_api import BybitAPI           # используем только для загрузки свечей
from state_manager import StateManager

# -------------------- Общие объекты --------------------
api = None
db = Database(memory=True)               # или Database("kwin_bot.db")
state = StateManager(db)

# ====================== Вспомогательные ======================
def _utc_now_ms() -> int:
    return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)

def _window_ms(days: int) -> Tuple[int, int]:
    end_ms = _utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    return start_ms, end_ms

def _ensure_ms(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts if ts > 1e11 else ts * 1000)
    if isinstance(ts, str):
        try:
            dt = pd.to_datetime(ts, utc=True)
            return int(dt.value // 10**6)
        except Exception:
            return None
    return None

def _normalize_klines(raw: List[Dict]) -> List[Dict]:
    if not raw:
        return []
    out = []
    for k in raw:
        ts = k.get("timestamp") or k.get("start") or k.get("open_time") or k.get("t")
        ts = _ensure_ms(ts)
        if ts is None:
            continue
        out.append({
            "timestamp": ts,  # МИЛЛИСЕКУНДЫ
            "open":  float(k.get("open",  k.get("o", 0.0))),
            "high":  float(k.get("high",  k.get("h", 0.0))),
            "low":   float(k.get("low",   k.get("l", 0.0))),
            "close": float(k.get("close", k.get("c", 0.0))),
            "volume": float(k.get("volume", k.get("v", 0.0))),
        })
    out.sort(key=lambda x: x["timestamp"])  # от старых к новым
    return out

@st.cache_data(show_spinner=False)
def load_klines_bybit_window(symbol: str, days: int) -> List[Dict]:
    """
    Реальные 15m свечи Bybit Futures за [UTC-сейчас - days, UTC-сейчас].
    Совместимо с разными обёртками: пробуем несколько сигнатур.
    Если пагинации нет – берём большой limit и обрезаем окно.
    """
    _api = BybitAPI(api_key=os.getenv("BYBIT_API_KEY"),
                    api_secret=os.getenv("BYBIT_API_SECRET"))

    # выбрать фьючерсную категорию, если метод есть
    try:
        if hasattr(_api, "set_market_type"):
            for mt in ("linear", "contract", "futures"):
                try:
                    _api.set_market_type(mt)
                    break
                except Exception:
                    continue
    except Exception:
        pass

    start_ms, end_ms = _window_ms(days)
    want_bars = days * 96  # 96 баров на день на 15м
    max_chunk = 1000

    bars: List[Dict] = []

    # ---------- Вариант А: простая выборка последнего большого куска ----------
    # многие обёртки поддерживают только (symbol, interval, limit)
    try:
        raw = _api.get_klines(symbol, "15", min(max_chunk, want_bars + 200)) or []
        if raw:
            chunk = _normalize_klines(raw)
            bars = [b for b in chunk if start_ms <= b["timestamp"] <= end_ms]
    except Exception:
        pass

    # если данных не хватило – попробуем пагинацию по from/start
    if len(bars) < want_bars * 0.9:
        cursor_from = start_ms
        safety = 0
        while cursor_from < end_ms and len(bars) < want_bars + 300 and safety < 30:
            safety += 1
            limit = min(max_chunk, want_bars - len(bars) + 300)

            raw = []
            # разные варианты параметров "начала"
            for kwargs in (
                {"symbol": symbol, "interval": "15", "limit": limit, "from": int(cursor_from // 1000)},   # сек
                {"symbol": symbol, "interval": "15", "limit": limit, "start": cursor_from},               # мс
                {"symbol": symbol, "interval": "15", "limit": limit, "startTime": cursor_from},           # мс
            ):
                try:
                    raw = _api.get_klines(**kwargs) or []
                    if raw:
                        break
                except TypeError:
                    continue
                except Exception:
                    raw = []

            if not raw:
                break

            chunk = _normalize_klines(raw)
            if not chunk:
                break

            # фильтруем строго по окну
            chunk = [b for b in chunk if start_ms <= b["timestamp"] <= end_ms]
            if not chunk:
                break

            bars.extend(chunk)
            # защита от дубликатов
            bars = sorted({b["timestamp"]: b for b in bars}.values(), key=lambda x: x["timestamp"])

            # сдвигаем курсор вперёд
            cursor_from = bars[-1]["timestamp"] + 1

    # финальная очистка окна
    bars = [b for b in bars if start_ms <= b["timestamp"] <= end_ms]
    bars.sort(key=lambda x: x["timestamp"])

    # диагностический вывод (помогает понять, почему пусто)
    if not bars:
        st.warning("Не удалось получить исторические свечи Bybit за выбранный период. "
                   "Скорее всего, обёртка get_klines поддерживает только (symbol, interval, limit).")
    else:
        first_dt = datetime.utcfromtimestamp(bars[0]["timestamp"]/1000)
        last_dt  = datetime.utcfromtimestamp(bars[-1]["timestamp"]/1000)
        st.caption(f"Свечи загружены: {len(bars)} шт.  "
                   f"окно: {first_dt:%Y-%m-%d %H:%M} — {last_dt:%Y-%m-%d %H:%M} UTC")

    return bars

# ====================== Paper API (эмулятор) ======================
class PaperBybitAPI:
    """Эмулятор методов, которые вызывает стратегия (никаких реальных запросов)."""
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

# ====================== Расчёт PnL и закрытия ======================
def _calc_trade_pnl(direction: str, entry_price: float, exit_price: float,
                    qty: float, taker_fee_rate: float) -> float:
    gross = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
    fees = (entry_price + exit_price) * qty * taker_fee_rate  # вход + выход
    return gross - fees

def _close_open_position(state: StateManager, db: Database, cfg: Config,
                         exit_price: float, exit_ts_ms: int):
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction   = pos["direction"]
    entry_price = float(pos["entry_price"])
    qty         = float(pos["size"])
    fee_rate    = float(getattr(cfg, "taker_fee_rate", 0.00055))

    pnl_net = _calc_trade_pnl(direction, entry_price, float(exit_price), qty, fee_rate)
    old_eq = float(state.get_equity() or 0.0)
    new_eq = old_eq + pnl_net
    state.set_equity(new_eq)

    trade = {
        "symbol": getattr(cfg, "symbol", "ETHUSDT"),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "stop_loss": pos.get("stop_loss"),
        "take_profit": pos.get("take_profit"),
        "quantity": qty,
        "pnl": pnl_net,
        "rr": None,
        "entry_time": datetime.utcfromtimestamp(int(pos.get("entry_time_ts", exit_ts_ms))/1000),
        "exit_time":  datetime.utcfromtimestamp(int(exit_ts_ms)/1000),
        "status": "closed",
    }
    try:
        if hasattr(db, "save_trade"):
            db.save_trade(trade)
        elif hasattr(db, "add_trade"):
            db.add_trade(trade)
        if hasattr(db, "save_equity_snapshot"):
            db.save_equity_snapshot(new_eq)
    except Exception:
        pass

    pos["status"]     = "closed"
    pos["exit_price"] = float(exit_price)
    pos["exit_time"]  = trade["exit_time"]
    state.set_position(pos)

# ====================== Реальный бэктест через стратегию ======================
def run_backtest(strategy: KWINStrategy, candles: List[Dict], initial_capital: float) -> Dict[str, pd.DataFrame]:
    """Прогоняем исторические 15m свечи по стратегии (paper API)."""
    state.set_equity(float(initial_capital))
    paper_api = PaperBybitAPI()
    strategy.api = paper_api  # подменяем API

    equity_points = []

    for bar in candles:  # от старых к новым
        paper_api.set_price(float(bar["close"]))

        # проверка SL/TP на текущем баре
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            sl = float(pos.get("stop_loss") or 0)
            tp = pos.get("take_profit")
            if pos["direction"] == "long" and sl > 0 and bar_low <= sl:
                _close_open_position(state, db, strategy.config, exit_price=sl, exit_ts_ms=bar["timestamp"])
            elif pos["direction"] == "short" and sl > 0 and bar_high >= sl:
                _close_open_position(state, db, strategy.config, exit_price=sl, exit_ts_ms=bar["timestamp"])
            else:
                if tp is not None:
                    tp = float(tp)
                    if pos["direction"] == "long" and bar_high >= tp:
                        _close_open_position(state, db, strategy.config, exit_price=tp, exit_ts_ms=bar["timestamp"])
                    if pos["direction"] == "short" and bar_low <= tp:
                        _close_open_position(state, db, strategy.config, exit_price=tp, exit_ts_ms=bar["timestamp"])

        # подадим бар в стратегию
        before_pos = state.get_current_position()
        strategy.on_bar_close_15m(bar)
        after_pos = state.get_current_position()
        if after_pos and after_pos is not before_pos and after_pos.get("status") == "open" and "entry_time_ts" not in after_pos:
            after_pos["entry_time_ts"] = int(bar["timestamp"])
            state.set_position(after_pos)

        equity_points.append({"timestamp": bar["timestamp"], "equity": float(state.get_equity() or initial_capital)})

    # закрыть возможную открытую позицию по последней цене
    last_bar = candles[-1]
    last_price = float(last_bar["close"])
    pos = state.get_current_position()
    if pos and pos.get("status") == "open":
        _close_open_position(state, db, strategy.config, exit_price=last_price, exit_ts_ms=last_bar["timestamp"])

    trades_list = []
    if hasattr(db, "get_recent_trades"):
        trades_list = db.get_recent_trades(100000)
    elif hasattr(db, "get_trades"):
        trades_list = db.get_trades()
    trades_df = pd.DataFrame(trades_list)
    equity_df = pd.DataFrame(equity_points)

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "final_equity": float(state.get_equity() or initial_capital),
        "initial_equity": float(initial_capital),
    }

# ====================== UI ======================
def main():
    st.set_page_config(page_title="KWIN Backtest", page_icon="📈", layout="wide")
    st.title("📊 KWIN Strategy Backtest — Bybit Futures 15m (от «сейчас» назад)")

    col1, col2 = st.columns(2)
    with col1:
        start_capital = st.number_input("Начальный капитал ($)", min_value=100, value=10_000, step=100)
        period_days   = st.selectbox("Период (дней назад от текущего UTC)", [7, 14, 30, 60, 90], index=2)
    with col2:
        symbol   = st.selectbox("Торговая пара", ["ETHUSDT", "BTCUSDT"], index=0)
        fee_rate = st.number_input("Комиссия (%)", min_value=0.01, max_value=1.0, value=0.055, step=0.005)

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
    with c3:
        use_sfp_quality = st.checkbox("SFP Quality Filter", value=True)
        wick_min_ticks  = st.number_input("Min Wick Ticks", min_value=1, max_value=20, value=7, step=1)
        close_back_pct  = st.number_input("Close Back (0..1)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    if st.button("🚀 Запустить бэктест", type="primary"):
        with st.spinner("Загружаю свечи и запускаю прогон стратегии..."):
            try:
                candles = load_klines_bybit_window(symbol, period_days)
                if not candles:
                    st.warning("Не удалось получить исторические свечи Bybit за выбранный период.")
                    return

                config = Config()
                config.symbol = symbol
                config.days_back = int(period_days)
                config.risk_reward = float(risk_reward)
                config.sfp_len = int(sfp_len)
                config.risk_pct = float(risk_pct)

                config.enable_smart_trail = bool(enable_smart_trail)
                config.trailing_perc = float(trailing_perc) / 100.0
                config.trailing_offset_perc = float(trailing_offset) / 100.0
                config.trailing_offset = float(trailing_offset)

                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct if close_back_pct <= 1 else close_back_pct / 100.0)
                config.taker_fee_rate = float(fee_rate) / 100.0

                strategy = KWINStrategy(config, api, state, db)
                results = run_backtest(strategy, candles, start_capital)
                display_backtest_results(results, f"Bybit Futures 15m — {symbol}")

            except Exception as e:
                st.error(f"Ошибка выполнения бэктеста: {e}")
                st.exception(e)

# ====================== вывод результатов ======================
def display_backtest_results(results, data_source_label: str):
    trades_df = results["trades_df"]
    equity_df = results["equity_df"].copy()
    final_equity = results["final_equity"]
    initial_equity = results["initial_equity"]

    # --- НОРМАЛИЗАЦИЯ ВРЕМЕНИ ДЛЯ ГРАФИКОВ/МЕТРИК ---
    if not equity_df.empty and "timestamp" in equity_df.columns:
        # если пришло число — это мс; если дата — просто делаем tz-naive
        if np.issubdtype(equity_df["timestamp"].dtype, np.number):
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], unit="ms", utc=True)
        else:
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce")
        equity_df["timestamp"] = equity_df["timestamp"].dt.tz_localize(None)
        equity_df = equity_df.sort_values("timestamp")

    if not trades_df.empty:
        for col in ("entry_time", "exit_time"):
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce").dt.tz_localize(None)

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
            eq["cummax"]  = eq["equity"].cummax()
            eq["drawdown"] = (eq["equity"] - eq["cummax"]) / eq["cummax"] * 100.0
            max_dd = float(eq["drawdown"].min())
        else:
            max_dd = 0.0

    total_return = ((final_equity - initial_equity) / initial_equity) * 100.0

    st.subheader("📈 Результаты бэктеста")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сделок", total_trades)
    c2.metric("Winrate", f"{win_rate:.1f}%")
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
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity", "Drawdown"),
                            shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=equity_df["timestamp"], y=equity_df["equity"], mode="lines",
                                 name="Equity", line=dict(color="green", width=2)), row=1, col=1)

        eq = equity_df.copy()
        eq["cummax"]  = eq["equity"].cummax()
        eq["drawdown"] = (eq["equity"] - eq["cummax"]) / eq["cummax"] * 100.0
        fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["drawdown"], mode="lines",
                                 name="Drawdown", line=dict(color="red", width=1),
                                 fill="tozeroy", fillcolor="rgba(255,0,0,0.2)"), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, title_text=f"Анализ производительности • {data_source_label}")
        fig.update_xaxes(title_text="Время", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Таблица сделок
    st.subheader("📋 Сделки")
    if not trades_df.empty:
        disp = trades_df.copy()
        for col in ("entry_time", "exit_time"):
            if col in disp.columns:
                disp[col] = pd.to_datetime(disp[col], errors="coerce").dt.tz_localize(None)
        for col in ("pnl", "rr", "entry_price", "exit_price"):
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce")
                if col in ("pnl", "rr", "entry_price", "exit_price"):
                    disp[col] = disp[col].round(2)
        if "quantity" in disp.columns:
            disp["quantity"] = pd.to_numeric(disp["quantity"], errors="coerce").round(4)
        view_cols_pref = ["entry_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "rr", "status"]
        view_cols = [c for c in view_cols_pref if c in disp.columns]
        st.dataframe(disp[view_cols].sort_values(by="entry_time", ascending=False), use_container_width=True)
    else:
        st.info("Сделок не найдено за выбранный период. Проверь условия/настройки стратегии.")

# ========================================================================
if __name__ == "__main__":
    main()
