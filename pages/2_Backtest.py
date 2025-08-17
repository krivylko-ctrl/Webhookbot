# 2_Backtest.py
# Реальный бэктест: исторические 15m OHLC -> on_bar_close_15m() -> сделки/статистика

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List

import streamlit as st
import matplotlib.pyplot as plt

# --- Импорты проекта ---
from config import Config
from state_manager import StateManager
from database import Database

# стратегия (с последними патчами)
from kwin_strategy import KWINStrategy

# Bybit API: пробуем несколько вариантов клиента
try:
    from bybit_api import BybitAPI
except Exception:
    try:
        from bybit_v5_fixed import BybitAPI  # если у тебя класс там
    except Exception:
        BybitAPI = None


# ====================== Утилиты загрузки данных ======================

def load_klines_bybit(api, symbol: str, interval: str, days: int) -> List[Dict]:
    """
    Грузим исторические свечи с Bybit и нормализуем формат под стратегию.
    Элемент:
      {"timestamp": ms, "start": ms, "open": float, "high": float, "low": float, "close": float, "volume": float}
    """
    if api is None:
        st.error("BybitAPI не инициализирован. Проверь импорт/креды.")
        return []

    # Примерно 4 бара на час для 15m
    bars = int(days * 24 * 4) + 20  # с запасом
    try:
        kl = api.get_klines(symbol, interval, bars) or []
    except Exception as e:
        st.error(f"Ошибка получения свечей: {e}")
        return []

    if not kl:
        return []

    # От старых к новым — чтобы кормить по закрытию баров
    try:
        kl.sort(key=lambda x: x.get("timestamp") or x.get("open_time") or x.get("start"))
    except Exception:
        pass

    norm = []
    for c in kl:
        ts = c.get("timestamp") or c.get("open_time") or c.get("start")
        if ts is None:
            continue
        try:
            norm.append({
                "timestamp": int(ts),
                "start": int(ts),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume", 0.0)),
            })
        except Exception:
            # запасной парсер
            try:
                norm.append({
                    "timestamp": int(ts),
                    "start": int(ts),
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": float(c.get("volume", 0.0)),
                })
            except Exception:
                continue
    return norm


# ================== P&L + equity helpers ==================

def _calc_trade_pnl(direction: str,
                    entry_price: float,
                    exit_price: float,
                    qty: float,
                    taker_fee_rate: float) -> float:
    """
    Возвращает ЧИСТЫЙ PnL в USDT с учётом двойной комиссии.
    qty — в базовой валюте (например, ETH), цены — в USDT.
    """
    gross = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
    fees = (entry_price + exit_price) * qty * taker_fee_rate  # вход + выход
    return gross - fees


def _close_open_position(state: StateManager,
                         db: Database,
                         cfg: Config,
                         exit_price: float,
                         exit_time):
    """
    Закрыть открытую позицию в бэктесте:
    - посчитать PnL (нетто),
    - обновить equity (compound),
    - записать сделку в БД.
    """
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction   = pos["direction"]
    entry_price = float(pos["entry_price"])
    qty         = float(pos["size"])  # qty в ETH (или другой базовой)
    fee_rate    = float(getattr(cfg, "taker_fee_rate", 0.00055))

    pnl_net = _calc_trade_pnl(direction, entry_price, exit_price, qty, fee_rate)

    # --- обновляем equity (compound) ---
    old_eq = float(state.get_equity() or 0.0)
    new_eq = old_eq + pnl_net
    state.set_equity(new_eq)

    # --- фиксируем сделку в БД (расчёт RR опционален; если нужно, можно вычислить) ---
    trade = {
        "symbol":      getattr(cfg, "symbol", "ETHUSDT"),
        "direction":   direction,
        "entry_price": entry_price,
        "exit_price":  float(exit_price),
        "stop_loss":   pos.get("stop_loss"),
        "take_profit": pos.get("take_profit"),
        "quantity":    qty,
        "pnl":         pnl_net,
        "rr":          None,
        "entry_time":  pos.get("entry_time"),
        "exit_time":   exit_time,
        "status":      "closed",
    }
    try:
        db.save_trade(trade)
        db.save_equity_snapshot(new_eq)
    except Exception:
        pass

    # --- закрываем локально позицию ---
    pos["status"]     = "closed"
    pos["exit_price"] = float(exit_price)
    pos["exit_time"]  = exit_time
    state.set_position(pos)

    print(f"[EXIT] {direction.upper()} qty={qty} @ {exit_price}  PnL={pnl_net:.2f}  equity: {old_eq:.2f} → {new_eq:.2f}")


def _dd_from_equity(equity_series: pd.Series) -> Dict[str, float]:
    """Расчёт максимальной просадки ($ и %) из серии equity (по времени)."""
    if equity_series.empty:
        return {"max_dd_abs": 0.0, "max_dd_pct": 0.0}

    roll_max = equity_series.cummax()
    dd = equity_series - roll_max
    dd_pct = equity_series / roll_max - 1.0
    max_dd_abs = float(dd.min())  # отрицательное значение
    max_dd_pct = float(dd_pct.min()) * 100.0  # %
    return {"max_dd_abs": round(max_dd_abs, 2), "max_dd_pct": round(max_dd_pct, 2)}


def _profit_factor_from_trades(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    losses = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 3)


# ====================== Реальный бэктест через стратегию ======================

def run_backtest_ohlc(period_days: int,
                      initial_capital: float,
                      commission_rate: float,
                      symbol: str,
                      config: Config) -> Dict[str, pd.DataFrame]:
    """
    1) In-memory БД и стейт (не трогаем прод).
    2) Загрузка исторических 15m свечей.
    3) Кормим стратегию закрытиями баров: on_bar_close_15m().
    4) Внутри прохода по барам эмулируем SL/TP на следующем баре (как в Pine).
    5) Собираем сделки из БД и считаем эквити/метрики.
    """
    # In-memory DB
    try:
        db = Database(memory=True)
    except TypeError:
        db = Database(db_path="backtest_tmp.sqlite")

    state = StateManager(db)
    state.set_equity(float(initial_capital))  # стартовый капитал для компаундинга

    if BybitAPI is None:
        st.error("Не найден BybitAPI. Убедись, что bybit_api.py или bybit_v5_fixed.py доступны.")
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    api = BybitAPI(
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET")
    )

    # Синхронизируем параметры
    config.days_back = int(period_days)
    config.taker_fee_rate = float(commission_rate)
    config.symbol = symbol

    # Инициализируем стратегию
    strat = KWINStrategy(config, api, state, db)

    # Подтянем equity (если есть на счёте) — но старт компаундинга = initial_capital
    try:
        strat._update_equity()
    except Exception:
        pass

    # Исторические 15m свечи (от старых к новым)
    candles = load_klines_bybit(api, symbol, "15", period_days)
    if not candles:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    # === Основной проход по барам ===
    # Логика Pine: вход на закрытии бара t, а SL/TP могут сработать на баре t+1 по его high/low.
    for bar in candles:
        # 1) Если позиция УЖЕ открыта с прошлого бара — проверим SL/TP на текущем баре по high/low.
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            cur_time = bar.get("close_time") or bar["timestamp"]

            sl = float(pos.get("stop_loss") or 0)
            tp = pos.get("take_profit")

            # Порядок: сначала SL, затем TP (как в Pine при касании обеих зон)
            if pos["direction"] == "long" and sl > 0 and bar_low <= sl:
                _close_open_position(state, db, config, exit_price=sl, exit_time=cur_time)
            elif pos["direction"] == "short" and sl > 0 and bar_high >= sl:
                _close_open_position(state, db, config, exit_price=sl, exit_time=cur_time)
            else:
                if tp is not None:
                    tp = float(tp)
                    if pos["direction"] == "long" and bar_high >= tp:
                        _close_open_position(state, db, config, exit_price=tp, exit_time=cur_time)
                    if pos["direction"] == "short" and bar_low <= tp:
                        _close_open_position(state, db, config, exit_price=tp, exit_time=cur_time)

        # 2) Теперь передаём бар в стратегию — это «закрытие» бара, где стратегия может ВОЙТИ
        #    и/или подвинуть Smart Trailing (он применяется в run_cycle, вызываемом внутри on_bar_close_15m).
        strat.on_bar_close_15m(bar)

    # Достаём сделки из БД
    try:
        trades = db.get_trades_by_period(period_days)
    except Exception:
        trades = db.get_all_trades() if hasattr(db, "get_all_trades") else []

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Строим equity-линию по сохранённым снимкам (если есть) или по сделкам
    capital = float(initial_capital)
    eq_times, eq_values = [], []

    # Попробуем взять equity snapshots из БД (если реализовано)
    equity_df = pd.DataFrame()
    try:
        snaps = db.get_equity_snapshots(period_days) if hasattr(db, "get_equity_snapshots") else []
        if snaps:
            equity_df = pd.DataFrame(snaps)
            if {"time", "equity"} <= set(equity_df.columns):
                equity_df["time"] = pd.to_datetime(equity_df["time"], utc=True, errors="coerce")
                equity_df = equity_df.sort_values("time")
    except Exception:
        pass

    if equity_df.empty:
        # fallback: строим по закрытым сделкам
        if not trades_df.empty:
            if "entry_time" in trades_df.columns:
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
            if "exit_time" in trades_df.columns:
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

            sort_key = "exit_time" if "exit_time" in trades_df.columns else "entry_time"
            trades_df = trades_df.sort_values(by=sort_key)

            wins = 0
            total_pnl = 0.0
            for _, tr in trades_df.iterrows():
                entry = float(tr.get("entry_price", np.nan))
                qty = float(tr.get("quantity", np.nan))
                side = tr.get("direction", "")
                exit_p = tr.get("exit_price", np.nan)

                if np.isnan(entry) or np.isnan(qty) or pd.isna(exit_p):
                    continue
                exit_p = float(exit_p)

                gross = (exit_p - entry) * qty if side == "long" else (entry - exit_p) * qty
                fee_in = entry * qty * commission_rate
                fee_out = exit_p * qty * commission_rate
                pnl = gross - fee_in - fee_out

                capital += pnl
                total_pnl += pnl
                if pnl > 0:
                    wins += 1

                t = tr.get("exit_time") or tr.get("entry_time")
                eq_times.append(pd.to_datetime(t, utc=True, errors="coerce"))
                eq_values.append(capital)

            winrate = (wins / len(trades_df)) * 100 if len(trades_df) else 0.0
        else:
            winrate = 0.0
            total_pnl = 0.0

        equity_df = pd.DataFrame({"time": eq_times, "equity": eq_values}) if eq_values else pd.DataFrame()
    else:
        # если есть снимки — метрики ниже посчитаем по сделкам (как обычно)
        total_pnl = 0.0
        winrate = 0.0
        if not trades_df.empty:
            wins = 0
            for _, tr in trades_df.iterrows():
                if "pnl" in tr and not pd.isna(tr["pnl"]):
                    total_pnl += float(tr["pnl"])
                    if float(tr["pnl"]) > 0:
                        wins += 1
            winrate = (wins / len(trades_df)) * 100 if len(trades_df) else 0.0
        capital = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_capital

    # --- Доп. метрики ---
    if not equity_df.empty:
        dd = _dd_from_equity(equity_df["equity"].astype(float))
    else:
        dd = {"max_dd_abs": 0.0, "max_dd_pct": 0.0}

    profit_factor = _profit_factor_from_trades(trades_df) if not trades_df.empty else 0.0
    avg_trade_pnl = float(trades_df["pnl"].mean()) if ("pnl" in trades_df.columns and not trades_df.empty) else 0.0

    stats = {
        "final_capital": round(float(capital), 2),
        "trades": int(len(trades_df)),
        "winrate_pct": round(float(winrate), 2),
        "total_pnl": round(float(total_pnl), 2),
        "max_dd_abs": dd["max_dd_abs"],   # $
        "max_dd_pct": dd["max_dd_pct"],   # %
        "profit_factor": profit_factor,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
    }
    return {"trades": trades_df, "equity": equity_df, "stats": stats}


# ====================== UI (Streamlit page) ======================

def main():
    st.set_page_config(page_title="Backtest — KWIN", layout="wide")
    st.title("KWIN — Backtest (15m OHLC → Strategy)")

    # Сайдбар: параметры
    st.sidebar.header("Параметры бэктеста")
    symbol = st.sidebar.text_input("Symbol", value="ETHUSDT")
    period_days = st.sidebar.selectbox("Период", options=[30, 60, 90, 180], index=0)
    start_capital = st.sidebar.number_input("Initial Capital (USDT)", min_value=1.0, value=100.0, step=10.0)
    commission_rate = st.sidebar.number_input("Commission (taker, decimal)", min_value=0.0, value=0.00055, step=0.00005, format="%.5f")

    # Секция конфигурации стратегии
    st.sidebar.header("Strategy Config (ключевые)")
    risk_pct = st.sidebar.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=3.0, step=0.1, format="%.1f")
    risk_reward = st.sidebar.number_input("TP Risk/Reward Ratio", min_value=0.5, value=1.3, step=0.1)
    sfp_len = st.sidebar.number_input("SFP Length", min_value=2, value=2, step=1)
    use_sfp_quality = st.sidebar.checkbox("Filter: SFP quality (wick+closeback)", value=True)
    wick_min_ticks = st.sidebar.number_input("SFP: min wick depth (ticks)", min_value=0, value=7, step=1)
    close_back_pct = st.sidebar.number_input("SFP: min close-back (0..1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Smart Trailing")
    enable_smart_trail = st.sidebar.checkbox("Enable Smart Trailing TP", value=True)
    use_arm_after_rr = st.sidebar.checkbox("Enable Arm after RR≥X", value=True)
    arm_rr = st.sidebar.number_input("Arm RR (R)", min_value=0.1, value=0.5, step=0.1, format="%.1f")
    # ВАЖНО: добавили недостающие поля процентного трейла
    trailing_perc = st.sidebar.number_input("Percent trailing, % of entry", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    trailing_offset_perc = st.sidebar.number_input("Trailing offset, % of entry", min_value=0.0, value=0.4, step=0.1, format="%.2f")
    use_bar_trail = st.sidebar.checkbox("Use Bar-Low/High Smart Trail", value=True)
    trail_lookback = st.sidebar.number_input("Trail lookback bars", min_value=1, value=50, step=1)
    trail_buf_ticks = st.sidebar.number_input("Trail buffer (ticks)", min_value=0, value=40, step=1)

    st.sidebar.markdown("---")
    limit_qty_enabled = st.sidebar.checkbox("Limit Max Position Qty", value=True)
    max_qty_manual = st.sidebar.number_input("Max Qty (asset units)", min_value=0.01, value=50.0, step=0.01, format="%.2f")

    min_net_profit = st.sidebar.number_input("Min Net Profit (USDT)", min_value=0.0, value=1.2, step=0.1)
    min_order_qty = st.sidebar.number_input("Min Order Qty", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    qty_step = st.sidebar.number_input("Qty Step", min_value=0.0, value=0.01, step=0.01, format="%.2f")

    # Собираем Config
    config = Config()
    config.symbol = symbol
    config.days_back = int(period_days)
    config.risk_pct = float(risk_pct)
    config.risk_reward = float(risk_reward)
    config.sfp_len = int(sfp_len)

    config.use_sfp_quality = bool(use_sfp_quality)
    config.wick_min_ticks = int(wick_min_ticks)
    config.close_back_pct = float(close_back_pct)  # 0..1!

    config.enable_smart_trail = bool(enable_smart_trail)
    config.use_arm_after_rr = bool(use_arm_after_rr)
    config.arm_rr = float(arm_rr)
    config.use_bar_trail = bool(use_bar_trail)
    config.trail_lookback = int(trail_lookback)
    config.trail_buf_ticks = int(trail_buf_ticks)

    # НОВОЕ: проценты для первичного трейла
    config.trailing_perc = float(trailing_perc)
    config.trailing_offset_perc = float(trailing_offset_perc)

    config.limit_qty_enabled = bool(limit_qty_enabled)
    config.max_qty_manual = float(max_qty_manual)

    config.min_net_profit = float(min_net_profit)
    config.min_order_qty = float(min_order_qty)
    config.qty_step = float(qty_step)

    config.taker_fee_rate = float(commission_rate)

    run_btn = st.button("Запустить бэктест")

    if run_btn:
        with st.spinner("Запускаю бэктест..."):
            res = run_backtest_ohlc(
                period_days=period_days,
                initial_capital=start_capital,
                commission_rate=commission_rate,
                symbol=symbol,
                config=config
            )

        trades_df = res.get("trades", pd.DataFrame())
        equity_df = res.get("equity", pd.DataFrame())
        stats = res.get("stats", {})

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Сделок", value=stats.get("trades", 0))
        col2.metric("Winrate", f"{stats.get('winrate_pct', 0.0)}%")
        col3.metric("Итоговый капитал", f"{stats.get('final_capital', start_capital):.2f}")
        col4.metric("Total PnL", f"{stats.get('total_pnl', 0.0):.2f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Max DD ($)", f"{stats.get('max_dd_abs', 0.0):.2f}")
        col6.metric("Max DD (%)", f"{stats.get('max_dd_pct', 0.0):.2f}%")
        pf = stats.get("profit_factor", 0.0)
        pf_str = "∞" if pf == float("inf") else f"{pf}"
        col7.metric("Profit Factor", pf_str)

        # График эквити
        st.subheader("Equity Curve")
        if not equity_df.empty:
            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(equity_df["time"]), equity_df["equity"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Equity (USDT)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Нет данных эквити (в БД отсутствуют закрытия сделок). Убедись, что стратегия записывает exit.")

        # Таблица сделок
        st.subheader("Сделки")
        if not trades_df.empty:
            # немного приводим формат
            for col in ("entry_time", "exit_time"):
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce").dt.tz_convert(None)
            for col in ("entry_price", "exit_price", "quantity", "pnl", "rr"):
                if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")
                    if col in ("entry_price", "exit_price", "pnl", "rr"):
                        trades_df[col] = trades_df[col].round(2)
                    if col == "quantity":
                        trades_df[col] = trades_df[col].round(4)

            view_cols = [c for c in ["entry_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "rr", "status", "exit_reason"] if c in trades_df.columns]
            st.dataframe(trades_df[view_cols].sort_values("entry_time", ascending=False), use_container_width=True)
        else:
            st.info("Сделок не найдено за выбранный период. Проверь условия детекции/окно бэктеста.")


if __name__ == "__main__":
    main()
