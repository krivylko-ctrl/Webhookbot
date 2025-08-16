# 2_Backtest.py
# Реальный бэктест: исторические 15m OHLC -> on_bar_close_15m() -> сделки/статистика
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List

import streamlit as st
import matplotlib.pyplot as plt

# --- Импорты проекта ---
from config import Config
from state_manager import StateManager
from database import Database

# стратегия (с твоими последними патчами)
from kwin_strategy import KWINStrategy

# Bybit API: пробуем несколько вариантов клиента (под твои файлы)
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
    Ожидаемый формат элемента:
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
            # пропускаем мусор
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
            # На случай строковых значений
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


# ====================== Реальный бэктест через стратегию ======================

def run_backtest_ohlc(period_days: int,
                      initial_capital: float,
                      commission_rate: float,
                      symbol: str,
                      config: Config) -> Dict[str, pd.DataFrame]:
    """
    1) In-memory БД и стейт (чтобы не трогать прод).
    2) Загрузка исторических 15m свечей.
    3) Кормим стратегию закрытиями баров: on_bar_close_15m().
    4) Собираем сделки из БД и считаем эквити/метрики.
    """
    # In-memory DB (сделай поддержу memory=True в Database; если нет — временную SQLite)
    try:
        db = Database(memory=True)
    except TypeError:
        # если твой Database не умеет memory=True — создаём обычный, но с отдельным файлом
        db = Database(db_path="backtest_tmp.sqlite")

    state = StateManager(db)

    if BybitAPI is None:
        st.error("Не найден BybitAPI. Убедись, что bybit_api.py или bybit_v5_fixed.py доступны.")
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    api = BybitAPI( api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"))


    # Синхронизируем ключевые параметры из UI
    config.days_back = int(period_days)
    config.taker_fee_rate = float(commission_rate)
    config.symbol = symbol

    # Инициализируем стратегию
    strat = KWINStrategy(config, api, state, db)

    # Подтянем equity (для риск-менеджмента/компаундинга)
    try:
        strat._update_equity()
    except Exception:
        pass

    # Исторические 15m свечи
    candles = load_klines_bybit(api, symbol, "15", period_days)
    if not candles:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    # Кормим стратегию по закрытию (как Pine)
    for c in candles:
        strat.on_bar_close_15m(c)

    # Достаём сделки из in-memory БД
    try:
        trades = db.get_trades_by_period(period_days)
    except Exception:
        # если нет такого метода — забери все сделки, реализуй свой метод в БД
        trades = db.get_all_trades() if hasattr(db, "get_all_trades") else []

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Если в БД нет факта выхода — лучше стратегию доработать.
    # Пока для эквити берём только сделки с exit_price/exit_time.
    capital = initial_capital
    eq_times, eq_values = [], []

    if not trades_df.empty:
        if "entry_time" in trades_df.columns:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
        if "exit_time" in trades_df.columns:
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

        # сорт по времени выхода (или входа)
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

            if side == "long":
                gross = (exit_p - entry) * qty
            else:
                gross = (entry - exit_p) * qty

            fee_in = entry * qty * commission_rate
            fee_out = exit_p * qty * commission_rate
            pnl = gross - fee_in - fee_out

            capital += pnl
            total_pnl += pnl
            if pnl > 0:
                wins += 1

            t = tr.get("exit_time") or tr.get("entry_time")
            eq_times.append(pd.to_datetime(t))
            eq_values.append(capital)

        winrate = (wins / len(trades_df)) * 100 if len(trades_df) else 0.0
    else:
        winrate = 0.0
        total_pnl = 0.0

    equity_df = pd.DataFrame({"time": eq_times, "equity": eq_values}) if eq_values else pd.DataFrame()
    stats = {
        "final_capital": capital,
        "trades": int(len(trades_df)),
        "winrate_pct": round(winrate, 2),
        "total_pnl": round(total_pnl, 2),
    }
    return {"trades": trades_df, "equity": equity_df, "stats": stats}


# ====================== UI (Streamlit page) ======================

def main():
    st.set_page_config(page_title="Backtest — KWIN", layout="wide")
    st.title("KWIN — Backtest (15m OHLC → Strategy)")

    # Сайдбар: параметры
    st.sidebar.header("Параметры бэктеста")
    symbol = st.sidebar.text_input("Symbol", value="ETHUSDT")
    period_days = st.sidebar.selectbox("Период", options=[30, 60, 180], index=0)
    start_capital = st.sidebar.number_input("Initial Capital (USDT)", min_value=1.0, value=100.0, step=10.0)
    commission_rate = st.sidebar.number_input("Commission (taker, decimal)", min_value=0.0, value=0.00055, step=0.00005, format="%.5f")

    # Секция конфигурации стратегии (дубли твоих настроек из UI)
    st.sidebar.header("Strategy Config (ключевые)")
    risk_pct = st.sidebar.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=3.0, step=0.1, format="%.1f")
    risk_reward = st.sidebar.number_input("TP Risk/Reward Ratio", min_value=0.5, value=1.3, step=0.1)
    sfp_len = st.sidebar.number_input("SFP Length", min_value=2, value=2, step=1)
    use_sfp_quality = st.sidebar.checkbox("Filter: SFP quality (wick+closeback)", value=True)
    wick_min_ticks = st.sidebar.number_input("SFP: min wick depth (ticks)", min_value=0, value=7, step=1)
    close_back_pct = st.sidebar.number_input("SFP: min close-back (0..1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    enable_smart_trail = st.sidebar.checkbox("Enable Smart Trailing TP", value=True)
    use_arm_after_rr = st.sidebar.checkbox("Enable Arm after RR≥X", value=True)
    arm_rr = st.sidebar.number_input("Arm RR (R)", min_value=0.1, value=0.5, step=0.1, format="%.1f")
    use_bar_trail = st.sidebar.checkbox("Use Bar-Low/High Smart Trail", value=True)
    trail_lookback = st.sidebar.number_input("Trail lookback bars", min_value=1, value=50, step=1)
    trail_buf_ticks = st.sidebar.number_input("Trail buffer (ticks)", min_value=0, value=40, step=1)

    limit_qty_enabled = st.sidebar.checkbox("Limit Max Position Qty", value=True)
    max_qty_manual = st.sidebar.number_input("Max Qty (asset units)", min_value=0.01, value=50.0, step=0.01, format="%.2f")

    min_net_profit = st.sidebar.number_input("Min Net Profit (USDT)", min_value=0.0, value=1.2, step=0.1)
    min_order_qty = st.sidebar.number_input("Min Order Qty", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    qty_step = st.sidebar.number_input("Qty Step", min_value=0.0, value=0.01, step=0.01, format="%.2f")

    # Собираем Config так, чтобы он совпадал с твоей стратегией
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

        # График эквити
        st.subheader("Equity Curve")
        if not equity_df.empty:
            fig, ax = plt.subplots()
            ax.plot(equity_df["time"], equity_df["equity"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Equity (USDT)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Нет данных эквити (в БД отсутствуют закрытия сделок). Убедись, что стратегия записывает exit.")

        # Таблица сделок
        st.subheader("Сделки")
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.info("Сделок не найдено за выбранный период. Проверь условия детекции/окно бэктеста.")


if __name__ == "__main__":
    main()
