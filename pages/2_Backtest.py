# 2_Backtest.py
# Реальный бэктест: Bybit Futures 15m OHLC -> демо-рандом сделки/статистика (окно от "сейчас" назад)

import os
import io
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

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
from bybit_api import BybitAPI  # используется только для совместимости импортов
from state_manager import StateManager

# -------------------- Глобальные заглушки для бэктеста --------------------
api = None
db = Database(memory=True)            # или Database("kwin_bot.db")
state = StateManager(db)

# ====================== вспомогательные ======================
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
            "timestamp": ts,
            "open":  float(k.get("open",  k.get("o", 0.0))),
            "high":  float(k.get("high",  k.get("h", 0.0))),
            "low":   float(k.get("low",   k.get("l", 0.0))),
            "close": float(k.get("close", k.get("c", 0.0))),
            "volume": float(k.get("volume", k.get("v", 0.0))),
        })
    # по времени от старых к новым (для корректного "прогона вперёд")
    out.sort(key=lambda x: x["timestamp"])
    return out

@st.cache_data(show_spinner=False)
def load_klines_bybit_window(_api, symbol: str, days: int) -> List[Dict]:
    """
    Реальные свечи Bybit (фьючерсы/перпетуалы): берём запас по лимиту и режем последние N дней (UTC).
    Важно: параметр называется _api, чтобы st.cache_data не пытался его хэшировать.
    """
    if _api is None:
        return []

    # Попросим клиент работать с фьючерсами, если он это поддерживает
    try:
        if hasattr(_api, "set_market_type"):
            # наиболее безопасные варианты: "linear" (USDT-перп), "contract" или "futures"
            for mt in ("linear", "contract", "futures"):
                try:
                    _api.set_market_type(mt)
                    break
                except Exception:
                    continue
    except Exception:
        pass

    start_ms, end_ms = _window_ms(days)
    # 15m ≈ 96 баров/день. Возьмём запас, потом отфильтруем по окну.
    need = int(days * 96 * 1.2) + 50

    try:
        raw = _api.get_klines(symbol, "15", need) or []
    except Exception:
        return []

    kl = _normalize_klines(raw)
    # режем строго по окну [now - days, now] (UTC)
    kl = [b for b in kl if start_ms <= b["timestamp"] <= end_ms]

    # приведём timestamp в pandas-дату для графиков/таблиц
    for b in kl:
        b["timestamp"] = pd.to_datetime(b["timestamp"], unit="ms", utc=True)
    return kl

# ========================================================================
def main():
    st.set_page_config(page_title="KWIN Backtest", page_icon="📈", layout="wide")

    st.title("📊 KWIN Strategy Backtest")
    st.markdown("Тестирование стратегии на **реальных 15-мин свечах Bybit Futures** (окно от текущего момента назад).")

    # Параметры бэктеста
    col1, col2 = st.columns(2)
    with col1:
        start_capital = st.number_input("Начальный капитал ($)", min_value=100, value=10_000, step=100)
        period_days   = st.selectbox("Период тестирования (дней назад от сейчас)", [7, 14, 30, 60, 90], index=2)
    with col2:
        symbol   = st.selectbox("Торговая пара (USDT Perp)", ["ETHUSDT", "BTCUSDT"], index=0)
        fee_rate = st.number_input("Комиссия (%)", min_value=0.01, max_value=1.0, value=0.055, step=0.005)

    # Настройки стратегии (оставлены без изменений)
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
        with st.spinner("Выполняется бэктест на реальных свечах Bybit..."):
            try:
                # Конфигурация стратегии
                config = Config()
                config.symbol = symbol
                config.risk_reward = float(risk_reward)
                config.sfp_len = int(sfp_len)
                config.risk_pct = float(risk_pct)

                config.enable_smart_trail = bool(enable_smart_trail)
                config.trailing_perc = float(trailing_perc)
                config.trailing_offset_perc = float(trailing_offset)
                config.trailing_offset = float(trailing_offset)

                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct)
                config.taker_fee_rate = float(fee_rate) / 100.0  # 0.055% -> 0.00055

                # Источник свечей: ТОЛЬКО Bybit (фьючерсы), окно от "сейчас" назад
                _api = BybitAPI(api_key=os.getenv("BYBIT_API_KEY"),
                                api_secret=os.getenv("BYBIT_API_SECRET"))
                candles = load_klines_bybit_window(_api, symbol, period_days)

                if not candles:
                    st.warning("Не удалось получить исторические свечи Bybit за выбранный период.")
                    return

                # Инициализируем стратегию с существующими db/state (api остаётся None, это демо-симуляция)
                strategy = KWINStrategy(config, api, state, db)

                # Бэктест
                results = run_backtest(strategy, candles, start_capital)

                # Вывод
                display_backtest_results(results, f"Bybit Futures 15m — {symbol}")

            except Exception as e:
                st.error(f"Ошибка выполнения бэктеста: {e}")
                st.exception(e)

# ========================================================================
def run_backtest(strategy: KWINStrategy, candles: List[Dict], start_capital: float):
    """Демо-симуляция: используем реальные 15m свечи (по списку candles)."""
    current_equity = float(start_capital)
    equity_points = []

    for i in range(2, len(candles)):  # с 3-й свечи, чтобы был контекст
        candle = candles[i]

        # ДЕМО: случайный сигнал (логика стратегии не задействована)
        if np.random.random() < 0.05:
            direction   = "long" if np.random.random() > 0.5 else "short"
            entry_price = candle["close"]
            stop_loss   = entry_price * (0.98 if direction == "long" else 1.02)
            take_profit = entry_price * (1.026 if direction == "long" else 0.974)

            risk_amount   = current_equity * (strategy.config.risk_pct / 100.0)
            stop_distance = abs(entry_price - stop_loss)
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0.0

            if quantity > 0:
                win = (np.random.random() < 0.55)
                exit_price = take_profit if win else stop_loss

                pnl = (exit_price - entry_price) * quantity if direction == "long" else (entry_price - exit_price) * quantity
                commission = (entry_price + exit_price) * quantity * strategy.config.taker_fee_rate
                net_pnl = pnl - commission
                current_equity += net_pnl

                rr = abs(pnl) / (quantity * stop_distance) if stop_distance > 0 else 0.0

                trade_data = {
                    "symbol": strategy.config.symbol,
                    "direction": direction,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "quantity": float(quantity),
                    "pnl": float(net_pnl),
                    "rr": float(rr),
                    "entry_time": candle["timestamp"],
                    "exit_time": candle["timestamp"] + timedelta(minutes=int(np.random.randint(15, 240))),
                    "exit_reason": "TP" if net_pnl > 0 else "SL",
                    "status": "closed",
                }
                if hasattr(strategy.db, "add_trade"):
                    strategy.db.add_trade(trade_data)
                elif hasattr(strategy.db, "save_trade"):
                    strategy.db.save_trade(trade_data)

        equity_points.append({"timestamp": candle["timestamp"], "equity": current_equity})

    trades_list = []
    if hasattr(strategy.db, "get_recent_trades"):
        trades_list = strategy.db.get_recent_trades(100000)
    elif hasattr(strategy.db, "get_trades"):
        trades_list = strategy.db.get_trades()
    trades_df = pd.DataFrame(trades_list)
    equity_df = pd.DataFrame(equity_points)

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "final_equity": current_equity,
        "initial_equity": start_capital,
    }

# ========================================================================
def display_backtest_results(results, data_source_label: str):
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
    if not trades_df.empty:
        st.subheader("📋 История сделок")
        disp = trades_df.copy()
        for col in ("entry_time", "exit_time"):
            if col in disp.columns:
                disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d %H:%M")
        for col in ("pnl", "rr", "entry_price", "exit_price"):
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)
        if "quantity" in disp.columns:
            disp["quantity"] = pd.to_numeric(disp["quantity"], errors="coerce").round(4)
        st.dataframe(disp.tail(20), use_container_width=True)

    st.markdown("---")
    st.info(
        "Режим демонстрации: логика входов/выходов случайная — **меняется только источник свечей** "
        "(реальные 15m Bybit Futures, окно от текущего момента назад). Для полноценного бэктеста "
        "прогоняйте по свечам вашу стратегию."
    )

# ========================================================================
if __name__ == "__main__":
    main()
