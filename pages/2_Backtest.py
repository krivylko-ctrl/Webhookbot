import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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

# ========================================================================
def main():
    st.set_page_config(page_title="KWIN Backtest", page_icon="📈", layout="wide")

    st.title("📊 KWIN Strategy Backtest")
    st.markdown("Тестирование стратегии на исторических данных (демо-симуляция)")

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
                config.trailing_perc = float(trailing_perc)
                # чтобы совпадало и с TrailEngine, и со старым кодом
                config.trailing_offset_perc = float(trailing_offset)
                config.trailing_offset = float(trailing_offset)

                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct)
                config.taker_fee_rate = float(fee_rate) / 100.0  # 0.055% -> 0.00055

                # Инициализируем стратегию с существующими db/state
                strategy = KWINStrategy(config, api, state, db)

                # Бэктест
                results = run_backtest(strategy, period_days, start_capital)

                # Вывод
                display_backtest_results(results)

            except Exception as e:
                st.error(f"Ошибка выполнения бэктеста: {e}")
                st.exception(e)

# ========================================================================
def run_backtest(strategy: KWINStrategy, period_days: int, start_capital: float):
    """Простая демо-симуляция: синтетические 15m данные + сделки."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    # 15m таймфрейм
    bars = period_days * 24 * 4
    dates = pd.date_range(start=start_date, periods=bars, freq="15T")

    base_price = 4500 if strategy.config.symbol == "ETHUSDT" else 118000
    price_changes = np.random.randn(len(dates)) * 0.002  # ~0.2% вола
    prices = base_price * np.exp(np.cumsum(price_changes))

    candles = []
    for i, (dt, p) in enumerate(zip(dates, prices)):
        vol = abs(np.random.randn() * 0.001)
        high = p * (1 + vol)
        low  = p * (1 - vol)
        open_p  = prices[i-1] if i > 0 else p
        close_p = p
        candles.append({
            "timestamp": dt,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close_p,
            "volume": float(np.random.uniform(1_000, 10_000)),
        })

    current_equity = float(start_capital)
    equity_points = []  # [{'timestamp': ..., 'equity': ...}]

    # цикл по барам
    for i in range(2, len(candles)):  # с 3-й свечи, чтобы был контекст
        candle = candles[i]

        # Демо: вероятность сигнала 5%
        if np.random.random() < 0.05:
            direction   = "long" if np.random.random() > 0.5 else "short"
            entry_price = candle["close"]
            stop_loss   = entry_price * (0.98 if direction == "long" else 1.02)
            take_profit = entry_price * (1.026 if direction == "long" else 0.974)

            risk_amount   = current_equity * (strategy.config.risk_pct / 100.0)
            stop_distance = abs(entry_price - stop_loss)
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0.0

            if quantity > 0:
                # результат сделки
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
                strategy.db.add_trade(trade_data)

        # ровно один снэпшот equity на бар
        equity_points.append({"timestamp": candle["timestamp"], "equity": current_equity})

    # результаты
    trades_df = pd.DataFrame(strategy.db.get_all_trades())
    equity_df = pd.DataFrame(equity_points)

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "final_equity": current_equity,
        "initial_equity": start_capital,
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
                disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d %H:%M")
        for col in ("pnl", "rr", "entry_price", "exit_price"):
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)
        if "quantity" in disp.columns:
            disp["quantity"] = pd.to_numeric(disp["quantity"], errors="coerce").round(4)
        st.dataframe(disp.tail(20), use_container_width=True)

    st.markdown("---")
    st.info(
        "Демо-бэктест: данные синтетические. Для реального бэктеста подключи исторические OHLC Bybit."
    )

# ========================================================================
if __name__ == "__main__":
    main()
