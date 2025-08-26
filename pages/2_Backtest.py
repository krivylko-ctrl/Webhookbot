# pages/3_Backtrader.py
import os
import io
import time
import datetime as dt

import streamlit as st
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt

st.set_page_config(page_title="Backtrader — Backtest", page_icon="📈", layout="wide")

st.title("📈 Backtrader — быстрый бэктест")

# ---------- 1) Источник данных ----------
with st.expander("Источник данных", expanded=True):
    src = st.radio("Выбери источник", ["CSV upload", "CSV из Volume (/data)", "YFinance"], horizontal=True)

    df = None
    if src == "CSV upload":
        f = st.file_uploader("Загрузи CSV (должны быть колонки: datetime, open, high, low, close, volume)", type=["csv"])
        if f is not None:
            df = pd.read_csv(f)
    elif src == "CSV из Volume (/data)":
        path = st.text_input("Путь к CSV (в контейнере)", "/data/eth_15m.csv")
        if st.button("Загрузить CSV"):
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                st.error(f"Файл не найден: {path}")
    else:  # YFinance
        import yfinance as yf
        ticker = st.text_input("Ticker", "ETH-USD")
        tf = st.selectbox("TF", ["15m", "30m", "1h", "4h", "1d"], index=0)
        period = st.selectbox("Период", ["7d", "14d", "30d", "60d", "1y"], index=2)
        if st.button("Скачать с Yahoo"):
            data = yf.download(ticker, period=period, interval=tf, auto_adjust=False, progress=False)
            if data is None or data.empty:
                st.error("Данных нет")
            else:
                data = data.reset_index()
                data.rename(columns={
                    "Datetime": "datetime",
                    "Date": "datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }, inplace=True)
                df = data[["datetime", "open", "high", "low", "close", "volume"]]

    if df is not None:
        # нормализуем формат
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df.dropna(subset=["datetime"], inplace=True)
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
        st.success(f"Данных строк: {len(df)}")
        st.dataframe(df.head(10))

# ---------- 2) Простая стратегия для smoke-test ----------
class SMACross(bt.Strategy):
    params = dict(fast=10, slow=20, risk=0.02)

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = self.broker.getvalue() * self.p.risk / self.data.close[0]
                self.buy(size=size)
            elif self.crossover < 0:
                size = self.broker.getvalue() * self.p.risk / self.data.close[0]
                self.sell(size=size)
        else:
            # простейший выход — обратный сигнал
            if self.position.size > 0 and self.crossover < 0:
                self.close()
            elif self.position.size < 0 and self.crossover > 0:
                self.close()

# ---------- 3) Обёртка под PandasData ----------
class PandasData(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
    )

# ---------- 4) Запуск бэктеста ----------
st.markdown("---")
st.header("Запуск бэктеста")

colA, colB, colC = st.columns(3)
with colA:
    cash = st.number_input("Начальный капитал", 1000.0, step=100.0, value=10000.0)
with colB:
    fast = st.number_input("SMA fast", 2, 200, 10)
with colC:
    slow = st.number_input("SMA slow", 2, 300, 20)

run = st.button("🚀 Запустить")

if run:
    if df is None or df.empty:
        st.error("Нет данных.")
    else:
        cerebro = bt.Cerebro()
        datafeed = PandasData(dataname=df.set_index("datetime"))
        cerebro.adddata(datafeed)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.00055)  # пример комиссии
        cerebro.addstrategy(SMACross, fast=fast, slow=slow)

        # Аналайзеры: как в TV — equity curve + basic stats
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='rets', timeframe=bt.TimeFrame.Days)

        result = cerebro.run(maxcpus=1)
        strat = result[0]
        val = cerebro.broker.getvalue()

        st.success(f"Финальный капитал: {val:.2f}")

        ta = strat.analyzers.ta.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        rets = strat.analyzers.rets.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        # короткая сводка
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total = ta.total.closed if 'total' in ta and 'closed' in ta.total else 0
            st.metric("Сделок", total)
        with c2:
            won = ta.won.total if 'won' in ta and 'total' in ta.won else 0
            wr = (won / total * 100) if total else 0
            st.metric("WinRate", f"{wr:.1f}%")
        with c3:
            st.metric("Max DD", f"{dd.max.drawdown:.2f}%")
        with c4:
            sr = sharpe.get("sharperatio", None)
            st.metric("Sharpe", f"{sr:.2f}" if sr is not None else "—")

        # график equity backtrader-овский (matplotlib)
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]
        st.pyplot(fig)
