# pages/3_Backtrader.py
import os
import io
import datetime as dt

import streamlit as st
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Backtrader — Backtest", page_icon="📈", layout="wide")
st.title("📈 Backtrader — быстрый бэктест")

# ---------- 1) Источник данных ----------
with st.expander("Источник данных", expanded=True):
    src = st.radio("Выбери источник", ["CSV upload", "CSV из Volume (/data)", "YFinance"], horizontal=True)
    df = None

    if src == "CSV upload":
        f = st.file_uploader("Загрузи CSV (нужны колонки: datetime, open, high, low, close, volume)", type=["csv"])
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
                    "Datetime": "datetime", "Date": "datetime",
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume"
                }, inplace=True)
                df = data[["datetime", "open", "high", "low", "close", "volume"]]

    if df is not None:
        # нормализация времени
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df.dropna(subset=["datetime"], inplace=True)
            # Backtrader любит naive-индекс
            df["datetime"] = df["datetime"].dt.tz_convert(None)
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
        st.success(f"Данных строк: {len(df)}")
        st.dataframe(df.head(10))

# ---------- 2) Стратегии ----------
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
            if self.position.size > 0 and self.crossover < 0:
                self.close()
            elif self.position.size < 0 and self.crossover > 0:
                self.close()

class LuxSFP(bt.Strategy):
    """
    Lux SFP (как в Pine), упрощённая валидация:
    - Пивот len, right=1 (берём high[-1]/low[-1] как swing)
    - Bear: high[0] > swingHigh и close[0] < swingHigh  -> SHORT по close
    - Bull: low[0]  < swingLow  и close[0] > swingLow   -> LONG  по close
    - SL = экстремум текущей свечи ± буфер_тиков * tick_size
    - TP = RR * риск (через bracket-orders)
    """
    params = dict(
        swings=2,            # len из Lux
        risk_pct=0.03,       # риск от капитала
        rr=1.3,              # risk:reward
        tick_size=0.01,
        sl_buf_ticks=0,      # буфер на SL в тиках
        print_signals=True,
        commission=0.00055,
    )

    def __init__(self):
        self.orders = []
        self.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        # комиссии
        self.broker.setcommission(commission=self.p.commission)

    # helper: размер позиции от риска и стопа
    def _calc_size(self, entry, sl):
        risk_amt = self.broker.getvalue() * float(self.p.risk_pct)
        stop = abs(entry - sl)
        if stop <= 0:
            return 0.0
        size = risk_amt / stop
        return max(0.0, size)

    def next(self):
        # нужно минимум swings+2 баров, чтобы существовал bar[-1] как центр пивота
        if len(self.data) < (int(self.p.swings) + 2):
            return

        L = int(self.p.swings)

        # --- определяем pivot-high/low на баре -1 (как ta.pivothigh/low len,1)
        highs = [self.data.high[-i] for i in range(L+2)]   # 0..L+1
        lows  = [self.data.low[-i]  for i in range(L+2)]

        swingH = highs[1] if highs[1] == max(highs) else None
        swingL = lows[1]  if lows[1]  == min(lows)  else None

        hi0, lo0, cl0 = float(self.data.high[0]), float(self.data.low[0]), float(self.data.close[0])
        ts = self.data.datetime.datetime(0)

        # --- Bearish SFP
        if swingH is not None and hi0 > swingH and cl0 < swingH and not self.position:
            tick = float(self.p.tick_size)
            sl = hi0 + float(self.p.sl_buf_ticks) * tick
            size = self._calc_size(cl0, sl)
            if size > 0:
                # TP от RR
                tp = cl0 - (sl - cl0) * float(self.p.rr)
                o = self.sell_bracket(size=size, price=None, exectype=bt.Order.Market,
                                      stopprice=sl, limitprice=tp)
                self.orders.extend(o)
                if self.p.print_signals:
                    print(f"[{ts}] SFP BEAR entry @ {cl0:.6f} SL={sl:.6f} TP={tp:.6f} size={size:.6f}")

        # --- Bullish SFP
        if swingL is not None and lo0 < swingL and cl0 > swingL and not self.position:
            tick = float(self.p.tick_size)
            sl = lo0 - float(self.p.sl_buf_ticks) * tick
            size = self._calc_size(cl0, sl)
            if size > 0:
                tp = cl0 + (cl0 - sl) * float(self.p.rr)
                o = self.buy_bracket(size=size, price=None, exectype=bt.Order.Market,
                                     stopprice=sl, limitprice=tp)
                self.orders.extend(o)
                if self.p.print_signals:
                    print(f"[{ts}] SFP BULL entry @ {cl0:.6f} SL={sl:.6f} TP={tp:.6f} size={size:.6f}")

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

colS, colA, colB, colC = st.columns([1.2,1,1,1])
with colS:
    strat_name = st.selectbox("Стратегия", ["Lux SFP", "SMA Cross"], index=0)
with colA:
    cash = st.number_input("Начальный капитал", 1000.0, step=100.0, value=10_000.0)
with colB:
    commission = st.number_input("Комиссия (taker, dec.)", value=0.00055, min_value=0.0, max_value=0.01, step=0.00005)
with colC:
    slippage = st.number_input("Слиппедж (dec.)", value=0.0, min_value=0.0, max_value=0.01, step=0.0001)

if strat_name == "SMA Cross":
    colF, colL = st.columns(2)
    with colF:
        fast = st.number_input("SMA fast", 2, 200, 10)
    with colL:
        slow = st.number_input("SMA slow", 2, 300, 20)
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        swings = st.number_input("Swings (len)", 1, 20, 2)
    with c2:
        risk_pct = st.number_input("Risk %", 0.1, 20.0, 3.0, step=0.1)
    with c3:
        rr = st.number_input("R:R", 0.5, 10.0, 1.3, step=0.1)
    with c4:
        tick_size = st.number_input("tick_size", 0.0001, 10.0, 0.01, step=0.0001, format="%.4f")
    with c5:
        sl_buf_ticks = st.number_input("SL buf (ticks)", 0, 200, 0, step=1)

run = st.button("🚀 Запустить")

if run:
    if df is None or df.empty:
        st.error("Нет данных.")
    else:
        cerebro = bt.Cerebro()
        # slippage (простая модель, если нужно)
        if slippage and slippage > 0:
            cerebro.broker.set_slippage_fixed(slippage=0.0)  # можно заменить на perc
        cerebro.broker.setcash(float(cash))
        cerebro.broker.setcommission(commission=float(commission))

        datafeed = PandasData(dataname=df.set_index("datetime"))
        cerebro.adddata(datafeed)

        if strat_name == "SMA Cross":
            cerebro.addstrategy(SMACross, fast=fast, slow=slow, risk=0.02)
        else:
            cerebro.addstrategy(
                LuxSFP,
                swings=int(swings),
                risk_pct=float(risk_pct)/100.0,
                rr=float(rr),
                tick_size=float(tick_size),
                sl_buf_ticks=int(sl_buf_ticks),
                commission=float(commission),
            )

        # Аналайзеры (похожий набор на TV)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='rets', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)

        result = cerebro.run(maxcpus=1)
        strat = result[0]
        val = cerebro.broker.getvalue()

        st.success(f"Финальный капитал: {val:.2f}")

        ta = strat.analyzers.ta.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        rets = strat.analyzers.rets.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total = ta.total.closed if 'total' in ta and 'closed' in ta.total else 0
            st.metric("Сделок", total)
        with c2:
            won = ta.won.total if 'won' in ta and 'total' in ta.won else 0
            wr = (won / total * 100) if total else 0
            st.metric("WinRate", f"{wr:.1f}%")
        with c3:
            st.metric("Max DD", f"{getattr(dd.max, 'drawdown', 0):.2f}%")
        with c4:
            sr = sharpe.get("sharperatio", None)
            st.metric("Sharpe", f"{sr:.2f}" if sr is not None else "—")

        # график
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]
        st.pyplot(fig)
