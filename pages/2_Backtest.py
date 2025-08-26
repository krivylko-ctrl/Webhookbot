# pages/3_Backtrader.py
import os
import io
import datetime as dt
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# опционально: твой API для Bybit
try:
    from bybit_api import BybitAPI
except Exception:
    BybitAPI = None

st.set_page_config(page_title="Backtrader — Бэктест", page_icon="📈", layout="wide")
st.title("📈 Backtrader — быстрый бэктест (с Lux SFP)")

# =========================================================
# Источники данных
# =========================================================
def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    req = ["datetime", "open", "high", "low", "close", "volume"]
    df = df.copy()
    # переименовываем если нужно
    rename = {}
    for a,b in [("Datetime","datetime"),("Date","datetime"),
                ("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
        if a in df.columns and b not in df.columns:
            rename[a] = b
    if rename:
        df.rename(columns=rename, inplace=True)

    if "datetime" not in df.columns:
        raise ValueError("В CSV нет колонки 'datetime'")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df = df[req]
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_from_csv_upload() -> Optional[pd.DataFrame]:
    f = st.file_uploader("Загрузи CSV (datetime, open, high, low, close, volume)", type=["csv"])
    if f is None:
        return None
    df = pd.read_csv(f)
    return _norm_df(df)

def load_from_csv_path(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Файл не найден: {path}")
        return None
    return _norm_df(pd.read_csv(path))

def load_from_yahoo(ticker: str, tf: str, period: str) -> Optional[pd.DataFrame]:
    import yfinance as yf
    data = yf.download(ticker, period=period, interval=tf, auto_adjust=False, progress=False)
    if data is None or data.empty:
        st.error("Yahoo вернул пусто.")
        return None
    data = data.reset_index()
    return _norm_df(data)

def load_from_bybit(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if BybitAPI is None:
        st.error("bybit_api не найден в окружении.")
        return None
    api = BybitAPI("", "", testnet=False)
    # простой путь: берём limit*chunks назад
    # у Bybit лимиты разные; для 15m 1000 свечей ~ 10 дней. Скачаем кусками.
    tf_ms = int(interval) * 60_000
    end = int(pd.Timestamp.utcnow().value // 10**6)
    start = end - days * 24 * 60 * 60 * 1000
    # если у тебя есть get_klines_window — лучше через него; иначе get_klines циклом назад
    rows: List[Dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + tf_ms * 1000)
        chunk = api.get_klines_window(symbol, interval, start_ms=cursor, end_ms=chunk_end, limit=1000) \
                if hasattr(api, "get_klines_window") else api.get_klines(symbol, interval, 1000)
        if not chunk:
            break
        for r in chunk:
            ts = int(r.get("timestamp") or r.get("open_time") or 0)
            if ts and ts < 1_000_000_000_000:
                ts *= 1000
            rows.append({
                "datetime": pd.to_datetime(ts, unit="ms", utc=True),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("volume") or 0.0)
            })
        # защитимся от дубликатов и шагаем
        cursor += tf_ms * 900
        if len(rows) > 0 and len(rows) > days*24*60//int(interval)+50:
            break
    if not rows:
        st.error("Bybit вернул пусто.")
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return df

# =========================================================
# Backtrader обёртка
# =========================================================
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

# =========================================================
# Lux SFP (BT-стратегия)
# =========================================================
class LuxSFP_BT(bt.Strategy):
    params = dict(
        swings=2,           # len у пивотов
        risk_pct=3.0,       # % от капитала в риск
        rr=1.3,             # RR для тейка (опционально)
        use_tp=True,        # ставить TP по RR
        tick_size=0.01,     # шаг цены
        sl_buf_ticks=0,     # буфер SL в тиках
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_high  = self.datas[0].high
        self.data_low   = self.datas[0].low

        # массивы для ручного поиска пивотов
        self.h = self.data_high
        self.l = self.data_low
        self.c = self.data_close

        self.can_enter_on_bar = True  # не более 1 входа на бар

    # --- вспомогательные ---
    def _pivot_high_value(self, left: int, right: int = 1):
        n = left + right + 1
        if len(self) < n:
            return None
        highs = [self.h[-i] for i in range(n)]
        center = highs[right]
        return center if center == max(highs) else None

    def _pivot_low_value(self, left: int, right: int = 1):
        n = left + right + 1
        if len(self) < n:
            return None
        lows = [self.l[-i] for i in range(n)]
        center = lows[right]
        return center if center == min(lows) else None

    # --- позиционирование ---
    def _risk_qty(self, entry, sl):
        equity = self.broker.getvalue()
        risk_amt = equity * (self.p.risk_pct / 100.0)
        stop = abs(entry - sl)
        if stop <= 0:
            return 0.0
        qty = risk_amt / stop
        return max(0.0, qty)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            pass

    def next(self):
        # сбрасываем флаг на каждом новом баре
        self.can_enter_on_bar = True

        # если позиция открыта — ничего, (выходы/трейлинг можно дописать)
        if self.position:
            return

        L = int(self.p.swings)
        ph = self._pivot_high_value(L, 1)
        pl = self._pivot_low_value(L, 1)

        hi, lo, cl, op = self.h[0], self.l[0], self.c[0], self.datas[0].open[0]

        # ---- bearish SFP: high > sw & close < sw
        if ph is not None:
            sw = ph
            if hi > sw and cl < sw and self.can_enter_on_bar:
                sl = sw if hi <= sw else hi
                sl += self.p.sl_buf_ticks * self.p.tick_size
                entry = cl
                qty = self._risk_qty(entry, sl)
                if qty > 0:
                    # в backtrader size — количество единиц базового актива
                    tp_price = entry - (sl - entry) * self.p.rr if self.p.use_tp else None
                    o = self.buy(size=-qty)  # short = отрицательный size
                    # стоп и TP
                    self.sell(exectype=bt.Order.Stop, price=sl, size=o.size)  # стоп на увеличение (покрыть шорт)
                    if tp_price:
                        self.buy(exectype=bt.Order.Limit, price=tp_price, size=-o.size)
                    self.can_enter_on_bar = False

        # ---- bullish SFP: low < sw & close > sw
        if pl is not None and self.can_enter_on_bar and not self.position:
            sw = pl
            if lo < sw and cl > sw:
                sl = sw if lo >= sw else lo
                sl -= self.p.sl_buf_ticks * self.p.tick_size
                entry = cl
                qty = self._risk_qty(entry, sl)
                if qty > 0:
                    o = self.buy(size=qty)
                    tp_price = entry + (entry - sl) * self.p.rr if self.p.use_tp else None
                    self.sell(exectype=bt.Order.Stop, price=sl, size=o.size)  # стоп
                    if tp_price:
                        self.sell(exectype=bt.Order.Limit, price=tp_price, size=o.size)
                    self.can_enter_on_bar = False

# =========================================================
# UI — выбор источника
# =========================================================
with st.expander("Источник данных", expanded=True):
    src = st.radio("Выбери источник", ["CSV upload", "CSV из Volume (/data)", "YFinance", "Bybit API"], horizontal=True)

    df: Optional[pd.DataFrame] = None
    if src == "CSV upload":
        df = load_from_csv_upload()
    elif src == "CSV из Volume (/data)":
        path = st.text_input("Путь к CSV (в контейнере)", "/data/eth_15m.csv")
        if st.button("Загрузить CSV"):
            df = load_from_csv_path(path)
    elif src == "YFinance":
        import yfinance as yf  # гарантируем импорт
        ticker = st.text_input("Ticker", "ETH-USD")
        tf = st.selectbox("TF", ["15m", "30m", "1h", "4h", "1d"], index=0)
        period = st.selectbox("Период", ["7d", "14d", "30d", "60d", "1y"], index=2)
        if st.button("Скачать с Yahoo"):
            df = load_from_yahoo(ticker, tf, period)
    else:
        symbol = st.text_input("Bybit symbol", "ETHUSDT")
        interval = st.selectbox("TF", ["1", "3", "5", "15", "30", "60"], index=3)
        days = st.slider("Период (дней)", 7, 120, 30)
        if st.button("Скачать с Bybit"):
            df = load_from_bybit(symbol, interval, days)

    if df is not None:
        st.success(f"Строк: {len(df)} | c {df['datetime'].iloc[0]} по {df['datetime'].iloc[-1]}")
        st.dataframe(df.head(10), use_container_width=True)

# =========================================================
# Запуск бэктеста
# =========================================================
st.markdown("---")
st.header("Запуск бэктеста")

col0, col1, col2, col3 = st.columns(4)
with col0:
    strategy_name = st.selectbox("Стратегия", ["Lux SFP", "SMA Cross"], index=0)
with col1:
    cash = st.number_input("Начальный капитал", 1000.0, step=100.0, value=10_000.0)
with col2:
    commission = st.number_input("Комиссия (taker, dec.)", 0.0, 0.01, 0.00055, 0.00005)
with col3:
    slippage = st.number_input("Слиппедж (dec.)", 0.0, 0.01, 0.0, 0.0001)

st.subheader("Параметры стратегии")
c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    swings = st.number_input("Swings (len)", 1, 20, 2)
with c2:
    risk_pct = st.number_input("Risk %", 0.1, 10.0, 3.0, 0.1)
with c3:
    rr = st.number_input("R", 0.5, 5.0, 1.3, 0.1)
with c4:
    tick_size = st.number_input("tick_size", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
with c5:
    sl_buf_ticks = st.number_input("SL buf (ticks)", 0, 200, 0, 1)
with c6:
    use_tp = st.checkbox("Use TP (RR)", True)

run = st.button("🚀 Запустить")

# доп. простая стратегия для сравнения
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

if run:
    if df is None or df.empty:
        st.error("Нет данных.")
    else:
        cerebro = bt.Cerebro()
        datafeed = PandasData(dataname=df.set_index("datetime"))
        cerebro.adddata(datafeed)
        cerebro.broker.setcash(cash)
        cerebro.broker.set_slippage_perc(slippage)
        cerebro.broker.setcommission(commission=commission)

        if strategy_name == "Lux SFP":
            cerebro.addstrategy(
                LuxSFP_BT,
                swings=swings, risk_pct=risk_pct, rr=rr,
                use_tp=use_tp, tick_size=tick_size, sl_buf_ticks=sl_buf_ticks
            )
        else:
            cerebro.addstrategy(SMACross)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='rets', timeframe=bt.TimeFrame.Days)

        result = cerebro.run(maxcpus=1)
        strat = result[0]
        final_val = cerebro.broker.getvalue()
        st.success(f"Финальный капитал: {final_val:.2f}")

        ta = strat.analyzers.ta.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        cA,cB,cC,cD = st.columns(4)
        with cA:
            total = ta.total.closed if 'total' in ta and 'closed' in ta.total else 0
            st.metric("Сделок", total)
        with cB:
            won = ta.won.total if 'won' in ta and 'total' in ta.won else 0
            wr = (won/total*100) if total else 0
            st.metric("WinRate", f"{wr:.1f}%")
        with cC:
            st.metric("Max DD", f"{getattr(dd.max, 'drawdown', 0):.2f}%")
        with cD:
            sr = sharpe.get("sharperatio", None)
            st.metric("Sharpe", f"{sr:.2f}" if sr is not None else "—")

        # график (matplotlib)
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]
        st.pyplot(fig, clear_figure=True, use_container_width=True)
