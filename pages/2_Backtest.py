# pages/3_Backtrader.py — Бэктест твоей KWINStrategy с dual-TF (15m + 1m)
import os
from typing import List, Dict, Optional, Tuple
import io

import streamlit as st
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # для привязки к оси времени

# ---- боевые модули ----
from kwin_strategy import KWINStrategy
from config import Config
from state_manager import StateManager
from database import Database

# (опционально) реальный BybitAPI для загрузки истории
try:
    from bybit_api import BybitAPI as LiveBybitAPI
except Exception:
    LiveBybitAPI = None

st.set_page_config(page_title="Backtrader — Бэктест KWIN (15m + 1m)", page_icon="📈", layout="wide")
st.title("📈 Бэктест KWINStrategy (Lux SFP) — 15m + 1m интрабар")

# =========================== утилиты загрузки ===========================
REQ_COLS = ["datetime", "open", "high", "low", "close", "volume"]

def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for a, b in [("Datetime","datetime"),("Date","datetime"),
                 ("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
        if a in df.columns and b not in df.columns:
            rename[a] = b
    if rename:
        df.rename(columns=rename, inplace=True)
    if "datetime" not in df.columns:
        raise ValueError("В источнике нет колонки 'datetime'")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[REQ_COLS]
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _df_to_list(df: pd.DataFrame) -> List[Dict]:
    return [{
        "datetime": pd.to_datetime(r.datetime),
        "open": float(r.open),
        "high": float(r.high),
        "low": float(r.low),
        "close": float(r.close),
        "volume": float(r.volume or 0.0),
    } for r in df.itertuples(index=False)]

def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Файл не найден: {path}")
        return None
    return _norm_df(pd.read_csv(path))

def load_csv_upload(label: str) -> Optional[pd.DataFrame]:
    f = st.file_uploader(label, type=["csv"])
    if not f:
        return None
    return _norm_df(pd.read_csv(f))

def load_yahoo(ticker: str, tf: str, period: str) -> Optional[pd.DataFrame]:
    import yfinance as yf
    data = yf.download(ticker, period=period, interval=tf, auto_adjust=False, progress=False)
    if data is None or data.empty:
        st.error("Yahoo: данных нет")
        return None
    data = data.reset_index()
    return _norm_df(data)

def load_bybit(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if LiveBybitAPI is None:
        st.error("bybit_api не найден")
        return None
    api = LiveBybitAPI("", "", testnet=False)
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=int(days))
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)
    rows = api.get_klines_window(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=1000)
    if not rows:
        st.error("Bybit: пусто")
        return None
    df = pd.DataFrame([{
        "datetime": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
        "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]),
        "volume": float(r.get("volume", 0.0))
    } for r in rows]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return df

def load_bybit_dual(symbol: str, main_interval: str, ltf_interval: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Грузим оба ТФ с Bybit за один и тот же период (по часам)."""
    df_main = load_bybit(symbol, main_interval, days)
    df_ltf  = load_bybit(symbol, ltf_interval, days)
    return df_main, df_ltf

# =========================== Feed ===========================
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

# =========================== Backtrader↔KWIN мост ===========================
class BTApiAdapter:
    """
    Мини-API для KWINStrategy: place_order/update_position_stop_loss/get_price/get_ticker/get_instruments_info.
    Работает поверх Backtrader (ctx).
    """
    def __init__(self, ctx: 'BT_KwinAdapter', symbol: str, tick_size: float, qty_step: float, min_order_qty: float):
        self.ctx = ctx
        self.symbol = symbol
        self._tick = float(tick_size)
        self._step = float(qty_step)
        self._minq = float(min_order_qty)
        self._sl_order = None
        self._tp_order = None

    def get_instruments_info(self, symbol: str) -> Dict:
        return {
            "priceFilter": {"tickSize": self._tick},
            "lotSizeFilter": {"qtyStep": self._step, "minOrderQty": self._minq},
        }

    def get_ticker(self, symbol: str) -> Dict:
        px = float(self.ctx.data1.close[0]) if self.ctx.data1_present else float(self.ctx.data0.close[0])
        return {"lastPrice": px, "markPrice": px, "last_price": px, "mark_price": px}

    def get_price(self, symbol: str, source: str = "last") -> float:
        return float(self.ctx.data1.close[0]) if self.ctx.data1_present else float(self.ctx.data0.close[0])

    def place_order(self, symbol: str, side: str, orderType: str, qty: float,
                    price: Optional[float] = None, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None, order_link_id: Optional[str] = None,
                    reduce_only: bool = False, trigger_by_source: str = "mark",
                    time_in_force: Optional[str] = None, position_idx: Optional[int] = None,
                    tpsl_mode: Optional[str] = None) -> Dict:

        size = float(max(qty, 0.0))
        if side.lower().startswith("b"):   # Buy = long
            main_order = self.ctx.buy(size=size)
            if stop_loss is not None:
                self._sl_order = self.ctx.sell(exectype=bt.Order.Stop,  price=float(stop_loss), size=size)
            if take_profit is not None:
                self._tp_order = self.ctx.sell(exectype=bt.Order.Limit, price=float(take_profit), size=size)
        else:                               # Sell = short
            main_order = self.ctx.sell(size=size)
            if stop_loss is not None:
                self._sl_order = self.ctx.buy(exectype=bt.Order.Stop,  price=float(stop_loss), size=size)
            if take_profit is not None:
                self._tp_order = self.ctx.buy(exectype=bt.Order.Limit, price=float(take_profit), size=size)
        return {"ok": True, "order": getattr(main_order, "ref", None)}

    def update_position_stop_loss(self, symbol: str, new_sl: float, trigger_by_source: str = "mark",
                                  position_idx: Optional[int] = None) -> bool:
        try:
            if self._sl_order is not None:
                try:
                    self.ctx.broker.cancel(self._sl_order)
                except Exception:
                    pass
                self._sl_order = None
            sz = abs(float(self.ctx.position.size))
            if sz <= 0:
                return True
            if self.ctx.position.size > 0:
                self._sl_order = self.ctx.sell(exectype=bt.Order.Stop, price=float(new_sl), size=sz)
            else:
                self._sl_order = self.ctx.buy(exectype=bt.Order.Stop, price=float(new_sl), size=sz)
            return True
        except Exception:
            return False

class BT_KwinAdapter(bt.Strategy):
    """
    Обёртка backtrader, которая создаёт твою KWINStrategy и кормит её:
      - data0: 15m (основной) -> on_bar_close_15m
      - data1: 1m (интрабар)  -> on_bar_close_1m  (если есть)
    """
    params = dict(
        symbol="ETHUSDT",
        tick_size=0.01,
        qty_step=0.01,
        min_order_qty=0.01,
        # KWIN config
        risk_pct=3.0,
        risk_reward=1.3,
        sl_buf_ticks=0,
        lux_swings=2,
        lux_volume_validation="none",
        use_intrabar=True,
        intrabar_tf="1",
        intrabar_pull_limit=2000,
        price_for_logic="last",
    )

    def __init__(self):
        self.data1_present = (len(self.datas) > 1)

        cfg = Config()
        cfg.symbol = self.p.symbol
        cfg.tick_size = float(self.p.tick_size)
        cfg.qty_step = float(self.p.qty_step)
        cfg.min_order_qty = float(self.p.min_order_qty)

        cfg.risk_pct = float(self.p.risk_pct)
        cfg.risk_reward = float(self.p.risk_reward)
        cfg.use_take_profit = True

        cfg.lux_swings = int(self.p.lux_swings)
        cfg.sfp_len = int(cfg.lux_swings)
        cfg.lux_volume_validation = str(self.p.lux_volume_validation or "none").lower()

        cfg.use_intrabar = bool(self.p.use_intrabar)
        cfg.intrabar_tf = str(self.p.intrabar_tf)
        cfg.intrabar_pull_limit = int(self.p.intrabar_pull_limit)

        cfg.price_for_logic = str(self.p.price_for_logic).lower()

        self.db = Database(memory=True)
        self.state = StateManager(self.db)

        self.bt_api = BTApiAdapter(
            ctx=self,
            symbol=cfg.symbol,
            tick_size=cfg.tick_size,
            qty_step=cfg.qty_step,
            min_order_qty=cfg.min_order_qty,
        )
        self.kwin = KWINStrategy(cfg, api=self.bt_api, state_manager=self.state, db=self.db)
        self.state.set_equity(float(self.broker.getvalue()))

        self._last_dt0 = None
        self._last_dt1 = None

    def _bar_to_candle(self, data) -> Dict:
        dt = data.datetime.datetime(0)  # naive UTC
        ts_ms = int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)
        return {
            "timestamp": ts_ms,
            "open":  float(data.open[0]),
            "high":  float(data.high[0]),
            "low":   float(data.low[0]),
            "close": float(data.close[0]),
            "volume": float(getattr(data, "volume", [0])[0] or 0.0),
        }

    def next(self):
        self.state.set_equity(float(self.broker.getvalue()))

        if self.data1_present:
            dt1 = self.data1.datetime.datetime(0)
            if self._last_dt1 != dt1:
                self._last_dt1 = dt1
                candle_1m = self._bar_to_candle(self.data1)
                self.kwin.on_bar_close_1m(candle_1m)

        dt0 = self.data0.datetime.datetime(0)
        if self._last_dt0 != dt0:
            self._last_dt0 = dt0
            candle_15m = self._bar_to_candle(self.data0)
            self.kwin.on_bar_close_15m(candle_15m)

        self.kwin.process_trailing()

# =========================== UI: источники ===========================
with st.expander("Источник данных", expanded=True):
    src = st.radio("Выбери источник", ["CSV (15m) + опц. CSV(1m)", "YFinance (только 15m+)", "Bybit API (15m+1m)"], horizontal=True)

    df15: Optional[pd.DataFrame] = None
    df1: Optional[pd.DataFrame] = None

    if src.startswith("CSV"):
        df15 = load_csv_upload("CSV для 15m")
        use_ltf = st.checkbox("Добавить LTF CSV (1m)", False)
        if use_ltf:
            df1 = load_csv_upload("CSV для 1m")
        if df15 is not None:
            st.success(f"15m: {len(df15)} строк | {df15['datetime'].iloc[0]} → {df15['datetime'].iloc[-1]}")
            st.dataframe(df15.head(5), use_container_width=True)
        if df1 is not None:
            st.info(f"1m: {len(df1)} строк | {df1['datetime'].iloc[0]} → {df1['datetime'].iloc[-1]}")
    elif src.startswith("YFinance"):
        import yfinance as yf  # noqa
        ticker = st.text_input("Ticker", "ETH-USD")
        tf = st.selectbox("TF", ["15m", "30m", "1h", "4h", "1d"], index=0)
        period = st.selectbox("Период", ["7d", "14d", "30d", "60d", "1y"], index=2)
        if st.button("Скачать с Yahoo"):
            df15 = load_yahoo(ticker, tf, period)
            df1 = None
        if df15 is not None:
            st.success(f"{tf}: {len(df15)} строк | {df15['datetime'].iloc[0]} → {df15['datetime'].iloc[-1]}")
            st.dataframe(df15.head(5), use_container_width=True)
    else:
        symbol_in = st.text_input("Bybit symbol", "ETHUSDT")
        days = st.slider("Период (дней)", 7, 180, 60)
        main_tf = st.selectbox("Main TF", ["15","30","60"], index=0)
        ltf_tf = st.selectbox("LTF (интрабар)", ["1","3","5"], index=0)
        if st.button("Скачать с Bybit"):
            df15, df1 = load_bybit_dual(symbol_in, main_tf, ltf_tf, days)
        if df15 is not None:
            st.success(f"{main_tf}m: {len(df15)} строк | {df15['datetime'].iloc[0]} → {df15['datetime'].iloc[-1]}")
            if df1 is not None:
                st.info(f"{ltf_tf}m: {len(df1)} строк | {df1['datetime'].iloc[0]} → {df1['datetime'].iloc[-1]}")
            st.dataframe(df15.head(5), use_container_width=True)

# =========================== Параметры и запуск ===========================
st.markdown("---")
st.header("Запуск бэктеста (KWINStrategy) — dual-TF")

c0, c1, c2, c3, c4 = st.columns(5)
with c0:
    symbol = st.text_input("Символ", "ETHUSDT")
with c1:
    cash = st.number_input("Начальный капитал", 1000.0, step=100.0, value=10_000.0)
with c2:
    commission = st.number_input("Комиссия (taker, dec.)", 0.0, 0.01, 0.00055, 0.00005)
with c3:
    slippage = st.number_input("Слиппедж (dec.)", 0.0, 0.01, 0.0, 0.0001)
with c4:
    cheat_on_close = st.checkbox("Исполнять по close текущего бара (Cheat-On-Close)", True)

st.subheader("Параметры стратегии")
d1, d2, d3, d4 = st.columns(4)
with d1:
    swings = st.number_input("Lux Swings (len)", 1, 20, 2)
with d2:
    risk_pct = st.number_input("Risk %", 0.1, 10.0, 3.0, 0.1)
with d3:
    rr = st.number_input("Risk/Reward (R)", 0.5, 5.0, 1.3, 0.1)
with d4:
    sl_buf_ticks = st.number_input("SL buf (ticks)", 0, 200, 0, 1)

e1, e2, e3, e4 = st.columns(4)
with e1:
    tick_size = st.number_input("tick_size", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
with e2:
    qty_step = st.number_input("qty_step", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
with e3:
    min_order_qty = st.number_input("min_order_qty", 0.0001, 10.0, 0.01, 0.0001, format="%.4f")
with e4:
    price_for_logic = st.selectbox("Источник цены для логики", ["last", "mark"], index=0)

# Оверлеи на графике
st.subheader("Оверлеи")
ov1, ov2, ov3 = st.columns(3)
with ov1:
    show_trade_markers = st.checkbox("Показывать стрелочки вход/выход", True)
with ov2:
    show_sl_tp = st.checkbox("Показывать линии SL/TP", True)
with ov3:
    show_labels = st.checkbox("Подписи (R, PnL)", True)

run = st.button("🚀 Запустить")

class PandasData15(PandasData): pass
class PandasData1(PandasData): pass

# ---------- Рисовалка стрелок + SL/TP + подписи ----------
def _plot_trade_markers(ax, df15: pd.DataFrame, trades: List[Dict],
                        *, show_sl_tp: bool = True, show_labels: bool = True) -> None:
    """Рисует стрелочки входов/выходов, SL/TP линии и подписи (R, PnL) из KWIN Database."""
    if not trades or df15 is None or df15.empty:
        return

    # баровая ширина в днях для линий SL/TP
    dn = mdates.date2num(pd.to_datetime(df15["datetime"]).dt.tz_convert(None).to_pydatetime())
    if len(dn) >= 2:
        bar_w = float(np.median(np.diff(dn)))
    else:
        bar_w = 1/96.0  # ~15m

    mid_price_by_dt = {pd.to_datetime(d).tz_convert(None).to_pydatetime(): float(c)
                       for d, c in zip(df15["datetime"], (df15["high"]+df15["low"])/2)}

    xs_in, ys_in, colors_in, markers_in = [], [], [], []
    xs_out, ys_out, colors_out, markers_out = [], [], [], []

    for tr in trades:
        try:
            side = (tr.get("direction") or "").lower()
            e_ts = tr.get("entry_time")
            x_ts = tr.get("exit_time")
            e_px = tr.get("entry_price")
            x_px = tr.get("exit_price")
            sl   = tr.get("stop_loss")
            tp   = tr.get("take_profit")
            pnl  = tr.get("pnl")
            rr   = tr.get("rr")

            if not e_ts:
                continue

            # точки входа/выхода
            edt = pd.to_datetime(e_ts, utc=True, errors="coerce").tz_convert(None).to_pydatetime()
            xedt = mdates.date2num(edt)
            y_entry = float(e_px) if e_px is not None else float(mid_price_by_dt.get(edt))

            if side == "long":
                xs_in.append(xedt); ys_in.append(y_entry); colors_in.append("#10B981"); markers_in.append("^")
            elif side == "short":
                xs_in.append(xedt); ys_in.append(y_entry); colors_in.append("#EF4444"); markers_in.append("v")

            exit_xnum = None
            if x_ts and x_px is not None:
                xdt = pd.to_datetime(x_ts, utc=True, errors="coerce").tz_convert(None).to_pydatetime()
                exit_xnum = mdates.date2num(xdt)
                if side == "long":
                    xs_out.append(exit_xnum); ys_out.append(float(x_px)); colors_out.append("#10B981"); markers_out.append("v")
                elif side == "short":
                    xs_out.append(exit_xnum); ys_out.append(float(x_px)); colors_out.append("#EF4444"); markers_out.append("^")

            # SL/TP линии (горизонтальные) от входа до выхода (или на 2 бара вперёд)
            if show_sl_tp and (sl is not None or tp is not None):
                x_start = xedt - 0.15*bar_w  # чуть левее входа
                x_end   = exit_xnum if exit_xnum is not None else (xedt + 2*bar_w)

                if sl is not None:
                    ax.hlines(y=float(sl), xmin=x_start, xmax=x_end,
                              colors="#EF4444", linestyles="dashed", linewidth=1.2, zorder=3)
                if tp is not None:
                    ax.hlines(y=float(tp), xmin=x_start, xmax=x_end,
                              colors="#10B981", linestyles="dashed", linewidth=1.2, zorder=3)

            # подписи возле входа
            if show_labels:
                label_parts = []
                if rr is not None:
                    try:
                        label_parts.append(f"R={float(rr):.2f}")
                    except Exception:
                        pass
                if pnl is not None:
                    try:
                        label_parts.append(f"PnL={float(pnl):.2f}")
                    except Exception:
                        pass
                if label_parts:
                    text = " • ".join(label_parts)
                    # смещение подписи: над long, под short
                    dy = (0.004 if side == "long" else -0.004) * y_entry
                    ax.annotate(text, (xedt, y_entry),
                                xytext=(0, 12 if side == "long" else -14),
                                textcoords="offset points",
                                ha="center", va="bottom" if side == "long" else "top",
                                fontsize=8, color="#111827",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="#D1FAE5" if side == "long" else "#FEE2E2",
                                          ec="#10B981" if side == "long" else "#EF4444",
                                          lw=0.8, alpha=0.9))
        except Exception:
            continue

    # входы — сплошные маркеры
    for x, y, c, m in zip(xs_in, ys_in, colors_in, markers_in):
        ax.scatter(x, y, marker=m, s=70, c=c, edgecolors="black", linewidths=0.6, zorder=5)

    # выходы — полые маркеры
    for x, y, c, m in zip(xs_out, ys_out, colors_out, markers_out):
        ax.scatter(x, y, marker=m, s=90, facecolors="white", edgecolors=c, linewidths=1.2, zorder=5)

    # легенда
    import matplotlib.lines as mlines
    lg_long_in  = mlines.Line2D([], [], color="#10B981", marker="^", linestyle="None", markersize=8, label="Long entry")
    lg_long_out = mlines.Line2D([], [], color="#10B981", marker="v", markerfacecolor="white", linestyle="None", markersize=8, label="Long exit")
    lg_sh_in    = mlines.Line2D([], [], color="#EF4444", marker="v", linestyle="None", markersize=8, label="Short entry")
    lg_sh_out   = mlines.Line2D([], [], color="#EF4444", marker="^", markerfacecolor="white", linestyle="None", markersize=8, label="Short exit")
    ax.legend(handles=[lg_long_in, lg_long_out, lg_sh_in, lg_sh_out], loc="upper left")

if run:
    if df15 is None or df15.empty:
        st.error("Нет данных 15m.")
    else:
        cerebro = bt.Cerebro()
        cerebro.broker.set_coc(bool(cheat_on_close))

        data0 = PandasData15(dataname=df15.set_index("datetime"))
        cerebro.adddata(data0)  # data0 = 15m

        data1 = None
        if df1 is not None and not df1.empty:
            data1 = PandasData1(dataname=df1.set_index("datetime"))
            cerebro.adddata(data1)  # data1 = 1m

        cerebro.broker.setcash(float(cash))
        cerebro.broker.set_slippage_perc(float(slippage))
        cerebro.broker.setcommission(commission=float(commission))

        cerebro.addstrategy(
            BT_KwinAdapter,
            symbol=symbol,
            tick_size=tick_size,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            risk_pct=risk_pct,
            risk_reward=rr,
            sl_buf_ticks=sl_buf_ticks,
            lux_swings=swings,
            lux_volume_validation="none",
            use_intrabar=(data1 is not None),
            intrabar_tf="1",
            price_for_logic=price_for_logic,
        )

        # Аналайзеры
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='rets', timeframe=bt.TimeFrame.Days)

        result = cerebro.run(maxcpus=1)
        strat: BT_KwinAdapter = result[0]
        final_val = cerebro.broker.getvalue()
        st.success(f"Финальный капитал: {final_val:.2f}")

        ta = strat.analyzers.ta.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        cA,cB,cC,cD = st.columns(4)
        with cA:
            total = ta.total.closed if 'total' in ta and 'closed' in ta.total else 0
            st.metric("Сделок (закрытых)", total or 0)
        with cB:
            won = ta.won.total if 'won' in ta and 'total' in ta.won else 0
            wr = (won/total*100) if total else 0
            st.metric("WinRate", f"{wr:.1f}%")
        with cC:
            st.metric("Max DD", f"{getattr(dd.max, 'drawdown', 0):.2f}%")
        with cD:
            sr = sharpe.get("sharperatio", None)
            st.metric("Sharpe", f"{sr:.2f}" if sr is not None else "—")

        # график Backtrader
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]

        # ---- стрелочки/SL-TP/подписи из твоей БД ----
        try:
            trades = strat.kwin.db.get_all_trades() if hasattr(strat, "kwin") else []
            ax_price = fig.axes[0] if fig.axes else None
            if trades and ax_price is not None and show_trade_markers:
                _plot_trade_markers(ax_price, df15, trades,
                                    show_sl_tp=show_sl_tp, show_labels=show_labels)
        except Exception as e:
            st.warning(f"Не удалось наложить метки сделок: {e}")

        st.pyplot(fig, clear_figure=True, use_container_width=True)

        # ===================== Экспорт сделок / логов из твоей БД =====================
        st.markdown("### 📋 Сделки (из KWIN Database)")
        try:
            trades = strat.kwin.db.get_all_trades() if hasattr(strat, "kwin") else []
            if trades:
                df_tr = pd.DataFrame(trades)
                st.dataframe(df_tr, use_container_width=True)
                csv_buf = io.StringIO()
                df_tr.to_csv(csv_buf, index=False)
                st.download_button("⬇️ Экспорт сделок CSV", data=csv_buf.getvalue(), file_name="trades_kwin.csv", mime="text/csv")
            else:
                st.info("Сделок нет.")
        except Exception as e:
            st.warning(f"Не удалось получить сделки: {e}")

        st.markdown("### 🧾 Логи стратегии")
        try:
            logs = strat.kwin.db.get_logs(500) if hasattr(strat, "kwin") else []
            if logs:
                df_lg = pd.DataFrame(logs)
                st.dataframe(df_lg, use_container_width=True)
                csv_buf2 = io.StringIO()
                df_lg.to_csv(csv_buf2, index=False)
                st.download_button("⬇️ Экспорт логов CSV", data=csv_buf2.getvalue(), file_name="logs_kwin.csv", mime="text/csv")
            else:
                st.caption("Логи пусты.")
        except Exception as e:
            st.warning(f"Не удалось получить логи: {e}")
