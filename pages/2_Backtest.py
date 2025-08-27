# pages/3_Backtrader.py — Бэктест KWINStrategy (15m + 1m, Bybit-only) + стрелочки вход/выход

import os
from typing import List, Dict, Optional, Tuple
import io

import streamlit as st
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- боевые модули ----
from kwin_strategy import KWINStrategy
from config import Config
from state_manager import StateManager
from database import Database

# реальный BybitAPI для загрузки истории
try:
    from bybit_api import BybitAPI as LiveBybitAPI
except Exception:
    LiveBybitAPI = None

st.set_page_config(page_title="Backtrader — Бэктест KWIN (15m + 1m)", page_icon="📈", layout="wide")
st.title("📈 Бэктест KWINStrategy — 15m + 1m (Bybit API)")

# -------- persist датафреймы между перезапусками Streamlit --------
if "data_15m" not in st.session_state:
    st.session_state["data_15m"] = None
if "data_1m" not in st.session_state:
    st.session_state["data_1m"] = None

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

# ---------- чанковая загрузка с Bybit ----------
_MINUTES = {"1":1,"3":3,"5":5,"15":15,"30":30,"60":60}
def _tf_ms(interval: str) -> int:
    return int(_MINUTES.get(str(interval), 1)) * 60_000

def _rows_to_df(rows: List[Dict]) -> pd.DataFrame:
    def norm_ts(v):
        if v is None:
            return None
        v = int(v)
        if v < 1_000_000_000_000:  # sec -> ms
            v *= 1000
        return v
    out = []
    for r in rows:
        ts = norm_ts(r.get("timestamp") or r.get("open_time"))
        if not ts:
            continue
        out.append({
            "datetime": pd.to_datetime(ts, unit="ms", utc=True),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume") or r.get("turnover") or 0.0),
        })
    if not out:
        return pd.DataFrame(columns=REQ_COLS)
    df = pd.DataFrame(out).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return df

def _fetch_bybit_range(symbol: str, interval: str, start_ms: int, end_ms: int, max_per_call: int = 1000) -> pd.DataFrame:
    """Качаем все бары в [start_ms, end_ms] кусками, объединяем (без обрезки по 1000 баров)."""
    if LiveBybitAPI is None:
        st.error("bybit_api не найден")
        return pd.DataFrame(columns=REQ_COLS)
    api = LiveBybitAPI("", "", testnet=False)

    tfms = _tf_ms(interval)
    cursor = start_ms
    all_rows: List[Dict] = []
    last_seen_ms = None

    while cursor < end_ms:
        approx_window = tfms * max_per_call
        chunk_end = min(end_ms, cursor + approx_window - 1)

        # предпочтительно window-метод
        rows = []
        try:
            if hasattr(api, "get_klines_window"):
                rows = api.get_klines_window(symbol, interval, start_ms=cursor, end_ms=chunk_end, limit=max_per_call) or []
            else:
                rows = api.get_klines(symbol, interval, max_per_call) or []
        except Exception:
            rows = []

        if not rows:
            cursor += tfms
            continue

        for r in rows:
            ts = r.get("timestamp") or r.get("open_time")
            if ts is None:
                continue
            ts = int(ts)
            if ts < 1_000_000_000_000:
                ts *= 1000
            if start_ms <= ts <= end_ms and ((last_seen_ms is None) or ts > last_seen_ms):
                all_rows.append(r)
                last_seen_ms = ts

        cursor = (int(last_seen_ms) + tfms) if last_seen_ms is not None else (cursor + approx_window)

    return _rows_to_df(all_rows)

def _utc_now():
    """UTC-aware pandas.Timestamp."""
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        return now.tz_localize("UTC")
    return now.tz_convert("UTC")

def load_bybit(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    end = _utc_now()
    start = end - pd.Timedelta(days=int(days))
    df = _fetch_bybit_range(symbol, str(interval), int(start.timestamp()*1000), int(end.timestamp()*1000))
    return df if not df.empty else None

def load_bybit_dual(symbol: str, main_interval: str, ltf_interval: str, days: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Грузим оба ТФ с Bybit за один и тот же период (полное покрытие)."""
    end = _utc_now()
    start = end - pd.Timedelta(days=int(days))
    s_ms, e_ms = int(start.timestamp()*1000), int(end.timestamp()*1000)
    df_main = _fetch_bybit_range(symbol, str(main_interval), s_ms, e_ms)
    df_ltf  = _fetch_bybit_range(symbol, str(ltf_interval), s_ms, e_ms)
    return (df_main if not df_main.empty else None,
            df_ltf  if not df_ltf.empty  else None)

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
    """Мини-API для KWINStrategy поверх Backtrader (без реальных BT-ордеров)."""
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

    # никакие реальные ордера в BT не отправляем — KWIN сама ведёт учёт
    def place_order(self, *args, **kwargs) -> Dict:
        return {"ok": True, "order": None}

    def update_position_stop_loss(self, *args, **kwargs) -> bool:
        return True

class BT_KwinAdapter(bt.Strategy):
    """Backtrader-обёртка, которая кормит твою KWINStrategy 15m и 1m данными."""
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
        # при первом проходе выравниваем локальную equity
        if self.state.get_equity() is None:
            self.state.set_equity(float(self.broker.getvalue()))

        if self.data1_present:
            dt1 = self.data1.datetime.datetime(0)
            if self._last_dt1 != dt1:
                self._last_dt1 = dt1
                self.kwin.on_bar_close_1m(self._bar_to_candle(self.data1))

        dt0 = self.data0.datetime.datetime(0)
        if self._last_dt0 != dt0:
            self._last_dt0 = dt0
            self.kwin.on_bar_close_15m(self._bar_to_candle(self.data0))

# =========================== Источник данных (Bybit только) ===========================
with st.expander("Источник данных (Bybit API)", expanded=True):
    symbol_in = st.text_input("Bybit symbol", "ETHUSDT")
    days = st.slider("Период (дней)", 7, 180, 60)
    main_tf = st.selectbox("Main TF", ["15","30","60"], index=0)
    ltf_tf = st.selectbox("LTF (интрабар)", ["1","3","5"], index=0)

    col_dl, col_clr = st.columns([1, 1])
    with col_dl:
        if st.button("Скачать с Bybit"):
            with st.spinner("Качаем историю с Bybit..."):
                df15, df1 = load_bybit_dual(symbol_in, main_tf, ltf_tf, days)
            st.session_state["data_15m"] = df15
            st.session_state["data_1m"] = df1
            if df15 is None:
                st.error("Bybit вернул пусто по 15m.")
            else:
                st.success("История загружена.")
    with col_clr:
        if st.button("🧹 Очистить загруженные данные"):
            st.session_state["data_15m"] = None
            st.session_state["data_1m"] = None
            st.success("Данные очищены.")

    # Показываем, что сейчас лежит в сессии
    df15 = st.session_state["data_15m"]
    df1  = st.session_state["data_1m"]
    if df15 is not None:
        start_dt, end_dt = df15["datetime"].iloc[0], df15["datetime"].iloc[-1]
        st.success(f"{main_tf}m: {len(df15)} строк | {start_dt} → {end_dt}")
        st.dataframe(df15.head(5), use_container_width=True)
        if df1 is not None and not df1.empty:
            s1, e1 = df1["datetime"].iloc[0], df1["datetime"].iloc[-1]
            st.info(f"{ltf_tf}m: {len(df1)} строк | {s1} → {e1}")
        # быстрый sanity-check
        exp_bars = int((pd.Timedelta(days=int(days)).total_seconds() // (_tf_ms(main_tf)/1000)))
        if len(df15) < 0.9 * exp_bars:
            st.warning("Покрытие меньше ожидаемого — лимиты API или пропуски.")

# =========================== Параметры и запуск ===========================
st.markdown("---")
st.header("Запуск бэктеста")

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
    cheat_on_close = st.checkbox("Cheat-On-Close (исполнять по close текущего бара)", True)

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

# Оверлеи
st.subheader("Оверлеи")
show_markers = st.checkbox("Показывать стрелочки вход/выход", True)

run = st.button("🚀 Запустить")

class PandasData15(PandasData): pass
class PandasData1(PandasData): pass

# ---------- Рисовалка стрелок поверх графика ----------
def _plot_trade_markers(ax, df15: pd.DataFrame, trades: List[Dict]) -> None:
    if not trades or df15 is None or df15.empty:
        return
    mid_price_by_dt = {pd.to_datetime(d).to_pydatetime(): float(c)
                       for d, c in zip(df15["datetime"], (df15["high"]+df15["low"])/2)}

    xs_in, ys_in, colors_in, markers_in = [], [], [], []
    xs_out, ys_out, colors_out, markers_out = [], [], [], []

    for tr in trades:
        try:
            ets = tr.get("entry_time")
            epx = tr.get("entry_price")
            side = (tr.get("direction") or "").lower()
            if ets:
                edt = pd.to_datetime(ets, utc=True, errors="coerce").tz_convert(None).to_pydatetime()
                x = mdates.date2num(edt)
                y = float(epx) if epx else float(mid_price_by_dt.get(edt))
                if side == "long":
                    xs_in.append(x); ys_in.append(y); colors_in.append("#10B981"); markers_in.append("^")
                elif side == "short":
                    xs_in.append(x); ys_in.append(y); colors_in.append("#EF4444"); markers_in.append("v")

            xts = tr.get("exit_time")
            xpx = tr.get("exit_price")
            if xts and xpx is not None:
                xdt = pd.to_datetime(xts, utc=True, errors="coerce").tz_convert(None).to_pydatetime()
                xnum = mdates.date2num(xdt)
                yv = float(xpx)
                if side == "long":
                    xs_out.append(xnum); ys_out.append(yv); colors_out.append("#10B981"); markers_out.append("v")
                elif side == "short":
                    xs_out.append(xnum); ys_out.append(yv); colors_out.append("#EF4444"); markers_out.append("^")
        except Exception:
            continue

    for x, y, c, m in zip(xs_in, ys_in, colors_in, markers_in):
        ax.scatter(x, y, marker=m, s=70, c=c, edgecolors="black", linewidths=0.6, zorder=5)
    for x, y, c, m in zip(xs_out, ys_out, colors_out, markers_out):
        ax.scatter(x, y, marker=m, s=90, facecolors="white", edgecolors=c, linewidths=1.2, zorder=5)

    import matplotlib.lines as mlines
    lg_long_in  = mlines.Line2D([], [], color="#10B981", marker="^", linestyle="None", markersize=8, label="Long entry")
    lg_long_out = mlines.Line2D([], [], color="#10B981", marker="v", markerfacecolor="white", linestyle="None", markersize=8, label="Long exit")
    lg_sh_in    = mlines.Line2D([], [], color="#EF4444", marker="v", linestyle="None", markersize=8, label="Short entry")
    lg_sh_out   = mlines.Line2D([], [], color="#EF4444", marker="^", markerfacecolor="white", linestyle="None", markersize=8, label="Short exit")
    ax.legend(handles=[lg_long_in, lg_long_out, lg_sh_in, lg_sh_out], loc="upper left")

# ---------- Хелперы метрик KWIN ----------
def _compute_kwin_metrics(trades: List[Dict], initial_cash: float) -> Dict[str, float]:
    """Считаем PnL, PnL%, WinRate, MaxDD, Кол-во сделок по данным KWIN DB."""
    if not trades:
        return {"pnl": 0.0, "pnl_pct": 0.0, "wr": 0.0, "count": 0, "maxdd_pct": 0.0}

    # берём ТОЛЬКО закрытые сделки
    rows = []
    for t in trades:
        try:
            if t.get("exit_price") is None:
                continue
            direction = (t.get("direction") or "").lower()
            size = float(t.get("size") or t.get("qty") or 0.0)
            entry = float(t.get("entry_price") or 0.0)
            exitp = float(t.get("exit_price") or 0.0)
            if size <= 0 or entry <= 0 or exitp <= 0:
                continue
            etime = pd.to_datetime(t.get("exit_time"), utc=True, errors="coerce")
            pnl = (exitp - entry) * size if direction == "long" else (entry - exitp) * size
            rows.append({"exit_time": etime, "pnl": float(pnl)})
        except Exception:
            continue

    if not rows:
        return {"pnl": 0.0, "pnl_pct": 0.0, "wr": 0.0, "count": 0, "maxdd_pct": 0.0}

    df = pd.DataFrame(rows).sort_values("exit_time").reset_index(drop=True)
    total_pnl = float(df["pnl"].sum())
    count = int(len(df))
    wins = int((df["pnl"] > 0).sum())
    wr = (wins / count * 100.0) if count else 0.0

    # equity curve
    eq = float(initial_cash) + df["pnl"].cumsum()
    roll_max = eq.cummax()
    dd = (eq - roll_max).min()  # отрицательное число
    maxdd_abs = float(dd) if pd.notnull(dd) else 0.0
    maxdd_pct = (abs(maxdd_abs) / float(roll_max.max()) * 100.0) if float(roll_max.max()) > 0 else 0.0

    pnl_pct = (total_pnl / float(initial_cash) * 100.0) if float(initial_cash) > 0 else 0.0
    return {"pnl": total_pnl, "pnl_pct": pnl_pct, "wr": wr, "count": count, "maxdd_pct": maxdd_pct}

if run:
    # забираем данные из session_state (устойчиво между перезапусками)
    df15 = st.session_state.get("data_15m")
    df1  = st.session_state.get("data_1m")

    if df15 is None or df15.empty:
        st.error("Нет данных 15m. Сначала нажми «Скачать с Bybit».")
        st.stop()
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
            intrabar_tf=str(1 if data1 is not None else main_tf),  # если нет LTF — не критично
            price_for_logic=price_for_logic,
        )

        # Аналайзеры BT (справочные; реальные метрики ниже считаем из KWIN DB)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='rets', timeframe=bt.TimeFrame.Days)

        result = cerebro.run(maxcpus=1)
        strat: BT_KwinAdapter = result[0]

        # ---------- МЕТРИКИ по KWIN DB ----------
        try:
            trades_kwin = strat.kwin.db.get_all_trades() if hasattr(strat, "kwin") else []
        except Exception:
            trades_kwin = []

        metr = _compute_kwin_metrics(trades_kwin, float(cash))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PnL (USDT)", f"{metr['pnl']:.2f}")
        m2.metric("PnL %", f"{metr['pnl_pct']:.2f}%")
        m3.metric("WinRate", f"{metr['wr']:.1f}%")
        m4.metric("Max DD", f"{metr['maxdd_pct']:.2f}%")
        m5.metric("Сделок", f"{int(metr['count'])}")

        # ---------- график Backtrader ----------
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]

        # стрелочки входов/выходов из БД KWIN
        if show_markers:
            try:
                ax_price = fig.axes[0] if fig.axes else None
                if trades_kwin and ax_price is not None:
                    _plot_trade_markers(ax_price, df15, trades_kwin)
            except Exception as e:
                st.warning(f"Не удалось наложить метки сделок: {e}")

        st.pyplot(fig, clear_figure=True, use_container_width=True)

        # ---------- таблицы ----------
        st.markdown("### 📋 Сделки (из KWIN Database)")
        try:
            if trades_kwin:
                df_tr = pd.DataFrame(trades_kwin)
                # удобный PnL по сделке (gross)
                if {"entry_price","exit_price","size","direction"}.issubset(df_tr.columns):
                    pnl = (df_tr["exit_price"] - df_tr["entry_price"]) * df_tr["size"]
                    pnl[df_tr["direction"].str.lower()=="short"] *= -1
                    df_tr["pnl"] = pnl
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
            logs = strat.kwin.db.get_logs(1000) if hasattr(strat, "kwin") else []
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
