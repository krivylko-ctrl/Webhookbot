# pages/3_Backtrader.py — Backtrader бэктест твоей боевой KWINStrategy
import os
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- твои боевые модули ----
from kwin_strategy import KWINStrategy
from config import Config
from state_manager import StateManager
from database import Database

# (опционально) реальный BybitAPI только для загрузки истории
try:
    from bybit_api import BybitAPI as LiveBybitAPI
except Exception:
    LiveBybitAPI = None

st.set_page_config(page_title="Backtrader — Бэктест KWIN", page_icon="📈", layout="wide")
st.title("📈 Бэктест: KWINStrategy (Lux SFP) — через Backtrader")

# =========================== Источники данных ===========================
def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    req = ["datetime", "open", "high", "low", "close", "volume"]
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
    df = df[req]
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_from_csv_upload() -> Optional[pd.DataFrame]:
    f = st.file_uploader("Загрузи CSV (datetime, open, high, low, close, volume)", type=["csv"])
    if f is None:
        return None
    return _norm_df(pd.read_csv(f))

def load_from_csv_path(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Файл не найден: {path}")
        return None
    return _norm_df(pd.read_csv(path))

def load_from_yahoo(ticker: str, tf: str, period: str) -> Optional[pd.DataFrame]:
    import yfinance as yf
    data = yf.download(ticker, period=period, interval=tf, auto_adjust=False, progress=False)
    if data is None or data.empty:
        st.error("Yahoo: данных нет")
        return None
    data = data.reset_index()
    return _norm_df(data)

def load_from_bybit(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if LiveBybitAPI is None:
        st.error("bybit_api не найден")
        return None
    api = LiveBybitAPI("", "", testnet=False)
    # используем оконный метод, чтобы точно покрыть период
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=int(days))
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)
    rows = api.get_klines_window(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=1000)
    if not rows:
        st.error("Bybit вернул пусто")
        return None
    df = pd.DataFrame([{
        "datetime": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
        "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]),
        "volume": float(r.get("volume", 0.0))
    } for r in rows]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return df

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
    Мини-API, реализующий подмножество методов BybitAPI,
    которые вызывает KWINStrategy: place_order, update_position_stop_loss,
    get_price/get_ticker, get_instruments_info.
    Внутри — обращается к backtrader-стратегии (ctx), исполняя заявки.
    """
    def __init__(self, ctx: 'BT_KwinAdapter', symbol: str, tick_size: float, qty_step: float, min_order_qty: float):
        self.ctx = ctx
        self.symbol = symbol
        self._tick = float(tick_size)
        self._step = float(qty_step)
        self._minq = float(min_order_qty)
        # активные защитные ордера
        self._sl_order = None
        self._tp_order = None

    # ---- справочные ----
    def get_instruments_info(self, symbol: str) -> Dict:
        return {
            "priceFilter": {"tickSize": self._tick},
            "lotSizeFilter": {"qtyStep": self._step, "minOrderQty": self._minq},
        }

    def get_ticker(self, symbol: str) -> Dict:
        px = float(self.ctx.data.close[0])
        return {"lastPrice": px, "markPrice": px, "last_price": px, "mark_price": px}

    def get_price(self, symbol: str, source: str = "last") -> float:
        return float(self.ctx.data.close[0])

    # ---- торговые ----
    def place_order(self, symbol: str, side: str, orderType: str, qty: float,
                    price: Optional[float] = None, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None, order_link_id: Optional[str] = None,
                    reduce_only: bool = False, trigger_by_source: str = "mark",
                    time_in_force: Optional[str] = None, position_idx: Optional[int] = None,
                    tpsl_mode: Optional[str] = None) -> Dict:

        size = float(qty)
        # market вход
        if side.lower().startswith("b"):   # Buy = long
            main_order = self.ctx.buy(size=size)
            # защитные
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
        return {"ok": True, "order": main_order.ref}

    def update_position_stop_loss(self, symbol: str, new_sl: float, trigger_by_source: str = "mark",
                                  position_idx: Optional[int] = None) -> bool:
        # отменим старый SL и поставим новый
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
                # long → SL = sell stop
                self._sl_order = self.ctx.sell(exectype=bt.Order.Stop, price=float(new_sl), size=sz)
            else:
                # short → SL = buy stop
                self._sl_order = self.ctx.buy(exectype=bt.Order.Stop, price=float(new_sl), size=sz)
            return True
        except Exception:
            return False

class BT_KwinAdapter(bt.Strategy):
    """
    Обёртка backtrader, которая внутри создаёт твою KWINStrategy
    и прокармливает ей закрытые бары 15m.
    """
    params = dict(
        symbol="ETHUSDT",
        tick_size=0.01,
        qty_step=0.01,
        min_order_qty=0.01,
        # KWIN config-параметры (минимум, что нужно здесь)
        risk_pct=3.0,
        risk_reward=1.3,
        sl_buf_ticks=0,
        lux_swings=2,
        lux_volume_validation="none",  # по ТЗ — байпас по умолчанию
    )

    def __init__(self):
        # подготовим окружение KWIN
        cfg = Config()
        cfg.symbol = self.p.symbol
        cfg.tick_size = float(self.p.tick_size)
        cfg.qty_step = float(self.p.qty_step)
        cfg.min_order_qty = float(self.p.min_order_qty)

        # риск/TP
        cfg.risk_pct = float(self.p.risk_pct)
        cfg.risk_reward = float(self.p.risk_reward)
        cfg.use_take_profit = True

        # Lux
        cfg.lux_swings = int(self.p.lux_swings)
        cfg.lux_volume_validation = str(self.p.lux_volume_validation or "none").lower()
        cfg.sfp_len = int(cfg.lux_swings)
        cfg.sl_buf_ticks = int(self.p.sl_buf_ticks)

        # state/db
        self.db = Database(memory=True)
        self.state = StateManager(self.db)

        # backtrader api-адаптер
        self.bt_api = BTApiAdapter(
            ctx=self,
            symbol=cfg.symbol,
            tick_size=cfg.tick_size,
            qty_step=cfg.qty_step,
            min_order_qty=cfg.min_order_qty,
        )

        # создаём твою боевую стратегию
        self.kwin = KWINStrategy(cfg, api=self.bt_api, state_manager=self.state, db=self.db)

        # начальное equity = брокер value
        self.state.set_equity(float(self.broker.getvalue()))

    def next(self):
        """
        Backtrader вызывает next() на закрытии каждого бара ⇒ это и есть «закрытый бар».
        Формируем свечу под KWIN и передаём.
        """
        # обновим equity в state (для расчёта позиции)
        self.state.set_equity(float(self.broker.getvalue()))

        # формируем 15m бар (или иной TF — что в источнике)
        # timestamp возьмём как «конец бара» (pandas index + freq неизвестна, но BT хранит dt)
        dt = self.data.datetime.datetime(0)  # naive UTC в BT
        ts_ms = int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)

        candle = {
            "timestamp": ts_ms,
            "open":  float(self.data.open[0]),
            "high":  float(self.data.high[0]),
            "low":   float(self.data.low[0]),
            "close": float(self.data.close[0]),
            "volume": float(self.data.volume[0]) if hasattr(self.data, "volume") else 0.0,
        }

        # отдать бар в твою стратегию
        self.kwin.on_bar_close_15m(candle)

        # смарт-трейлинг (цену возьмёт через bt_api.get_price)
        self.kwin.process_trailing()

# =========================== UI: источники ===========================
with st.expander("Источник данных", expanded=True):
    src = st.radio("Выбери источник", ["CSV upload", "CSV из Volume (/data)", "YFinance", "Bybit API"], horizontal=True)

    df: Optional[pd.DataFrame] = None
    if src == "CSV upload":
        df = load_from_csv_upload()
    elif src == "CSV из Volume (/data)":
        path = st.text_input("Путь к CSV", "/data/eth_15m.csv")
        if st.button("Загрузить CSV"):
            df = load_from_csv_path(path)
    elif src == "YFinance":
        ticker = st.text_input("Ticker", "ETH-USD")
        tf = st.selectbox("TF", ["15m", "30m", "1h", "4h", "1d"], index=0)
        period = st.selectbox("Период", ["7d", "14d", "30d", "60d", "1y"], index=2)
        if st.button("Скачать с Yahoo"):
            df = load_from_yahoo(ticker, tf, period)
    else:
        symbol_in = st.text_input("Bybit symbol", "ETHUSDT")
        interval = st.selectbox("TF", ["1", "3", "5", "15", "30", "60", "120", "240"], index=3)
        days = st.slider("Период (дней)", 7, 180, 60)
        if st.button("Скачать с Bybit"):
            df = load_from_bybit(symbol_in, interval, days)

    if df is not None:
        st.success(f"Строк: {len(df)} | c {df['datetime'].iloc[0]} по {df['datetime'].iloc[-1]}")
        st.dataframe(df.head(10), use_container_width=True)

# =========================== Параметры и запуск ===========================
st.markdown("---")
st.header("Запуск бэктеста (KWINStrategy)")

c0, c1, c2, c3 = st.columns(4)
with c0:
    symbol = st.text_input("Символ", "ETHUSDT")
with c1:
    cash = st.number_input("Начальный капитал", 1000.0, step=100.0, value=10_000.0)
with c2:
    commission = st.number_input("Комиссия (taker, dec.)", 0.0, 0.01, 0.00055, 0.00005)
with c3:
    slippage = st.number_input("Слиппедж (dec.)", 0.0, 0.01, 0.0, 0.0001)

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

e1, e2, e3 = st.columns(3)
with e1:
    tick_size = st.number_input("tick_size", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
with e2:
    qty_step = st.number_input("qty_step", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
with e3:
    min_order_qty = st.number_input("min_order_qty", 0.0001, 10.0, 0.01, 0.0001, format="%.4f")

run = st.button("🚀 Запустить")

if run:
    if df is None or df.empty:
        st.error("Нет данных")
    else:
        cerebro = bt.Cerebro()
        datafeed = PandasData(dataname=df.set_index("datetime"))
        cerebro.adddata(datafeed)
        cerebro.broker.setcash(float(cash))
        cerebro.broker.set_slippage_perc(float(slippage))
        cerebro.broker.setcommission(commission=float(commission))

        # запускаем обёртку, которая внутри создаёт и гоняет твою KWINStrategy
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
            lux_volume_validation="none",  # важно: байпас объёма как по ТЗ
        )

        # Аналайзеры
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

        # график
        fig = cerebro.plot(style='candlestick', iplot=False, volume=False)[0][0]
        st.pyplot(fig, clear_figure=True, use_container_width=True)
