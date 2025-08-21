# pages/03_backtest.py
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ===== PYTHONPATH (если запускается из подпапки) =====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ===== ТВОИ МОДУЛИ =====
from config import Config
from database import Database
from state_manager import StateManager
from kwin_strategy import KWINStrategy
from analytics import TradingAnalytics
from bybit_api import BybitAPI

st.set_page_config(
    page_title="KWIN Bot — Бэктест",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Бэктест KWIN Strategy")

# =============== Бумажный API-адаптер (ТОЛЬКО для бэктеста) ===============
class PaperAPI:
    """
    Минимальный адаптер, чтобы стратегия работала без реального биржевого API:
    - подтверждает place_order (стратегия сама пишет trade в БД и set_position в state)
    - обновляет SL (возвращает ok, чтобы стратегия обновила state)
    - get_price / get_ticker отдаются из текущего контекста (текущий бар)
    - get_klines делегирует в реальный BybitAPI (только для загрузки истории)
    """
    def __init__(self, real_market_api: Optional[BybitAPI] = None):
        self.real = real_market_api
        self._last_price: Dict[str, float] = {}

    # ---- маркет-данные ----
    def get_klines(self, symbol: str, interval: str, limit: int = 200):
        if self.real:
            return self.real.get_klines(symbol, interval, limit)
        return []

    def set_last_price(self, symbol: str, price: float):
        self._last_price[symbol] = float(price)

    def get_price(self, symbol: str, source: str = "last") -> float:
        return float(self._last_price.get(symbol, 0.0))

    def get_ticker(self, symbol: str) -> Dict:
        p = float(self._last_price.get(symbol, 0.0))
        return {"symbol": symbol, "lastPrice": p, "markPrice": p}

    def get_instruments_info(self, symbol: str):
        # делегируем если есть
        return self.real.get_instruments_info(symbol) if self.real else None

    # ---- ордера / позиция ----
    def place_order(self, **_kwargs):
        # достаточно вернуть правдивый объект, т.к. стратегия сама фиксирует позицию/сделку в БД
        return {"ok": True, "msg": "paper filled"}

    def update_position_stop_loss(self, symbol: str, new_sl: float):
        # сообщаем стратегии "ок", чтобы она локально обновила SL в state_manager
        return True

    def modify_order(self, **_kwargs):
        # не нужен для бэктеста
        return {"ok": True}


# =============== УТИЛИТЫ ДЛЯ БЭКТЕСТА ===============
@dataclass
class BtData:
    m15: pd.DataFrame                # 15m свечи (возрастающий порядок)
    m1: Optional[pd.DataFrame]       # 1m свечи (может быть None/пусто)


@st.cache_data(show_spinner=False)
def load_history(api: PaperAPI, symbol: str, m15_limit: int, m1_limit: int, intrabar_tf: str = "1") -> BtData:
    """Грузим историю с рынка (через реальный BybitAPI внутри paper-адаптера).
    Возвращаем pandas-таблицы в возрастающем порядке времени.
    """
    m15_raw = api.get_klines(symbol, "15", m15_limit) or []
    df15 = pd.DataFrame(m15_raw)
    if not df15.empty:
        df15 = df15.sort_values("timestamp").reset_index(drop=True)
    else:
        df15 = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df1 = pd.DataFrame()
    if m1_limit > 0:
        m1_raw = api.get_klines(symbol, intrabar_tf, m1_limit) or []
        df1 = pd.DataFrame(m1_raw)
        if not df1.empty:
            df1 = df1.sort_values("timestamp").reset_index(drop=True)
        else:
            df1 = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    return BtData(m15=df15, m1=df1 if not df1.empty else None)


def iter_m1_between(df1: Optional[pd.DataFrame], t_from: int, t_to: int) -> List[Dict]:
    """Отдать 1m свечи строго в (t_from, t_to] (оба в ms)."""
    if df1 is None or df1.empty:
        return []
    # фильтрация по диапазону
    mask = (df1["timestamp"] > t_from) & (df1["timestamp"] <= t_to)
    sub = df1.loc[mask]
    if sub.empty:
        return []
    return sub.to_dict("records")


def simulate_exits_on_m1(state: StateManager, db: Database, cfg: Config, m1: Dict):
    """
    Если позиция открыта — проверяем срабатывание SL/TP на минутке (как биржа):
    long:  low<=SL -> SL,  high>=TP -> TP
    short: high>=SL -> SL,  low<=TP  -> TP
    Приоритет: SL срабатывает, если достигнут (как защитный), затем TP. Это приближённо, но честно.
    """
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction = pos.get("direction")
    sl = float(pos.get("stop_loss") or 0.0)
    tp = float(pos.get("take_profit") or 0.0) if getattr(cfg, "use_take_profit", True) else 0.0

    hi = float(m1["high"]); lo = float(m1["low"])

    # SL приоритетнее TP (как “защитный”)
    if direction == "long":
        if sl and lo <= sl:
            state.close_position(exit_price=sl, exit_reason="SL")
            return
        if tp and hi >= tp:
            state.close_position(exit_price=tp, exit_reason="TP")
            return
    else:
        if sl and hi >= sl:
            state.close_position(exit_price=sl, exit_reason="SL")
            return
        if tp and lo <= tp:
            state.close_position(exit_price=tp, exit_reason="TP")
            return


def run_backtest(symbol: str,
                 m15_limit: int,
                 m1_limit: int,
                 init_equity: float,
                 price_source_for_logic: str = "last") -> Tuple[Database, StateManager, KWINStrategy]:
    """
    Ядро бэктеста: прогоняем стратегию по 15m закрытиям, между ними — 1m для трейлинга и срабатываний SL/TP.
    НИКАКОЙ логики из Pine не меняем — только эмуляция биржи (fills + SL/TP).
    """

    # ==== конфиг ====
    cfg = Config()
    cfg.symbol = symbol
    cfg.price_for_logic = price_source_for_logic
    # respect текущие параметры (sfp_len, wick_min_ticks, close_back_pct, use_take_profit, trailing_perc, arm_rr и т.д.)

    # отдельная БД под бэктест (чтобы не мешать live)
    bt_db_path = f"kwin_backtest_{symbol}.db"
    db = Database(db_path=bt_db_path)
    state = StateManager(db)
    state.set_equity(float(init_equity))

    # реальный маркет-API для свечей (паблик), paper-обёртка для ордеров/SL
    real_market = BybitAPI(api_key="", api_secret="", testnet=False)
    paper_api = PaperAPI(real_market_api=real_market)

    # стратегия как есть
    strat = KWINStrategy(cfg, api=paper_api, state_manager=state, db=db)

    # загрузка истории
    intrabar_tf = str(getattr(cfg, "intrabar_tf", "1"))
    data = load_history(paper_api, symbol, m15_limit=m15_limit, m1_limit=m1_limit, intrabar_tf=intrabar_tf)
    if data.m15.empty:
        st.error("Не удалось загрузить 15m историю. Проверь подключение к Bybit (паблик).")
        return db, state, strat

    # основной прогон: по 15m закрытым барам (в возрастающем порядке)
    m15 = data.m15.reset_index(drop=True)
    # чтобы корректно отделять интервалы, возьмём t_{i} и t_{i+1} (последний интервал интрабара пропустим)
    for i in range(0, len(m15) - 1):
        bar = m15.iloc[i].to_dict()
        t_curr = int(bar["timestamp"])
        t_next = int(m15.iloc[i + 1]["timestamp"])

        # выставим текущую "рыночную" цену (=close 15m бара)
        paper_api.set_last_price(symbol, float(bar["close"]))

        # закрытие 15m бара — стратегия проверит вход/ARM и т.д.
        strat.on_bar_close_15m({
            "timestamp": t_curr,
            "open":  float(bar["open"]),
            "high":  float(bar["high"]),
            "low":   float(bar["low"]),
            "close": float(bar["close"]),
        })

        # теперь проигрываем 1m бары между этими 15m закрытиями (для трейлинга/срабатываний SL/TP)
        m1_set = iter_m1_between(data.m1, t_curr, t_next)
        for m1 in m1_set:
            # обновляем "цену рынка" для логики
            paper_api.set_last_price(symbol, float(m1["close"]))

            # минута закрылась — стратегия подтянет трейл, если позиция есть
            strat.on_bar_close_1m({
                "timestamp": int(m1["timestamp"]),
                "open":  float(m1["open"]),
                "high":  float(m1["high"]),
                "low":   float(m1["low"]),
                "close": float(m1["close"]),
            })

            # биржевое срабатывание SL/TP на этой минуте
            simulate_exits_on_m1(state, db, cfg, m1)

    # финальная синхронизация equity
    # (если осталась открытая позиция — считаем, что её не закрывали в бэктесте)
    return db, state, strat


# =============== UI ===============
with st.sidebar:
    st.header("⚙️ Параметры бэктеста")
    cfg = Config()
    symbol = st.text_input("Символ", value=str(getattr(cfg, "symbol", "ETHUSDT")))
    init_eq = st.number_input("Начальный equity ($)", min_value=10.0, max_value=1_000_000.0, value=1000.0, step=10.0)
    # лимиты истории (зависят от ограничений публичного API)
    m15_limit = st.slider("15m бары (limit)", min_value=200, max_value=5000, value=1500, step=100)
    m1_limit  = st.slider("Intrabar 1m (limit)", min_value=0, max_value=5000, value=int(getattr(cfg, "intrabar_pull_limit", 1500)), step=100)
    price_src = st.selectbox("Источник цены для логики", options=["last", "mark"], index=0)

    run_btn = st.button("🚀 Запустить бэктест", use_container_width=True)


def show_equity_curve(db: Database):
    eq = db.get_equity_history(days=365)
    if not eq:
        st.info("Нет истории equity.")
        return
    df = pd.DataFrame(eq)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines", name="Equity"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def show_trades_table(db: Database):
    trades = db.get_recent_trades(500)
    if not trades:
        st.info("Сделок нет.")
        return
    df = pd.DataFrame(trades)
    for col in ("entry_time","exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("pnl","rr","entry_price","exit_price","quantity","qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # унификация quantity
    if "quantity" not in df.columns and "qty" in df.columns:
        df["quantity"] = df["qty"]
    cols = [c for c in ["entry_time","direction","entry_price","exit_price","quantity","pnl","rr","status","exit_reason"] if c in df.columns]
    st.dataframe(df[cols].round(6), use_container_width=True)


def show_stats(db_path: str, days: int = 365):
    analytics = TradingAnalytics(db_path=db_path)
    stats = analytics.get_comprehensive_stats(days_back=days) or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trades", stats.get("total_trades", 0))
        st.metric("WinRate", f"{(stats.get('winrate') or {}).get('total', 0)}%")
    with c2:
        pnl = stats.get("pnl") or {}
        st.metric("Net PnL", f"{pnl.get('total_pnl', 0):.2f}")
        st.metric("Profit Factor", f"{pnl.get('profit_factor', 0):.2f}")
    with c3:
        rr = stats.get("risk_reward") or {}
        st.metric("Avg R:R", f"{rr.get('avg_rr', 0):.2f}")
        st.metric("Max R", f"{rr.get('max_rr', 0):.2f}")
    with c4:
        dd = stats.get("drawdown") or {}
        st.metric("Max DD", f"{dd.get('max_drawdown', 0):.2f}%")
        st.metric("Curr DD", f"{dd.get('current_drawdown', 0):.2f}%")

    st.caption(f"Обновлено: {stats.get('updated_at','—')}")

# =============== RUN ===============
if run_btn:
    with st.spinner("Грузим историю и запускаем бэктест…"):
        db, state, strat = run_backtest(
            symbol=symbol.strip().upper(),
            m15_limit=int(m15_limit),
            m1_limit=int(m1_limit),
            init_equity=float(init_eq),
            price_source_for_logic=str(price_src),
        )

    st.success("Готово ✅")

    # Метрики
    st.markdown("### 📊 Статистика")
    show_stats(db_path=db.db_path, days=365)

    # Эквити
    st.markdown("### 💰 Equity Curve")
    show_equity_curve(db)

    # Сделки
    st.markdown("### 📋 Сделки")
    show_trades_table(db)

else:
    st.info("Задай параметры слева и нажми **«Запустить бэктест»**.")
