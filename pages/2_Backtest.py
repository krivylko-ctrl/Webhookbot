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
    """Минимальный адаптер для эмуляции биржи в бэктесте."""

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
        return self.real.get_instruments_info(symbol) if self.real else None

    # ---- ордера / позиция ----
    def place_order(self, **_kwargs):
        return {"ok": True, "msg": "paper filled"}

    def update_position_stop_loss(self, symbol: str, new_sl: float):
        return True

    def modify_order(self, **_kwargs):
        return {"ok": True}


# =============== УТИЛИТЫ ДЛЯ БЭКТЕСТА ===============
@dataclass
class BtData:
    m15: pd.DataFrame
    m1: Optional[pd.DataFrame]


@st.cache_data(show_spinner=False)
def load_history(_api: PaperAPI, symbol: str, m15_limit: int, m1_limit: int, intrabar_tf: str = "1") -> BtData:
    """Грузим историю с рынка (Bybit API)."""
    m15_raw = _api.get_klines(symbol, "15", m15_limit) or []
    df15 = pd.DataFrame(m15_raw)
    if not df15.empty:
        df15 = df15.sort_values("timestamp").reset_index(drop=True)
    else:
        df15 = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df1 = pd.DataFrame()
    if m1_limit > 0:
        m1_raw = _api.get_klines(symbol, intrabar_tf, m1_limit) or []
        df1 = pd.DataFrame(m1_raw)
        if not df1.empty:
            df1 = df1.sort_values("timestamp").reset_index(drop=True)
        else:
            df1 = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    return BtData(m15=df15, m1=df1 if not df1.empty else None)


def iter_m1_between(df1: Optional[pd.DataFrame], t_from: int, t_to: int) -> List[Dict]:
    """1m свечи строго в (t_from, t_to]."""
    if df1 is None or df1.empty:
        return []
    mask = (df1["timestamp"] > t_from) & (df1["timestamp"] <= t_to)
    sub = df1.loc[mask]
    if sub.empty:
        return []
    return sub.to_dict("records")


def simulate_exits_on_m1(state: StateManager, db: Database, cfg: Config, m1: Dict):
    """Проверка срабатывания SL/TP на минутках (биржевой приоритет: SL > TP)."""
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction = pos.get("direction")
    sl = float(pos.get("stop_loss") or 0.0)
    tp = float(pos.get("take_profit") or 0.0) if cfg.use_take_profit else 0.0
    hi, lo = float(m1["high"]), float(m1["low"])

    if direction == "long":
        if sl and lo <= sl:
            state.close_position(exit_price=sl, exit_reason="SL"); return
        if tp and hi >= tp:
            state.close_position(exit_price=tp, exit_reason="TP"); return
    else:
        if sl and hi >= sl:
            state.close_position(exit_price=sl, exit_reason="SL"); return
        if tp and lo <= tp:
            state.close_position(exit_price=tp, exit_reason="TP"); return


def run_backtest(symbol: str,
                 m15_limit: int,
                 m1_limit: int,
                 init_equity: float,
                 cfg: Config,
                 price_source_for_logic: str = "last") -> Tuple[Database, StateManager, KWINStrategy]:
    """Ядро бэктеста: 15m + 1m интрабары, с точной Pine-механикой."""

    # отдельная БД под бэктест
    bt_db_path = f"kwin_backtest_{symbol}.db"
    db = Database(db_path=bt_db_path)
    state = StateManager(db)
    state.set_equity(float(init_equity))

    real_market = BybitAPI(api_key="", api_secret="", testnet=False)
    paper_api = PaperAPI(real_market_api=real_market)

    strat = KWINStrategy(cfg, api=paper_api, state_manager=state, db=db)

    intrabar_tf = str(getattr(cfg, "intrabar_tf", "1"))
    data = load_history(paper_api, symbol, m15_limit, m1_limit, intrabar_tf)
    if data.m15.empty:
        st.error("Не удалось загрузить 15m историю.")
        return db, state, strat

    m15 = data.m15.reset_index(drop=True)
    for i in range(0, len(m15) - 1):
        bar = m15.iloc[i].to_dict()
        t_curr = int(bar["timestamp"])
        t_next = int(m15.iloc[i + 1]["timestamp"])

        paper_api.set_last_price(symbol, float(bar["close"]))

        strat.on_bar_close_15m({
            "timestamp": t_curr,
            "open":  float(bar["open"]),
            "high":  float(bar["high"]),
            "low":   float(bar["low"]),
            "close": float(bar["close"]),
        })

        m1_set = iter_m1_between(data.m1, t_curr, t_next)
        for m1 in m1_set:
            paper_api.set_last_price(symbol, float(m1["close"]))

            strat.on_bar_close_1m({
                "timestamp": int(m1["timestamp"]),
                "open":  float(m1["open"]),
                "high":  float(m1["high"]),
                "low":   float(m1["low"]),
                "close": float(m1["close"]),
            })

            simulate_exits_on_m1(state, db, cfg, m1)

    return db, state, strat


# ========================= UI: центральная форма =========================
st.markdown("## ⚙️ Настройки бэктеста")

with st.form("backtest_form"):
    cfg = Config()

    # ----- верхняя строка (символ/капитал/источник цены/период) -----
    c0a, c0b, c0c, c0d = st.columns(4)
    with c0a:
        symbol = st.text_input("Символ", value=str(getattr(cfg, "symbol", "ETHUSDT")))
    with c0b:
        init_eq = st.number_input("Начальный equity ($)", min_value=10.0, max_value=1_000_000.0, value=1000.0, step=10.0)
    with c0c:
        price_src = st.selectbox("Источник цены", options=["last", "mark"], index=0)
    with c0d:
        bt_days = st.selectbox("Период бэктеста (дней)", [7, 14, 30, 60], index=2)

    st.markdown("---")

    # ====== Группа: Основные ======
    st.subheader("📌 Основные параметры")
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_reward = st.number_input("TP Risk/Reward Ratio", min_value=0.5, max_value=5.0,
                                      value=float(getattr(cfg, "risk_reward", 1.3)), step=0.1)
    with c2:
        sfp_len = st.number_input("Swing Length (SFP length)", min_value=1, max_value=10,
                                  value=int(getattr(cfg, "sfp_len", 2)), step=1)
    with c3:
        risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0,
                                   value=float(getattr(cfg, "risk_pct", 3.0)), step=0.1)

    st.markdown("---")

    # ====== Группа: Smart Trailing TP ======
    st.subheader("📌 Smart Trailing TP")
    c4, c5, c6 = st.columns(3)
    with c4:
        enable_smart_trail = st.checkbox("💚 Enable Smart Trailing TP",
                                         value=bool(getattr(cfg, "enable_smart_trail", True)))
    with c5:
        trailing_perc = st.number_input("Trailing %", min_value=0.0, max_value=5.0,
                                        value=float(getattr(cfg, "trailing_perc", 0.5)), step=0.1)
    with c6:
        trailing_offset_perc = st.number_input("Trailing Offset %", min_value=0.0, max_value=5.0,
                                               value=float(getattr(cfg, "trailing_offset_perc", 0.4)), step=0.1)

    st.markdown("---")

    # ====== Группа: ARM RR ======
    st.subheader("📌 ARM RR")
    c7, c8 = st.columns(2)
    with c7:
        use_arm_after_rr = st.checkbox("💚 Enable Arm after RR≥X",
                                       value=bool(getattr(cfg, "use_arm_after_rr", True)))
    with c8:
        arm_rr = st.number_input("Arm RR (R)", min_value=0.1, max_value=5.0,
                                 value=float(getattr(cfg, "arm_rr", 0.5)), step=0.1)

    st.markdown("---")

    # ====== Группа: Bar-Low/High Smart Trail ======
    st.subheader("📌 Use Bar-Low/High Smart Trail")
    c9, c10, c11 = st.columns(3)
    with c9:
        use_bar_trail = st.checkbox("💚 Use Bar-Low/High Smart Trail",
                                    value=bool(getattr(cfg, "use_bar_trail", True)))
    with c10:
        trail_lookback = st.number_input("Trail lookback bars", min_value=1, max_value=300,
                                         value=int(getattr(cfg, "trail_lookback", 50)), step=1)
    with c11:
        trail_buf_ticks = st.number_input("Trail buffer (ticks)", min_value=0, max_value=500,
                                          value=int(getattr(cfg, "trail_buf_ticks", 40)), step=1)

    st.markdown("---")

    # ====== Группа: Лимиты позиции ======
    st.subheader("📌 Лимиты позиции")
    c12, c13 = st.columns(2)
    with c12:
        limit_qty_enabled = st.checkbox("💚 Limit Max Position Qty",
                                        value=bool(getattr(cfg, "limit_qty_enabled", True)))
    with c13:
        max_qty_manual = st.number_input("Max Qty (ETH)", min_value=0.001, max_value=10_000.0,
                                         value=float(getattr(cfg, "max_qty_manual", 50.0)), step=0.001)

    st.markdown("---")

    # ====== Группа: Фильтры SFP ======
    st.subheader("📌 Фильтр SFP (wick + closeback)")
    c14, c15, c16 = st.columns(3)
    with c14:
        use_sfp_quality = st.checkbox("Filter: SFP quality (wick+closeback)",
                                      value=bool(getattr(cfg, "use_sfp_quality", True)))
    with c15:
        wick_min_ticks = st.number_input("SFP: min wick depth (ticks)", min_value=0, max_value=100,
                                         value=int(getattr(cfg, "wick_min_ticks", 7)), step=1)
    with c16:
        close_back_pct = st.number_input("SFP: min close-back % of wick", min_value=0.0, max_value=1.0,
                                         value=float(getattr(cfg, "close_back_pct", 1.0)), step=0.05)

    st.markdown("---")

    submitted = st.form_submit_button("🚀 Запустить бэктест", use_container_width=True)


# ========================= запуск бэктеста =========================
def _compute_limits_from_days(days: int) -> Tuple[int, int]:
    """конвертируем дни в лимиты баров (ограничим верхние лимиты API)."""
    m15_per_day = 24 * 4         # 96
    m1_per_day  = 24 * 60        # 1440
    m15_limit = min(5000, days * m15_per_day + 2)
    m1_limit  = min(5000, days * m1_per_day + 2)
    return m15_limit, m1_limit


if submitted:
    # применяем значения в конфиг (строго без изменения механики)
    cfg.symbol = symbol.strip().upper()
    cfg.risk_reward = float(risk_reward)
    cfg.sfp_len = int(sfp_len)
    cfg.risk_pct = float(risk_pct)

    cfg.enable_smart_trail = bool(enable_smart_trail)
    cfg.trailing_perc = float(trailing_perc)
    cfg.trailing_offset_perc = float(trailing_offset_perc)
    cfg.trailing_offset = float(trailing_offset_perc)  # alias

    cfg.use_arm_after_rr = bool(use_arm_after_rr)
    cfg.arm_rr = float(arm_rr)

    cfg.use_bar_trail = bool(use_bar_trail)
    cfg.trail_lookback = int(trail_lookback)
    cfg.trail_buf_ticks = int(trail_buf_ticks)

    cfg.limit_qty_enabled = bool(limit_qty_enabled)
    cfg.max_qty_manual = float(max_qty_manual)

    cfg.use_sfp_quality = bool(use_sfp_quality)
    cfg.wick_min_ticks = int(wick_min_ticks)
    cfg.close_back_pct = float(close_back_pct)

    cfg.price_for_logic = str(price_src).lower()
    cfg.intrabar_tf = "1"  # интрабар — минутки, как и раньше
    # лимиты истории из выбора периода
    m15_limit, m1_limit = _compute_limits_from_days(int(bt_days))

    with st.spinner("Грузим историю и запускаем бэктест…"):
        db, state, strat = run_backtest(
            symbol=cfg.symbol,
            m15_limit=int(m15_limit),
            m1_limit=int(m1_limit),
            init_equity=float(init_eq),
            cfg=cfg,
            price_source_for_logic=str(price_src),
        )

    st.success("Готово ✅")
    st.markdown("### 📊 Статистика")
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
    show_stats(db_path=db.db_path, days=365)

    st.markdown("### 💰 Equity Curve")
    def show_equity_curve(db: Database):
        eq = db.get_equity_history(days=365)
        if not eq:
            st.info("Нет истории equity."); return
        df = pd.DataFrame(eq)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    show_equity_curve(db)

    st.markdown("### 📋 Сделки")
    def show_trades_table(db: Database):
        trades = db.get_recent_trades(500)
        if not trades:
            st.info("Сделок нет."); return
        df = pd.DataFrame(trades)
        for col in ("entry_time","exit_time"):
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ("pnl","rr","entry_price","exit_price","quantity","qty"):
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        if "quantity" not in df.columns and "qty" in df.columns:
            df["quantity"] = df["qty"]
        cols = [c for c in ["entry_time","direction","entry_price","exit_price","quantity","pnl","rr","status","exit_reason"] if c in df.columns]
        st.dataframe(df[cols].round(6), use_container_width=True)
    show_trades_table(db)

else:
    st.info("Заполни параметры и нажми **«🚀 Запустить бэктест»**.")
