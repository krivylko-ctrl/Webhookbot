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
    def __init__(self, real_market_api: Optional[BybitAPI] = None):
        self.real = real_market_api
        self._last_price: Dict[str, float] = {}

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
    """Используем _api → Streamlit не будет пытаться его хэшировать"""
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
    if df1 is None or df1.empty:
        return []
    mask = (df1["timestamp"] > t_from) & (df1["timestamp"] <= t_to)
    sub = df1.loc[mask]
    if sub.empty:
        return []
    return sub.to_dict("records")

def simulate_exits_on_m1(state: StateManager, db: Database, cfg: Config, m1: Dict):
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return
    direction = pos.get("direction")
    sl = float(pos.get("stop_loss") or 0.0)
    tp = float(pos.get("take_profit") or 0.0) if getattr(cfg, "use_take_profit", True) else 0.0
    hi = float(m1["high"]); lo = float(m1["low"])
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

def run_backtest(symbol: str, m15_limit: int, m1_limit: int, init_equity: float, price_source_for_logic: str = "last") -> Tuple[Database, StateManager, KWINStrategy]:
    cfg = Config()
    cfg.symbol = symbol
    cfg.price_for_logic = price_source_for_logic
    db = Database(db_path=f"kwin_backtest_{symbol}.db")
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
        t_curr = int(bar["timestamp"]); t_next = int(m15.iloc[i+1]["timestamp"])
        paper_api.set_last_price(symbol, float(bar["close"]))
        strat.on_bar_close_15m({k: float(bar[k]) if k!="timestamp" else t_curr for k in ["timestamp","open","high","low","close"]})
        m1_set = iter_m1_between(data.m1, t_curr, t_next)
        for m1 in m1_set:
            paper_api.set_last_price(symbol, float(m1["close"]))
            strat.on_bar_close_1m({k: float(m1[k]) if k!="timestamp" else int(m1["timestamp"]) for k in ["timestamp","open","high","low","close"]})
            simulate_exits_on_m1(state, db, cfg, m1)
    return db, state, strat

# =============== UI ===============
with st.sidebar:
    st.header("⚙️ Параметры бэктеста")
    cfg = Config()
    symbol = st.text_input("Символ", value=str(getattr(cfg,"symbol","ETHUSDT")))
    init_eq = st.number_input("Начальный equity ($)", 10.0, 1_000_000.0, 1000.0, 10.0)
    m15_limit = st.slider("15m бары (limit)", 200, 5000, 1500, 100)
    m1_limit = st.slider("Intrabar 1m (limit)", 0, 5000, int(getattr(cfg,"intrabar_pull_limit",1500)), 100)
    price_src = st.selectbox("Источник цены", ["last","mark"], 0)
    run_btn = st.button("🚀 Запустить бэктест", use_container_width=True)

def show_equity_curve(db: Database):
    eq = db.get_equity_history(days=365)
    if not eq: st.info("Нет истории equity."); return
    df = pd.DataFrame(eq); df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df["timestamp"],y=df["equity"],mode="lines",name="Equity"))
    fig.update_layout(height=340,margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig,use_container_width=True)

def show_trades_table(db: Database):
    trades=db.get_recent_trades(500)
    if not trades: st.info("Сделок нет."); return
    df=pd.DataFrame(trades)
    for col in ("entry_time","exit_time"): 
        if col in df: df[col]=pd.to_datetime(df[col],errors="coerce")
    for col in ("pnl","rr","entry_price","exit_price","quantity","qty"): 
        if col in df: df[col]=pd.to_numeric(df[col],errors="coerce")
    if "quantity" not in df and "qty" in df: df["quantity"]=df["qty"]
    cols=[c for c in ["entry_time","direction","entry_price","exit_price","quantity","pnl","rr","status","exit_reason"] if c in df]
    st.dataframe(df[cols].round(6), use_container_width=True)

def show_stats(db_path: str, days: int = 365):
    analytics=TradingAnalytics(db_path=db_path); stats=analytics.get_comprehensive_stats(days_back=days) or {}
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Trades",stats.get("total_trades",0)); st.metric("WinRate",f"{(stats.get('winrate') or {}).get('total',0)}%")
    with c2: pnl=stats.get("pnl") or {}; st.metric("Net PnL",f"{pnl.get('total_pnl',0):.2f}"); st.metric("Profit Factor",f"{pnl.get('profit_factor',0):.2f}")
    with c3: rr=stats.get("risk_reward") or {}; st.metric("Avg R:R",f"{rr.get('avg_rr',0):.2f}"); st.metric("Max R",f"{rr.get('max_rr',0):.2f}")
    with c4: dd=stats.get("drawdown") or {}; st.metric("Max DD",f"{dd.get('max_drawdown',0):.2f}%"); st.metric("Curr DD",f"{dd.get('current_drawdown',0):.2f}%")
    st.caption(f"Обновлено: {stats.get('updated_at','—')}")

if run_btn:
    with st.spinner("Грузим историю и запускаем бэктест…"):
        db,state,strat=run_backtest(symbol.strip().upper(),int(m15_limit),int(m1_limit),float(init_eq),str(price_src))
    st.success("Готово ✅")
    st.markdown("### 📊 Статистика"); show_stats(db_path=db.db_path, days=365)
    st.markdown("### 💰 Equity Curve"); show_equity_curve(db)
    st.markdown("### 📋 Сделки"); show_trades_table(db)
else:
    st.info("Задай параметры и нажми **Запустить бэктест**.")
