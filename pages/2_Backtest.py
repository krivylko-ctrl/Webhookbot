# pages/03_backtest.py
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import Config
from database import Database
from state_manager import StateManager
from kwin_strategy import KWINStrategy
from analytics import TradingAnalytics
from bybit_api import BybitAPI
from trail_engine import TrailEngine


st.set_page_config(page_title="KWIN Bot — Бэктест", page_icon="📈", layout="wide")
st.title("📈 Бэктест KWIN Strategy")


# ====== Paper API ======
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


# ====== История ======
@dataclass
class BtData:
    m15: pd.DataFrame
    m1: Optional[pd.DataFrame]


@st.cache_data(show_spinner=False)
def load_history(_api: PaperAPI, symbol: str, m15_limit: int, m1_limit: int, intrabar_tf: str = "1") -> BtData:
    m15_raw = _api.get_klines(symbol, "15", m15_limit) or []
    df15 = pd.DataFrame(m15_raw)
    if not df15.empty:
        df15 = df15.sort_values("timestamp").reset_index(drop=True)
    else:
        st.error("❌ История M15 пуста — проверь Bybit API или подгрузи CSV")
        df15 = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df1 = pd.DataFrame()
    if m1_limit > 0:
        m1_raw = _api.get_klines(symbol, intrabar_tf, m1_limit) or []
        df1 = pd.DataFrame(m1_raw)
        if not df1.empty:
            df1 = df1.sort_values("timestamp").reset_index(drop=True)

    return BtData(m15=df15, m1=df1 if not df1.empty else None)


def iter_m1_between(df1: Optional[pd.DataFrame], t_from: int, t_to: int) -> List[Dict]:
    if df1 is None or df1.empty:
        return []
    mask = (df1["timestamp"] > t_from) & (df1["timestamp"] <= t_to)
    return df1.loc[mask].to_dict("records")


def simulate_exits_on_m1(state: StateManager, db: Database, cfg: Config, m1: Dict, trail: TrailEngine):
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction = pos.get("direction")
    sl = float(pos.get("stop_loss") or 0.0)
    tp = float(pos.get("take_profit") or 0.0) if cfg.use_take_profit else 0.0
    hi, lo = float(m1["high"]), float(m1["low"])

    # Smart trail update
    trail.check_and_update(state, float(m1["close"]), hi, lo)

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


def run_backtest(symbol: str, m15_limit: int, m1_limit: int, init_equity: float, cfg: Config,
                 price_source_for_logic: str = "last") -> Tuple[Database, StateManager, KWINStrategy]:
    db = Database(db_path=f"kwin_backtest_{symbol}.db")
    state = StateManager(db)
    state.set_equity(float(init_equity))

    real_market = BybitAPI(api_key="", api_secret="", testnet=False)
    paper_api = PaperAPI(real_market_api=real_market)

    strat = KWINStrategy(cfg, api=paper_api, state_manager=state, db=db)
    trail = TrailEngine(cfg, api=paper_api, state_manager=state, db=db)

    data = load_history(paper_api, symbol, m15_limit, m1_limit, "1")
    if data.m15.empty:
        return db, state, strat

    m15 = data.m15.reset_index(drop=True)
    for i in range(0, len(m15) - 1):
        bar = m15.iloc[i].to_dict()
        t_curr = int(bar["timestamp"])
        t_next = int(m15.iloc[i + 1]["timestamp"])

        paper_api.set_last_price(symbol, float(bar["close"]))
        strat.on_bar_close_15m(bar)

        m1_set = iter_m1_between(data.m1, t_curr, t_next)
        for m1 in m1_set:
            paper_api.set_last_price(symbol, float(m1["close"]))
            strat.on_bar_close_1m(m1)
            simulate_exits_on_m1(state, db, cfg, m1, trail)

    return db, state, strat
