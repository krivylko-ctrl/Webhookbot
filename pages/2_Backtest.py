import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

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


# ======================= Брокер под БЭКТЕСТ =======================
class BacktestBroker:
    """
    Брокер для бэктеста:
      • отдаёт РЕАЛЬНЫЕ бары Bybit,
      • хранит «текущую» цену для стратегии,
      • ордерные методы — безопасные заглушки.
    """
    def __init__(self, market: BybitAPI):
        self.market = market
        self.market.force_linear()
        self._last_price: Dict[str, float] = {}

    # ---- маркет-данные ----
    def get_klines(self, symbol: str, interval: str, limit: int = 200):
        return self.market.get_klines(symbol, interval, limit) or []

    def get_klines_window(self, symbol: str, interval: str,
                          start_ms: Optional[int], end_ms: Optional[int], limit: int = 1000):
        return self.market.get_klines_window(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=limit) or []

    def get_instruments_info(self, symbol: str):
        return self.market.get_instruments_info(symbol)

    def get_ticker(self, symbol: str) -> Dict:
        p = float(self._last_price.get(symbol, 0.0))
        return {"symbol": symbol, "lastPrice": p, "markPrice": p}

    def set_current_price(self, symbol: str, price: float):
        self._last_price[symbol] = float(price)

    def get_price(self, symbol: str, source: str = "last") -> float:
        return float(self._last_price.get(symbol, 0.0))

    # ---- ордерные заглушки ----
    def place_order(self, **_kwargs):
        return {"ok": True, "filled": True, "msg": "backtest fill"}

    def update_position_stop_loss(self, symbol: str, new_sl: float):
        return True

    def modify_order(self, **_kwargs):
        return {"ok": True}


# =============== УТИЛИТЫ ДЛЯ БЭКТЕСТА ===============
@dataclass
class BtData:
    m15: pd.DataFrame


def _utc_midnight(dt: Optional[datetime] = None) -> datetime:
    dt = dt or datetime.now(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _align_floor(ts_ms: int, tf_ms: int) -> int:
    return (int(ts_ms) // tf_ms) * tf_ms


def _align_ceil(ts_ms: int, tf_ms: int) -> int:
    return ((int(ts_ms) + tf_ms - 1) // tf_ms) * tf_ms


def _norm_ts_ms(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Приводим timestamp к миллисекундам, если пришли секунды."""
    if col in df.columns and not df.empty:
        mx = pd.to_numeric(df[col], errors="coerce").max()
        if pd.notna(mx) and mx < 1_000_000_000_000:  # секунды
            df[col] = pd.to_numeric(df[col], errors="coerce") * 1000
    return df


def _fetch_aligned_window(
    _api: BacktestBroker,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    overlap_bars: int = 2,
) -> List[Dict]:
    """
    Универсальный лоадер через broker.get_klines_window с:
      • выравниванием start/end к сетке ТФ,
      • чанками с перекрытием (overlap_bars),
      • дедупом и сортировкой,
      • нормализацией timestamp в мс.
    """
    tf_ms = int(interval) * 60_000
    start_ms = _align_floor(start_ms, tf_ms)
    end_ms = _align_ceil(end_ms, tf_ms) - 1

    out: List[Dict] = []
    step_ms = limit * tf_ms + overlap_bars * tf_ms
    cursor = start_ms
    while cursor <= end_ms:
        chunk_end = min(end_ms, cursor + step_ms - 1)
        rows = _api.get_klines_window(symbol, interval, start_ms=cursor, end_ms=chunk_end, limit=limit) or []
        if rows:
            # в dict-форму и ms-нормализация
            for r in rows:
                ts = int(r.get("timestamp") or r.get("open_time") or 0)
                if ts and ts < 1_000_000_000_000:  # сек -> мс
                    ts *= 1000
                out.append({
                    "timestamp": ts,
                    "open":  float(r["open"]),
                    "high":  float(r["high"]),
                    "low":   float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume") or 0.0),
                })
        cursor = chunk_end + 1

    # дедуп + сортировка
    out = sorted({b["timestamp"]: b for b in out if b["timestamp"]}.values(), key=lambda x: x["timestamp"])
    return out


@st.cache_data(show_spinner=False)
def load_m15_window(_api: BacktestBroker, symbol: str, days: int, sfp_len: int) -> BtData:
    """
    15m история по ЖЁСТКОМУ окну времени + тёплый старт для пивотов.
    Возвращаем DataFrame по возрастанию времени.
    """
    # окно бэктеста: [UTC-полночь - days, now]
    utc_midnight = _utc_midnight()
    start_dt = utc_midnight - timedelta(days=int(days))
    end_dt = datetime.now(timezone.utc)

    # warm-up: L + 2 + 10 баров 15m
    warmup_15m = int(sfp_len) + 12
    start_ms = int(start_dt.timestamp() * 1000) - warmup_15m * 15 * 60 * 1000
    end_ms = int(end_dt.timestamp() * 1000)

    # надёжная подгрузка с выравниванием и перекрытием
    raw = _fetch_aligned_window(_api, symbol, "15", start_ms=start_ms, end_ms=end_ms, limit=1000, overlap_bars=2)
    df = pd.DataFrame(raw or [])
    if df.empty:
        df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    else:
        df = _norm_ts_ms(df, "timestamp").sort_values("timestamp").reset_index(drop=True)
    return BtData(m15=df)


@st.cache_data(show_spinner=False)
def load_m1_day(_api: BacktestBroker, symbol: str, intrabar_tf: str, day_start_ms: int) -> pd.DataFrame:
    """
    Минутки/интрабар за ОДИН ДЕНЬ: [day_start .. day_start+24h], с выравниванием и перекрытием.
    Кэшируется поминутно по дням => ~N запросов на N дней.
    """
    tf_ms = int(intrabar_tf) * 60_000
    day_start_ms = _align_floor(day_start_ms, 24 * 60 * 60 * 1000)
    day_end_ms = day_start_ms + 24 * 60 * 60 * 1000 - 1

    raw = _fetch_aligned_window(_api, symbol, intrabar_tf, start_ms=day_start_ms, end_ms=day_end_ms,
                                limit=1000, overlap_bars=5)
    df = pd.DataFrame(raw or [])
    if df.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return _norm_ts_ms(df, "timestamp").sort_values("timestamp").reset_index(drop=True)


def iter_m1_between_by_day(_api: BacktestBroker, symbol: str, intrabar_tf: str, t_from: int, t_to: int) -> List[Dict]:
    """
    Возвращает список m1-свечей строго в (t_from, t_to], подгружая данные
    суточными пачками и вырезая нужный интервал. Время — в МИЛЛИСЕКУНДАХ.
    """
    if t_to <= t_from:
        return []

    # нормализуем к UTC-полуночам исходя из t_from/t_to
    from_dt = datetime.utcfromtimestamp(int(t_from) / 1000).replace(tzinfo=timezone.utc)
    to_dt   = datetime.utcfromtimestamp(int(t_to)   / 1000).replace(tzinfo=timezone.utc)

    day = _utc_midnight(from_dt)
    end_day = _utc_midnight(to_dt)

    out: List[Dict] = []
    while day <= end_day:
        day_ms = int(day.timestamp() * 1000)
        df_day = load_m1_day(_api, symbol, intrabar_tf, day_ms)
        if not df_day.empty:
            mask = (df_day["timestamp"] > t_from) & (df_day["timestamp"] <= t_to)
            sub = df_day.loc[mask]
            if not sub.empty:
                out.extend(sub.to_dict("records"))
        day += timedelta(days=1)

    return out


def _compute_net_pnl(pos: Dict, exit_price: float, fee_rate: float) -> float:
    entry = float(pos.get("entry_price") or 0.0)
    qty   = float(pos.get("size") or pos.get("quantity") or 0.0)
    if qty <= 0:
        return 0.0
    direction = str(pos.get("direction"))
    gross = (exit_price - entry) * qty if direction == "long" else (entry - exit_price) * qty
    fee_in  = entry * qty * fee_rate
    fee_out = exit_price * qty * fee_rate
    return float(gross - fee_in - fee_out)


def _book_close_and_update_equity(state: StateManager, db: Database, cfg: Config, pos: Dict, exit_px: float, reason: str):
    state.close_position(exit_price=float(exit_px), exit_reason=reason)
    net = _compute_net_pnl(pos, exit_px, float(getattr(cfg, "taker_fee_rate", 0.00055)))
    new_eq = float(state.get_equity()) + net
    state.set_equity(new_eq)
    db.save_equity_snapshot(new_eq)


def simulate_exits_on_m1(state: StateManager, db: Database, cfg: Config, m1: Dict):
    """Биржевой приоритет: сначала SL, потом TP."""
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction = pos.get("direction")
    sl = float(pos.get("stop_loss") or 0.0)
    tp = float(pos.get("take_profit") or 0.0) if getattr(cfg, "use_take_profit", True) else 0.0
    hi, lo = float(m1["high"]), float(m1["low"])

    if direction == "long":
        if sl and lo <= sl:
            _book_close_and_update_equity(state, db, cfg, pos, sl, "SL"); return
        if tp and hi >= tp:
            _book_close_and_update_equity(state, db, cfg, pos, tp, "TP"); return
    else:
        if sl and hi >= sl:
            _book_close_and_update_equity(state, db, cfg, pos, sl, "SL"); return
        if tp and lo <= tp:
            _book_close_and_update_equity(state, db, cfg, pos, tp, "TP"); return


def run_backtest(symbol: str,
                 days: int,
                 init_equity: float,
                 cfg: Config,
                 price_source_for_logic: str = "last") -> Tuple[Database, StateManager, KWINStrategy]:
    """15m + интрабар M1 (по дням), Pine-точные входы/трейл, реальные бары Bybit."""

    # отдельная БД под бэктест (полный сброс перед стартом)
    bt_db_path = f"kwin_backtest_{symbol}.db"
    db = Database(db_path=bt_db_path)
    db.drop_and_recreate()

    state = StateManager(db)
    state.set_equity(float(init_equity))
    db.save_equity_snapshot(float(init_equity))

    # маркет-источник + брокер для бэктеста
    real_market = BybitAPI(api_key="", api_secret="", testnet=False)
    broker = BacktestBroker(market=real_market)

    # стратегия
    cfg.price_for_logic = str(price_source_for_logic).lower()
    cfg.start_time_ms = None                    # не ограничиваем «isActive» в bt
    strat = KWINStrategy(cfg, api=broker, state_manager=state, db=db)

    # 15m история (с выравниванием и тёплым стартом)
    data15 = load_m15_window(broker, symbol, days=int(days), sfп_len=int(getattr(cfg, "sfp_len", 2)))
    if data15.m15.empty:
        st.error("Не удалось загрузить 15m историю.")
        return db, state, strat

    m15 = data15.m15.reset_index(drop=True)


    # основной цикл по закрытым 15m барам
    intrabar_tf = str(getattr(cfg, "intrabar_tf", "1"))
    for i in range(0, len(m15) - 1):
        bar = m15.iloc[i].to_dict()
        t_curr = int(bar["timestamp"])
        t_next = int(m15.iloc[i + 1]["timestamp"])

        # «текущая» цена — клоуз 15m
        broker.set_current_price(symbol, float(bar["close"]))

        strat.on_bar_close_15m({
            "timestamp": t_curr,
            "open":  float(bar["open"]),
            "high":  float(bar["high"]),
            "low":   float(bar["low"]),
            "close": float(bar["close"]),
        })

        # интрабары из кэша "по дням"
        m1_set = iter_m1_between_by_day(broker, symbol, intrabar_tf, t_curr, t_next)
        for m1 in m1_set:
            broker.set_current_price(symbol, float(m1["close"]))
            strat.on_bar_close_1m({
                "timestamp": int(m1["timestamp"]),
                "open":  float(m1["open"]),
                "high":  float(m1["high"]),
                "low":   float(m1["low"]),
                "close": float(m1["close"]),
            })
            simulate_exits_on_m1(state, db, cfg, m1)

    # если осталась открытая позиция — закроем по последнему close 15m
    pos = state.get_current_position()
    if pos and pos.get("status") == "open":
        last_close = float(m15.iloc[-1]["close"])
        _book_close_and_update_equity(state, db, cfg, pos, last_close, "bt_end")

    return db, state, strat


# ========================= UI: центральная форма =========================
st.markdown("## ⚙️ Настройки бэктеста")

with st.form("backtest_form"):
    cfg = Config()

    c0a, c0b, c0c, c0d = st.columns(4)
    with c0a:
        symbol = st.text_input("Символ", value=str(getattr(cfg, "symbol", "ETHUSDT")))
    with c0b:
        init_eq = st.number_input("Начальный equity ($)", min_value=10.0, max_value=1_000_000.0,
                                  value=1000.0, step=10.0)
    with c0c:
        price_src = st.selectbox("Источник цены для логики", options=["last", "mark"], index=0)
    with c0d:
        bt_days = st.selectbox("Период бэктеста (дней)", [7, 14, 30, 39, 60], index=3)

    st.markdown("---")

    # ====== Основные ======
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

    # ====== Stop-Loss зона (Pine-подход) ======
    st.subheader("📌 Stop-Loss Zone (Pine-like)")
    z1, z2, z3, z4, z5 = st.columns(5)
    with z1:
        use_swing_sl = st.checkbox("SL от свинга (pivot)", value=bool(getattr(cfg, "use_swing_sl", True)))
    with z2:
        use_prev_candle_sl = st.checkbox("SL от свечи [1]", value=bool(getattr(cfg, "use_prev_candle_sl", False)))
    with z3:
        sl_buf_ticks = st.number_input("Буфер к SL (ticks)", min_value=0, max_value=1000,
                                       value=int(getattr(cfg, "sl_buf_ticks", 40)), step=1)
    with z4:
        use_atr_buffer = st.checkbox("ATR-буфер", value=bool(getattr(cfg, "use_atr_buffer", False)))
    with z5:
        atr_mult = st.number_input("ATR Mult", min_value=0.0, max_value=10.0,
                                   value=float(getattr(cfg, "atr_mult", 0.0)), step=0.1)

    tps = st.selectbox("Триггер стопа/тейка (биржа)", options=["mark", "last"],
                       index=0 if str(getattr(cfg, "trigger_price_source", "mark")).lower() == "mark" else 1,
                       help="По какой цене биржа срабатывает SL/TP. Рекомендуется Mark.")

    st.markdown("---")

    # ====== Smart Trailing ======
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

    # ====== ARM RR ======
    st.subheader("📌 ARM RR")
    c7, c8 = st.columns(2)
    with c7:
        use_arm_after_rr = st.checkbox("💚 Enable Arm after RR≥X",
                                       value=bool(getattr(cfg, "use_arm_after_rr", True)))
    with c8:
        arm_rr = st.number_input("Arm RR (R)", min_value=0.1, max_value=5.0,
                                 value=float(getattr(cfg, "arm_rr", 0.5)), step=0.1)

    st.markdown("---")

    # ====== Bar-Low/High Smart Trail ======
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

    # ====== Лимиты позиции ======
    st.subheader("📌 Лимиты позиции")
    c12, c13 = st.columns(2)
    with c12:
        limit_qty_enabled = st.checkbox("💚 Limit Max Position Qty",
                                        value=bool(getattr(cfg, "limit_qty_enabled", True)))
    with c13:
        max_qty_manual = st.number_input("Max Qty (ETH)", min_value=0.001, max_value=10_000.0,
                                         value=float(getattr(cfg, "max_qty_manual", 50.0)), step=0.001)

    st.markdown("---")

    # ====== Фильтры SFP ======
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

    # ====== TP / комиссия ======
    c17, c18 = st.columns(2)
    with c17:
        use_take_profit = st.checkbox("Use Take Profit", value=bool(getattr(cfg, "use_take_profit", True)))
    with c18:
        taker_fee = st.number_input("Taker fee (decimal)", min_value=0.0, max_value=0.01,
                                    value=float(getattr(cfg, "taker_fee_rate", 0.00055)), step=0.00005)

    st.markdown("---")

    # ====== Intrabar entries ======
    intrabar_entries = st.checkbox("🔁 Intrabar entries (calc_on_every_tick)", value=True)

    submitted = st.form_submit_button("🚀 Запустить бэктест", use_container_width=True)


# ========================= запуск бэктеста =========================
if submitted:
    # полный сброс кэша перед каждым запуском (пробиваем @st.cache_data/@st.cache_resource)
    try: st.cache_data.clear()
    except Exception: pass
    try: st.cache_resource.clear()
    except Exception: pass

    # применяем значения в конфиг
    cfg = Config()
    cfg.symbol = symbol.strip().upper()
    cfg.risk_reward = float(risk_reward)
    cfg.sfp_len = int(sfp_len)
    cfg.risk_pct = float(risk_pct)

    # ===== SL-зона (новые параметры) ====
    cfg.use_swing_sl = bool(use_swing_sl)
    cfg.use_prev_candle_sl = bool(use_prev_candle_sl)
    cfg.sl_buf_ticks = int(sl_buf_ticks)
    cfg.use_atr_buffer = bool(use_atr_buffer)
    cfg.atr_mult = float(atr_mult)
    cfg.trigger_price_source = str(tps).lower()  # "mark" | "last"

    # ===== Smart trail / ARM / bar-trail / лимиты =====
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

    # ===== фильтры / комиссия / источники цены =====
    cfg.use_sfp_quality = bool(use_sfp_quality)
    cfg.wick_min_ticks = int(wick_min_ticks)
    cfg.close_back_pct = float(close_back_pct)

    cfg.use_take_profit = bool(use_take_profit)
    cfg.taker_fee_rate = float(taker_fee)

    cfg.price_for_logic = str(price_src).lower()
    cfg.intrabar_tf = "1"
    cfg.days_back = int(bt_days)
    cfg.use_intrabar_entries = bool(intrabar_entries)
    cfg.start_time_ms = None  # никаких ограничений «isActive» в бэктесте

    with st.spinner("Грузим историю и запускаем бэктест…"):
        db, state, strat = run_backtest(
            symbol=cfg.symbol,
            days=int(bt_days),
            init_equity=float(init_eq),
            cfg=cfg,
            price_source_for_logic=str(price_src),
        )

    st.success("Готово ✅")

    # ---------- Статистика ----------
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

    # ---------- Кривая капитала ----------
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

    # ---------- Таблица сделок ----------
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
