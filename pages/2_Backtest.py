# 2_Backtest.py
# Реальный бэктест: исторические 15m OHLC -> on_bar_close_15m() -> сделки/статистика

import os
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple

import streamlit as st
import matplotlib.pyplot as plt

# --- Импорты проекта ---
from config import Config
from state_manager import StateManager
from database import Database

# стратегия (с последними патчами)
from kwin_strategy import KWINStrategy

# Bybit API: пробуем несколько вариантов клиента
try:
    from bybit_api import BybitAPI
except Exception:
    try:
        from bybit_v5_fixed import BybitAPI  # если у тебя класс там
    except Exception:
        BybitAPI = None


# ====================== Вспомогательные (время/нормализация) ======================

def _utc_now_ms() -> int:
    return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)

def _window_ms(days: int) -> Tuple[int, int]:
    end_ms = _utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    return start_ms, end_ms

def _ensure_ms(ts):
    """Принимаем сек/мс/iso -> возвращаем unix ms."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # если похоже на секунды — домножим
        return int(ts if ts > 1e11 else ts * 1000)
    if isinstance(ts, str):
        try:
            # ISO / pandas parse
            dt = pd.to_datetime(ts, utc=True)
            return int(dt.value // 10**6)
        except Exception:
            return None
    return None

def _normalize_klines(raw: List[Dict]) -> List[Dict]:
    if not raw:
        return []
    out = []
    for k in raw:
        ts = k.get("timestamp") or k.get("start") or k.get("open_time") or k.get("t")
        ts = _ensure_ms(ts)
        if ts is None:
            continue
        out.append({
            "timestamp": ts,
            "open":  float(k.get("open",  k.get("o", 0.0))),
            "high":  float(k.get("high",  k.get("h", 0.0))),
            "low":   float(k.get("low",   k.get("l", 0.0))),
            "close": float(k.get("close", k.get("c", 0.0))),
            "volume": float(k.get("volume", k.get("v", 0.0)))
        })
    # от старых к новым для «прохода вперёд»
    out.sort(key=lambda x: x["timestamp"])
    return out


# ====================== Загрузчики OHLC (якорные окна) ======================

@st.cache_data(show_spinner=False)
def load_klines_bybit_window(_api, symbol: str, interval: str, days: int) -> List[Dict]:
    """
    Загружает ровно последние N дней (UTC): [now - N, now] с Bybit.
    ВАЖНО: параметр _api начинается с подчёркивания, чтобы Streamlit
    не пытался его хэшировать в cache_data.
    """
    if _api is None:
        return []

    start_ms, end_ms = _window_ms(days)

    # Для 15m: ~96 баров в день. Берём небольшой запас и потом фильтруем окном.
    need = int(days * 96 * 1.2) + 50

    try:
        raw = _api.get_klines(symbol, interval, need) or []
    except Exception:
        return []

    kl = _normalize_klines(raw)
    # Строго обрезаем по якорному окну
    kl = [b for b in kl if start_ms <= b["timestamp"] <= end_ms]
    return kl

@st.cache_data(show_spinner=False)
def load_klines_from_csv(file_bytes: bytes, tz_aware: bool = True) -> List[Dict]:
    """
    Поддержка CSV с полями: timestamp/open/high/low/close[/volume]
    timestamp может быть в мс, сек или ISO.
    """
    if not file_bytes:
        return []
    df = pd.read_csv(io.BytesIO(file_bytes))
    # нормализуем timestamp -> ms
    if "timestamp" not in df.columns:
        # иногда ts поле называется 'time'/'t'
        for cand in ("time", "t", "open_time"):
            if cand in df.columns:
                df.rename(columns={cand: "timestamp"}, inplace=True)
                break
    if "timestamp" not in df.columns:
        return []

    df["timestamp"] = df["timestamp"].apply(_ensure_ms)
    df = df.dropna(subset=["timestamp"])
    # ожидаемые ценовые поля
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    out = df[["timestamp", "open", "high", "low", "close"]].copy()
    if "volume" in df.columns:
        out["volume"] = df["volume"].astype(float)
    else:
        out["volume"] = 0.0
    # сортировка по времени (asc)
    out = out.sort_values("timestamp")
    return out.to_dict(orient="records")

@st.cache_data(show_spinner=False)
def load_klines_synthetic(symbol: str, days: int, seed: int = 42, base_price: Optional[float] = None) -> List[Dict]:
    """
    Детерминированная синтетика для отладки (сид управляет генерацией).
    """
    np.random.seed(seed)
    start_ms, end_ms = _window_ms(days)
    # 15m фрейм
    freq = "15T"
    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_dt   = pd.to_datetime(end_ms, unit="ms", utc=True)
    idx = pd.date_range(start=start_dt, end=end_dt, freq=freq, inclusive="left")
    if base_price is None:
        base_price = 4500.0 if symbol.upper().startswith("ETH") else 68000.0
    # геометр. броун. движение с малой волатильностью
    steps = len(idx)
    ret = np.random.randn(steps) * 0.002
    price = base_price * np.exp(np.cumsum(ret))
    out = []
    prev_close = price[0]
    for t, p in zip(idx, price):
        # простая свеча вокруг p
        vol = abs(np.random.randn()) * 0.001
        high = p * (1 + vol)
        low = p * (1 - vol)
        o = prev_close
        c = p
        out.append({
            "timestamp": int(t.value // 10**6),
            "open": float(o),
            "high": float(high),
            "low": float(low),
            "close": float(c),
            "volume": float(np.random.uniform(100, 10000))
        })
        prev_close = c
    return out


# ================== P&L + equity helpers ==================

def _calc_trade_pnl(direction: str,
                    entry_price: float,
                    exit_price: float,
                    qty: float,
                    taker_fee_rate: float) -> float:
    gross = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
    fees = (entry_price + exit_price) * qty * taker_fee_rate  # вход + выход
    return gross - fees


def _close_open_position(state: StateManager,
                         db: Database,
                         cfg: Config,
                         exit_price: float,
                         exit_time):
    pos = state.get_current_position()
    if not pos or pos.get("status") != "open":
        return

    direction   = pos["direction"]
    entry_price = float(pos["entry_price"])
    qty         = float(pos["size"])
    fee_rate    = float(getattr(cfg, "taker_fee_rate", 0.00055))

    pnl_net = _calc_trade_pnl(direction, entry_price, exit_price, qty, fee_rate)

    old_eq = float(state.get_equity() or 0.0)
    new_eq = old_eq + pnl_net
    state.set_equity(new_eq)

    trade = {
        "symbol":      getattr(cfg, "symbol", "ETHUSDT"),
        "direction":   direction,
        "entry_price": entry_price,
        "exit_price":  float(exit_price),
        "stop_loss":   pos.get("stop_loss"),
        "take_profit": pos.get("take_profit"),
        "quantity":    qty,
        "pnl":         pnl_net,
        "rr":          None,
        "entry_time":  pos.get("entry_time"),
        "exit_time":   datetime.fromtimestamp(int(exit_time)/1000, tz=timezone.utc) if isinstance(exit_time, (int,float)) else exit_time,
        "status":      "closed",
    }
    try:
        db.save_trade(trade)
        db.save_equity_snapshot(new_eq)
    except Exception:
        pass

    pos["status"]     = "closed"
    pos["exit_price"] = float(exit_price)
    pos["exit_time"]  = trade["exit_time"]
    state.set_position(pos)

    print(f"[EXIT] {direction.upper()} qty={qty} @ {exit_price}  PnL={pnl_net:.2f}  equity: {old_eq:.2f} → {new_eq:.2f}")


def _dd_from_equity(equity_series: pd.Series) -> Dict[str, float]:
    if equity_series.empty:
        return {"max_dd_abs": 0.0, "max_dd_pct": 0.0}
    roll_max = equity_series.cummax()
    dd = equity_series - roll_max
    dd_pct = equity_series / roll_max - 1.0
    max_dd_abs = float(dd.min())
    max_dd_pct = float(dd_pct.min()) * 100.0
    return {"max_dd_abs": round(max_dd_abs, 2), "max_dd_pct": round(max_dd_pct, 2)}

def _profit_factor_from_trades(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    losses = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 3)


# ====================== Реальный бэктест через стратегию ======================

def run_backtest_ohlc(period_days: int,
                      initial_capital: float,
                      commission_rate: float,
                      symbol: str,
                      config: Config,
                      candles_override: Optional[List[Dict]] = None) -> Dict[str, pd.DataFrame]:
    """
    Если candles_override передан — используем его (строгое окно),
    иначе загрузим по-старому.
    """
    # In-memory DB
    try:
        db = Database(memory=True)
    except TypeError:
        db = Database(db_path="backtest_tmp.sqlite")

    state = StateManager(db)
    state.set_equity(float(initial_capital))

    if BybitAPI is None:
        st.error("Не найден BybitAPI. Убедись, что bybit_api.py или bybit_v5_fixed.py доступны.")
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    api = BybitAPI(
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET")
    )

    config.days_back = int(period_days)
    config.taker_fee_rate = float(commission_rate)
    config.symbol = symbol

    strat = KWINStrategy(config, api, state, db)

    try:
        strat._update_equity()
    except Exception:
        pass

    if candles_override is not None:
        candles = candles_override
    else:
        # прошлый вариант (приближённый)
        candles = load_klines_bybit_window(api, symbol, "15", period_days)

    if not candles:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    # === Основной проход по барам (по времени: от старых к новым) ===
    for bar in candles:
        pos = state.get_current_position()
        if pos and pos.get("status") == "open":
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            cur_time = bar.get("timestamp")

            sl = float(pos.get("stop_loss") or 0)
            tp = pos.get("take_profit")

            if pos["direction"] == "long" and sl > 0 and bar_low <= sl:
                _close_open_position(state, db, config, exit_price=sl, exit_time=cur_time)
            elif pos["direction"] == "short" and sl > 0 and bar_high >= sl:
                _close_open_position(state, db, config, exit_price=sl, exit_time=cur_time)
            else:
                if tp is not None:
                    tp = float(tp)
                    if pos["direction"] == "long" and bar_high >= tp:
                        _close_open_position(state, db, config, exit_price=tp, exit_time=cur_time)
                    if pos["direction"] == "short" and bar_low <= tp:
                        _close_open_position(state, db, config, exit_price=tp, exit_time=cur_time)

        strat.on_bar_close_15m(bar)

    # Достаём сделки из БД
    try:
        trades = db.get_trades_by_period(period_days)
    except Exception:
        trades = db.get_recent_trades(10_000)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    capital = float(initial_capital)
    eq_times, eq_values = [], []

    equity_df = pd.DataFrame()
    try:
        snaps = db.get_equity_history(days=period_days) if hasattr(db, "get_equity_history") else []
        if snaps:
            equity_df = pd.DataFrame(snaps)
            if {"timestamp", "equity"} <= set(equity_df.columns):
                equity_df["time"] = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce")
                equity_df = equity_df.sort_values("time")[["time", "equity"]]
    except Exception:
        pass

    if equity_df.empty:
        if not trades_df.empty:
            if "entry_time" in trades_df.columns:
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
            if "exit_time" in trades_df.columns:
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

            sort_key = "exit_time" if "exit_time" in trades_df.columns else "entry_time"
            trades_df = trades_df.sort_values(by=sort_key)

            wins = 0
            total_pnl = 0.0
            for _, tr in trades_df.iterrows():
                entry = float(tr.get("entry_price", np.nan))
                qty = float(tr.get("quantity", np.nan) or tr.get("qty", np.nan))
                side = tr.get("direction", "")
                exit_p = tr.get("exit_price", np.nan)

                if np.isnan(entry) or np.isnan(qty) or pd.isna(exit_p):
                    continue
                exit_p = float(exit_p)

                gross = (exit_p - entry) * qty if side == "long" else (entry - exit_p) * qty
                fee_in = entry * qty * commission_rate
                fee_out = exit_p * qty * commission_rate
                pnl = gross - fee_in - fee_out

                capital += pnl
                total_pnl += pnl
                if pnl > 0:
                    wins += 1

                t = tr.get("exit_time") or tr.get("entry_time")
                eq_times.append(pd.to_datetime(t, utc=True, errors="coerce"))
                eq_values.append(capital)

            winrate = (wins / len(trades_df)) * 100 if len(trades_df) else 0.0
        else:
            winrate = 0.0
            total_pnl = 0.0

        equity_df = pd.DataFrame({"time": eq_times, "equity": eq_values}) if eq_values else pd.DataFrame()
    else:
        total_pnl = 0.0
        winrate = 0.0
        if not trades_df.empty:
            wins = 0
            for _, tr in trades_df.iterrows():
                if "pnl" in tr and not pd.isna(tr["pnl"]):
                    total_pnl += float(tr["pnl"])
                    if float(tr["pnl"]) > 0:
                        wins += 1
            winrate = (wins / len(trades_df)) * 100 if len(trades_df) else 0.0
        capital = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_capital

    # --- Доп. метрики ---
    if not equity_df.empty:
        dd = _dd_from_equity(equity_df["equity"].astype(float))
    else:
        dd = {"max_dd_abs": 0.0, "max_dd_pct": 0.0}

    profit_factor = _profit_factor_from_trades(trades_df) if not trades_df.empty else 0.0
    avg_trade_pnl = float(trades_df["pnl"].mean()) if ("pnl" in trades_df.columns and not trades_df.empty) else 0.0

    stats = {
        "final_capital": round(float(capital), 2),
        "trades": int(len(trades_df)),
        "winrate_pct": round(float(winrate), 2),
        "total_pnl": round(float(total_pnl), 2),
        "max_dd_abs": dd["max_dd_abs"],
        "max_dd_pct": dd["max_dd_pct"],
        "profit_factor": profit_factor,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
    }
    # привели equity_df к ожидаемому виду
    if not equity_df.empty and "time" in equity_df.columns:
        equity_df = equity_df.rename(columns={"time": "time"})
    return {"trades": trades_df, "equity": equity_df, "stats": stats}


# ====================== UI (Streamlit page) ======================

def main():
    st.set_page_config(page_title="Backtest — KWIN", layout="wide")
    st.title("KWIN — Backtest (15m OHLC → Strategy)")

    # --- Источник данных и окно ---
    st.sidebar.header("Источник данных")
    data_source = st.sidebar.selectbox(
        "Data source",
        ["Bybit (live)", "CSV (upload)", "Synthetic"],
        index=0
    )
    symbol = st.sidebar.text_input("Symbol", value="ETHUSDT")
    period_days = st.sidebar.selectbox("Период (якорный, назад от текущего UTC)", options=[30, 60, 90, 180], index=0)

    csv_file = None
    syn_seed = 42
    if data_source == "CSV (upload)":
        up = st.sidebar.file_uploader("Загрузить CSV (timestamp,open,high,low,close[,volume])", type=["csv"])
        if up is not None:
            csv_file = up.read()
    elif data_source == "Synthetic":
        syn_seed = st.sidebar.number_input("Synthetic seed", min_value=0, value=42, step=1)

    st.sidebar.header("Комиссии / Капитал")
    start_capital = st.sidebar.number_input("Initial Capital (USDT)", min_value=1.0, value=100.0, step=10.0)
    commission_rate = st.sidebar.number_input("Commission (taker, decimal)", min_value=0.0, value=0.00055, step=0.00005, format="%.5f")

    # Секция конфигурации стратегии
    st.sidebar.header("Strategy Config (ключевые)")
    risk_pct = st.sidebar.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=3.0, step=0.1, format="%.1f")
    risk_reward = st.sidebar.number_input("TP Risk/Reward Ratio", min_value=0.5, value=1.3, step=0.1)
    sfp_len = st.sidebar.number_input("SFP Length", min_value=2, value=2, step=1)
    use_sfp_quality = st.sidebar.checkbox("Filter: SFP quality (wick+closeback)", value=True)
    wick_min_ticks = st.sidebar.number_input("SFP: min wick depth (ticks)", min_value=0, value=7, step=1)
    close_back_pct = st.sidebar.number_input("SFP: min close-back (0..1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Smart Trailing")
    enable_smart_trail = st.sidebar.checkbox("Enable Smart Trailing TP", value=True)
    use_arm_after_rr = st.sidebar.checkbox("Enable Arm after RR≥X", value=True)
    arm_rr = st.sidebar.number_input("Arm RR (R)", min_value=0.1, value=0.5, step=0.1, format="%.1f")
    trailing_perc = st.sidebar.number_input("Percent trailing, % of entry", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    trailing_offset_perc = st.sidebar.number_input("Trailing offset, % of entry", min_value=0.0, value=0.4, step=0.1, format="%.2f")
    use_bar_trail = st.sidebar.checkbox("Use Bar-Low/High Smart Trail", value=True)
    trail_lookback = st.sidebar.number_input("Trail lookback bars", min_value=1, value=50, step=1)
    trail_buf_ticks = st.sidebar.number_input("Trail buffer (ticks)", min_value=0, value=40, step=1)

    st.sidebar.markdown("---")
    limit_qty_enabled = st.sidebar.checkbox("Limit Max Position Qty", value=True)
    max_qty_manual = st.sidebar.number_input("Max Qty (asset units)", min_value=0.01, value=50.0, step=0.01, format="%.2f")

    min_net_profit = st.sidebar.number_input("Min Net Profit (USDT)", min_value=0.0, value=1.2, step=0.1)
    min_order_qty = st.sidebar.number_input("Min Order Qty", min_value=0.0, value=0.01, step=0.01, format="%.2f")
    qty_step = st.sidebar.number_input("Qty Step", min_value=0.0, value=0.01, step=0.01, format="%.2f")

    # Собираем Config
    config = Config()
    config.symbol = symbol
    config.days_back = int(period_days)
    config.risk_pct = float(risk_pct)
    config.risk_reward = float(risk_reward)
    config.sfp_len = int(sfp_len)

    config.use_sfp_quality = bool(use_sfp_quality)
    config.wick_min_ticks = int(wick_min_ticks)
    config.close_back_pct = float(close_back_pct)

    config.enable_smart_trail = bool(enable_smart_trail)
    config.use_arm_after_rr = bool(use_arm_after_rr)
    config.arm_rr = float(arm_rr)
    config.use_bar_trail = bool(use_bar_trail)
    config.trail_lookback = int(trail_lookback)
    config.trail_buf_ticks = int(trail_buf_ticks)

    config.trailing_perc = float(trailing_perc)
    config.trailing_offset_perc = float(trailing_offset_perc)

    config.limit_qty_enabled = bool(limit_qty_enabled)
    config.max_qty_manual = float(max_qty_manual)

    config.min_net_profit = float(min_net_profit)
    config.min_order_qty = float(min_order_qty)
    config.qty_step = float(qty_step)

    config.taker_fee_rate = float(commission_rate)

    # --- Загрузка данных согласно выбору ---
    candles_for_run: List[Dict] = []
    if data_source == "Bybit (live)":
        if BybitAPI is None:
            st.error("BybitAPI недоступен.")
        else:
            api = BybitAPI(api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"))
            candles_for_run = load_klines_bybit_window(api, symbol, "15", period_days)
    elif data_source == "CSV (upload)":
        if not csv_file:
            st.info("Загрузите CSV, чтобы запустить бэктест.")
        else:
            candles_for_run = load_klines_from_csv(csv_file)
            # отрежем окно последних N дней (якорно от «сейчас»)
            start_ms, end_ms = _window_ms(period_days)
            candles_for_run = [b for b in candles_for_run if start_ms <= b["timestamp"] <= end_ms]
    else:  # Synthetic
        candles_for_run = load_klines_synthetic(symbol, period_days, seed=syn_seed)

    run_btn = st.button("Запустить бэктест", type="primary")

    if run_btn:
        if not candles_for_run:
            st.warning("Нет данных свечей для выбранного источника/периода.")
            return

        with st.spinner("Запускаю бэктест..."):
            res = run_backtest_ohlc(
                period_days=period_days,
                initial_capital=start_capital,
                commission_rate=commission_rate,
                symbol=symbol,
                config=config,
                candles_override=candles_for_run
            )

        trades_df = res.get("trades", pd.DataFrame())
        equity_df = res.get("equity", pd.DataFrame())
        stats = res.get("stats", {})

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Сделок", value=stats.get("trades", 0))
        col2.metric("Winrate", f"{stats.get('winrate_pct', 0.0)}%")
        col3.metric("Итоговый капитал", f"{stats.get('final_capital', start_capital):.2f}")
        col4.metric("Total PnL", f"{stats.get('total_pnl', 0.0):.2f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Max DD ($)", f"{stats.get('max_dd_abs', 0.0):.2f}")
        col6.metric("Max DD (%)", f"{stats.get('max_dd_pct', 0.0):.2f}%")
        pf = stats.get("profit_factor", 0.0)
        pf_str = "∞" if pf == float("inf") else f"{pf}"
        col7.metric("Profit Factor", pf_str)

        # График эквити
        st.subheader("Equity Curve")
        if not equity_df.empty:
            fig, ax = plt.subplots()
            # совместимость: поле может называться time
            x = equity_df["time"] if "time" in equity_df.columns else equity_df.iloc[:, 0]
            ax.plot(pd.to_datetime(x), equity_df["equity"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Equity (USDT)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Нет данных эквити (в БД отсутствуют закрытия сделок). Убедись, что стратегия записывает exit.")

        # Таблица сделок
        st.subheader("Сделки")
        if not trades_df.empty:
            # немного приводим формат
            for col in ("entry_time", "exit_time"):
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce").dt.tz_convert(None)
            for col in ("entry_price", "exit_price", "quantity", "qty", "pnl", "rr"):
                if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")
                    if col in ("entry_price", "exit_price", "pnl", "rr"):
                        trades_df[col] = trades_df[col].round(2)
                    if col in ("quantity", "qty"):
                        trades_df[col] = trades_df[col].round(4)

            view_cols_pref = ["entry_time", "direction", "entry_price", "exit_price", "quantity", "qty", "pnl", "rr", "status", "exit_reason"]
            view_cols = [c for c in view_cols_pref if c in trades_df.columns]
            st.dataframe(trades_df[view_cols].sort_values("entry_time", ascending=False), use_container_width=True)
        else:
            st.info("Сделок не найдено за выбранный период. Проверь условия детекции/окно бэктеста.")


if __name__ == "__main__":
    main()
