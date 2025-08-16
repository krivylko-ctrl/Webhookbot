import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone

# ==== UI ====
st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")
st.title("Бэктест стратегии KWIN")

colp1, colp2, colp3 = st.columns(3)
period_map = {"30D":30, "60D":60, "180D":180}
period_key = colp1.selectbox("Период бэктеста", list(period_map.keys()), index=1)
start_capital = colp2.number_input("Начальный капитал ($)", min_value=50.0, value=300.0, step=50.0)
commission_pct = colp3.number_input("Комиссия (%) за сделку (taker side)", min_value=0.0, value=0.055, step=0.005, format="%.3f")

col1, col2, col3, col4 = st.columns(4)
symbol = col1.text_input("Символ (Bybit Linear)", value="ETHUSDT")
interval = col2.selectbox("TF", ["1","3","5","15","30","60","240"], index=3)
risk_pct = col3.number_input("Риск % на сделку", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
rr_input = col4.number_input("RR фикса для фильтра NetPnL, R", min_value=0.5, max_value=5.0, value=1.3, step=0.1)

colt1, colt2, colt3, colt4 = st.columns(4)
enable_smart_trail = colt1.checkbox("✅ Smart Trailing", value=True)
arm_after_rr = colt2.checkbox("Arm после RR≥X", value=True)
arm_rr = colt3.number_input("Arm RR (R)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
use_bar_trail = colt4.checkbox("Bar High/Low Trail", value=True)

colt5, colt6, colt7 = st.columns(3)
trailing_perc = colt5.number_input("Trailing %", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
trailing_offset = colt6.number_input("Offset %", min_value=0.1, max_value=5.0, value=0.4, step=0.1)
trail_lookback = colt7.number_input("Trail lookback bars", min_value=1, max_value=200, value=50, step=1)

colq1, colq2, colq3 = st.columns(3)
use_quality = colq1.checkbox("Фильтр качества SFP", value=True)
wick_min_ticks = colq2.number_input("Min глубина тени (ticks)", min_value=0, value=7, step=1)
close_back_pct = colq3.number_input("Min close-back % от тени", min_value=0.0, max_value=1.0, value=1.00, step=0.05)

run_btn = st.button("🚀 Запустить бэктест", type="primary", use_container_width=True)

# ==== DATA: Bybit REST v5 ====
BYBIT_BASE = "https://api.bybit.com"
KL_ENDPOINT = "/v5/market/kline"

@st.cache_data(show_spinner=False)
def fetch_bybit_klines(symbol:str, interval:str, start_ms:int, end_ms:int)->pd.DataFrame:
    """Тянем историю свечей циклом (Bybit v5 /v5/market/kline, max 1000 баров за запрос)."""
    rows = []
    cursor = None
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": 1000
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(BYBIT_BASE + KL_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {j}")
        result = j.get("result", {})
        kl = result.get("list", [])
        if not kl:
            break
        rows.extend(kl)
        cursor = result.get("nextPageCursor")
        if not cursor:
            break
        # анти rate-limit
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    # Bybit возвращает строки как [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
    df["start"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_convert("UTC")
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("start").reset_index(drop=True)
    df = df.rename(columns={"start":"timestamp"})
    return df[["timestamp","open","high","low","close","volume"]]

# ==== STRATEGY (приближенно к твоему Pine) ====
def run_backtest(df: pd.DataFrame)->dict:
    if df is None or df.empty:
        return {"trades": pd.DataFrame(), "equity": pd.DataFrame(), "stats": {}}

    df = df.copy()
    df["mTick"] = (df["close"] * 0 + 0.01)  # приближение; реально mintick зависят от символа/биржи

    # pivot SFP на 15m = sfpLen=2, lookback 1 бар между плечами
    sfpLen = 2
    # делаем свинг-экстремумы
    def pivot_low(i):
        if i-sfpLen-1 < 0 or i+1 >= len(df): return False
        window = df["low"].iloc[i-sfpLen-1:i+2]
        return df["low"].iloc[i] == window.min() and df["low"].iloc[i] < df["low"].iloc[i-sfpLen]
    def pivot_high(i):
        if i-sfpLen-1 < 0 or i+1 >= len(df): return False
        window = df["high"].iloc[i-sfpLen-1:i+2]
        return df["high"].iloc[i] == window.max() and df["high"].iloc[i] > df["high"].iloc[i-sfpLen]

    # сигналы
    bull_sig = []
    bear_sig = []
    for i in range(len(df)):
        bl = pivot_low(i) and (df["open"].iloc[i] > df["low"].iloc[i-sfpLen]) and (df["close"].iloc[i] > df["low"].iloc[i-sfpLen]) and (df["low"].iloc[i] < df["low"].iloc[i-sfpLen])
        bh = pivot_high(i) and (df["open"].iloc[i] < df["high"].iloc[i-sfpLen]) and (df["close"].iloc[i] < df["high"].iloc[i-sfpLen]) and (df["high"].iloc[i] > df["high"].iloc[i-sfpLen])
        bull_sig.append(bl)
        bear_sig.append(bh)
    df["bull"] = bull_sig
    df["bear"] = bear_sig

    # фильтр качества SFP
    df["bullWickDepth"] = np.where(df["bull"], df["low"].shift(sfpLen) - df["low"], 0.0)
    df["bearWickDepth"] = np.where(df["bear"], df["high"] - df["high"].shift(sfpLen), 0.0)
    df["bullCloseBackOK"] = (df["close"] - df["low"]) >= df["bullWickDepth"] * close_back_pct
    df["bearCloseBackOK"] = (df["high"] - df["close"]) >= df["bearWickDepth"] * close_back_pct
    if use_quality:
        df["bull"] = df["bull"] & (df["bullWickDepth"] >= wick_min_ticks * df["mTick"]) & df["bullCloseBackOK"]
        df["bear"] = df["bear"] & (df["bearWickDepth"] >= wick_min_ticks * df["mTick"]) & df["bearCloseBackOK"]

    # симуляция
    capital = start_capital
    taker_fee = commission_pct / 100.0
    trades = []
    pos = 0  # 0 / +qty / -qty
    entry = sl = qty = 0.0
    long_armed = short_armed = (not arm_after_rr)

    for i in range(1, len(df)):
        o,h,l,c = df.loc[df.index[i], ["open","high","low","close"]]
        ts = df.loc[df.index[i], "timestamp"]

        # exits: trailing/bar-based
        if pos > 0:
            # bar trail
            if enable_smart_trail and use_bar_trail and long_armed:
                lb_low = df["low"].iloc[max(0, i-int(trail_lookback)):i].min()
                bar_stop = max(lb_low, sl)
                if l <= bar_stop:  # выбило стопом
                    exit_price = bar_stop
                    pnl = (exit_price - entry) * qty
                    fee = (entry + exit_price) * qty * taker_fee
                    capital += pnl - fee
                    trades.append([ts, "LONG", entry, exit_price, qty, pnl - fee, capital])
                    pos = 0
                    qty = 0
                    continue
            # simple trailing % (fallback)
            if enable_smart_trail and not use_bar_trail:
                trail = entry * (trailing_perc/100.0)
                trail_px = c - trail
                ts_px = max(trail_px, sl)
                if l <= ts_px:
                    exit_price = ts_px
                    pnl = (exit_price - entry) * qty
                    fee = (entry + exit_price) * qty * taker_fee
                    capital += pnl - fee
                    trades.append([ts, "LONG", entry, exit_price, qty, pnl - fee, capital])
                    pos = 0
                    qty = 0
                    continue

        if pos < 0:
            if enable_smart_trail and use_bar_trail and short_armed:
                lb_high = df["high"].iloc[max(0, i-int(trail_lookback)):i].max()
                bar_stop = min(lb_high, sl)
                if h >= bar_stop:
                    exit_price = bar_stop
                    pnl = (entry - exit_price) * abs(qty)
                    fee = (entry + exit_price) * abs(qty) * taker_fee
                    capital += pnl - fee
                    trades.append([ts, "SHORT", entry, exit_price, abs(qty), pnl - fee, capital])
                    pos = 0; qty = 0
                    continue
            if enable_smart_trail and not use_bar_trail:
                trail = entry * (trailing_perc/100.0)
                trail_px = c + trail
                ts_px = min(trail_px, sl)
                if h >= ts_px:
                    exit_price = ts_px
                    pnl = (entry - exit_price) * abs(qty)
                    fee = (entry + exit_price) * abs(qty) * taker_fee
                    capital += pnl - fee
                    trades.append([ts, "SHORT", entry, exit_price, abs(qty), pnl - fee, capital])
                    pos = 0; qty = 0
                    continue

        # arm RR
        if arm_after_rr and pos > 0 and not long_armed:
            moved = c - entry
            need = (entry - sl) * arm_rr
            long_armed = moved >= need
        if arm_after_rr and pos < 0 and not short_armed:
            moved = entry - c
            need = (sl - entry) * arm_rr
            short_armed = moved >= need

        # entries (только если flat)
        if pos == 0:
            if df["bull"].iloc[i]:
                sl = df["low"].shift(1).iloc[i]
                entry = c
                stop_size = entry - sl
                if stop_size > 0:
                    risk_amt = capital * (risk_pct/100.0)
                    qty = risk_amt / stop_size
                    pos = +1
                    long_armed = (not arm_after_rr)
            elif df["bear"].iloc[i]:
                sl = df["high"].shift(1).iloc[i]
                entry = c
                stop_size = sl - entry
                if stop_size > 0:
                    risk_amt = capital * (risk_pct/100.0)
                    qty = risk_amt / stop_size
                    pos = -1
                    short_armed = (not arm_after_rr)

    trades_df = pd.DataFrame(trades, columns=["time","side","entry","exit","qty","net_pnl","equity"])
    eq = trades_df[["time","equity"]].copy()
    return {"trades": trades_df, "equity": eq, "stats": {"final_capital": (trades_df["equity"].iloc[-1] if len(trades_df)>0 else start_capital)}}

# ==== RUN ====
if run_btn:
    if not period_key or not symbol or not interval:
        st.error("Заполните все параметры")
        st.stop()
    
    days = period_map[period_key]
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    st.write(f"Загрузка данных Bybit: {symbol} {interval}m  {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")
    df = fetch_bybit_klines(str(symbol), str(interval), int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000))
    if df is None or df.empty:
        st.error("Не удалось получить исторические данные. Проверь SYMBOL/interval/доступность Bybit.")
        st.stop()

    res = run_backtest(df)
    trades = res["trades"]
    equity = res["equity"]

    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Equity")
        if not equity.empty:
            equity = equity.set_index("time")
            st.line_chart(equity)
        else:
            st.info("Сделок нет за выбранный период.")
    with colB:
        st.subheader("Итог")
        st.metric("Финальный капитал", f"${res['stats'].get('final_capital', start_capital):,.2f}")
        st.metric("Сделок", len(trades))

    st.subheader("Сделки")
    st.dataframe(trades, use_container_width=True)