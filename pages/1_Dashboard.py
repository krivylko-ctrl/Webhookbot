# pages/1_Dashboard.py
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config

# --- мягкие утилиты (fallback, если в проекте нет utils или там нет нужных функций)
try:
    import utils  # type: ignore
    _fmt_cur = getattr(utils, "format_currency", None)
    _fmt_pct = getattr(utils, "format_percentage", None)
    _pnl_calc = getattr(utils, "calculate_pnl", None)
    _rr_calc  = getattr(utils, "calculate_rr", None)
except Exception:
    utils = None
    _fmt_cur = _fmt_pct = _pnl_calc = _rr_calc = None

def format_currency(x: float) -> str:
    if _fmt_cur:
        return _fmt_cur(x)
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"

def format_percentage(x: float) -> str:
    if _fmt_pct:
        return _fmt_pct(x)
    return f"{float(x):.2f}%"

def calc_pnl(entry: float, price: float, qty: float, direction: str, include_fees: bool = True, fee_rate: float = 0.00055) -> float:
    if _pnl_calc:
        return _pnl_calc(entry, price, qty, direction, include_fees=include_fees)
    gross = (price - entry) * qty if direction == "long" else (entry - price) * qty
    fees = (entry + price) * qty * (fee_rate if include_fees else 0.0)
    return gross - fees

def calc_rr(entry: float, price: float, sl: float, direction: str) -> float:
    if _rr_calc:
        return _rr_calc(entry, price, sl, direction)
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return (price - entry) / risk if direction == "long" else (entry - price) / risk

# --- Streamlit page config
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ---------- helpers ----------
def utc_today_str() -> str:
    return datetime.utcnow().date().isoformat()

def safe_get_trades_today(db: Database) -> int:
    """Кросс-совместимо: если метода нет — считаем по таблице сделок."""
    if hasattr(db, "get_trades_count_today"):
        try:
            return int(db.get_trades_count_today())  # type: ignore[attr-defined]
        except Exception:
            pass
    # fallback
    try:
        trades = db.get_recent_trades(500)  # достаточно для дня
        today = utc_today_str()
        return sum(1 for t in trades if str(t.get("entry_time", "")).startswith(today))
    except Exception:
        return 0

def safe_get_pnl_today(db: Database) -> float:
    """Кросс-совместимо: если метода нет — считаем суммарный PnL закрытых сделок за сегодня."""
    if hasattr(db, "get_pnl_today"):
        try:
            return float(db.get_pnl_today())  # type: ignore[attr-defined]
        except Exception:
            pass
    # fallback
    try:
        trades = db.get_recent_trades(1000)
        today = utc_today_str()
        pnl = 0.0
        for t in trades:
            if str(t.get("exit_time", "")).startswith(today) and str(t.get("status", "")).lower() == "closed":
                try:
                    pnl += float(t.get("pnl") or 0.0)
                except Exception:
                    pass
        return pnl
    except Exception:
        return 0.0

def safe_get_price(bybit_api: BybitAPI, symbol: str) -> float:
    try:
        t = bybit_api.get_ticker(symbol) or {}
        # поддерживаем разные ключи
        for k in ("last_price", "lastPrice", "last", "mark_price", "markPrice"):
            if k in t and t[k] not in (None, "", 0, "0"):
                return float(t[k])
    except Exception:
        pass
    return 0.0

# ---------- init singletons ----------
@st.cache_resource
def init_components():
    cfg = Config()
    db = Database()
    state = StateManager(db)

    api_key = os.getenv("BYBIT_API_KEY", "") or ""
    api_sec = os.getenv("BYBIT_API_SECRET", "") or ""
    testnet = (os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "yes"))

    bybit_api = None
    if api_key and api_sec:
        bybit_api = BybitAPI(api_key, api_sec, testnet=testnet)
        # гарантируем фьючерсы (деривативы)
        bybit_api.set_market_type("linear")

    strategy = KWINStrategy(cfg, bybit_api, state, db)
    # легкий старт: подтянуть свечи один раз (если API есть)
    try:
        if bybit_api:
            strategy.update_candles()
    except Exception:
        pass

    return cfg, db, state, bybit_api, strategy

# ======================= MAIN =======================
def main():
    st.title("📊 Dashboard")

    config, db, state_manager, bybit_api, strategy = init_components()

    # автообновление раз в 5 сек (без бесконечного цикла)
    if "last_update" not in st.session_state:
        st.session_state.last_update = 0.0
    now = time.time()
    if now - st.session_state.last_update > 5:
        st.session_state.last_update = now
        try:
            if bybit_api:
                strategy.update_candles()
                strategy.run_cycle()
        except Exception:
            pass
        # не делаем st.rerun() здесь, чтобы не ловить гонки – стримлит сам перерисует

    # === ОСНОВНЫЕ МЕТРИКИ ===
    st.markdown("### 📈 Основные метрики")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Equity
    with col1:
        equity = state_manager.get_equity()
        st.metric("💰 Equity", f"${equity:.2f}")

    # Position
    position = state_manager.get_current_position()
    with col2:
        if position:
            pos_qty = position.get("size") or position.get("quantity") or 0.0
            pos_text = f"{float(pos_qty):.4f}"
            pos_direction = (position.get("direction") or "").upper()
            st.metric("📍 Позиция", f"{pos_direction} {pos_text}")
        else:
            st.metric("📍 Позиция", "Нет позиции")

    # Price
    with col3:
        if bybit_api:
            price = safe_get_price(bybit_api, config.symbol if hasattr(config, "symbol") else "ETHUSDT")
            if price > 0:
                st.metric(f"💹 Цена {config.symbol if hasattr(config,'symbol') else 'ETHUSDT'}", f"${price:,.2f}")
            else:
                st.metric("💹 Цена", "—")
        else:
            st.metric("💹 Цена", "API отключён")

    # Trades today
    with col4:
        trades_today = safe_get_trades_today(db)
        st.metric("📊 Сделки сегодня", trades_today)

    # PnL today
    with col5:
        pnl_today = safe_get_pnl_today(db)
        st.metric("💵 PnL сегодня", format_currency(pnl_today))

    # === ТЕКУЩАЯ ПОЗИЦИЯ ===
    if position:
        st.markdown("### 🎯 Текущая позиция")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.write(f"**Направление:** {(position.get('direction') or '').upper()}")
            qty = position.get("size") or position.get("quantity") or 0.0
            st.write(f"**Размер:** {float(qty):.4f}")

        with c2:
            entry_price = float(position.get("entry_price") or 0.0)
            st.write(f"**Вход:** ${entry_price:,.2f}")
            if bybit_api and entry_price > 0:
                cur_price = safe_get_price(bybit_api, config.symbol if hasattr(config,"symbol") else "ETHUSDT")
                if cur_price > 0:
                    pnl = calc_pnl(entry_price, cur_price, float(qty), position.get("direction", "long"))
                    color = "green" if pnl >= 0 else "red"
                    st.markdown(
                        f"**Текущий PnL:** <span style='color:{color}'>{format_currency(pnl)}</span>",
                        unsafe_allow_html=True,
                    )

        with c3:
            st.write(f"**Stop Loss:** ${float(position.get('stop_loss') or 0.0):,.2f}")
            tp = position.get("take_profit")
            st.write(f"**Take Profit:** {'—' if tp is None else f'${float(tp):,.2f}'}")

        with c4:
            armed_status = "🟢 Armed" if bool(position.get("armed", False)) else "🔴 Not Armed"
            st.write(f"**Статус:** {armed_status}")
            if bybit_api:
                cur_price = safe_get_price(bybit_api, config.symbol if hasattr(config,"symbol") else "ETHUSDT")
                if cur_price > 0 and float(position.get("entry_price") or 0) > 0:
                    rr = calc_rr(
                        float(position.get("entry_price") or 0),
                        cur_price,
                        float(position.get("stop_loss") or 0),
                        position.get("direction", "long"),
                    )
                    st.write(f"**Текущий RR:** {rr:.2f}")

    # === СТАТИСТИКА ЗА ПЕРИОДЫ ===
    st.markdown("### 📊 Статистика производительности")
    tabs = st.tabs(["30 дней", "60 дней", "180 дней"])
    for i, days in enumerate([30, 60, 180]):
        with tabs[i]:
            try:
                stats = db.get_performance_stats(days=days) if hasattr(db, "get_performance_stats") else {}
            except Exception:
                stats = {}
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Всего сделок", int(stats.get("total_trades", 0)))
                st.metric("Прибыльных", int(stats.get("winning_trades", 0)))

            with c2:
                st.metric("Win Rate", format_percentage(stats.get("win_rate", 0.0)))
                st.metric("Средний RR", f"{float(stats.get('avg_rr', 0.0)):.2f}")

            with c3:
                st.metric("Общий PnL", format_currency(float(stats.get("total_pnl", 0.0))))
                st.metric("Средний PnL", format_currency(float(stats.get("avg_pnl", 0.0))))

            with c4:
                st.metric("Макс. прибыль", format_currency(float(stats.get("max_win", 0.0))))
                st.metric("Макс. убыток", format_currency(float(stats.get("max_loss", 0.0))))

    # === ГРАФИК EQUITY ===
    st.markdown("### 💰 Кривая Equity")
    try:
        equity_data = db.get_equity_history(days=30) if hasattr(db, "get_equity_history") else []
    except Exception:
        equity_data = []

    if equity_data:
        df_eq = pd.DataFrame(equity_data)
        if "timestamp" in df_eq:
            df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_eq["timestamp"], y=df_eq["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=380, xaxis_title="Дата", yaxis_title="Equity ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения кривой equity")

    # === ПОСЛЕДНИЕ СДЕЛКИ ===
    st.markdown("### 📋 Последние сделки")
    try:
        recent_trades = db.get_recent_trades(20)
    except Exception:
        recent_trades = []

    if recent_trades:
        df_tr = pd.DataFrame(recent_trades)
        # дружественное отображение
        cols = ["entry_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "rr", "status"]
        for c in cols:
            if c not in df_tr.columns:
                continue
            if c in ("entry_time", "exit_time"):
                df_tr[c] = pd.to_datetime(df_tr[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            elif c in ("entry_price", "exit_price", "pnl", "rr"):
                df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce").round(2)
            elif c == "quantity":
                df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce").round(4)

        rename = {
            "entry_time": "Время входа",
            "direction": "Направление",
            "entry_price": "Цена входа",
            "exit_price": "Цена выхода",
            "quantity": "Количество",
            "pnl": "PnL ($)",
            "rr": "RR",
            "status": "Статус",
        }
        st.dataframe(df_tr[ [c for c in cols if c in df_tr.columns] ].rename(columns=rename),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Нет сделок для отображения")

    # === УПРАВЛЕНИЕ БОТОМ ===
    st.markdown("### 🎛️ Управление ботом")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶️ Запустить бота", use_container_width=True):
            state_manager.set_bot_status("running")
            st.success("Бот запущен!")

    with c2:
        if st.button("⏹️ Остановить бота", use_container_width=True):
            state_manager.set_bot_status("stopped")
            st.warning("Бот остановлен!")

    with c3:
        if st.button("🔄 Обновить данные", use_container_width=True):
            try:
                if bybit_api:
                    strategy.update_candles()
                    strategy.run_cycle()
                st.success("Данные обновлены")
            except Exception as e:
                st.error(f"Ошибка обновления: {e}")

    # Статус бота
    bot_status = state_manager.get_bot_status()
    status_color = "green" if bot_status == "running" else "red"
    st.markdown(
        f"**Статус бота:** <span style='color:{status_color}'>{bot_status.upper()}</span>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
