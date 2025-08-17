import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config
import config as cfg

# ===================== Страница =====================
st.set_page_config(
    page_title="KWIN Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Kwin Trading Bot")

# ===================== Проверка ENV =====================
try:
    cfg.must_have()
    st.success(f"ENV OK: SYMBOL={cfg.SYMBOL} | INTERVALS={cfg.INTERVALS} | ACCOUNT={cfg.BYBIT_ACCOUNT_TYPE}")
except Exception as e:
    st.error(f"⛔ Настройки не заданы: {e}")
    st.info("Добавьте переменные окружения: BYBIT_API_KEY, BYBIT_API_SECRET")
    st.stop()

# ===================== Инициализация =====================
@st.cache_resource
def init_components():
    config = Config()
    db = Database()
    state_manager = StateManager(db)

    # Подключение к Bybit или демо-API
    if cfg.BYBIT_API_KEY and cfg.BYBIT_API_SECRET:
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                st.warning("⚠️ Bybit API недоступен (гео/сеть). Включен демо-режим.")
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
        except Exception:
            st.warning("⚠️ Проблема с Bybit API. Включен демо-режим.")
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
    else:
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        st.info("ℹ️ API ключи не настроены. Работаем в демо-режиме.")

    # Стратегия
    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    return config, db, state_manager, bybit_api, strategy

# ===================== Вспомогательные =====================
def _fmt_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "$0.00"

# ===================== Основной рендер =====================
def main():
    config, db, state_manager, bybit_api, strategy = init_components()

    # Боковая панель
    with st.sidebar:
        st.header("🎛️ Управление ботом")

        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Старт", use_container_width=True):
                if not st.session_state.bot_running:
                    st.session_state.bot_running = True
                    st.success("Бот запущен!")
        with c2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
                    st.error("Бот остановлен!")

        st.button("🔄 Обновить сейчас", use_container_width=True)

        st.markdown("### 📡 Статус подключения")
        try:
            # Признак демо-API
            if hasattr(bybit_api, 'current_price'):
                st.warning("🎮 Демо-режим активен")
                st.caption("Bybit API недоступен — используются тестовые данные.")
            else:
                server_time = bybit_api.get_server_time()
                if server_time:
                    st.success("✅ Подключение к Bybit активно")
                else:
                    st.error("❌ Ошибка подключения к Bybit")
        except Exception as e:
            st.error(f"❌ Ошибка API: {e}")

        st.markdown("### ⚙️ Текущие настройки")
        st.write(f"**Символ:** {getattr(strategy, 'symbol', cfg.SYMBOL)}")
        st.write(f"**Риск на сделку:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Макс. позиция:** {config.max_qty_manual} ETH")

        st.markdown("### 🧠 Smart Trailing")
        st.write(f"Включен: {'✅' if config.enable_smart_trail else '❌'}")
        st.write(f"Percent trailing: {getattr(config, 'trailing_perc', 0.5)}%")
        st.write(f"Offset trailing: {getattr(config, 'trailing_offset_perc', 0.4)}%")
        st.write(f"Bar lookback: {getattr(config, 'trail_lookback', 50)}")
        st.write(f"Buffer (ticks): {getattr(config, 'trail_buf_ticks', 0)}")
        st.write(f"ARM RR: {getattr(config, 'arm_rr', 0.5)} | after RR: {'✅' if getattr(config, 'use_arm_after_rr', True) else '❌'}")

    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Дашборд", "📈 График", "💰 Equity", "📋 Сделки"])

    with tab1:
        show_dashboard(db, state_manager, strategy)

    with tab2:
        show_chart(bybit_api, db, strategy)

    with tab3:
        show_equity_curve(db)

    with tab4:
        show_trades_table(db)

# ===================== Разделы =====================
def show_dashboard(db: Database, state_manager: StateManager, strategy: KWINStrategy):
    col1, col2, col3, col4 = st.columns(4)

    equity_val = state_manager.get_equity() or 0.0
    with col1:
        st.metric("💰 Equity", _fmt_money(equity_val))

    cur_pos = state_manager.get_current_position()
    with col2:
        pos_text = "нет"
        if cur_pos and float(cur_pos.get("size") or 0) > 0:
            pos_text = f"{cur_pos.get('direction', '-')} {float(cur_pos.get('size', 0)):.4f} @ {float(cur_pos.get('entry_price', 0)):.2f}"
        st.metric("📍 Позиция", pos_text)

    with col3:
        trades_today = db.get_trades_count_today()
        st.metric("📊 Сделки сегодня", int(trades_today or 0))

    with col4:
        pnl_today = db.get_pnl_today() or 0.0
        st.metric("💵 PnL сегодня", _fmt_money(pnl_today))

    st.markdown("### 📈 Статистика за 30 дней")
    stats = db.get_performance_stats(days=30)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Win Rate", f"{float(stats.get('win_rate', 0) or 0):.1f}%")
    with c2:
        st.metric("📊 Avg RR", f"{float(stats.get('avg_rr', 0) or 0):.2f}")
    with c3:
        st.metric("⏱️ Avg Hold Time", f"{float(stats.get('avg_hold_time', 0) or 0):.1f}h")

def show_chart(bybit_api, db: Database, strategy: KWINStrategy):
    symbol = getattr(strategy, "symbol", "ETHUSDT")
    st.markdown(f"### 📈 График {symbol} (15m)")

    try:
        # Получаем последние 100 свечей по символу стратегии
        klines = bybit_api.get_klines(symbol, "15", 100)
        if not klines:
            st.warning("Не удалось получить данные свечей")
            return

        df = pd.DataFrame(klines)
        # нормализация таймстампа
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        elif "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
        else:
            st.warning("Нет поля timestamp в данных свечей")
            return

        df = df.sort_values("timestamp")
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol
        )])

        # Добавляем сделки (входы/выходы)
        trades = db.get_recent_trades(200)
        if trades:
            entries_x, entries_y, entries_color, entries_symbol, entries_text = [], [], [], [], []
            exits_x, exits_y, exits_color, exits_symbol, exits_text = [], [], [], [], []

            for tr in trades:
                # Entry
                if tr.get("entry_time") and tr.get("entry_price") is not None:
                    et = pd.to_datetime(tr["entry_time"], errors="coerce")
                    if pd.notna(et):
                        entries_x.append(et)
                        entries_y.append(float(tr["entry_price"]))
                        entries_color.append("green" if tr.get("direction") == "long" else "red")
                        entries_symbol.append("triangle-up" if tr.get("direction") == "long" else "triangle-down")
                        qty = tr.get("quantity") or 0.0
                        entries_text.append(f"Entry {tr.get('direction', '-')}"
                                            f"<br>qty={float(qty):.4f}")

                # Exit
                if tr.get("exit_time") and tr.get("exit_price") is not None:
                    xt = pd.to_datetime(tr["exit_time"], errors="coerce")
                    if pd.notna(xt):
                        exits_x.append(xt)
                        exits_y.append(float(tr["exit_price"]))
                        pnl = tr.get("pnl")
                        rr = tr.get("rr")
                        c = "green" if (pnl or 0) >= 0 else "red"
                        exits_color.append(c)
                        exits_symbol.append("x")
                        pnl_s = f"{float(pnl):.2f}" if pnl is not None else "—"
                        rr_s = f"{float(rr):.2f}" if rr is not None else "—"
                        exits_text.append(f"Exit ({tr.get('exit_reason','-')})"
                                          f"<br>PNL={pnl_s} | RR={rr_s}")

            if entries_x:
                fig.add_trace(go.Scatter(
                    x=entries_x, y=entries_y, mode="markers",
                    marker=dict(size=10, color=entries_color, symbol=entries_symbol, line=dict(width=1)),
                    name="Entries", hovertext=entries_text, hoverinfo="text"
                ))

            if exits_x:
                fig.add_trace(go.Scatter(
                    x=exits_x, y=exits_y, mode="markers",
                    marker=dict(size=10, color=exits_color, symbol=exits_symbol, line=dict(width=1)),
                    name="Exits", hovertext=exits_text, hoverinfo="text"
                ))

        fig.update_layout(
            title=f"{symbol} 15m со сделками",
            xaxis_title="Время",
            yaxis_title="Цена",
            height=640,
            legend=dict(orientation="h", y=1.02, x=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка загрузки графика: {e}")

def show_equity_curve(db: Database):
    st.markdown("### 💰 Кривая Equity")
    data = db.get_equity_history(days=30)
    if not data:
        st.info("Нет данных для отображения кривой equity")
        return

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["equity"], mode="lines",
        name="Equity", line=dict(width=2)
    ))
    fig.update_layout(
        title="Изменение Equity за последние 30 дней",
        xaxis_title="Дата",
        yaxis_title="Equity ($)",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

def show_trades_table(db: Database):
    st.markdown("### 📋 История сделок")
    trades = db.get_recent_trades(200)
    if not trades:
        st.info("Нет сделок для отображения")
        return

    df = pd.DataFrame(trades)
    # приведение типов/форматирование
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

    for col in ("entry_price", "exit_price", "quantity", "pnl", "rr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # округления
    for col in ("entry_price", "exit_price", "pnl", "rr"):
        if col in df.columns:
            df[col] = df[col].round(2)
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].round(4)

    view_cols = ["entry_time", "direction", "entry_price", "exit_price", "quantity", "pnl", "rr", "status", "exit_reason"]
    view_cols = [c for c in view_cols if c in df.columns]

    st.dataframe(df[view_cols].sort_values("entry_time", ascending=False), use_container_width=True)

# ===================== Запуск =====================
if __name__ == "__main__":
    main()
