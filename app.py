# app.py
import os
import time
import threading
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config
import config as cfg


# ============================ UI / PAGE ============================

st.set_page_config(
    page_title="KWIN Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Kwin Trading Bot")

# ============================ ENV CHECK ============================

try:
    cfg.must_have()
    st.success(f"ENV OK: {cfg.SYMBOL} | {cfg.INTERVALS} | {cfg.BYBIT_ACCOUNT_TYPE}")
except Exception as e:
    st.error(f"⛔ Настройки не заданы: {e}")
    st.info("Добавьте переменные окружения: BYBIT_API_KEY, BYBIT_API_SECRET")
    st.stop()

# ============================ UTILS ============================

def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _append_log(message: str):
    # Потокобезопасный «мягкий» лог — просто дописываем строчку в session_state
    if "logs" not in st.session_state:
        st.session_state.logs = []
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {message}")
    # ограничим буфер
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


# ============================ INIT COMPONENTS ============================

@st.cache_resource
def init_components():
    config = Config()
    db = Database()
    state_manager = StateManager(db)

    # ----- Прокидываем ключевые настройки из cfg/ENV -----
    config.enable_smart_trail   = bool(getattr(cfg, "ENABLE_SMART_TRAIL", True))
    config.trailing_perc        = float(getattr(cfg, "TRAILING_PERC", 0.5))            # %
    config.trailing_offset_perc = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))     # %
    config.trailing_offset      = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))     # совместимость

    config.use_arm_after_rr     = bool(getattr(cfg, "USE_ARM_AFTER_RR", True))
    config.arm_rr               = float(getattr(cfg, "ARM_RR", 0.5))

    config.risk_pct             = float(getattr(cfg, "RISK_PCT", getattr(config, "risk_pct", 3.0)))
    config.risk_reward          = float(getattr(cfg, "RISK_REWARD", getattr(config, "risk_reward", 1.3)))

    if hasattr(cfg, "SYMBOL"):
        config.symbol = cfg.SYMBOL

    # ----- API -----
    if getattr(cfg, "BYBIT_API_KEY", None) and getattr(cfg, "BYBIT_API_SECRET", None):
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                _append_log("Bybit API недоступен (гео/сеть) → демо-режим.")
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
        except Exception:
            _append_log("Ошибка подключения к Bybit → демо-режим.")
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
    else:
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        _append_log("API ключи не заданы → демо-режим.")

    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    return config, db, state_manager, bybit_api, strategy


# ============================ BACKGROUND LOOP ============================

def _bg_bot_loop(bybit_api, strategy: KWINStrategy, state_manager: StateManager, config: Config,
                 stop_event: threading.Event, poll_sec: float):

    last_closed_15m_ts = 0   # защита от дублей закрытых 15m баров
    last_closed_1m_ts  = 0   # защита от дублей закрытых 1m баров
    last_equity_pull   = 0

    _append_log("Фоновый цикл стартовал.")

    try:
        while not stop_event.is_set():
            # === 0) Тикер/цена (для ARM/трейлинга) ===
            try:
                if hasattr(bybit_api, "get_ticker"):
                    _ = bybit_api.get_ticker(config.symbol)
            except Exception:
                pass

            # === 1) 15m: отдать РОВНО один новый закрытый бар в стратегию ===
            try:
                if hasattr(bybit_api, "get_klines"):
                    kl_15 = bybit_api.get_klines(config.symbol, "15", 3) or []
                    if kl_15:
                        df = pd.DataFrame(kl_15)
                        if "timestamp" in df.columns:
                            df = df.sort_values("timestamp")
                            # Последняя строка может быть формирующимся баром. Возьмём предпоследний как «точно закрытый».
                            row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
                            ts = int(row.get("timestamp") or 0)
                            # Закрытый бар не должен быть слишком «свежим»: минимум 1 минута назад
                            if ts and (ts != last_closed_15m_ts) and (_now_utc_ms() - ts > 60_000):
                                strategy.on_bar_close_15m({
                                    "timestamp": ts,
                                    "open":  _safe_float(row.get("open")),
                                    "high":  _safe_float(row.get("high")),
                                    "low":   _safe_float(row.get("low")),
                                    "close": _safe_float(row.get("close")),
                                })
                                last_closed_15m_ts = ts
                                _append_log(f"15m бар отдан в стратегию: {datetime.utcfromtimestamp(ts/1000)} UTC")
            except Exception as e:
                _append_log(f"Ошибка на шаге 15m: {e}")

            # === 2) 1m: интрабар трейлинг (если включен) ===
            try:
                if strategy.config.enable_smart_trail and hasattr(bybit_api, "get_klines"):
                    kl_1 = bybit_api.get_klines(config.symbol, "1", 2) or []
                    if kl_1:
                        df1 = pd.DataFrame(kl_1).sort_values("timestamp")
                        row1 = df1.iloc[-1]       # последняя минута (обычно закрытая)
                        ts1 = int(row1.get("timestamp") or 0)
                        if ts1 and ts1 != last_closed_1m_ts:
                            strategy.on_bar_close_1m({
                                "timestamp": ts1,
                                "open":  _safe_float(row1.get("open")),
                                "high":  _safe_float(row1.get("high")),
                                "low":   _safe_float(row1.get("low")),
                                "close": _safe_float(row1.get("close")),
                            })
                            last_closed_1m_ts = ts1
            except Exception as e:
                _append_log(f"Ошибка на шаге 1m: {e}")

            # === 3) Трейлинг на каждом тике (эмуляция calc_on_every_tick) ===
            try:
                strategy.process_trailing()
            except Exception as e:
                _append_log(f"Ошибка process_trailing: {e}")

            # === 4) Equity периодически (раз в 30 сек) ===
            try:
                now = time.time()
                if now - last_equity_pull > 30:
                    last_equity_pull = now
                    if hasattr(strategy, "_update_equity"):
                        strategy._update_equity()
            except Exception:
                pass

            # === пауза опроса ===
            stop_event.wait(poll_sec)

    finally:
        _append_log("Фоновый цикл остановлен.")


def _ensure_bg_controls():
    # Инициализация управляющих флагов/объектов
    if "bot_running" not in st.session_state:
        st.session_state.bot_running = False
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if "bot_thread" not in st.session_state:
        st.session_state.bot_thread = None


def _start_bot_thread(bybit_api, strategy, state_manager, config, poll_sec: float):
    _ensure_bg_controls()

    # Если поток уже жив — не дублируем
    th = st.session_state.bot_thread
    if th and th.is_alive():
        return

    # На всякий случай сбросим старый stop_event
    st.session_state.stop_event.clear()

    th = threading.Thread(
        target=_bg_bot_loop,
        name="kwin-bg-loop",
        args=(bybit_api, strategy, state_manager, config, st.session_state.stop_event, poll_sec),
        daemon=True,
    )
    st.session_state.bot_thread = th
    th.start()


def _stop_bot_thread():
    _ensure_bg_controls()
    st.session_state.stop_event.set()
    # Не блокируем UI .join(), поток выключится сам


# ============================ MAIN UI ============================

def main():
    config, db, state_manager, bybit_api, strategy = init_components()

    # -------- SIDEBAR --------
    with st.sidebar:
        st.header("🎛️ Управление ботом")

        _ensure_bg_controls()

        # Частота опроса
        poll_sec = st.slider("Частота опроса (сек)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Старт", use_container_width=True):
                if not st.session_state.bot_running:
                    st.session_state.bot_running = True
                    _start_bot_thread(bybit_api, strategy, state_manager, config, poll_sec)
                    st.success("Бот запущен!")

        with col2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
                    _stop_bot_thread()
                    st.error("Бот остановлен!")

        # Быстрые тумблеры (живые — влияют сразу)
        st.markdown("### ⚙️ Быстрые настройки")
        config.enable_smart_trail = st.toggle("✅ Smart Trailing", value=bool(getattr(config, "enable_smart_trail", True)))
        config.use_arm_after_rr   = st.toggle("🛡️ Arm after RR", value=bool(getattr(config, "use_arm_after_rr", True)))
        config.arm_rr             = st.number_input("Arm RR (R)", min_value=0.1, max_value=5.0, value=float(getattr(config, "arm_rr", 0.5)), step=0.1)
        config.trailing_perc      = st.number_input("Trailing %", min_value=0.1, max_value=5.0, value=float(getattr(config, "trailing_perc", 0.5)), step=0.1)
        config.trailing_offset_perc = st.number_input("Trailing Offset %", min_value=0.0, max_value=5.0, value=float(getattr(config, "trailing_offset_perc", 0.4)), step=0.1)

        # Статус подключения
        st.markdown("### 📡 Подключение")
        try:
            if hasattr(bybit_api, 'current_price'):  # демо-API
                st.warning("🎮 Демо-режим активен")
            else:
                server_time = bybit_api.get_server_time()
                if server_time:
                    st.success("✅ Bybit API")
                else:
                    st.error("❌ Ошибка подключения к Bybit")
        except Exception as e:
            st.error(f"❌ Ошибка API: {e}")

        # Инфо по символу/риску
        st.markdown("### 📋 Текущие параметры")
        st.write(f"**Символ:** {config.symbol}")
        st.write(f"**Риск/сделка:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Макс. позиция:** {getattr(config, 'max_qty_manual', 0)}")
        st.write(f"**Трейлинг активен:** {'✅' if config.enable_smart_trail else '❌'}")

        # Логи
        with st.expander("🪵 Логи (последние)"):
            if "logs" in st.session_state and st.session_state.logs:
                st.text("\n".join(st.session_state.logs[-200:]))
            else:
                st.caption("Пока пусто.")

    # -------- TABS --------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Дашборд", "📈 График", "💰 Equity", "📋 Сделки"])

    with tab1:
        show_dashboard(db, state_manager, strategy)

    with tab2:
        show_chart(bybit_api, db, strategy)

    with tab3:
        show_equity_curve(db)

    with tab4:
        show_trades_table(db)


# ============================ VIEWS ============================

def show_dashboard(db, state_manager, strategy):
    col1, col2, col3, col4 = st.columns(4)

    eq = float(state_manager.get_equity() or 0.0)
    with col1:
        st.metric("💰 Equity", f"${eq:.2f}")

    with col2:
        current_pos = state_manager.get_current_position()
        if current_pos:
            sz = float(current_pos.get('size') or 0)
            base = getattr(strategy.config, "symbol", "BASE")
            st.metric("📍 Позиция", f"{sz:.4f} ({base})")
        else:
            st.metric("📍 Позиция", "0")

    with col3:
        try:
            trades_today = db.get_trades_count_today()
        except Exception:
            trades_today = 0
        st.metric("📊 Сделки сегодня", trades_today)

    with col4:
        try:
            pnl_today = float(db.get_pnl_today() or 0.0)
        except Exception:
            pnl_today = 0.0
        st.metric("💵 PnL сегодня", f"${pnl_today:.2f}")

    st.markdown("### 📈 Статистика за 30 дней")
    try:
        stats = db.get_performance_stats(days=30) or {}
    except Exception:
        stats = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Win Rate", f"{float(stats.get('win_rate', 0)):.1f}%")
    with c2:
        st.metric("📊 Avg RR", f"{float(stats.get('avg_rr', 0)):.2f}")
    with c3:
        st.metric("⏱️ Avg Hold Time", f"{float(stats.get('avg_hold_time', 0)):.1f}h")


def show_chart(bybit_api, db, strategy):
    symbol = getattr(strategy.config, "symbol", "ETHUSDT")
    st.markdown(f"### 📈 График {symbol}")

    if not hasattr(bybit_api, "get_klines"):
        st.info("API не поддерживает свечи")
        return

    try:
        klines = bybit_api.get_klines(symbol, "15", 120) or []
        if not klines:
            st.warning("Не удалось получить данные свечей")
            return

        df = pd.DataFrame(klines)
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )])

        trades = db.get_recent_trades(80) or []
        for t in trades:
            try:
                if t.get('entry_time'):
                    entry_time = pd.to_datetime(t['entry_time'], errors='coerce')
                    fig.add_trace(go.Scatter(
                        x=[entry_time],
                        y=[float(t['entry_price'])],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if t.get('direction') == 'long' else 'triangle-down',
                            size=10,
                            color='green' if t.get('direction') == 'long' else 'red'
                        ),
                        name=f"Entry {t.get('direction')}"
                    ))
            except Exception:
                pass

        fig.update_layout(
            title=f"{symbol} 15m с входами",
            xaxis_title="Время",
            yaxis_title="Цена",
            height=600,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка загрузки графика: {e}")


def show_equity_curve(db):
    st.markdown("### 💰 Кривая Equity")
    try:
        equity_data = db.get_equity_history(days=30)
    except Exception:
        equity_data = None

    if not equity_data:
        st.info("Нет данных для отображения кривой equity")
        return

    df = pd.DataFrame(equity_data)
    if "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['equity'],
        mode='lines',
        name='Equity'
    ))
    fig.update_layout(
        title="Изменение Equity за последние 30 дней",
        xaxis_title="Дата",
        yaxis_title="Equity ($)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_trades_table(db):
    st.markdown("### 📋 История сделок")
    try:
        trades = db.get_recent_trades(200)
    except Exception:
        trades = None

    if not trades:
        st.info("Нет сделок для отображения")
        return

    df = pd.DataFrame(trades)
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in ("pnl", "rr", "entry_price", "exit_price", "quantity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols = [c for c in ['entry_time', 'direction', 'entry_price', 'exit_price', 'quantity', 'pnl', 'rr', 'status'] if c in df.columns]
    st.dataframe(df[cols].round(4), use_container_width=True)


# ============================ ENTRY ============================

if __name__ == "__main__":
    main()
