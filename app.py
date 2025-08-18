import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
import os

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config
import config as cfg

# Настройка страницы
st.set_page_config(
    page_title="KWIN Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Kwin Trading Bot")

# Проверяем переменные окружения сразу
try:
    cfg.must_have()
    st.success(f"ENV OK: {cfg.SYMBOL} | {cfg.INTERVALS} | {cfg.BYBIT_ACCOUNT_TYPE}")
except Exception as e:
    st.error(f"⛔ Настройки не заданы: {e}")
    st.info("Добавьте переменные окружения: BYBIT_API_KEY, BYBIT_API_SECRET")
    st.stop()

# Инициализация компонентов
@st.cache_resource
def init_components():
    config = Config()
    db = Database()
    state_manager = StateManager(db)

    # ====== COMPAT: shims for save_bot_state / get_bot_state ======
    # (стратегия их вызывает; в твоём Database может не быть этих методов)
    if not hasattr(db, "save_bot_state"):
        def _save_bot_state(state: dict):
            try:
                # если есть какие-то KV/мета методы — используем
                if hasattr(db, "set_meta"):
                    db.set_meta("bot_state", state)
                elif hasattr(db, "save_kv"):
                    db.save_kv("bot_state", state)
                else:
                    # тихий no-op, чтобы не спамить ошибками
                    pass
            except Exception:
                # никакого падения — просто не сохраняем
                pass
        db.save_bot_state = _save_bot_state

    if not hasattr(db, "get_bot_state"):
        def _get_bot_state():
            try:
                if hasattr(db, "get_meta"):
                    return db.get_meta("bot_state") or {}
                elif hasattr(db, "get_kv"):
                    return db.get_kv("bot_state") or {}
            except Exception:
                pass
            return {}
        db.get_bot_state = _get_bot_state
    # =============================================================

    # ====== Точечно: прокидываем настройки Smart Trailing / ARM ======
    config.enable_smart_trail      = bool(getattr(cfg, "ENABLE_SMART_TRAIL", True))
    config.trailing_perc           = float(getattr(cfg, "TRAILING_PERC", 0.5))
    config.trailing_offset_perc    = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))
    config.trailing_offset         = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))  # совместимость

    config.use_arm_after_rr        = bool(getattr(cfg, "USE_ARM_AFTER_RR", True))
    config.arm_rr                  = float(getattr(cfg, "ARM_RR", 0.5))

    # Базовые риски/символ (оставляю как есть, с безопасными дефолтами)
    config.risk_pct                = float(getattr(cfg, "RISK_PCT", getattr(config, "risk_pct", 3.0)))
    config.risk_reward             = float(getattr(cfg, "RISK_REWARD", getattr(config, "risk_reward", 1.3)))
    if hasattr(cfg, "SYMBOL"):
        config.symbol = cfg.SYMBOL
    # ==================================================================

    # Используем API ключи из конфига
    if getattr(cfg, "BYBIT_API_KEY", None) and getattr(cfg, "BYBIT_API_SECRET", None):
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                st.warning("⚠️ Bybit API недоступен из-за гео-ограничений. Включен демо-режим.")
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
        except:
            st.warning("⚠️ Проблема с подключением к Bybit API. Включен демо-режим.")
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
    else:
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        st.info("ℹ️ API ключи не настроены. Работаем в демо-режиме.")

    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    return config, db, state_manager, bybit_api, strategy

def main():
    config, db, state_manager, bybit_api, strategy = init_components()

    if bybit_api is None:
        st.error("⚠️ API ключи Bybit не настроены. Добавьте BYBIT_API_KEY и BYBIT_API_SECRET.")
        st.stop()

    # Боковая панель
    with st.sidebar:
        st.header("🎛️ Управление ботом")

        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Старт", use_container_width=True):
                if not st.session_state.bot_running:
                    st.session_state.bot_running = True
                    st.success("Бот запущен!")
        with col2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
                    st.error("Бот остановлен!")

        st.markdown("### 📡 Статус подключения")
        try:
            if hasattr(bybit_api, 'current_price'):  # Демо API
                st.warning("🎮 Демо-режим активен")
                st.caption("⚠️ Bybit API заблокирован географически. Используются тестовые данные.")
                if st.button("ℹ️ Подробнее о проблеме"):
                    st.info("""
                    **Проблема:** Сервер находится в регионе, заблокированном Bybit.
                    **Решения:** локальный запуск, VPS в разрешённом регионе, прокси/VPN.
                    """)
            else:
                server_time = bybit_api.get_server_time()
                if server_time:
                    st.success("✅ Подключение к Bybit активно")
                else:
                    st.error("❌ Ошибка подключения к Bybit")
        except Exception as e:
            st.error(f"❌ Ошибка API: {e}")

        st.markdown("### ⚙️ Текущие настройки")
        st.write(f"**Риск:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Макс. позиция:** {getattr(config, 'max_qty_manual', 0)} ETH")
        st.write(f"**Трейлинг активен:** {'✅' if config.enable_smart_trail else '❌'}")

        with st.expander("🔧 Smart Trailing / ARM (текущие)"):
            st.write(f"**Trailing % (от entry):** {config.trailing_perc}%")
            st.write(f"**Trailing Offset %:** {config.trailing_offset_perc}%")
            st.write(f"**Arm after RR:** {'Да' if config.use_arm_after_rr else 'Нет'}")
            st.write(f"**ARM RR (R):** {config.arm_rr}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Дашборд", "📈 График", "💰 Equity", "📋 Сделки"])

    with tab1:
        show_dashboard(db, state_manager, strategy)
    with tab2:
        show_chart(bybit_api, db, strategy)
    with tab3:
        show_equity_curve(db)
    with tab4:
        show_trades_table(db)

def show_dashboard(db, state_manager, strategy):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Equity", f"${state_manager.get_equity():.2f}")
    with col2:
        current_pos = state_manager.get_current_position()
        pos_text = f"{current_pos.get('size', 0):.4f} ETH" if current_pos else "0 ETH"
        st.metric("📍 Позиция", pos_text)
    with col3:
        trades_today = db.get_trades_count_today()
        st.metric("📊 Сделки сегодня", trades_today)
    with col4:
        pnl_today = db.get_pnl_today()
        st.metric("💵 PnL сегодня", f"${pnl_today:.2f}")

    st.markdown("### 📈 Статистика за 30 дней")
    stats = db.get_performance_stats(days=30)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col2:
        st.metric("📊 Avg RR", f"{stats.get('avg_rr', 0):.2f}")
    with col3:
        st.metric("⏱️ Avg Hold Time", f"{stats.get('avg_hold_time', 0):.1f}h")

def show_chart(bybit_api, db, strategy):
    st.markdown("### 📈 График ETH/USDT")
    if bybit_api:
        try:
            klines = bybit_api.get_klines("ETHUSDT", "15", 100)
            if klines:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="ETH/USDT"
                )])
                trades = db.get_recent_trades(50)
                for trade
