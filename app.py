import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import threading
from datetime import datetime
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


@st.cache_resource
def init_components():
    config = Config()
    db = Database()
    state_manager = StateManager(db)

    # Smart Trailing / ARM — строго как в Pine
    config.enable_smart_trail   = bool(getattr(cfg, "ENABLE_SMART_TRAIL", True))
    config.trailing_perc        = float(getattr(cfg, "TRAILING_PERC", 0.5))
    config.trailing_offset_perc = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))

    config.use_arm_after_rr     = bool(getattr(cfg, "USE_ARM_AFTER_RR", True))
    config.arm_rr               = float(getattr(cfg, "ARM_RR", 0.5))
    config.arm_rr_basis         = str(getattr(cfg, "ARM_RR_BASIS", getattr(config, "arm_rr_basis", "extremum"))).lower()
    if config.arm_rr_basis not in ("extremum", "last"):
        config.arm_rr_basis = "extremum"

    config.risk_pct             = float(getattr(cfg, "RISK_PCT", getattr(config, "risk_pct", 3.0)))
    config.risk_reward          = float(getattr(cfg, "RISK_REWARD", getattr(config, "risk_reward", 1.3)))

    if hasattr(cfg, "SYMBOL"):
        config.symbol = cfg.SYMBOL

    # API init
    if getattr(cfg, "BYBIT_API_KEY", None) and getattr(cfg, "BYBIT_API_SECRET", None):
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                st.warning("⚠️ Bybit API недоступен из-за гео. Включен демо-режим.")
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
        except:
            st.warning("⚠️ Проблема с Bybit API. Включен демо-режим.")
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
    else:
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        st.info("ℹ️ API ключи не настроены. Демо-режим.")

    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    return config, db, state_manager, bybit_api, strategy


# ---------- ФОНОВЫЙ ЦИКЛ (15m + 1m интрабар + equity) ----------
def _bg_bot_loop(bybit_api, strategy: KWINStrategy, state_manager: StateManager, config: Config, poll_sec: float = 2.0):
    last_15m_ts = 0
    last_1m_ts  = 0
    loop_i = 0

    while getattr(st.session_state, "bot_running", False):
        try:
            # 0) Подтягиваем цену (для ARM/трейлинга)
            try:
                _ = bybit_api.get_ticker(config.symbol)
            except Exception:
                pass

            # 1) Закрытые 15m бары
            try:
                kl = bybit_api.get_klines(config.symbol, "15", 3) if hasattr(bybit_api, "get_klines") else []
                if kl:
                    df = pd.DataFrame(kl)
                    if "timestamp" in df.columns:
                        df = df.sort_values("timestamp")
                        last = df.iloc[-1].to_dict()
                        ts = int(last.get("timestamp", 0))
                        if ts and ts != last_15m_ts:
                            strategy.on_bar_close_15m({
                                "timestamp": int(last["timestamp"]),
                                "open": float(last["open"]),
                                "high": float(last["high"]),
                                "low":  float(last["low"]),
                                "close": float(last["close"])
                            })
                            last_15m_ts = ts
            except Exception:
                pass

            # 2) Интрабар (обычно 1m) — для Smart Trail
            try:
                intrabar_tf = str(getattr(config, "intrabar_tf", "1"))
                kl1 = bybit_api.get_klines(config.symbol, intrabar_tf, 3) if hasattr(bybit_api, "get_klines") else []
                if kl1:
                    df1 = pd.DataFrame(kl1)
                    if "timestamp" in df1.columns:
                        df1 = df1.sort_values("timestamp")
                        last1 = df1.iloc[-1].to_dict()
                        ts1 = int(last1.get("timestamp", 0))
                        if ts1 and ts1 != last_1m_ts:
                            strategy.on_bar_close_1m({
                                "timestamp": int(last1["timestamp"]),
                                "open": float(last1["open"]),
                                "high": float(last1["high"]),
                                "low":  float(last1["low"]),
                                "close": float(last1["close"])
                            })
                            last_1m_ts = ts1
            except Exception:
                pass

            # 3) Обновление трейлинга
            try:
                strategy.process_trailing()
            except Exception:
                pass

            # 4) Периодическое обновление equity
            loop_i += 1
            if loop_i % 30 == 0:
                try:
                    strategy._update_equity()
                except Exception:
                    pass

        except Exception:
            pass

        time.sleep(poll_sec)


def _start_bot_thread(bybit_api, strategy, state_manager, config):
    if "bot_thread" in st.session_state and st.session_state.bot_thread and st.session_state.bot_thread.is_alive():
        return
    th = threading.Thread(
        target=_bg_bot_loop,
        args=(bybit_api, strategy, state_manager, config, 2.0),
        daemon=True
    )
    st.session_state.bot_thread = th
    th.start()


def _stop_bot_thread():
    th = st.session_state.get("bot_thread")
    if th and th.is_alive():
        pass


def main():
    config, db, state_manager, bybit_api, strategy = init_components()

    if bybit_api is None:
        st.error("⚠️ Нет API. Добавьте ключи BYBIT_API_KEY/SECRET.")
        st.stop()

    # Сайдбар
    with st.sidebar:
        st.header("🎛️ Управление ботом")
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Старт", use_container_width=True):
                if not st.session_state.bot_running:
                    st.session_state.bot_running = True
                    _start_bot_thread(bybit_api, strategy, state_manager, config)
                    st.success("Бот запущен!")
        with c2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
                    _stop_bot_thread()
                    st.error("Бот остановлен!")

        st.markdown("### 📡 Статус подключения")
        try:
            if hasattr(bybit_api, 'current_price'):  # демо
                st.warning("🎮 Демо-режим")
                st.caption("Bybit API недоступен по гео; используются тест-данные.")
            else:
                if bybit_api.get_server_time():
                    st.success("✅ Bybit OK")
                else:
                    st.error("❌ Ошибка подключения")
        except Exception as e:
            st.error(f"❌ Ошибка API: {e}")

        st.markdown("### ⚙️ Текущие настройки")
        st.write(f"**Символ:** {config.symbol}")
        st.write(f"**Риск:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Трейлинг:** {'✅' if config.enable_smart_trail else '❌'}")
        with st.expander("🔧 Smart Trailing / ARM (текущие)"):
            st.write(f"**Trailing %:** {config.trailing_perc}%")
            st.write(f"**Trailing Offset %:** {config.trailing_offset_perc}%")
            st.write(f"**ARM after RR:** {'Да' if config.use_arm_after_rr else 'Нет'}")
            st.write(f"**ARM basis:** {config.arm_rr_basis}")
            st.write(f"**ARM RR:** {config.arm_rr}")

        # Debug: Smart Trail
        with st.expander("🧪 Debug: Trailing state"):
            try:
                d = strategy.get_trailing_debug()
                if d:
                    st.write(d)
                else:
                    st.caption("Нет открытой позиции.")
            except Exception as e:
                st.caption(f"нет данных ({e})")

    # Основные вкладки
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

    eq = state_manager.get_equity() or 0.0
    with col1:
        st.metric("💰 Equity", f"${float(eq):.2f}")

    with col2:
        current_pos = state_manager.get_current_position()
        if current_pos:
            sz = float(current_pos.get('size') or 0)
            base = getattr(strategy.config, "symbol", "BASE")
            st.metric("📍 Позиция", f"{sz:.4f} ({base})")
        else:
            st.metric("📍 Позиция", "0")

    with col3:
        trades_today = db.get_trades_count_today()
        st.metric("📊 Сделки сегодня", trades_today)

    with col4:
        pnl_today = db.get_pnl_today()
        st.metric("💵 PnL сегодня", f"${float(pnl_today):.2f}")

    st.markdown("### 📈 Статистика за 30 дней")
    stats = db.get_performance_stats(days=30) or {}
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
    if bybit_api:
        try:
            klines = bybit_api.get_klines(symbol, "15", 100)
            if klines:
                df = pd.DataFrame(klines)
                if "timestamp" in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name=symbol
                )])
                trades = db.get_recent_trades(50) or []
                for tr in trades:
                    try:
                        if tr.get('entry_time'):
                            entry_time = pd.to_datetime(tr['entry_time'], errors='coerce')
                            fig.add_trace(go.Scatter(
                                x=[entry_time],
                                y=[float(tr['entry_price'])],
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up' if tr.get('direction') == 'long' else 'triangle-down',
                                    size=10,
                                    color='green' if tr.get('direction') == 'long' else 'red'
                                ),
                                name=f"Entry {tr.get('direction')}"
                            ))
                    except Exception:
                        pass
                fig.update_layout(title=f"{symbol} 15m с входами", xaxis_title="Время", yaxis_title="Цена", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Не удалось получить данные свечей")
        except Exception as e:
            st.error(f"Ошибка загрузки графика: {e}")


def show_equity_curve(db):
    st.markdown("### 💰 Кривая Equity")
    eq = db.get_equity_history(days=30)
    if eq:
        df = pd.DataFrame(eq)
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['equity'], mode='lines', name='Equity',
                                 line=dict(width=2)))
        fig.update_layout(title="Изменение Equity за 30 дней", xaxis_title="Дата", yaxis_title="Equity ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения кривой equity")


def show_trades_table(db):
    st.markdown("### 📋 История сделок")
    trades = db.get_recent_trades(100)
    if trades:
        df = pd.DataFrame(trades)
        if 'entry_time' in df.columns: df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
        if 'exit_time' in df.columns:  df['exit_time']  = pd.to_datetime(df['exit_time'], errors='coerce')
        for col in ('pnl', 'rr', 'entry_price', 'exit_price', 'quantity'):
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        cols = [c for c in ['entry_time','direction','entry_price','exit_price','quantity','pnl','rr','status'] if c in df.columns]
        st.dataframe(df[cols].round(4), use_container_width=True)
    else:
        st.info("Нет сделок для отображения")


if __name__ == "__main__":
    main()
