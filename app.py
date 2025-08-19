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
    st.stop()  # Не идём дальше — вместо «белого экрана» получишь понятное сообщение

# Инициализация компонентов
@st.cache_resource
def init_components():
    config = Config()
    db = Database()
    state_manager = StateManager(db)

    # ====== ТОЧЕЧНО: прокидываем трейлинг-настройки и ARM в config из cfg/env ======
    config.enable_smart_trail      = bool(getattr(cfg, "ENABLE_SMART_TRAIL", True))
    config.trailing_perc           = float(getattr(cfg, "TRAILING_PERC", 0.5))           # в %
    config.trailing_offset_perc    = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))    # в %
    # совместимость со старым именем (если где-то внутри стратегии используется)
    config.trailing_offset         = float(getattr(cfg, "TRAILING_OFFSET_PERC", 0.4))

    # ARM-логика (активация трейла после достижения R-множителя)
    config.use_arm_after_rr        = bool(getattr(cfg, "USE_ARM_AFTER_RR", True))
    config.arm_rr                  = float(getattr(cfg, "ARM_RR", 0.5))

    # Базовые параметры риска
    config.risk_pct                = float(getattr(cfg, "RISK_PCT", getattr(config, "risk_pct", 3.0)))
    config.risk_reward             = float(getattr(cfg, "RISK_REWARD", getattr(config, "risk_reward", 1.3)))

    # Символ/рынок
    if hasattr(cfg, "SYMBOL"):
        config.symbol = cfg.SYMBOL
    # ==============================================================================

    # Используем API ключи из конфига
    if getattr(cfg, "BYBIT_API_KEY", None) and getattr(cfg, "BYBIT_API_SECRET", None):
        # Пробуем подключиться к Bybit
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        # Проверка доступности API
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                st.warning("⚠️ Bybit API недоступен из-за географических ограничений. Включен демо-режим.")
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
        except:
            st.warning("⚠️ Проблема с подключением к Bybit API. Включен демо-режим.")
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
    else:
        # Если нет API ключей, используем демо
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        st.info("ℹ️ API ключи не настроены. Работаем в демо-режиме.")

    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    return config, db, state_manager, bybit_api, strategy

# ---------- ФОНОВЫЙ ЦИКЛ ДЛЯ ЛАЙВ-РЕЖИМА (чаще обновляет трейлинг и закрытые бары) ----------
def _bg_bot_loop(bybit_api, strategy: KWINStrategy, state_manager: StateManager, config: Config, poll_sec: float = 2.0):
    """
    Лёгкий поток:
      - раз в poll_sec обновляет «текущую цену» через get_ticker
      - пытается взять последние свечи 15m (2-3 штуки) и отдать закрытый бар стратегии ровно 1 раз
      - вызывает strategy.process_trailing() для имитации calc_on_every_tick
    """
    # последний отданный в стратегию таймстамп закрытого бара (мс), для защиты от дублей
    last_closed_ts = 0

    while getattr(st.session_state, "bot_running", False):
        try:
            # 1) Обновим текущую цену для ARM/трейлинга
            try:
                t = bybit_api.get_ticker(config.symbol) if hasattr(bybit_api, "get_ticker") else {}
                # ничего не делаем: внутренняя стратегия сама читает через self.api.get_ticker()
                _ = t
            except Exception:
                pass

            # 2) Возьмём последние 2-3 закрытые 15m свечи
            closed_bar = None
            try:
                kl = bybit_api.get_klines(config.symbol, "15", 3) if hasattr(bybit_api, "get_klines") else []
                if kl:
                    # предполагаем, что API возвращает закрытые бары, отсортируем по времени по возрастанию
                    df = pd.DataFrame(kl)
                    if "timestamp" in df.columns:
                        df = df.sort_values("timestamp")
                        # берём самый новый закрытый бар
                        closed_bar = df.iloc[-1].to_dict()
                        ts = int(closed_bar.get("timestamp", 0))
                        # защита от повторов
                        if ts and ts != last_closed_ts:
                            strategy.on_bar_close_15m({
                                "timestamp": int(closed_bar["timestamp"]),
                                "open": float(closed_bar["open"]),
                                "high": float(closed_bar["high"]),
                                "low":  float(closed_bar["low"]),
                                "close": float(closed_bar["close"])
                            })
                            last_closed_ts = ts
            except Exception:
                pass

            # 3) По месту — обновим трейлинг (имитация calc_on_every_tick)
            try:
                strategy.process_trailing()
            except Exception:
                pass

        except Exception:
            # не рушим поток
            pass

        # пауза опроса
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
    # сам поток закончится, когда bot_running станет False
    th = st.session_state.get("bot_thread")
    if th and th.is_alive():
        # дадим мягко завершиться
        pass

# ---------------------------------------------------------------------------------------------

def main():

    # Инициализация компонентов
    config, db, state_manager, bybit_api, strategy = init_components()

    # Проверка подключения к API
    if bybit_api is None:
        st.error("⚠️ API ключи Bybit не настроены. Добавьте BYBIT_API_KEY и BYBIT_API_SECRET в переменные окружения.")
        st.stop()

    # Боковая панель с основной информацией
    with st.sidebar:
        st.header("🎛️ Управление ботом")

        # Статус бота
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Старт", use_container_width=True):
                if not st.session_state.bot_running:
                    st.session_state.bot_running = True
                    _start_bot_thread(bybit_api, strategy, state_manager, config)
                    st.success("Бот запущен!")

        with col2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
                    _stop_bot_thread()
                    st.error("Бот остановлен!")

        # Статус подключения
        st.markdown("### 📡 Статус подключения")
        try:
            # Проверяем тип API (демо или реальный)
            if hasattr(bybit_api, 'current_price'):  # Демо API
                st.warning("🎮 Демо-режим активен")
                st.caption("⚠️ Bybit API заблокирован географически. Используются тестовые данные.")
                if st.button("ℹ️ Подробнее о проблеме"):
                    st.info("""
                    **Проблема:** Сервер Replit находится в регионе, заблокированном Bybit.

                    **Решения:**
                    1. 🏠 Запустите бота локально на своем компьютере
                    2. 🌐 Используйте VPS в разрешенном регионе  
                    3. 🔧 Настройте прокси/VPN для обхода блокировки

                    **Текущий статус:** Демо-режим позволяет протестировать интерфейс и логику бота.
                    """)
            else:
                server_time = bybit_api.get_server_time()
                if server_time:
                    st.success("✅ Подключение к Bybit активно")
                else:
                    st.error("❌ Ошибка подключения к Bybit")
        except Exception as e:
            st.error(f"❌ Ошибка API: {e}")

        # Текущие настройки
        st.markdown("### ⚙️ Текущие настройки")
        st.write(f"**Символ:** {config.symbol}")
        st.write(f"**Риск:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Макс. позиция:** {getattr(config, 'max_qty_manual', 0)}")
        st.write(f"**Трейлинг активен:** {'✅' if config.enable_smart_trail else '❌'}")

        # ====== ТОЧЕЧНО: блок с трейлинг-настройками/ARM для наглядности ======
        with st.expander("🔧 Smart Trailing / ARM (текущие)"):
            st.write(f"**Trailing % (от цены входа):** {config.trailing_perc}%")
            st.write(f"**Trailing Offset %:** {config.trailing_offset_perc}%")
            st.write(f"**Arm after RR:** {'Да' if config.use_arm_after_rr else 'Нет'}")
            st.write(f"**ARM RR (R):** {config.arm_rr}")
        # ======================================================================

    # Основная область
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
    """Показать основную информацию дашборда"""
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

    # Статистика за последние 30 дней
    st.markdown("### 📈 Статистика за 30 дней")

    stats = db.get_performance_stats(days=30) or {}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Win Rate", f"{float(stats.get('win_rate', 0)):.1f}%")
    with col2:
        st.metric("📊 Avg RR", f"{float(stats.get('avg_rr', 0)):.2f}")
    with col3:
        st.metric("⏱️ Avg Hold Time", f"{float(stats.get('avg_hold_time', 0)):.1f}h")

def show_chart(bybit_api, db, strategy):
    """Показать график с сделками"""
    symbol = getattr(strategy.config, "symbol", "ETHUSDT")
    st.markdown(f"### 📈 График {symbol}")

    # Получаем данные свечей
    if bybit_api:
        try:
            klines = bybit_api.get_klines(symbol, "15", 100)
            if klines:
                df = pd.DataFrame(klines)
                if "timestamp" in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

                # Создаем график свечей
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol
                )])

                # Добавляем сделки
                trades = db.get_recent_trades(50) or []
                for trade in trades:
                    try:
                        if trade.get('entry_time'):
                            entry_time = pd.to_datetime(trade['entry_time'], errors='coerce')
                            fig.add_trace(go.Scatter(
                                x=[entry_time],
                                y=[float(trade['entry_price'])],
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up' if trade.get('direction') == 'long' else 'triangle-down',
                                    size=10,
                                    color='green' if trade.get('direction') == 'long' else 'red'
                                ),
                                name=f"Entry {trade.get('direction')}"
                            ))
                    except Exception:
                        pass

                fig.update_layout(
                    title=f"{symbol} 15m с входами",
                    xaxis_title="Время",
                    yaxis_title="Цена",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Не удалось получить данные свечей")
        except Exception as e:
            st.error(f"Ошибка загрузки графика: {e}")

def show_equity_curve(db):
    """Показать кривую equity"""
    st.markdown("### 💰 Кривая Equity")

    equity_data = db.get_equity_history(days=30)

    if equity_data:
        df = pd.DataFrame(equity_data)
        if "timestamp" in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title="Изменение Equity за последние 30 дней",
            xaxis_title="Дата",
            yaxis_title="Equity ($)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения кривой equity")

def show_trades_table(db):
    """Показать таблицу сделок"""
    st.markdown("### 📋 История сделок")

    trades = db.get_recent_trades(100)

    if trades:
        df = pd.DataFrame(trades)

        # Форматируем данные для отображения
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        for col in ('pnl', 'rr', 'entry_price', 'exit_price', 'quantity'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Отображаем таблицу
        cols = [c for c in ['entry_time', 'direction', 'entry_price', 'exit_price', 'quantity', 'pnl', 'rr', 'status'] if c in df.columns]
        st.dataframe(df[cols].round(4), use_container_width=True)
    else:
        st.info("Нет сделок для отображения")

if __name__ == "__main__":
    main()
