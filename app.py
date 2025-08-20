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
    
    # Используем API ключи из конфига
    if cfg.BYBIT_API_KEY and cfg.BYBIT_API_SECRET:
        # Пробуем подключиться к Bybit
        bybit_api = BybitAPI(cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=False)
        
        # Проверяем доступность API
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
                    st.success("Бот запущен!")
                    
        with col2:
            if st.button("⏹️ Стоп", use_container_width=True):
                if st.session_state.bot_running:
                    st.session_state.bot_running = False
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
        st.write(f"**Риск:** {config.risk_pct}%")
        st.write(f"**RR:** {config.risk_reward}")
        st.write(f"**Макс. позиция:** {config.max_qty_manual} ETH")
        st.write(f"**Трейлинг:** {'✅' if config.enable_smart_trail else '❌'}")
    
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
    
    # Статистика за последние 30 дней
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
    """Показать график с сделками"""
    st.markdown("### 📈 График ETH/USDT")
    
    # Получаем данные свечей
    if bybit_api:
        try:
            klines = bybit_api.get_klines("ETHUSDT", "15", 100)
            if klines:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Создаем график свечей
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="ETH/USDT"
                )])
                
                # Добавляем сделки
                trades = db.get_recent_trades(50)
                for trade in trades:
                    if trade['entry_time']:
                        entry_time = pd.to_datetime(trade['entry_time'])
                        fig.add_trace(go.Scatter(
                            x=[entry_time],
                            y=[trade['entry_price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up' if trade['direction'] == 'long' else 'triangle-down',
                                size=10,
                                color='green' if trade['direction'] == 'long' else 'red'
                            ),
                            name=f"Entry {trade['direction']}"
                        ))
                
                fig.update_layout(
                    title="ETH/USDT 15m с входами",
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
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
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['pnl'] = df['pnl'].round(2)
        df['rr'] = df['rr'].round(2)
        
        # Отображаем таблицу
        st.dataframe(
            df[['entry_time', 'direction', 'entry_price', 'exit_price', 'quantity', 'pnl', 'rr', 'status']],
            use_container_width=True
        )
    else:
        st.info("Нет сделок для отображения")

if __name__ == "__main__":
    main()
