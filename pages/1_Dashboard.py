import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config
import utils

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

def main():
    st.title("📊 Dashboard")
    
    # Инициализация компонентов (используем кэширование)
    @st.cache_resource
    def init_components():
        config = Config()
        db = Database()
        state_manager = StateManager(db)
        
        # API ключи из environment variables
        import os
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        
        if api_key and api_secret:
            bybit_api = BybitAPI(api_key, api_secret, testnet=False)
        else:
            bybit_api = None
        
        strategy = KWINStrategy(config, bybit_api, state_manager, db)
        
        return config, db, state_manager, bybit_api, strategy
    
    config, db, state_manager, bybit_api, strategy = init_components()
    
    # Проверка подключения
    if bybit_api is None:
        st.error("⚠️ API ключи Bybit не настроены")
        st.stop()
    
    # Автообновление каждые 5 секунд
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0
    
    current_time = time.time()
    if current_time - st.session_state.last_update > 5:
        st.session_state.last_update = current_time
        if bybit_api:
            strategy.run_cycle()
        st.rerun()
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    st.markdown("### 📈 Основные метрики")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        equity = state_manager.get_equity()
        st.metric("💰 Equity", f"${equity:.2f}")
    
    with col2:
        position = state_manager.get_current_position()
        if position:
            pos_text = f"{position.get('size', 0):.4f} ETH"
            pos_direction = position.get('direction', '').upper()
            st.metric("📍 Позиция", f"{pos_direction} {pos_text}")
        else:
            st.metric("📍 Позиция", "Нет позиции")
    
    with col3:
        try:
            ticker = bybit_api.get_ticker("ETHUSDT")
            current_price = ticker['last_price'] if ticker else 0
            st.metric("💹 Цена ETH", f"${current_price:.2f}")
        except:
            st.metric("💹 Цена ETH", "Ошибка")
    
    with col4:
        trades_today = db.get_trades_count_today()
        st.metric("📊 Сделки сегодня", trades_today)
    
    with col5:
        pnl_today = db.get_pnl_today()
        st.metric("💵 PnL сегодня", utils.format_currency(pnl_today))
    
    # === ТЕКУЩАЯ ПОЗИЦИЯ ===
    if position:
        st.markdown("### 🎯 Текущая позиция")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**Направление:** {position.get('direction', '').upper()}")
            st.write(f"**Размер:** {position.get('size', 0):.4f} ETH")
        
        with col2:
            entry_price = position.get('entry_price', 0)
            st.write(f"**Вход:** ${entry_price:.2f}")
            
            # Текущий PnL
            if current_price and entry_price:
                current_pnl = utils.calculate_pnl(
                    entry_price, current_price, position.get('size', 0), 
                    position.get('direction'), include_fees=True
                )
                pnl_color = "green" if current_pnl >= 0 else "red"
                st.markdown(f"**Текущий PnL:** <span style='color:{pnl_color}'>{utils.format_currency(current_pnl)}</span>", 
                           unsafe_allow_html=True)
        
        with col3:
            st.write(f"**Stop Loss:** ${position.get('stop_loss', 0):.2f}")
            st.write(f"**Take Profit:** ${position.get('take_profit', 0):.2f}")
        
        with col4:
            armed_status = "🟢 Armed" if position.get('armed', False) else "🔴 Not Armed"
            st.write(f"**Статус:** {armed_status}")
            
            # Текущий RR
            if current_price and entry_price:
                current_rr = utils.calculate_rr(
                    entry_price, current_price, position.get('stop_loss', 0), 
                    position.get('direction')
                )
                st.write(f"**Текущий RR:** {current_rr:.2f}")
    
    # === СТАТИСТИКА ЗА ПЕРИОДЫ ===
    st.markdown("### 📊 Статистика производительности")
    
    period_tabs = st.tabs(["30 дней", "60 дней", "180 дней"])
    
    for i, period in enumerate([30, 60, 180]):
        with period_tabs[i]:
            stats = db.get_performance_stats(days=period)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего сделок", stats.get('total_trades', 0))
                st.metric("Прибыльных", stats.get('winning_trades', 0))
            
            with col2:
                win_rate = stats.get('win_rate', 0)
                st.metric("Win Rate", utils.format_percentage(win_rate))
                avg_rr = stats.get('avg_rr', 0)
                st.metric("Средний RR", f"{avg_rr:.2f}")
            
            with col3:
                total_pnl = stats.get('total_pnl', 0)
                st.metric("Общий PnL", utils.format_currency(total_pnl))
                avg_pnl = stats.get('avg_pnl', 0)
                st.metric("Средний PnL", utils.format_currency(avg_pnl))
            
            with col4:
                max_win = stats.get('max_win', 0)
                st.metric("Макс. прибыль", utils.format_currency(max_win))
                max_loss = stats.get('max_loss', 0)
                st.metric("Макс. убыток", utils.format_currency(max_loss))
    
    # === ГРАФИК EQUITY ===
    st.markdown("### 💰 Кривая Equity")
    
    equity_data = db.get_equity_history(days=30)
    
    if equity_data:
        df_equity = pd.DataFrame(equity_data)
        df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df_equity['timestamp'],
            y=df_equity['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_equity.update_layout(
            title="Изменение Equity за последние 30 дней",
            xaxis_title="Дата",
            yaxis_title="Equity ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Нет данных для отображения кривой equity")
    
    # === ПОСЛЕДНИЕ СДЕЛКИ ===
    st.markdown("### 📋 Последние сделки")
    
    recent_trades = db.get_recent_trades(10)
    
    if recent_trades:
        df_trades = pd.DataFrame(recent_trades)
        
        # Форматирование для отображения
        display_columns = ['entry_time', 'direction', 'entry_price', 'exit_price', 
                          'quantity', 'pnl', 'rr', 'status']
        
        for col in display_columns:
            if col in df_trades.columns:
                if col == 'entry_time':
                    df_trades[col] = pd.to_datetime(df_trades[col]).dt.strftime('%Y-%m-%d %H:%M')
                elif col in ['pnl']:
                    df_trades[col] = df_trades[col].round(2)
                elif col in ['rr']:
                    df_trades[col] = df_trades[col].round(2)
                elif col in ['entry_price', 'exit_price']:
                    df_trades[col] = df_trades[col].round(2)
                elif col in ['quantity']:
                    df_trades[col] = df_trades[col].round(4)
        
        # Переименование колонок для отображения
        column_mapping = {
            'entry_time': 'Время входа',
            'direction': 'Направление',
            'entry_price': 'Цена входа',
            'exit_price': 'Цена выхода',
            'quantity': 'Количество',
            'pnl': 'PnL ($)',
            'rr': 'RR',
            'status': 'Статус'
        }
        
        df_display = df_trades[display_columns].rename(columns=column_mapping)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("Нет сделок для отображения")
    
    # === УПРАВЛЕНИЕ БОТОМ ===
    st.markdown("### 🎛️ Управление ботом")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Запустить бота", use_container_width=True):
            state_manager.set_bot_status("running")
            st.success("Бот запущен!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Остановить бота", use_container_width=True):
            state_manager.set_bot_status("stopped")
            st.warning("Бот остановлен!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Обновить данные", use_container_width=True):
            if bybit_api:
                strategy.run_cycle()
            st.success("Данные обновлены!")
            st.rerun()
    
    # Статус бота
    bot_status = state_manager.get_bot_status()
    status_color = "green" if bot_status == "running" else "red"
    st.markdown(f"**Статус бота:** <span style='color:{status_color}'>{bot_status.upper()}</span>", 
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
