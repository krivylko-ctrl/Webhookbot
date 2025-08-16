import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from database import Database
from config import Config
from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
import utils

st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")

def main():
    st.title("📈 Бэктест стратегии KWIN")
    
    # Инициализация компонентов
    @st.cache_resource
    def init_components():
        config = Config()
        db = Database()
        state_manager = StateManager(db)
        return config, db, state_manager
    
    config, db, state_manager = init_components()
    
    # === НАСТРОЙКИ БЭКТЕСТА ===
    st.markdown("### ⚙️ Настройки бэктеста")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period = st.selectbox(
            "📅 Период бэктеста",
            options=[30, 60, 180],
            index=0,
            help="Количество дней для анализа"
        )
    
    with col2:
        initial_capital = st.number_input(
            "💰 Начальный капитал ($)",
            min_value=100.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0
        )
    
    with col3:
        commission_rate = st.number_input(
            "💸 Комиссия (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.055,
            step=0.001,
            format="%.3f"
        )
    
    # === РЕЗУЛЬТАТЫ БЭКТЕСТА ===
    if st.button("🚀 Запустить бэктест", use_container_width=True):
        with st.spinner("Выполняется бэктест..."):
            results = run_backtest(period, initial_capital, commission_rate, db)
            
            if results:
                display_backtest_results(results)
            else:
                st.error("Не удалось выполнить бэктест. Недостаточно данных.")
    
    # === АНАЛИЗ СУЩЕСТВУЮЩИХ СДЕЛОК ===
    st.markdown("### 📊 Анализ реальных сделок")
    
    period_analysis = st.selectbox(
        "Период для анализа",
        options=[30, 60, 180],
        key="analysis_period"
    )
    
    trades = db.get_trades_by_period(period_analysis)
    
    if trades:
        display_trades_analysis(trades, initial_capital)
    else:
        st.info(f"Нет сделок за последние {period_analysis} дней")

def run_backtest(period_days: int, initial_capital: float, commission_rate: float, db: Database) -> dict:
    """Выполнение бэктеста на исторических данных"""
    try:
        # Получаем сделки за период
        trades = db.get_trades_by_period(period_days)
        
        if not trades:
            return None
        
        # Симуляция торговли
        capital = initial_capital
        equity_curve = [initial_capital]
        timestamps = [datetime.now() - timedelta(days=period_days)]
        
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        max_drawdown = 0
        peak_equity = initial_capital
        
        trade_results = []
        
        for trade in trades:
            if trade['status'] == 'closed' and trade['pnl'] is not None:
                # Применяем комиссию если она отличается от сохраненной
                pnl = trade['pnl']
                
                # Пересчитываем PnL с новой комиссией если нужно
                if commission_rate != 0.055:  # Стандартная комиссия Bybit
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price'] or entry_price
                    quantity = trade['quantity']
                    direction = trade['direction']
                    
                    pnl = utils.calculate_pnl(
                        entry_price, exit_price, quantity, direction,
                        include_fees=True, fee_rate=commission_rate/100
                    )
                
                capital += pnl
                total_pnl += pnl
                total_trades += 1
                
                if pnl > 0:
                    winning_trades += 1
                
                # Обновляем кривую equity
                equity_curve.append(capital)
                timestamps.append(pd.to_datetime(trade['exit_time']))
                
                # Рассчитываем максимальную просадку
                if capital > peak_equity:
                    peak_equity = capital
                
                current_drawdown = (peak_equity - capital) / peak_equity * 100
                max_drawdown = max(max_drawdown, current_drawdown)
                
                trade_results.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time'],
                    'direction': trade['direction'],
                    'pnl': pnl,
                    'equity': capital,
                    'rr': trade.get('rr', 0)
                })
        
        # Расчет метрик
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        total_return = ((capital - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'timestamps': timestamps,
            'trade_results': trade_results
        }
    
    except Exception as e:
        st.error(f"Ошибка при выполнении бэктеста: {e}")
        return None

def display_backtest_results(results: dict):
    """Отображение результатов бэктеста"""
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    st.markdown("### 📊 Результаты бэктеста")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Финальный капитал",
            utils.format_currency(results['final_capital']),
            delta=utils.format_currency(results['total_pnl'])
        )
        st.metric("📊 Всего сделок", results['total_trades'])
    
    with col2:
        st.metric(
            "📈 Общая доходность",
            utils.format_percentage(results['total_return']),
        )
        st.metric("🎯 Win Rate", utils.format_percentage(results['win_rate']))
    
    with col3:
        st.metric("💵 Общий PnL", utils.format_currency(results['total_pnl']))
        st.metric("✅ Прибыльных", results['winning_trades'])
    
    with col4:
        st.metric("📉 Макс. просадка", utils.format_percentage(results['max_drawdown']))
        st.metric("❌ Убыточных", results['losing_trades'])
    
    # === КРИВАЯ EQUITY ===
    st.markdown("### 💰 Кривая Equity")
    
    fig_equity = go.Figure()
    
    fig_equity.add_trace(go.Scatter(
        x=results['timestamps'],
        y=results['equity_curve'],
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_equity.update_layout(
        title="Развитие капитала во времени",
        xaxis_title="Дата",
        yaxis_title="Капитал ($)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # === РАСПРЕДЕЛЕНИЕ PnL ===
    st.markdown("### 📊 Распределение PnL")
    
    if results['trade_results']:
        pnl_values = [trade['pnl'] for trade in results['trade_results']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Гистограмма PnL
            fig_hist = go.Figure(data=[go.Histogram(x=pnl_values, nbinsx=20)])
            fig_hist.update_layout(
                title="Распределение PnL по сделкам",
                xaxis_title="PnL ($)",
                yaxis_title="Количество сделок",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Статистика PnL
            st.markdown("#### 📈 Статистика PnL")
            
            pnl_array = np.array(pnl_values)
            
            metrics_data = {
                "Среднее": np.mean(pnl_array),
                "Медиана": np.median(pnl_array),
                "Стандартное отклонение": np.std(pnl_array),
                "Минимум": np.min(pnl_array),
                "Максимум": np.max(pnl_array),
                "25-й процентиль": np.percentile(pnl_array, 25),
                "75-й процентиль": np.percentile(pnl_array, 75)
            }
            
            for metric, value in metrics_data.items():
                st.write(f"**{metric}:** {utils.format_currency(value)}")
    
    # === ТАБЛИЦА СДЕЛОК ===
    st.markdown("### 📋 Детали сделок")
    
    if results['trade_results']:
        df_trades = pd.DataFrame(results['trade_results'])
        
        # Форматирование
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        df_trades['pnl'] = df_trades['pnl'].round(2)
        df_trades['equity'] = df_trades['equity'].round(2)
        df_trades['rr'] = df_trades['rr'].round(2)
        
        # Переименование колонок
        column_mapping = {
            'entry_time': 'Время входа',
            'exit_time': 'Время выхода',
            'direction': 'Направление',
            'pnl': 'PnL ($)',
            'equity': 'Equity ($)',
            'rr': 'RR'
        }
        
        df_display = df_trades.rename(columns=column_mapping)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

def display_trades_analysis(trades: list, initial_capital: float):
    """Анализ реальных сделок"""
    
    if not trades:
        return
    
    df = pd.DataFrame(trades)
    closed_trades = df[df['status'] == 'closed'].copy()
    
    if closed_trades.empty:
        st.info("Нет закрытых сделок для анализа")
        return
    
    # Фильтруем сделки с валидными данными
    valid_trades = closed_trades.dropna(subset=['pnl', 'entry_time', 'exit_time'])
    
    if valid_trades.empty:
        st.info("Нет валидных данных для анализа")
        return
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = len(valid_trades)
    winning_trades = len(valid_trades[valid_trades['pnl'] > 0])
    total_pnl = valid_trades['pnl'].sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    with col1:
        st.metric("📊 Всего сделок", total_trades)
        st.metric("✅ Прибыльных", winning_trades)
    
    with col2:
        st.metric("🎯 Win Rate", utils.format_percentage(win_rate))
        st.metric("❌ Убыточных", total_trades - winning_trades)
    
    with col3:
        st.metric("💵 Общий PnL", utils.format_currency(total_pnl))
        avg_pnl = valid_trades['pnl'].mean()
        st.metric("📊 Средний PnL", utils.format_currency(avg_pnl))
    
    with col4:
        max_win = valid_trades['pnl'].max()
        max_loss = valid_trades['pnl'].min()
        st.metric("📈 Макс. прибыль", utils.format_currency(max_win))
        st.metric("📉 Макс. убыток", utils.format_currency(max_loss))
    
    # === ГРАФИК PnL ПО ВРЕМЕНИ ===
    st.markdown("### 📈 PnL по времени")
    
    valid_trades['exit_time'] = pd.to_datetime(valid_trades['exit_time'])
    valid_trades = valid_trades.sort_values('exit_time')
    valid_trades['cumulative_pnl'] = valid_trades['pnl'].cumsum()
    
    fig_pnl = go.Figure()
    
    # Кумулятивный PnL
    fig_pnl.add_trace(go.Scatter(
        x=valid_trades['exit_time'],
        y=valid_trades['cumulative_pnl'],
        mode='lines+markers',
        name='Кумулятивный PnL',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_pnl.update_layout(
        title="Кумулятивный PnL по времени",
        xaxis_title="Дата",
        yaxis_title="PnL ($)",
        height=400
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # === ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Распределение по направлениям")
        direction_counts = valid_trades['direction'].value_counts()
        for direction, count in direction_counts.items():
            st.write(f"**{direction.upper()}:** {count} сделок")
    
    with col2:
        st.markdown("#### ⏱️ Статистика времени удержания")
        if 'entry_time' in valid_trades.columns:
            valid_trades['entry_time'] = pd.to_datetime(valid_trades['entry_time'])
            valid_trades['hold_time'] = (valid_trades['exit_time'] - valid_trades['entry_time']).dt.total_seconds() / 3600
            
            avg_hold_time = valid_trades['hold_time'].mean()
            max_hold_time = valid_trades['hold_time'].max()
            min_hold_time = valid_trades['hold_time'].min()
            
            st.write(f"**Среднее время:** {utils.format_time_duration(avg_hold_time)}")
            st.write(f"**Максимальное:** {utils.format_time_duration(max_hold_time)}")
            st.write(f"**Минимальное:** {utils.format_time_duration(min_hold_time)}")

if __name__ == "__main__":
    main()
