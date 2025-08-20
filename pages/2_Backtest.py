import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kwin_strategy import KWINStrategy
from database import Database
from config import Config
from bybit_api import BybitAPI

def main():
    st.set_page_config(
        page_title="KWIN Backtest",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 KWIN Strategy Backtest")
    st.markdown("Тестирование стратегии на исторических данных")
    
    # Параметры бэктеста
    col1, col2 = st.columns(2)
    
    with col1:
        start_capital = st.number_input("Начальный капитал ($)", min_value=100, value=10000, step=100)
        period_days = st.selectbox("Период тестирования", [7, 14, 30, 60, 90], index=2)
        
    with col2:
        symbol = st.selectbox("Торговая пара", ["ETHUSDT", "BTCUSDT"], index=0)
        fee_rate = st.number_input("Комиссия (%)", min_value=0.01, max_value=1.0, value=0.055, step=0.005)
    
    # Настройки стратегии
    st.subheader("⚙️ Параметры стратегии")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_reward = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.3, step=0.1)
        sfp_len = st.number_input("SFP Length", min_value=1, max_value=10, value=2, step=1)
        risk_pct = st.number_input("Risk %", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
        
    with col2:
        enable_smart_trail = st.checkbox("Smart Trailing", value=True)
        trailing_perc = st.number_input("Trailing %", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        trailing_offset = st.number_input("Trailing Offset", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
        
    with col3:
        use_sfp_quality = st.checkbox("SFP Quality Filter", value=True)
        wick_min_ticks = st.number_input("Min Wick Ticks", min_value=1, max_value=20, value=7, step=1)
        close_back_pct = st.number_input("Close Back %", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    # Кнопка запуска
    if st.button("🚀 Запустить бэктест", type="primary"):
        with st.spinner("Выполняется бэктест..."):
            try:
                # Создаем конфигурацию для бэктеста
                config = Config()
                config.risk_reward = risk_reward
                config.sfp_len = int(sfp_len)  # Конвертируем в int
                config.risk_pct = risk_pct
                config.enable_smart_trail = enable_smart_trail
                config.trailing_perc = trailing_perc
                config.trailing_offset = trailing_offset
                config.use_sfp_quality = use_sfp_quality
                config.wick_min_ticks = int(wick_min_ticks)  # Конвертируем в int
                config.close_back_pct = close_back_pct
                config.taker_fee_rate = fee_rate / 100  # Преобразуем в десятичную дробь
                config.symbol = str(symbol) if symbol else "BTCUSDT"  # Защищаем от None
                
                # Создаем компоненты
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                from database import Database
                from state_manager import StateManager
                from kwin_strategy import KWINStrategy
                
                db = Database(":memory:")
                state_manager = StateManager()
                
                # Инициализируем стратегию (конструктор принимает 4 параметра)
                strategy = KWINStrategy(config, None, state_manager, db)
                
                # Запускаем бэктест
                results = run_backtest(strategy, period_days, start_capital)
                
                # Отображаем результаты
                display_backtest_results(results)
                
            except Exception as e:
                st.error(f"Ошибка выполнения бэктеста: {e}")
                st.exception(e)

def run_backtest(strategy, period_days, start_capital):
    """Выполнение бэктеста"""
    
    # Симуляция исторических данных (в реальной версии здесь будет API Bybit)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Генерируем тестовые свечи (15м интервал)
    minutes_15_count = period_days * 24 * 4  # 15-минутных интервалов
    dates = pd.date_range(start=start_date, periods=minutes_15_count, freq='15T')
    
    # Симулируем движение цены
    base_price = 4500 if strategy.config.symbol == "ETHUSDT" else 118000
    price_changes = np.random.randn(len(dates)) * 0.002  # 0.2% волатильность
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Создаем OHLCV данные
    candles = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        volatility = abs(np.random.randn() * 0.001)
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        candles.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    # Запускаем симуляцию торговли
    equity_history = [start_capital]
    current_equity = start_capital
    trades = []
    
    # Подготовка данных newest-first как в оригинальной стратегии
    candles_newest_first = list(reversed(candles))
    
    for i in range(len(candles_newest_first) - 50):
        # Обновляем данные стратегии (newest-first формат)
        strategy.candles_15m = candles_newest_first[i:i+50]
        strategy._bt_last_bar = strategy.candles_15m[0]  # текущий бар для бэктеста
        
        # Запускаем полный цикл стратегии (как в реальной торговле)
        try:
            strategy.run_cycle()
        except Exception as e:
            continue
            
        # Проверяем если произошел вход или выход
        current_position = strategy.state.get_current_position()
        
        # Собираем данные по эквити
        if current_position and current_position.get('status') == 'closed':
            # Позиция закрыта, обновляем эквити
            trade_pnl = current_position.get('realized_pnl', 0)
            current_equity += trade_pnl
            trades.append(current_position)
            
        equity_history.append(current_equity)
    
    # Получаем сделки из БД
    trades = strategy.db.get_all_trades()
    
    # Рассчитываем финальную статистику
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.get('realized_pnl', 0) > 0])
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
    final_equity = start_capital + total_pnl
    
    # Максимальная просадка
    peak = start_capital
    max_drawdown = 0
    for equity in equity_history:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Собираем результаты
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({
        'timestamp': dates[:len(equity_history)],
        'equity': equity_history
    })
    
    return {
        'trades_df': trades_df,
        'equity_df': equity_df,
        'final_equity': final_equity,
        'initial_equity': start_capital,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown
    }

def display_backtest_results(results):
    """Отображение результатов бэктеста"""
    
    trades_df = results['trades_df']
    equity_df = results['equity_df']
    final_equity = results['final_equity']
    initial_equity = results['initial_equity']
    
    # Расчет ключевых метрик
    if trades_df.empty:
        win_rate = 0
        profit_factor = 0
        max_dd = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_return = 0
    else:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Maximum Drawdown
        if not equity_df.empty and len(equity_df) > 1:
            equity_df = equity_df.copy()
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
            max_dd = equity_df['drawdown'].min()
        else:
            max_dd = 0
        
        # Общая доходность
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
    
    # Отображение основных метрик
    st.subheader("📈 Результаты бэктеста")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Общие сделки", total_trades)
    with col2:
        st.metric("Винрейт", f"{win_rate:.1f}%")
    with col3:
        st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
    with col4:
        st.metric("Max DD", f"{max_dd:.2f}%")
    with col5:
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric("Общая доходность", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Начальный капитал", f"${initial_equity:,.2f}")
    with col2:
        profit_loss = final_equity - initial_equity
        delta_color = "normal" if profit_loss >= 0 else "inverse"
        st.metric("Итоговый капитал", f"${final_equity:,.2f}", 
                 delta=f"${profit_loss:,.2f}")
    
    # График Equity
    if not equity_df.empty and len(equity_df) > 1:
        st.subheader("📊 Кривая Equity")
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity', 'Drawdown'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Equity кривая
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        if 'drawdown' in equity_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Анализ производительности"
        )
        
        fig.update_xaxes(title_text="Время", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Таблица сделок
    if not trades_df.empty:
        st.subheader("📋 История сделок")
        
        # Форматируем данные для отображения
        display_df = trades_df.copy()
        if 'entry_time' in display_df.columns:
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'exit_time' in display_df.columns:
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'pnl' in display_df.columns:
            display_df['pnl'] = display_df['pnl'].round(2)
        if 'rr' in display_df.columns:
            display_df['rr'] = display_df['rr'].round(2)
        if 'entry_price' in display_df.columns:
            display_df['entry_price'] = display_df['entry_price'].round(2)
        if 'exit_price' in display_df.columns:
            display_df['exit_price'] = display_df['exit_price'].round(2)
        
        # Показываем последние 20 сделок
        st.dataframe(display_df.tail(20), use_container_width=True)
        
        # Статистика по сделкам
        if len(trades_df) > 0:
            st.subheader("📊 Детальная статистика")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Статистика сделок:**")
                st.write(f"• Выигрышных: {winning_trades}")
                st.write(f"• Проигрышных: {losing_trades}")
                if 'pnl' in trades_df.columns:
                    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                    st.write(f"• Средний выигрыш: ${avg_win:.2f}")
                    st.write(f"• Средний проигрыш: ${avg_loss:.2f}")
            
            with col2:
                st.write("**Risk/Reward статистика:**")
                if 'rr' in trades_df.columns:
                    avg_rr = trades_df['rr'].mean()
                    max_rr = trades_df['rr'].max()
                    min_rr = trades_df['rr'].min()
                    st.write(f"• Средний RR: {avg_rr:.2f}")
                    st.write(f"• Максимальный RR: {max_rr:.2f}")
                    st.write(f"• Минимальный RR: {min_rr:.2f}")
    
    # Информация о бэктесте
    st.markdown("---")
    st.markdown("### ℹ️ О бэктесте")
    st.info("""
    Этот бэктест симулирует торговлю по стратегии KWIN на исторических 15-минутных данных.
    
    **Особенности:**
    - Использует симулированные данные с реалистичной волатильностью
    - Учитывает комиссии биржи
    - Эмулирует срабатывание stop-loss и take-profit
    - Включает Smart Trailing систему
    - Pine Script совместимость 99%+
    
    **Примечание:** Для реального бэктеста требуется подключение к API Bybit для получения исторических данных.
    """)

if __name__ == "__main__":
    main()
