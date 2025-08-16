"""
📈 Страница аналитики и статистики
Расширенная аналитика торговых результатов
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import TradingAnalytics, TrailingLogger
from database import Database

# Настройка страницы
st.set_page_config(
    page_title="KWIN Bot - Аналитика",
    page_icon="📈",
    layout="wide"
)

# Инициализация
@st.cache_resource
def init_analytics():
    """Инициализация модулей аналитики"""
    db = Database()
    analytics = TradingAnalytics()
    trail_logger = TrailingLogger()
    return analytics, trail_logger

def create_performance_chart(stats: dict):
    """Создание графика производительности"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Winrate по направлениям (%)',
            'PnL метрики (USDT)',
            'Risk/Reward статистика',
            'ROI динамика (%)'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Winrate
    winrate = stats.get('winrate', {})
    fig.add_trace(
        go.Bar(
            x=['Total', 'Long', 'Short'],
            y=[winrate.get('total', 0), winrate.get('long', 0), winrate.get('short', 0)],
            name='Winrate',
            marker_color=['#00D4AA', '#1f77b4', '#ff7f0e']
        ),
        row=1, col=1
    )
    
    # 2. PnL метрики
    pnl = stats.get('pnl', {})
    fig.add_trace(
        go.Bar(
            x=['Total PnL', 'Gross Profit', 'Gross Loss', 'Avg Win', 'Avg Loss'],
            y=[
                pnl.get('total_pnl', 0),
                pnl.get('gross_profit', 0),
                -abs(pnl.get('gross_loss', 0)),  # Отрицательное для убытков
                pnl.get('avg_win', 0),
                -abs(pnl.get('avg_loss', 0))
            ],
            name='PnL',
            marker_color=['#00D4AA' if x >= 0 else '#FF4B4B' for x in [
                pnl.get('total_pnl', 0),
                pnl.get('gross_profit', 0),
                -abs(pnl.get('gross_loss', 0)),
                pnl.get('avg_win', 0),
                -abs(pnl.get('avg_loss', 0))
            ]]
        ),
        row=1, col=2
    )
    
    # 3. Risk/Reward
    rr = stats.get('risk_reward', {})
    fig.add_trace(
        go.Bar(
            x=['Avg RR', 'Max RR', 'Min RR'],
            y=[rr.get('avg_rr', 0), rr.get('max_rr', 0), rr.get('min_rr', 0)],
            name='Risk/Reward',
            marker_color='#9C27B0'
        ),
        row=2, col=1
    )
    
    # 4. ROI
    roi = stats.get('roi', {})
    fig.add_trace(
        go.Scatter(
            x=['Total ROI', 'Monthly ROI', 'Daily ROI'],
            y=[roi.get('total_roi', 0), roi.get('monthly_roi', 0), roi.get('daily_roi', 0)],
            mode='lines+markers',
            name='ROI',
            line=dict(color='#FF6B35', width=3),
            marker=dict(size=10)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Комплексная аналитика торгов",
        title_x=0.5
    )
    
    return fig

def create_monthly_performance_chart(monthly_data: list):
    """График месячной производительности"""
    if not monthly_data:
        return go.Figure()
    
    months = [item['month'] for item in monthly_data]
    pnl = [item['total_pnl'] for item in monthly_data]
    winrates = [item['winrate'] for item in monthly_data]
    trades = [item['trades'] for item in monthly_data]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Месячный PnL (USDT)', 'Winrate & Количество сделок'],
        secondary_y=True
    )
    
    # PnL по месяцам
    fig.add_trace(
        go.Bar(
            x=months,
            y=pnl,
            name='PnL',
            marker_color=['#00D4AA' if x >= 0 else '#FF4B4B' for x in pnl]
        ),
        row=1, col=1
    )
    
    # Winrate
    fig.add_trace(
        go.Scatter(
            x=months,
            y=winrates,
            mode='lines+markers',
            name='Winrate (%)',
            line=dict(color='#1f77b4', width=2),
            yaxis='y2'
        ),
        row=2, col=1
    )
    
    # Количество сделок
    fig.add_trace(
        go.Bar(
            x=months,
            y=trades,
            name='Trades',
            marker_color='#ff7f0e',
            opacity=0.7,
            yaxis='y3'
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def create_sfp_pattern_analysis(pattern_stats: dict):
    """Анализ SFP паттернов"""
    bull_stats = pattern_stats.get('bull_sfp', {})
    bear_stats = pattern_stats.get('bear_sfp', {})
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Bull SFP Performance', 'Bear SFP Performance'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Bull SFP
    fig.add_trace(
        go.Bar(
            x=['Trades', 'Winrate %', 'Avg PnL', 'Total PnL'],
            y=[
                bull_stats.get('trades', 0),
                bull_stats.get('winrate', 0),
                bull_stats.get('avg_pnl', 0),
                bull_stats.get('total_pnl', 0)
            ],
            name='Bull SFP',
            marker_color='#00D4AA'
        ),
        row=1, col=1
    )
    
    # Bear SFP
    fig.add_trace(
        go.Bar(
            x=['Trades', 'Winrate %', 'Avg PnL', 'Total PnL'],
            y=[
                bear_stats.get('trades', 0),
                bear_stats.get('winrate', 0),
                bear_stats.get('avg_pnl', 0),
                bear_stats.get('total_pnl', 0)
            ],
            name='Bear SFP',
            marker_color='#FF4B4B'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def display_trailing_logs(trail_logger: TrailingLogger):
    """Отображение логов трейлинга"""
    st.subheader("🎯 Логи Smart Trailing")
    
    # Выбор периода
    hours_back = st.selectbox(
        "Период логов:",
        [1, 6, 12, 24, 48, 72],
        index=3,
        format_func=lambda x: f"Последние {x} часов"
    )
    
    # Получаем логи
    trail_logs = trail_logger.get_trailing_history(hours_back)
    
    if trail_logs:
        # Преобразуем в DataFrame для отображения
        df = pd.DataFrame(trail_logs)
        
        # Форматируем колонки
        display_columns = ['timestamp', 'direction', 'trigger_type', 'old_sl', 'new_sl', 
                         'current_price', 'trail_distance', 'unrealized_pnl', 'arm_status']
        
        if all(col in df.columns for col in display_columns):
            df_display = df[display_columns].copy()
            
            # Форматируем числовые значения
            for col in ['old_sl', 'new_sl', 'current_price', 'trail_distance', 'unrealized_pnl']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].round(2)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=300
            )
            
            # Статистика трейлинга
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего движений", len(df))
            
            with col2:
                long_trails = len(df[df['direction'] == 'long'])
                st.metric("Long трейлы", long_trails)
            
            with col3:
                short_trails = len(df[df['direction'] == 'short'])
                st.metric("Short трейлы", short_trails)
            
            with col4:
                avg_distance = df['trail_distance'].mean() if 'trail_distance' in df.columns else 0
                st.metric("Средняя дистанция", f"{avg_distance:.2f}")
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info(f"Нет логов трейлинга за последние {hours_back} часов")

def main():
    """Основная функция страницы аналитики"""
    st.title("📈 Аналитика и статистика KWIN Bot")
    st.markdown("---")
    
    # Инициализация
    analytics, trail_logger = init_analytics()
    
    # Выбор периода анализа
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Периодная аналитика")
    
    with col2:
        period = st.selectbox(
            "Период:",
            [30, 60, 90, 180],
            index=0,
            format_func=lambda x: f"{x} дней"
        )
    
    # Получаем статистику
    try:
        stats = analytics.get_comprehensive_stats(period)
        
        if stats and stats.get('total_trades', 0) > 0:
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_trades = stats.get('total_trades', 0)
                st.metric("Всего сделок", total_trades)
            
            with col2:
                winrate = stats.get('winrate', {}).get('total', 0)
                st.metric("Общий Winrate", f"{winrate}%")
            
            with col3:
                total_pnl = stats.get('pnl', {}).get('total_pnl', 0)
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Общий PnL", f"${total_pnl}", delta_color=pnl_color)
            
            with col4:
                profit_factor = stats.get('pnl', {}).get('profit_factor', 0)
                st.metric("Profit Factor", f"{profit_factor}")
            
            # Дополнительные метрики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_rr = stats.get('risk_reward', {}).get('avg_rr', 0)
                st.metric("Средний RR", f"{avg_rr}")
            
            with col2:
                total_roi = stats.get('roi', {}).get('total_roi', 0)
                st.metric("ROI", f"{total_roi}%")
            
            with col3:
                max_dd = stats.get('drawdown', {}).get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_dd}%")
            
            with col4:
                monthly_roi = stats.get('roi', {}).get('monthly_roi', 0)
                st.metric("Месячный ROI", f"{monthly_roi}%")
            
            # Графики производительности
            st.plotly_chart(
                create_performance_chart(stats),
                use_container_width=True
            )
            
            # Месячная производительность
            st.subheader("📅 Месячная производительность")
            monthly_data = analytics.get_monthly_performance()
            
            if monthly_data:
                st.plotly_chart(
                    create_monthly_performance_chart(monthly_data),
                    use_container_width=True
                )
            else:
                st.info("Недостаточно данных для месячной аналитики")
            
            # Анализ SFP паттернов
            st.subheader("🔍 Анализ SFP паттернов")
            pattern_stats = analytics.get_sfp_pattern_stats()
            
            if pattern_stats:
                st.plotly_chart(
                    create_sfp_pattern_analysis(pattern_stats),
                    use_container_width=True
                )
                
                # Детальная статистика паттернов
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Bull SFP статистика:**")
                    bull_stats = pattern_stats.get('bull_sfp', {})
                    if bull_stats:
                        st.write(f"• Сделок: {bull_stats.get('trades', 0)}")
                        st.write(f"• Winrate: {bull_stats.get('winrate', 0)}%")
                        st.write(f"• Средний PnL: ${bull_stats.get('avg_pnl', 0)}")
                        st.write(f"• Общий PnL: ${bull_stats.get('total_pnl', 0)}")
                
                with col2:
                    st.markdown("**Bear SFP статистика:**")
                    bear_stats = pattern_stats.get('bear_sfp', {})
                    if bear_stats:
                        st.write(f"• Сделок: {bear_stats.get('trades', 0)}")
                        st.write(f"• Winrate: {bear_stats.get('winrate', 0)}%")
                        st.write(f"• Средний PnL: ${bear_stats.get('avg_pnl', 0)}")
                        st.write(f"• Общий PnL: ${bear_stats.get('total_pnl', 0)}")
            
            # Логи трейлинга
            display_trailing_logs(trail_logger)
            
        else:
            st.info(f"Нет торговых данных за последние {period} дней")
            st.markdown("""
            **Для появления аналитики необходимо:**
            1. Включить бота в режиме торговли
            2. Дождаться первых сделок
            3. Данные обновятся автоматически
            """)
    
    except Exception as e:
        st.error(f"Ошибка загрузки аналитики: {e}")
        
        # Показываем демо данные
        st.info("Показ демонстрационных данных...")
        
        demo_stats = {
            'total_trades': 15,
            'winrate': {'total': 73.3, 'long': 80.0, 'short': 66.7},
            'pnl': {
                'total_pnl': 125.50,
                'gross_profit': 180.0,
                'gross_loss': 54.5,
                'avg_win': 16.4,
                'avg_loss': 13.6,
                'profit_factor': 3.3
            },
            'risk_reward': {'avg_rr': 1.4, 'max_rr': 2.1, 'min_rr': 0.8},
            'roi': {'total_roi': 12.55, 'monthly_roi': 4.2, 'daily_roi': 0.14},
            'drawdown': {'max_drawdown': 3.2, 'current_drawdown': 0.8}
        }
        
        st.plotly_chart(
            create_performance_chart(demo_stats),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
