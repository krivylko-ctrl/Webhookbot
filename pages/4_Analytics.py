import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from math import isnan

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import TradingAnalytics, TrailingLogger
from database import Database

# ===================== Вспомогательные =====================

def _safe_num(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _compute_max_drawdown_pct(equity_curve: pd.Series) -> float:
    """Максимальная просадка в % по equity_curve."""
    if equity_curve is None or equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max - 1.0) * 100.0
    return float(dd.min()) if not dd.empty else 0.0

def _compute_max_drawdown_abs(equity_curve: pd.Series) -> float:
    """Максимальная абсолютная просадка ($) по equity_curve."""
    if equity_curve is None or equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd_abs = equity_curve - roll_max
    return float(dd_abs.min()) if not dd_abs.empty else 0.0

def _to_equity_series(equity_curve_raw) -> pd.Series:
    """Преобразует что угодно в pd.Series для расчётов просадок."""
    try:
        if equity_curve_raw is None:
            return pd.Series(dtype=float)

        if isinstance(equity_curve_raw, dict):
            df = pd.DataFrame(
                [{"timestamp": k, "equity": v} for k, v in equity_curve_raw.items()]
            )
        else:
            # предполагаем list
            if len(equity_curve_raw) == 0:
                return pd.Series(dtype=float)
            if isinstance(equity_curve_raw[0], dict):
                df = pd.DataFrame(equity_curve_raw)
            else:
                # list of tuples
                df = pd.DataFrame(equity_curve_raw, columns=["timestamp", "equity"])

        if "timestamp" not in df.columns or "equity" not in df.columns:
            return pd.Series(dtype=float)

        # нормализация времени
        ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        if ts.isna().all():
            # попробуем без unit
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["ts"] = ts
        df = df.dropna(subset=["ts"]).sort_values("ts")
        s = pd.Series(df["equity"].astype(float).values, index=df["ts"].values)
        return s
    except Exception:
        return pd.Series(dtype=float)

# ===================== Графики =====================

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
    wr_total = _safe_num(winrate.get('total', 0))
    wr_long  = _safe_num(winrate.get('long', 0))
    wr_short = _safe_num(winrate.get('short', 0))

    fig.add_trace(
        go.Bar(
            x=['Total', 'Long', 'Short'],
            y=[wr_total, wr_long, wr_short],
            name='Winrate'
        ),
        row=1, col=1
    )
    
    # 2. PnL метрики
    pnl = stats.get('pnl', {})
    total_pnl   = _safe_num(pnl.get('total_pnl', 0))
    gross_profit= _safe_num(pnl.get('gross_profit', 0))
    gross_loss  = _safe_num(pnl.get('gross_loss', 0))
    avg_win     = _safe_num(pnl.get('avg_win', 0))
    avg_loss    = _safe_num(pnl.get('avg_loss', 0))

    bars = [total_pnl, gross_profit, -abs(gross_loss), avg_win, -abs(avg_loss)]
    colors = ['#00D4AA' if x >= 0 else '#FF4B4B' for x in bars]
    fig.add_trace(
        go.Bar(
            x=['Total PnL', 'Gross Profit', 'Gross Loss', 'Avg Win', 'Avg Loss'],
            y=bars,
            name='PnL',
            marker_color=colors
        ),
        row=1, col=2
    )
    
    # 3. Risk/Reward
    rr = stats.get('risk_reward', {})
    fig.add_trace(
        go.Bar(
            x=['Avg RR', 'Max RR', 'Min RR'],
            y=[_safe_num(rr.get('avg_rr', 0)),
               _safe_num(rr.get('max_rr', 0)),
               _safe_num(rr.get('min_rr', 0))],
            name='Risk/Reward'
        ),
        row=2, col=1
    )
    
    # 4. ROI
    roi = stats.get('roi', {})
    fig.add_trace(
        go.Scatter(
            x=['Total ROI', 'Monthly ROI', 'Daily ROI'],
            y=[_safe_num(roi.get('total_roi', 0)),
               _safe_num(roi.get('monthly_roi', 0)),
               _safe_num(roi.get('daily_roi', 0))],
            mode='lines+markers',
            name='ROI'
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

def create_equity_drawdown_chart(equity_curve: pd.Series):
    """Строит 2 оси: Equity ($) и Drawdown (%)"""
    if equity_curve is None or equity_curve.empty:
        return go.Figure()

    roll_max = equity_curve.cummax()
    dd_pct = (equity_curve / roll_max - 1.0) * 100.0

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity ($)", mode="lines"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=dd_pct.index, y=dd_pct.values, name="Drawdown (%)", mode="lines"),
        row=1, col=1, secondary_y=True
    )
    fig.update_layout(height=420, title_text="Equity & Drawdown")
    fig.update_yaxes(title_text="Equity ($)", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
    return fig

# ===================== Инициализация =====================

st.set_page_config(
    page_title="KWIN Bot - Аналитика",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def init_analytics():
    """Инициализация модулей аналитики"""
    db = Database()
    analytics = TradingAnalytics()
    trail_logger = TrailingLogger()
    return analytics, trail_logger, db

def main():
    """Основная функция страницы аналитики"""
    st.title("📈 Аналитика и статистика KWIN Bot")
    st.markdown("---")
    
    analytics, trail_logger, db = init_analytics()
    
    # Период анализа
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
        
        # Equity curve
        equity_curve_raw = None
        try:
            if hasattr(analytics, "get_equity_curve"):
                equity_curve_raw = analytics.get_equity_curve(period)
        except Exception:
            equity_curve_raw = None
        equity_series = _to_equity_series(equity_curve_raw)

        total_trades = int(_safe_num(stats.get('total_trades', 0)))
        
        if total_trades > 0:
            # Основные метрики
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Всего сделок", total_trades)
            
            with col2:
                winrate = _safe_num(stats.get('winrate', {}).get('total', 0))
                st.metric("Общий Winrate", f"{winrate:.2f}%")
            
            with col3:
                total_pnl = _safe_num(stats.get('pnl', {}).get('total_pnl', 0))
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Общий PnL", f"${total_pnl:.2f}", delta_color=pnl_color)
            
            with col4:
                profit_factor = _safe_num(stats.get('pnl', {}).get('profit_factor', 0))
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            with col5:
                avg_rr = _safe_num(stats.get('risk_reward', {}).get('avg_rr', 0))
                st.metric("Средний RR", f"{avg_rr:.2f}")
            
            # Графики производительности
            st.plotly_chart(
                create_performance_chart(stats),
                use_container_width=True
            )

            # Equity & Drawdown
            if equity_series is not None and not equity_series.empty:
                st.subheader("📉 Equity & Drawdown")
                st.plotly_chart(
                    create_equity_drawdown_chart(equity_series),
                    use_container_width=True
                )
        else:
            st.info("Нет торговых данных для анализа за выбранный период")
            
    except Exception as e:
        st.error(f"Ошибка получения статистики: {e}")

if __name__ == "__main__":
    main()
