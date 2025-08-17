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
    """
    Максимальная просадка в % по equity_curve.
    equity_curve: pd.Series с индексом времени и значениями equity.
    """
    if equity_curve is None or equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd = (equity_curve / roll_max - 1.0) * 100.0
    return float(dd.min()) if not dd.empty else 0.0

def _compute_max_drawdown_abs(equity_curve: pd.Series) -> float:
    """
    Максимальная абсолютная просадка ($) по equity_curve.
    """
    if equity_curve is None or equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd_abs = equity_curve - roll_max
    return float(dd_abs.min()) if not dd_abs.empty else 0.0

def _to_equity_series(equity_curve_raw) -> pd.Series:
    """
    Преобразует что угодно в pd.Series для расчётов просадок.
    Ожидаемые форматы:
      - list[dict{timestamp, equity}]
      - list[tuple(ts, equity)]
      - dict[ts] = equity
    """
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

def create_monthly_performance_chart(monthly_data: list):
    """График месячной производительности"""
    if not monthly_data:
        return go.Figure()
    
    months = [item.get('month') for item in monthly_data]
    pnl = [_safe_num(item.get('total_pnl', 0)) for item in monthly_data]
    winrates = [_safe_num(item.get('winrate', 0)) for item in monthly_data]
    trades = [_safe_num(item.get('trades', 0)) for item in monthly_data]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Месячный PnL (USDT)', 'Winrate & Количество сделок'],
        specs=[[{"type": "bar"}], [{"type": "xy"}]],
        shared_xaxes=True
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
    
    # Winrate и Количество сделок
    fig.add_trace(
        go.Scatter(
            x=months,
            y=winrates,
            mode='lines+markers',
            name='Winrate (%)'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=trades,
            name='Trades',
            opacity=0.6
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
                _safe_num(bull_stats.get('trades', 0)),
                _safe_num(bull_stats.get('winrate', 0)),
                _safe_num(bull_stats.get('avg_pnl', 0)),
                _safe_num(bull_stats.get('total_pnl', 0))
            ],
            name='Bull SFP',
        ),
        row=1, col=1
    )
    
    # Bear SFP
    fig.add_trace(
        go.Bar(
            x=['Trades', 'Winrate %', 'Avg PnL', 'Total PnL'],
            y=[
                _safe_num(bear_stats.get('trades', 0)),
                _safe_num(bear_stats.get('winrate', 0)),
                _safe_num(bear_stats.get('avg_pnl', 0)),
                _safe_num(bear_stats.get('total_pnl', 0))
            ],
            name='Bear SFP',
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_equity_drawdown_chart(equity_curve: pd.Series):
    """
    Строит 2 оси: Equity ($) и Drawdown (%)
    """
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

# ===================== UI =====================

def display_trailing_logs(trail_logger: TrailingLogger):
    """Отображение логов трейлинга"""
    st.subheader("🎯 Логи Smart Trailing")
    
    hours_back = st.selectbox(
        "Период логов:",
        [1, 6, 12, 24, 48, 72],
        index=3,
        format_func=lambda x: f"Последние {x} часов"
    )
    
    trail_logs = trail_logger.get_trailing_history(hours_back)
    
    if trail_logs:
        df = pd.DataFrame(trail_logs)
        display_columns = ['timestamp', 'direction', 'trigger_type', 'old_sl', 'new_sl', 
                           'current_price', 'trail_distance', 'unrealized_pnl', 'arm_status']
        if all(col in df.columns for col in display_columns):
            df_display = df[display_columns].copy()
            for col in ['old_sl', 'new_sl', 'current_price', 'trail_distance', 'unrealized_pnl']:
                if col in df_display.columns:
                    df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(2)
            st.dataframe(df_display, use_container_width=True, height=300)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего движений", len(df))
            with col2:
                st.metric("Long трейлы", int((df['direction'] == 'long').sum()) if 'direction' in df.columns else 0)
            with col3:
                st.metric("Short трейлы", int((df['direction'] == 'short').sum()) if 'direction' in df.columns else 0)
            with col4:
                avg_distance = pd.to_numeric(df.get('trail_distance', pd.Series(dtype=float)), errors="coerce").mean()
                st.metric("Средняя дистанция", f"{(avg_distance or 0):.2f}")
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info(f"Нет логов трейлинга за последние {hours_back} часов")

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
        stats = analytics.get_comprehensive_stats(period)  # ожидается словарь
        
        # --- Equity curve (для расчёта DD$, если нужно) ---
        equity_curve_raw = None
        try:
            if hasattr(analytics, "get_equity_curve"):
                equity_curve_raw = analytics.get_equity_curve(period)
        except Exception:
            equity_curve_raw = None
        equity_series = _to_equity_series(equity_curve_raw)

        # --- Проверяем/добавляем просадку в $ ---
        dd_block = stats.setdefault('drawdown', {})
        dd_pct = _safe_num(dd_block.get('max_drawdown', 0))
        dd_abs = dd_block.get('max_drawdown_abs', None)
        if dd_abs is None:
            # Попробуем посчитать из equity_curve
            if equity_series is not None and not equity_series.empty:
                dd_abs_calc = _compute_max_drawdown_abs(equity_series)
                dd_block['max_drawdown_abs'] = float(dd_abs_calc)
                # Если нет процента — тоже посчитаем
                if dd_pct == 0:
                    dd_block['max_drawdown'] = float(_compute_max_drawdown_pct(equity_series))
            else:
                dd_block['max_drawdown_abs'] = 0.0

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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_roi = _safe_num(stats.get('roi', {}).get('total_roi', 0))
                st.metric("ROI", f"{total_roi:.2f}%")
            with col2:
                max_dd_pct = _safe_num(stats.get('drawdown', {}).get('max_drawdown', 0))
                st.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}%")
            with col3:
                max_dd_abs = _safe_num(stats.get('drawdown', {}).get('max_drawdown_abs', 0))
                st.metric("Max Drawdown ($)", f"${max_dd_abs:.2f}")
            
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
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Bull SFP статистика:**")
                    bull_stats = pattern_stats.get('bull_sfp', {})
                    if bull_stats:
                        st.write(f"• Сделок: {int(_safe_num(bull_stats.get('trades', 0)))}")
                        st.write(f"• Winrate: {_safe_num(bull_stats.get('winrate', 0)):.2f}%")
                        st.write(f"• Средний PnL: ${_safe_num(bull_stats.get('avg_pnl', 0)):.2f}")
                        st.write(f"• Общий PnL: ${_safe_num(bull_stats.get('total_pnl', 0)):.2f}")
                with col2:
                    st.markdown("**Bear SFP статистика:**")
                    bear_stats = pattern_stats.get('bear_sfp', {})
                    if bear_stats:
                        st.write(f"• Сделок: {int(_safe_num(bear_stats.get('trades', 0)))}")
                        st.write(f"• Winrate: {_safe_num(bear_stats.get('winrate', 0)):.2f}%")
                        st.write(f"• Средний PnL: ${_safe_num(bear_stats.get('avg_pnl', 0)):.2f}")
                        st.write(f"• Общий PnL: ${_safe_num(bear_stats.get('total_pnl', 0)):.2f}")
            
            # Логи трейлинга
            display_trailing_logs(trail_logger)

            # ================= Проверка учёта сделок =================
            st.markdown("---")
            st.subheader("🧪 Проверка учёта сделок")
            try:
                trades_df = None
                # Предпочтительно через analytics, если есть удобный метод
                if hasattr(analytics, "get_trades_dataframe"):
                    trades_df = analytics.get_trades_dataframe(period_days=period)
                elif hasattr(analytics, "get_trades"):
                    trades_raw = analytics.get_trades(period_days=period)
                    trades_df = pd.DataFrame(trades_raw) if trades_raw else None
                else:
                    # fallback к БД
                    if hasattr(db, "get_trades"):
                        trades_raw = db.get_trades(period_days=period)
                        trades_df = pd.DataFrame(trades_raw) if trades_raw else None

                if trades_df is not None and not trades_df.empty:
                    # короткий summary
                    total = len(trades_df)
                    closed = int((trades_df.get("status") == "closed").sum()) if "status" in trades_df.columns else None
                    long_cnt = int((trades_df.get("direction") == "long").sum()) if "direction" in trades_df.columns else None
                    short_cnt = int((trades_df.get("direction") == "short").sum()) if "direction" in trades_df.columns else None

                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Всего сделок (таблица)", total)
                    with c2: st.metric("Закрыто", closed if closed is not None else 0)
                    with c3: st.metric("Long", long_cnt if long_cnt is not None else 0)
                    with c4: st.metric("Short", short_cnt if short_cnt is not None else 0)

                    # аккуратно форматируем PnL/qty/price
                    for col in ("entry_price","exit_price","pnl","qty"):
                        if col in trades_df.columns:
                            trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce").round(4)

                    st.dataframe(
                        trades_df.head(500),
                        use_container_width=True,
                        height=350
                    )
                else:
                    st.info("Таблица сделок пуста за выбранный период. Проверь, что закрытия сделок пишутся в БД/аналитику.")
            except Exception as te:
                st.warning(f"Не удалось загрузить таблицу сделок: {te}")

        else:
            st.info(f"Нет торговых данных за последние {period} дней")
            st.markdown("""
            **Для появления аналитики необходимо:**
            1. Включить бота в режиме торговли/бэктеста  
            2. Убедиться, что сделки **закрываются** и пишутся в БД  
            3. Дождаться первых закрытых сделок — данные обновятся автоматически
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
            'drawdown': {'max_drawdown': 3.2, 'current_drawdown': 0.8, 'max_drawdown_abs': -45.2}
        }
        
        st.plotly_chart(
            create_performance_chart(demo_stats),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
