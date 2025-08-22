import os
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config
import utils


st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")


def main():
    st.title("📊 Dashboard")

    # ------------------------ ИНИЦИАЛИЗАЦИЯ ------------------------
    @st.cache_resource
    def init_components():
        cfg = Config()
        db = Database()
        state = StateManager(db)

        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "yes")

        bybit = None
        if api_key and api_secret:
            bybit = BybitAPI(api_key, api_secret, testnet=testnet)
            # Мы работаем только с деривативами (фьючерсы)
            bybit.set_market_type("linear")

        # Стратегия получает API/DB/State и сама тянет свечи
        strat = KWINStrategy(cfg, api=bybit, state_manager=state, db=db)
        return cfg, db, state, bybit, strat

    config, db, state_manager, bybit_api, strategy = init_components()

    # Символ по умолчанию (можно расширить до селектора)
    symbol = getattr(config, "symbol", "ETHUSDT")

    if bybit_api is None:
        st.error("⚠️ API ключи Bybit не настроены (BYBIT_API_KEY/BYBIT_API_SECRET).")
        st.stop()

    # ------------------------ ПОМОЩНИКИ ------------------------
    def get_display_price() -> float:
        """
        Безопасно получить цену для отображения.
        Берём источник из config.price_for_logic: 'last' или 'mark'.
        """
        try:
            src = str(getattr(config, "price_for_logic", "last")).lower()
            # Унифицированный метод BybitAPI
            px = bybit_api.get_price(symbol, source=src)
            if px and px > 0:
                return float(px)

            # Фолбэк: прямой тикер
            t = bybit_api.get_ticker(symbol) or {}
            last = t.get("lastPrice")
            mark = t.get("markPrice")
            if src == "mark" and mark is not None:
                return float(mark)
            if last is not None:
                return float(last)
            if mark is not None:
                return float(mark)
        except Exception:
            pass
        return 0.0

    def poll_candles(allow_entries: bool):
        """
        Подтянуть свечи 15m/1m из деривативного (linear) эндпоинта и
        дать стратегии их обработать.
        Если бот остановлен, временно блокируем входы (но позволяем трейлить).
        """
        if not bybit_api:
            return
        if allow_entries:
            strategy.update_candles()
        else:
            # временно отключаем входы, чтобы при закрытии 15m не открылось позиции
            prev_flags = (strategy.can_enter_long, strategy.can_enter_short)
            strategy.can_enter_long = False
            strategy.can_enter_short = False
            try:
                strategy.update_candles()
            finally:
                strategy.can_enter_long, strategy.can_enter_short = prev_flags

    # ------------------------ АВТООБНОВЛЕНИЕ ------------------------
    if "last_update" not in st.session_state:
        st.session_state.last_update = 0.0

    bot_status = state_manager.get_bot_status()
    is_running = (bot_status == "running")

    # автообновление раз в 5 сек
    now = time.time()
    if now - st.session_state.last_update > 5:
        st.session_state.last_update = now
        # подтягиваем свечи; если бот остановлен — без входов
        poll_candles(allow_entries=is_running)
        st.rerun()

    # ------------------------ ОСНОВНЫЕ МЕТРИКИ ------------------------
    st.markdown("### 📈 Основные метрики")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        equity = state_manager.get_equity()
        st.metric("💰 Equity", f"${equity:.2f}")

    position = state_manager.get_current_position()
    with col2:
        if position:
            qty = position.get("size") or position.get("quantity") or 0.0
            pos_text = f"{qty:.4f} {symbol.replace('USDT','')}"
            pos_direction = (position.get("direction") or "").upper()
            st.metric("📍 Позиция", f"{pos_direction} {pos_text}")
        else:
            st.metric("📍 Позиция", "Нет позиции")

    with col3:
        try:
            current_price = get_display_price()
            if current_price > 0:
                st.metric(f"💹 Цена {symbol}", f"${current_price:.2f}")
            else:
                st.metric(f"💹 Цена {symbol}", "—")
        except Exception:
            st.metric(f"💹 Цена {symbol}", "Ошибка")

    with col4:
        trades_today = db.get_trades_count_today()
        st.metric("📊 Сделки сегодня", trades_today)

    with col5:
        pnl_today = db.get_pnl_today()
        st.metric("💵 PnL сегодня", utils.format_currency(pnl_today))

    # ------------------------ ТЕКУЩАЯ ПОЗИЦИЯ ------------------------
    if position:
        st.markdown("### 🎯 Текущая позиция")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.write(f"**Направление:** {(position.get('direction') or '').upper()}")
            qty = position.get("size") or position.get("quantity") or 0.0
            st.write(f"**Размер:** {qty:.4f} {symbol.replace('USDT','')}")

        with c2:
            entry_price = float(position.get("entry_price") or 0)
            st.write(f"**Вход:** ${entry_price:.2f}")

            # Текущий PnL
            if current_price and entry_price and qty:
                current_pnl = utils.calculate_pnl(
                    entry_price, current_price, qty,
                    position.get("direction"),
                    include_fees=True
                )
                pnl_color = "green" if current_pnl >= 0 else "red"
                st.markdown(
                    f"**Текущий PnL:** <span style='color:{pnl_color}'>{utils.format_currency(current_pnl)}</span>",
                    unsafe_allow_html=True
                )

        with c3:
            st.write(f"**Stop Loss:** ${float(position.get('stop_loss') or 0):.2f}")
            tp_val = position.get("take_profit")
            tp_txt = f"${float(tp_val):.2f}" if tp_val is not None else "—"
            st.write(f"**Take Profit:** {tp_txt}")

        with c4:
            armed_status = "🟢 Armed" if bool(position.get("armed", False)) else "🔴 Not Armed"
            st.write(f"**Статус:** {armed_status}")

            # Текущий RR
            if current_price and entry_price:
                current_rr = utils.calculate_rr(
                    entry_price, current_price, float(position.get("stop_loss") or 0),
                    position.get("direction")
                )
                st.write(f"**Текущий RR:** {current_rr:.2f}")

    # ------------------------ СТАТИСТИКА ------------------------
    st.markdown("### 📊 Статистика производительности")

    tabs = st.tabs(["30 дней", "60 дней", "180 дней"])
    for i, days in enumerate([30, 60, 180]):
        with tabs[i]:
            stats = db.get_performance_stats(days=days)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Всего сделок", stats.get("total_trades", 0))
                st.metric("Прибыльных", stats.get("winning_trades", 0))

            with c2:
                win_rate = stats.get("win_rate", 0.0)
                st.metric("Win Rate", utils.format_percentage(win_rate))
                avg_rr = stats.get("avg_rr", 0.0)
                st.metric("Средний RR", f"{avg_rr:.2f}")

            with c3:
                total_pnl = stats.get("total_pnl", 0.0)
                st.metric("Общий PnL", utils.format_currency(total_pnl))
                avg_pnl = stats.get("avg_pnl", 0.0)
                st.metric("Средний PnL", utils.format_currency(avg_pnl))

            with c4:
                max_win = stats.get("max_win", 0.0)
                st.metric("Макс. прибыль", utils.format_currency(max_win))
                max_loss = stats.get("max_loss", 0.0)
                st.metric("Макс. убыток", utils.format_currency(max_loss))

    # ------------------------ КРИВАЯ EQUITY ------------------------
    st.markdown("### 💰 Кривая Equity")

    equity_data = db.get_equity_history(days=30)
    if equity_data:
        df_eq = pd.DataFrame(equity_data)
        df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"], errors="coerce")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_eq["timestamp"],
            y=df_eq["equity"],
            mode="lines",
            name="Equity"
        ))
        fig.update_layout(
            title="Изменение Equity за последние 30 дней",
            xaxis_title="Дата",
            yaxis_title="Equity ($)",
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения кривой equity.")

    # ------------------------ ПОСЛЕДНИЕ СДЕЛКИ ------------------------
    st.markdown("### 📋 Последние сделки")

    recent = db.get_recent_trades(10)
    if recent:
        df_tr = pd.DataFrame(recent)

        display_columns = [
            "entry_time", "direction", "entry_price", "exit_price",
            "quantity", "pnl", "rr", "status"
        ]
        # Мягкая совместимость: если quantity пусто — возьмём qty
        if "quantity" in df_tr.columns and df_tr["quantity"].isna().all() and "qty" in df_tr.columns:
            df_tr["quantity"] = df_tr["qty"]

        for col in display_columns:
            if col in df_tr.columns:
                if col == "entry_time":
                    df_tr[col] = pd.to_datetime(df_tr[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                elif col in ("pnl", "rr", "entry_price", "exit_price"):
                    df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").round(2)
                elif col == "quantity":
                    df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").round(4)

        column_mapping = {
            "entry_time": "Время входа",
            "direction": "Направление",
            "entry_price": "Цена входа",
            "exit_price": "Цена выхода",
            "quantity": "Количество",
            "pnl": "PnL ($)",
            "rr": "RR",
            "status": "Статус",
        }
        df_display = df_tr[[c for c in display_columns if c in df_tr.columns]].rename(columns=column_mapping)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("Нет сделок для отображения.")

    # ------------------------ УПРАВЛЕНИЕ БОТОМ ------------------------
    st.markdown("### 🎛️ Управление ботом")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("▶️ Запустить бота", use_container_width=True):
            state_manager.set_bot_status("running")
            st.success("Бот запущен!")
            st.rerun()

    with c2:
        if st.button("⏹️ Остановить бота", use_container_width=True):
            state_manager.set_bot_status("stopped")
            st.warning("Бот остановлен!")
            st.rerun()

    with c3:
        if st.button("🔄 Обновить данные", use_container_width=True):
            poll_candles(allow_entries=(state_manager.get_bot_status() == "running"))
            st.success("Данные обновлены!")
            st.rerun()

    with c4:
        if st.button("📥 Снимок equity", use_container_width=True):
            # Опционально можно запросить баланс с биржи и сохранить снапшот
            try:
                strategy._update_equity()
                st.success("Equity синхронизирован.")
            except Exception:
                st.info("Не удалось синхронизировать equity.")
            st.rerun()

    # Статус бота (визуал)
    bot_status = state_manager.get_bot_status()
    status_color = "green" if bot_status == "running" else "red"
    st.markdown(f"**Статус бота:** <span style='color:{status_color}'>{bot_status.upper()}</span>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
