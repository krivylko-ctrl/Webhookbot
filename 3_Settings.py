import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

from config import Config
from database import Database
from state_manager import StateManager
from bybit_api import BybitAPI

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

def main():
    st.title("⚙️ Настройки стратегии KWIN")
    
    # Инициализация компонентов
    @st.cache_resource
    def init_components():
        config = Config()
        db = Database()
        state_manager = StateManager(db)
        return config, db, state_manager
    
    config, db, state_manager = init_components()
    
    # Создаем табы для разных категорий настроек
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Основные параметры", 
        "🔄 Smart Trailing", 
        "📊 SFP Фильтры", 
        "💰 Управление капиталом", 
        "🔧 API & Система"
    ])
    
    # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
    with tab1:
        st.markdown("### 🎯 Основные параметры стратегии")
        st.markdown("*Эквивалент основных inputs в TradingView*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_reward = st.number_input(
                "🎯 TP Risk/Reward Ratio",
                min_value=0.5,
                max_value=5.0,
                value=config.risk_reward,
                step=0.1,
                help="Соотношение риска к прибыли для тейк-профита"
            )
            
            sfp_len = st.number_input(
                "📏 Swing Length",
                min_value=1,
                max_value=10,
                value=config.sfp_len,
                step=1,
                help="Длина свинга для определения экстремумов"
            )
            
            risk_pct = st.number_input(
                "💸 Risk % per trade",
                min_value=0.1,
                max_value=10.0,
                value=config.risk_pct,
                step=0.1,
                format="%.1f",
                help="Процент от капитала, рискуемый в каждой сделке"
            )
        
        with col2:
            period_choice = st.selectbox(
                "📅 Backtest Period",
                options=["30", "60", "180"],
                index=["30", "60", "180"].index(config.period_choice),
                help="Период для бэктестирования в днях"
            )
            
            st.markdown("#### 📊 Текущие значения")
            st.info(f"""
            **Risk/Reward:** {config.risk_reward}
            **Swing Length:** {config.sfp_len}
            **Risk per Trade:** {config.risk_pct}%
            **Backtest Period:** {config.period_choice} дней
            """)
    
    # === SMART TRAILING ===
    with tab2:
        st.markdown("### 🔄 Smart Trailing System")
        st.markdown("*Настройки системы умного трейлинга стоп-лоссов*")
        
        enable_smart_trail = st.checkbox(
            "✅ Enable Smart Trailing TP",
            value=config.enable_smart_trail,
            help="Включить систему умного трейлинга"
        )
        
        if enable_smart_trail:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Основные параметры трейлинга")
                
                trailing_perc = st.number_input(
                    "📈 Trailing %",
                    min_value=0.1,
                    max_value=5.0,
                    value=config.trailing_perc,
                    step=0.1,
                    help="Процент для активации трейлинга"
                )
                
                trailing_offset = st.number_input(
                    "📏 Trailing Offset %",
                    min_value=0.1,
                    max_value=5.0,
                    value=config.trailing_offset,
                    step=0.1,
                    help="Процентное смещение для трейлинга"
                )
                
                use_arm_after_rr = st.checkbox(
                    "🎯 Enable Arm after RR≥X",
                    value=config.use_arm_after_rr,
                    help="Включать трейлинг только после достижения определенного RR"
                )
                
                if use_arm_after_rr:
                    arm_rr = st.number_input(
                        "🔢 Arm RR (R)",
                        min_value=0.1,
                        max_value=2.0,
                        value=config.arm_rr,
                        step=0.1,
                        help="RR для активации арминга"
                    )
                else:
                    arm_rr = config.arm_rr
            
            with col2:
                st.markdown("#### 📊 Bar Trail настройки")
                
                use_bar_trail = st.checkbox(
                    "📊 Use Bar-Low/High Smart Trail",
                    value=config.use_bar_trail,
                    help="Использовать трейлинг по барам"
                )
                
                if use_bar_trail:
                    trail_lookback = st.number_input(
                        "👀 Trail lookback bars",
                        min_value=1,
                        max_value=200,
                        value=config.trail_lookback,
                        step=1,
                        help="Количество баров для анализа трейлинга"
                    )
                    
                    trail_buf_ticks = st.number_input(
                        "📏 Trail buffer (ticks)",
                        min_value=0,
                        max_value=100,
                        value=config.trail_buf_ticks,
                        step=1,
                        help="Буфер в тиках для трейлинга"
                    )
                else:
                    trail_lookback = config.trail_lookback
                    trail_buf_ticks = config.trail_buf_ticks
        else:
            # Используем текущие значения если трейлинг отключен
            trailing_perc = config.trailing_perc
            trailing_offset = config.trailing_offset
            use_arm_after_rr = config.use_arm_after_rr
            arm_rr = config.arm_rr
            use_bar_trail = config.use_bar_trail
            trail_lookback = config.trail_lookback
            trail_buf_ticks = config.trail_buf_ticks
    
    # === SFP ФИЛЬТРЫ ===
    with tab3:
        st.markdown("### 📊 SFP Quality Filters")
        st.markdown("*Фильтры качества Swing Failure Pattern*")
        
        use_sfp_quality = st.checkbox(
            "🔍 Filter: SFP quality (wick+closeback)",
            value=config.use_sfp_quality,
            help="Включить фильтрацию качества SFP паттернов"
        )
        
        if use_sfp_quality:
            col1, col2 = st.columns(2)
            
            with col1:
                wick_min_ticks = st.number_input(
                    "📏 SFP: min wick depth (ticks)",
                    min_value=0,
                    max_value=100,
                    value=config.wick_min_ticks,
                    step=1,
                    help="Минимальная глубина тени в тиках"
                )
                
                close_back_pct = st.number_input(
                    "🔄 SFP: min close-back % of wick",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.close_back_pct,
                    step=0.01,
                    format="%.2f",
                    help="Минимальный процент отката закрытия от тени"
                )
            
            with col2:
                st.markdown("#### 📊 Объяснение фильтров")
                st.info("""
                **Wick Depth:** Минимальная глубина тени для валидного SFP
                
                **Close Back %:** Процент восстановления цены закрытия от экстремума тени
                
                Эти фильтры помогают отсеять слабые SFP сигналы и торговать только качественные паттерны.
                """)
        else:
            wick_min_ticks = config.wick_min_ticks
            close_back_pct = config.close_back_pct
    
    # === УПРАВЛЕНИЕ КАПИТАЛОМ ===
    with tab4:
        st.markdown("### 💰 Управление капиталом и рисками")
        st.markdown("*Ограничения размера позиций и минимальной прибыли*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Ограничения позиций")
            
            limit_qty_enabled = st.checkbox(
                "🚫 Limit Max Position Qty",
                value=config.limit_qty_enabled,
                help="Включить ограничение максимального размера позиции"
            )
            
            if limit_qty_enabled:
                max_qty_manual = st.number_input(
                    "📊 Max Qty (ETH)",
                    min_value=0.01,
                    max_value=1000.0,
                    value=config.max_qty_manual,
                    step=0.01,
                    format="%.2f",
                    help="Максимальный размер позиции в ETH"
                )
            else:
                max_qty_manual = config.max_qty_manual
        
        with col2:
            st.markdown("#### 💰 Прибыльность")
            
            min_net_profit = st.number_input(
                "💵 Min Net Profit ($)",
                min_value=0.0,
                max_value=100.0,
                value=config.min_net_profit,
                step=0.1,
                help="Минимальная чистая прибыль для открытия сделки"
            )
            
            taker_fee_rate = st.number_input(
                "💸 Taker Fee Rate",
                min_value=0.0,
                max_value=1.0,
                value=config.taker_fee_rate,
                step=0.00001,
                format="%.5f",
                help="Комиссия тейкера (обычно 0.00055 для Bybit)"
            )
        
        st.markdown("#### 📊 Текущие ограничения")
        current_equity = state_manager.get_equity()
        max_risk = current_equity * (risk_pct / 100)
        
        st.info(f"""
        **Текущий Equity:** ${current_equity:.2f}
        **Максимальный риск на сделку:** ${max_risk:.2f}
        **Максимальный размер позиции:** {max_qty_manual if limit_qty_enabled else 'Без ограничений'} ETH
        **Минимальная прибыль:** ${min_net_profit:.2f}
        """)
    
    # === API & СИСТЕМА ===
    with tab5:
        st.markdown("### 🔧 API и системные настройки")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 API Configuration")
            
            # API ключи (только отображение статуса)
            api_key = os.getenv("BYBIT_API_KEY", "")
            api_secret = os.getenv("BYBIT_API_SECRET", "")
            
            if api_key and api_secret:
                st.success("✅ API ключи настроены")
                st.write(f"**API Key:** {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
            else:
                st.error("❌ API ключи не настроены")
                st.markdown("""
                Добавьте переменные окружения:
                - `BYBIT_API_KEY`
                - `BYBIT_API_SECRET`
                """)
            
            # Тест подключения
            if st.button("🔍 Тест подключения к API"):
                if api_key and api_secret:
                    try:
                        api = BybitAPI(api_key, api_secret, testnet=False)
                        server_time = api.get_server_time()
                        if server_time:
                            st.success(f"✅ Подключение успешно! Время сервера: {datetime.fromtimestamp(server_time)}")
                        else:
                            st.error("❌ Ошибка подключения к API")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
                else:
                    st.warning("⚠️ Сначала настройте API ключи")
        
        with col2:
            st.markdown("#### 🗄️ Database Management")
            
            # Статистика базы данных
            try:
                total_trades = len(db.get_recent_trades(1000))
                equity_records = len(db.get_equity_history(365))
                
                st.info(f"""
                **Всего сделок в БД:** {total_trades}
                **Записей equity:** {equity_records}
                **Размер файла БД:** ~{os.path.getsize(db.db_path) / 1024:.1f} KB
                """)
            except Exception as e:
                st.error(f"Ошибка чтения БД: {e}")
            
            # Управление данными
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🧹 Очистить старые данные"):
                    try:
                        db.cleanup_old_data(days_to_keep=90)
                        st.success("✅ Старые данные очищены")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
            
            with col_b:
                if st.button("📊 Экспорт данных"):
                    try:
                        trades = db.get_recent_trades(1000)
                        if trades:
                            df = pd.DataFrame(trades)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Скачать CSV",
                                data=csv,
                                file_name=f"kwin_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("Нет данных для экспорта")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
    
    # === КНОПКИ УПРАВЛЕНИЯ ===
    st.markdown("---")
    st.markdown("### 💾 Управление настройками")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Сохранить настройки", use_container_width=True):
            try:
                # Собираем все настройки
                new_config = {
                    'risk_reward': risk_reward,
                    'sfp_len': sfp_len,
                    'risk_pct': risk_pct,
                    'period_choice': period_choice,
                    'enable_smart_trail': enable_smart_trail,
                    'trailing_perc': trailing_perc,
                    'trailing_offset': trailing_offset,
                    'use_arm_after_rr': use_arm_after_rr,
                    'arm_rr': arm_rr,
                    'use_bar_trail': use_bar_trail,
                    'trail_lookback': trail_lookback,
                    'trail_buf_ticks': trail_buf_ticks,
                    'use_sfp_quality': use_sfp_quality,
                    'wick_min_ticks': wick_min_ticks,
                    'close_back_pct': close_back_pct,
                    'limit_qty_enabled': limit_qty_enabled,
                    'max_qty_manual': max_qty_manual,
                    'min_net_profit': min_net_profit,
                    'taker_fee_rate': taker_fee_rate
                }
                
                # Валидация
                config.update_from_dict(new_config)
                if config.validate():
                    st.success("✅ Настройки сохранены успешно!")
                    st.rerun()
                else:
                    st.error("❌ Ошибка валидации настроек")
                    
            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {e}")
    
    with col2:
        if st.button("🔄 Сбросить к умолчанию", use_container_width=True):
            try:
                default_config = Config()
                config.update_from_dict(default_config.to_dict())
                st.success("✅ Настройки сброшены к умолчанию!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ошибка сброса: {e}")
    
    with col3:
        if st.button("📥 Импорт настроек", use_container_width=True):
            uploaded_file = st.file_uploader(
                "Выберите файл конфигурации",
                type=['json'],
                key="config_upload"
            )
            
            if uploaded_file is not None:
                try:
                    config_data = json.load(uploaded_file)
                    config.update_from_dict(config_data)
                    st.success("✅ Настройки импортированы!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка импорта: {e}")
    
    with col4:
        if st.button("📤 Экспорт настроек", use_container_width=True):
            try:
                config_json = json.dumps(config.to_dict(), indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Скачать конфигурацию",
                    data=config_json,
                    file_name=f"kwin_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Ошибка экспорта: {e}")
    
    # === ПРЕДУПРЕЖДЕНИЯ И ИНФОРМАЦИЯ ===
    st.markdown("---")
    st.markdown("### ⚠️ Важная информация")
    
    st.warning("""
    **Внимание!** 
    - Изменение настроек влияет только на новые сделки
    - Открытые позиции продолжат работать по старым настройкам
    - Рекомендуется тестировать новые настройки на небольших суммах
    - Всегда проверяйте настройки перед запуском автоматической торговли
    """)
    
    st.info("""
    **Совет:** Используйте бэктест для проверки эффективности новых настроек перед применением в реальной торговле.
    """)

if __name__ == "__main__":
    main()
