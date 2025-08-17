import streamlit as st
import sys
import os

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

st.set_page_config(
    page_title="KWIN Bot - Настройки",
    page_icon="⚙️",
    layout="wide"
)

def main():
    st.title("⚙️ Настройки KWIN Trading Bot")
    st.markdown("---")
    
    config = Config()
    
    # Основные настройки торговли
    st.subheader("🎯 Основные настройки торговли")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_pct = st.number_input(
            "Риск на сделку (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(config.risk_pct),
            step=0.1,
            help="Процент от капитала, рискуемый на одну сделку"
        )
        
        risk_reward = st.number_input(
            "Risk/Reward соотношение",
            min_value=0.5,
            max_value=5.0,
            value=float(config.risk_reward),
            step=0.1,
            help="Соотношение прибыли к убытку"
        )
        
        max_qty = st.number_input(
            "Максимальная позиция (ETH)",
            min_value=0.01,
            max_value=100.0,
            value=float(config.max_qty_manual),
            step=0.01,
            help="Максимальный размер позиции в ETH"
        )
    
    with col2:
        sfp_len = st.number_input(
            "SFP Length",
            min_value=1,
            max_value=10,
            value=int(getattr(config, 'sfp_len', 2)),
            help="Длина поиска Swing Failure Pattern"
        )
        
        use_sfp_quality = st.checkbox(
            "Фильтр качества SFP",
            value=getattr(config, 'use_sfp_quality', True),
            help="Включить дополнительную фильтрацию SFP по качеству"
        )
        
        wick_min_ticks = st.number_input(
            "Минимальная глубина фитиля (тики)",
            min_value=0,
            max_value=50,
            value=int(getattr(config, 'wick_min_ticks', 7)),
            help="Минимальная глубина фитиля для валидного SFP"
        )
    
    # Smart Trailing настройки
    st.subheader("🎯 Smart Trailing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_smart_trail = st.checkbox(
            "Включить Smart Trailing",
            value=getattr(config, 'enable_smart_trail', True),
            help="Включить систему умного трейлинга стоп-лосса"
        )
        
        use_arm_after_rr = st.checkbox(
            "Арминг после RR",
            value=getattr(config, 'use_arm_after_rr', True),
            help="Активировать трейлинг только после достижения определенного RR"
        )
        
        arm_rr = st.number_input(
            "RR для арминга",
            min_value=0.1,
            max_value=2.0,
            value=float(getattr(config, 'arm_rr', 0.5)),
            step=0.1,
            help="Risk/Reward для активации трейлинга"
        )
    
    with col2:
        trailing_perc = st.number_input(
            "Процент трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_perc', 0.5)),
            step=0.1,
            help="Процент от цены входа для трейлинга"
        )
        
        trailing_offset = st.number_input(
            "Offset трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_offset_perc', 0.4)),
            step=0.1,
            help="Дополнительный отступ для трейлинга"
        )
        
        use_bar_trail = st.checkbox(
            "Баровый трейлинг",
            value=getattr(config, 'use_bar_trail', True),
            help="Использовать трейлинг по максимумам/минимумам баров"
        )
    
    # Фильтры и гварды
    st.subheader("🛡️ Фильтры и защита")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_stop_guards = st.checkbox(
            "Гварды стоп-лосса",
            value=getattr(config, 'use_stop_guards', False),
            help="Включить дополнительную проверку корректности SL"
        )
        
        max_stop_pct = st.number_input(
            "Максимальный SL (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(getattr(config, 'max_stop_pct', 8.0)),
            step=0.5,
            help="Максимальный размер стоп-лосса в процентах"
        )
    
    with col2:
        close_back_pct = st.number_input(
            "Close-back процент",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(config, 'close_back_pct', 1.0)),
            step=0.05,
            help="Процент возврата цены закрытия для SFP (0.0-1.0)"
        )
        
        min_profit_usd = st.number_input(
            "Минимальная прибыль ($)",
            min_value=0.0,
            max_value=100.0,
            value=float(getattr(config, 'min_profit_usd', 0.0)),
            step=1.0,
            help="Минимальная ожидаемая прибыль в USD"
        )
    
    # Кнопка сохранения
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
            try:
                # Обновляем конфигурацию
                config.risk_pct = risk_pct
                config.risk_reward = risk_reward
                config.max_qty_manual = max_qty
                config.sfp_len = sfp_len
                config.use_sfp_quality = use_sfp_quality
                config.wick_min_ticks = wick_min_ticks
                config.enable_smart_trail = enable_smart_trail
                config.use_arm_after_rr = use_arm_after_rr
                config.arm_rr = arm_rr
                config.trailing_perc = trailing_perc
                config.trailing_offset_perc = trailing_offset
                config.use_bar_trail = use_bar_trail
                config.use_stop_guards = use_stop_guards
                config.max_stop_pct = max_stop_pct
                config.close_back_pct = close_back_pct
                config.min_profit_usd = min_profit_usd
                
                # Сохраняем в файл
                config.save_config()
                
                st.success("✅ Настройки успешно сохранены!")
                
            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {e}")
    
    # Информация о текущих настройках
    st.markdown("---")
    st.subheader("📋 Текущая конфигурация")
    
    with st.expander("Показать все параметры"):
        config_dict = {
            "Риск на сделку": f"{risk_pct}%",
            "Risk/Reward": risk_reward,
            "Максимальная позиция": f"{max_qty} ETH",
            "SFP Length": sfp_len,
            "Фильтр качества SFP": "Включен" if use_sfp_quality else "Выключен",
            "Smart Trailing": "Включен" if enable_smart_trail else "Выключен",
            "Арминг после RR": "Включен" if use_arm_after_rr else "Выключен",
            "RR для арминга": arm_rr,
            "Процент трейлинга": f"{trailing_perc}%",
            "Баровый трейлинг": "Включен" if use_bar_trail else "Выключен",
        }
        
        for key, value in config_dict.items():
            st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()
