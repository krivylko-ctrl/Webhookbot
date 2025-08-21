import streamlit as st
import sys
import os

# Если запускаешь страницу из подпапки — добавим корень в PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import Config

st.set_page_config(
    page_title="KWIN Bot - Настройки",
    page_icon="⚙️",
    layout="wide"
)

def main():
    st.title("⚙️ Настройки KWIN Trading Bot")
    st.markdown("---")

    # Загружаем текущую конфигурацию (внутри она подтягивает config.json, если есть)
    config = Config()

    # ========================= Основные настройки торговли =========================
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
            help="Соотношение прибыли к убытку (R:R)"
        )

        max_qty = st.number_input(
            "Максимальная позиция (в базовом активе)",
            min_value=0.001,
            max_value=1000.0,
            value=float(config.max_qty_manual),
            step=0.001,
            help="Максимальный размер позиции (учитывается, если включено ограничение количества)"
        )

    with col2:
        sfp_len = st.number_input(
            "SFP Length",
            min_value=1,
            max_value=10,
            value=int(getattr(config, 'sfp_len', 2)),
            help="Длина для поиска Swing Failure Pattern (как в Pine)"
        )

        use_sfp_quality = st.checkbox(
            "Фильтр качества SFP (wick + close-back)",
            value=bool(getattr(config, 'use_sfp_quality', True)),
            help="Включить фильтры качества SFP"
        )

        wick_min_ticks = st.number_input(
            "Минимальная глубина фитиля (в тиках)",
            min_value=0,
            max_value=100,
            value=int(getattr(config, 'wick_min_ticks', 7)),
            help="Минимальная глубина фитиля для валидного SFP (в тик-сайзах инструмента)"
        )

    # =============================== Smart Trailing ================================
    st.subheader("🎯 Smart Trailing")

    col1, col2 = st.columns(2)
    with col1:
        enable_smart_trail = st.checkbox(
            "Включить Smart Trailing",
            value=bool(getattr(config, 'enable_smart_trail', True)),
            help="Умный трейлинг SL (аналог Pine-логики)"
        )

        use_arm_after_rr = st.checkbox(
            "Арминг трейла после достижения RR",
            value=bool(getattr(config, 'use_arm_after_rr', True)),
            help="Активировать трейлинг только после достижения заданного RR"
        )

        arm_rr = st.number_input(
            "RR для арминга",
            min_value=0.1,
            max_value=5.0,
            value=float(getattr(config, 'arm_rr', 0.5)),
            step=0.1,
            help="Минимальное значение R, после которого включается трейл"
        )

        arm_rr_basis = st.selectbox(
            "База расчёта RR для арминга",
            options=["extremum", "last"],
            index=0 if getattr(config, "arm_rr_basis", "extremum") == "extremum" else 1,
            help="extremum — считаем от экстремума бара; last — от текущей цены"
        )

    with col2:
        trailing_perc = st.number_input(
            "Процент трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_perc', 0.5)),
            step=0.1,
            help="Процент от цены входа для вычисления дистанции трейла"
        )

        trailing_offset_perc = st.number_input(
            "Offset трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_offset_perc', 0.4)),
            step=0.1,
            help="Дополнительный отступ от дистанции трейла"
        )

        use_bar_trail = st.checkbox(
            "Баровый трейлинг (lowest/highest N закрытых баров)",
            value=bool(getattr(config, 'use_bar_trail', True)),
            help="Совместимость со старым режимом: lowest(low, N)[1] / highest(high, N)[1]"
        )

    # =========================== Фильтры (из Pine) ===========================
    st.subheader("🛡️ Фильтры SFP")

    col1, col2 = st.columns(2)
    with col1:
        close_back_pct = st.number_input(
            "Close-back (0.0 … 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(config, 'close_back_pct', 1.0)),
            step=0.05,
            help="Требуемая доля возврата закрытия относительно глубины фитиля (как в Pine)"
        )

    with col2:
        # опционально покажем текущие шаги инструмента (read-only)
        st.text_input(
            "Tick size (read-only)",
            value=str(getattr(config, 'tick_size', 0.01)),
            disabled=True
        )

    # ============================== Сохранение ==============================
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
            try:
                # Обновляем объект конфигурации
                config.risk_pct = float(risk_pct)
                config.risk_reward = float(risk_reward)
                config.max_qty_manual = float(max_qty)

                config.sfp_len = int(sfp_len)
                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct)

                config.enable_smart_trail = bool(enable_smart_trail)
                config.use_arm_after_rr = bool(use_arm_after_rr)
                config.arm_rr = float(arm_rr)
                config.arm_rr_basis = str(arm_rr_basis)

                config.trailing_perc = float(trailing_perc)
                config.trailing_offset_perc = float(trailing_offset_perc)
                # alias для обратной совместимости
                config.trailing_offset = float(trailing_offset_perc)

                config.use_bar_trail = bool(use_bar_trail)

                # Нормализация и валидация перед сохранением
                ok = config.validate()
                if not ok:
                    st.error("❌ Валидация конфигурации не пройдена. Проверь значения.")
                else:
                    config.save_config()
                    st.success("✅ Настройки успешно сохранены!")

            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {e}")

    # ============================== Просмотр текущих ==============================
    st.markdown("---")
    st.subheader("📋 Текущая конфигурация")

    with st.expander("Показать все параметры (config.json)"):
        try:
            st.json(config.to_dict())
        except Exception:
            st.write(config.to_dict())

if __name__ == "__main__":
    main()
