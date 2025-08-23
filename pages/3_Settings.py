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

    # ========================= Инструмент / источники =========================
    st.subheader("🧩 Инструмент и источники цен")
    i1, i2, i3 = st.columns(3)
    with i1:
        symbol = st.text_input("Символ", value=str(getattr(config, "symbol", "ETHUSDT")).upper())
    with i2:
        price_for_logic = st.selectbox(
            "Источник цены для логики",
            options=["last", "mark"],
            index=0 if str(getattr(config, "price_for_logic", "last")).lower() == "last" else 1,
            help="Какую цену использовать для внутренних расчётов стратегии."
        )
    with i3:
        trigger_price_source = st.selectbox(
            "Триггер для SL/TP на бирже",
            options=["mark", "last"],
            index=0 if str(getattr(config, "trigger_price_source", "mark")).lower() == "mark" else 1,
            help="По какой цене (Mark/Last) биржа будет срабатывать на SL/TP."
        )

    st.markdown("---")

    # ========================= Основные =========================
    st.subheader("🎯 Основные настройки торговли")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_pct = st.number_input(
            "Риск на сделку (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(getattr(config, "risk_pct", 3.0)),
            step=0.1,
            help="Процент от капитала, рискуемый на одну сделку"
        )
        max_qty = st.number_input(
            "Максимальная позиция (в базовом активе)",
            min_value=0.001,
            max_value=10000.0,
            value=float(getattr(config, "max_qty_manual", 50.0)),
            step=0.001,
            help="Максимальный размер позиции (учитывается, если включено ограничение количества)"
        )
        limit_qty_enabled = st.checkbox(
            "Ограничивать максимальную позицию",
            value=bool(getattr(config, "limit_qty_enabled", True))
        )
    with col2:
        risk_reward = st.number_input(
            "Risk/Reward соотношение",
            min_value=0.5,
            max_value=5.0,
            value=float(getattr(config, "risk_reward", 1.3)),
            step=0.1,
            help="Соотношение прибыли к убытку (R:R)"
        )
        taker_fee_rate = st.number_input(
            "Комиссия taker (десятичная)",
            min_value=0.0,
            max_value=0.01,
            value=float(getattr(config, "taker_fee_rate", 0.00055)),
            step=0.00005
        )
        use_take_profit = st.checkbox(
            "Использовать Take Profit",
            value=bool(getattr(config, "use_take_profit", True))
        )
    with col3:
        # SFP Length и Intrabar TF убраны — управляются через Lux (Swings/LTF)
        st.write("")
        st.caption("Параметры SFP (Swings, LTF) задаются в блоке **Lux SFP** ниже.")

    st.markdown("---")

    # =========================== Lux SFP ===========================
    st.subheader("✨ Lux SFP (как в LuxAlgo)")

    l0, = st.columns(1)
    with l0:
        lux_enabled = st.checkbox(
            "Включить Lux SFP",
            value=bool(getattr(config, "lux_mode", True)),
            help="Главный фильтр входа. При включении старая валидация входа отключена."
        )

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        lux_volume_validation = st.selectbox(
            "Validation",
            options=["outside_gt", "outside_lt", "none"],
            index={"outside_gt": 0, "outside_lt": 1, "none": 2}.get(
                str(getattr(config, "lux_volume_validation", "outside_gt")).lower(), 0
            ),
            help="Правило валидации объёма на младшем ТФ (доля объёма ‘вне свинга’ относительно порога)."
        )
        lux_swings = st.number_input(
            "Swings",
            min_value=1, max_value=20,
            value=int(getattr(config, "lux_swings", 2)),
            step=1,
            help="Аналог len в Lux: сдвиг свинга для построения уровня."
        )
    with l2:
        lux_volume_threshold_pct = st.number_input(
            "Volume Threshold %",
            min_value=0.0, max_value=100.0,
            value=float(getattr(config, "lux_volume_threshold_pct", 10.0)),
            step=0.5,
            help="% от суммарного объёма LTF-бара, приходящаяся на ‘вне свинга’."
        )
        lux_auto = st.checkbox(
            "Auto (ресемплинг LTF)",
            value=bool(getattr(config, "lux_auto", False)),
            help="Автоматический расчёт LTF из текущего ТФ (как у Lux)."
        )
    with l3:
        lux_mlt = st.number_input(
            "Auto mlt",
            min_value=1, max_value=120,
            value=int(getattr(config, "lux_mlt", 10)),
            step=1,
            help="Делитель для авто-выбора LTF (секунды текущего ТФ / mlt)."
        )
        lux_ltf = st.selectbox(
            "LTF (ручной)",
            options=["1", "3", "5"],
            index=["1", "3", "5"].index(str(getattr(config, "lux_ltf", "1"))),
            help="Если Auto выключен — используем этот младший ТФ."
        )
    with l4:
        lux_premium = st.checkbox(
            "Premium",
            value=bool(getattr(config, "lux_premium", False)),
            help="Ограничивает минимальный интервал LTF (как в Lux)."
        )
        lux_expire_bars = st.number_input(
            "Expire bars",
            min_value=10, max_value=2000,
            value=int(getattr(config, "lux_expire_bars", 500)),
            step=10,
            help="Через сколько баров уровень SFP перестаёт быть активным."
        )

    st.markdown("---")

    # =============================== Smart Trailing ===============================
    st.subheader("🎯 Smart Trailing")

    t1, t2 = st.columns(2)
    with t1:
        enable_smart_trail = st.checkbox(
            "Включить Smart Trailing",
            value=bool(getattr(config, 'enable_smart_trail', True)),
            help="Умный трейлинг SL (аналог Pine-логики)."
        )
        use_arm_after_rr = st.checkbox(
            "Арминг трейла после достижения RR",
            value=bool(getattr(config, 'use_arm_after_rr', True)),
            help="Активировать трейлинг только после достижения заданного RR."
        )
        arm_rr = st.number_input(
            "RR для арминга",
            min_value=0.1,
            max_value=5.0,
            value=float(getattr(config, 'arm_rr', 0.5)),
            step=0.1
        )
        arm_rr_basis = st.selectbox(
            "База расчёта RR для арминга",
            options=["extremum", "last"],
            index=0 if str(getattr(config, "arm_rr_basis", "extremum")).lower() == "extremum" else 1
        )
    with t2:
        trailing_perc = st.number_input(
            "Процент трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_perc', 0.5)),
            step=0.1
        )
        trailing_offset_perc = st.number_input(
            "Offset трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_offset_perc', 0.4)),
            step=0.1
        )

    st.markdown("---")

    # =============================== Баровый трейл / прочее ===============================
    st.subheader("🧱 Баровый трейл / прочее")
    b1, b2, b3 = st.columns(3)
    with b1:
        use_bar_trail = st.checkbox(
            "Баровый трейлинг (lowest/highest N закрытых баров)",
            value=bool(getattr(config, 'use_bar_trail', True)),
        )
    with b2:
        trail_lookback = st.number_input(
            "Trail lookback bars",
            min_value=1,
            max_value=300,
            value=int(getattr(config, "trail_lookback", 50)),
            step=1
        )
    with b3:
        trail_buf_ticks = st.number_input(
            "Trail buffer (ticks)",
            min_value=0,
            max_value=500,
            value=int(getattr(config, "trail_buf_ticks", 40)),
            step=1
        )

    st.markdown("---")

    # ============================== Сохранение ==============================
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
            try:
                # Источники
                config.symbol = str(symbol).upper()
                config.price_for_logic = str(price_for_logic).lower()
                config.trigger_price_source = str(trigger_price_source).lower()

                # Риск/ограничения
                config.risk_pct = float(risk_pct)
                config.risk_reward = float(risk_reward)
                config.limit_qty_enabled = bool(limit_qty_enabled)
                config.max_qty_manual = float(max_qty)
                config.taker_fee_rate = float(taker_fee_rate)
                config.use_take_profit = bool(use_take_profit)

                # Lux SFP
                config.lux_mode = bool(lux_enabled)
                config.lux_swings = int(lux_swings)
                config.lux_volume_validation = str(lux_volume_validation)
                config.lux_volume_threshold_pct = float(lux_volume_threshold_pct)
                config.lux_auto = bool(lux_auto)
                config.lux_mlt = int(lux_mlt)
                config.lux_ltf = str(lux_ltf)
                config.lux_premium = bool(lux_premium)
                config.lux_expire_bars = int(lux_expire_bars)

                # Smart trailing
                config.enable_smart_trail = bool(enable_smart_trail)
                config.use_arm_after_rr = bool(use_arm_after_rr)
                config.arm_rr = float(arm_rr)
                config.arm_rr_basis = str(arm_rr_basis)
                config.trailing_perc = float(trailing_perc)
                config.trailing_offset_perc = float(trailing_offset_perc)
                config.trailing_offset = float(trailing_offset_perc)  # alias

                # Баровый трейл/прочее
                config.use_bar_trail = bool(use_bar_trail)
                config.trail_lookback = int(trail_lookback)
                config.trail_buf_ticks = int(trail_buf_ticks)

                # Удалённые блоки НЕ сохраняем и не трогаем:
                # - старая SFP-валидация
                # - Stop-Loss Zone
                # - Cooldown
                # - SFP Length / Intrabar TF

                ok = config.validate()
                if not ok:
                    st.error("❌ Валидация конфигурации не пройдена. Проверь значения.")
                else:
                    config.save_config()
                    st.success("✅ Настройки успешно сохранены и будут применены в торговле!")
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
