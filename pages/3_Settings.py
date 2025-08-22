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

    # ========================= Идентификатор инструмента / источники =========================
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

    # ========================= Основные настройки торговли =========================
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
        sfp_len = st.number_input(
            "SFP Length",
            min_value=1,
            max_value=10,
            value=int(getattr(config, 'sfp_len', 2)),
            help="Длина для поиска Swing Failure Pattern (как в Pine)"
        )
        intrabar_tf = st.selectbox(
            "Интрабар TF",
            options=["1", "3", "5"],
            index=["1","3","5"].index(str(getattr(config, "intrabar_tf", "1"))),
            help="Таймфрейм для интрабарной логики/данных."
        )
        use_intrabar = st.checkbox(
            "Включить интрабар-трейл/обновления",
            value=bool(getattr(config, "use_intrabar", True))
        )

    st.markdown("---")

    # =========================== Фильтры (из Pine) ===========================
    st.subheader("🛡️ Фильтры SFP")

    f1, f2, f3 = st.columns(3)
    with f1:
        use_sfp_quality = st.checkbox(
            "Фильтр качества SFP (wick + close-back)",
            value=bool(getattr(config, 'use_sfp_quality', True)),
            help="Включить фильтры качества SFP"
        )
    with f2:
        wick_min_ticks = st.number_input(
            "Минимальная глубина фитиля (в тиках)",
            min_value=0,
            max_value=100,
            value=int(getattr(config, 'wick_min_ticks', 7)),
            help="Минимальная глубина фитиля для валидного SFP (в тик-сайзах инструмента)"
        )
    with f3:
        close_back_pct = st.number_input(
            "Close-back (0.0 … 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(config, 'close_back_pct', 1.0)),
            step=0.05,
            help="Требуемая доля возврата закрытия относительно глубины фитиля (как в Pine)"
        )

    st.markdown("---")

    # ======================= Stop-Loss Zone (Pine-like) =======================
    st.subheader("📌 Stop-Loss Zone (Pine-like)")
    z1, z2, z3, z4, z5 = st.columns(5)
    with z1:
        use_swing_sl = st.checkbox(
            "SL от свинга (pivot)",
            value=bool(getattr(config, "use_swing_sl", True)),
            help="База SL — свинговый high/low (pivot)."
        )
    with z2:
        use_prev_candle_sl = st.checkbox(
            "SL от свечи [1]",
            value=bool(getattr(config, "use_prev_candle_sl", False)),
            help="База SL — high[1]/low[1] (предыдущая свеча)."
        )
    with z3:
        sl_buf_ticks = st.number_input(
            "Буфер к SL (ticks)",
            min_value=0,
            max_value=2000,
            value=int(getattr(config, "sl_buf_ticks", 40)),
            step=1,
            help="Отступ от базы в тик-сайзах, добавляемый к SL."
        )
    with z4:
        use_atr_buffer = st.checkbox(
            "ATR-буфер",
            value=bool(getattr(config, "use_atr_buffer", False)),
            help="Добавлять к SL дополнительную подушку ATR*mult."
        )
    with z5:
        atr_mult = st.number_input(
            "ATR Mult",
            min_value=0.0,
            max_value=10.0,
            value=float(getattr(config, "atr_mult", 0.0)),
            step=0.1,
            help="Множитель ATR для дополнительного буфера к SL (если включено)."
        )

    st.markdown("---")

    # =============================== Smart Trailing ================================
    st.subheader("🎯 Smart Trailing")

    t1, t2, t3 = st.columns(3)
    with t1:
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
            index=0 if str(getattr(config, "arm_rr_basis", "extremum")).lower() == "extremum" else 1,
            help="extremum — считаем от экстремума бара; last — от текущей цены"
        )
    with t2:
        trailing_basis = st.selectbox(
            "Базис трейла",
            options=["risk_r", "entry_pct"],
            index=0 if str(getattr(config, "trailing_basis", "risk_r")) == "risk_r" else 1,
            help="risk_r — дистанция в R (риск-юнитах), entry_pct — проценты от цены входа."
        )
        trailing_r = st.number_input(
            "Trailing (в R)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, "trailing_r", 0.5)),
            step=0.1,
            help="Дистанция трейла в R (используется при базисе risk_r)."
        )
        trailing_offset_r = st.number_input(
            "Offset (в R)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, "trailing_offset_r", 0.0)),
            step=0.1,
            help="Дополнительный отступ в R при базисе risk_r."
        )
    with t3:
        trailing_perc = st.number_input(
            "Процент трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_perc', 0.5)),
            step=0.1,
            help="Процент от цены входа (используется при базисе entry_pct)."
        )
        trailing_offset_perc = st.number_input(
            "Offset трейлинга (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(getattr(config, 'trailing_offset_perc', 0.4)),
            step=0.1,
            help="Дополнительный отступ (используется при базисе entry_pct)."
        )
        # совместимость
        st.caption("Если выбран базис risk_r — проценты игнорируются, и наоборот.")

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

    # =============================== Контроль перезаходов ===============================
    st.subheader("🧊 Cooldown после закрытия позиции")
    cd1 = st.number_input(
        "Cooldown (минуты)",
        min_value=0,
        max_value=240,
        value=int(getattr(config, "cooldown_minutes", 0)),
        step=1,
        help="Запрещать новые входы в течение N минут после закрытия позиции (чтобы не переворачиваться сразу)."
    )

    st.markdown("---")

    # ============================== Сохранение ==============================
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
            try:
                # Обновляем объект конфигурации
                config.symbol = str(symbol).upper()

                # Источники
                config.price_for_logic = str(price_for_logic).lower()
                config.trigger_price_source = str(trigger_price_source).lower()

                # Риск/ограничения
                config.risk_pct = float(risk_pct)
                config.risk_reward = float(risk_reward)
                config.limit_qty_enabled = bool(limit_qty_enabled)
                config.max_qty_manual = float(max_qty)
                config.taker_fee_rate = float(taker_fee_rate)
                config.use_take_profit = bool(use_take_profit)

                # SFP
                config.sfp_len = int(sfp_len)
                config.use_sfp_quality = bool(use_sfp_quality)
                config.wick_min_ticks = int(wick_min_ticks)
                config.close_back_pct = float(close_back_pct)

                # SL zone
                config.use_swing_sl = bool(use_swing_sl)
                config.use_prev_candle_sl = bool(use_prev_candle_sl)
                config.sl_buf_ticks = int(sl_buf_ticks)
                config.use_atr_buffer = bool(use_atr_buffer)
                config.atr_mult = float(atr_mult)

                # Smart trailing
                config.enable_smart_trail = bool(enable_smart_trail)
                config.use_arm_after_rr = bool(use_arm_after_rr)
                config.arm_rr = float(arm_rr)
                config.arm_rr_basis = str(arm_rr_basis)

                # Базис трейла
                config.trailing_basis = str(trailing_basis)
                config.trailing_r = float(trailing_r)
                config.trailing_offset_r = float(trailing_offset_r)
                config.trailing_perc = float(trailing_perc)
                config.trailing_offset_perc = float(trailing_offset_perc)
                config.trailing_offset = float(trailing_offset_perc)  # alias

                # Баровый трейл/прочее
                config.use_bar_trail = bool(use_bar_trail)
                config.trail_lookback = int(trail_lookback)
                config.trail_buf_ticks = int(trail_buf_ticks)

                # Интрабар/TF
                config.intrabar_tf = str(intrabar_tf)
                config.use_intrabar = bool(use_intrabar)

                # Cooldown
                config.cooldown_minutes = int(cd1)

                # Нормализация и валидация перед сохранением
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
