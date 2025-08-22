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
    st.caption("Все параметры сохраняются в config.json и используются ботом в live/бэктесте.")
    st.markdown("---")

    # Загружаем текущую конфигурацию (внутри она подтягивает config.json, если есть)
    cfg = Config()

    # ─────────────────────────────────  ОСНОВНЫЕ ─────────────────────────────────
    st.subheader("🎯 Основные настройки торговли")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_pct = st.number_input("Риск на сделку (%)", 0.1, 10.0, float(cfg.risk_pct), 0.1,
                                   help="Процент от капитала, рискуемый на одну сделку.")
        risk_reward = st.number_input("Risk/Reward соотношение", 0.5, 5.0, float(cfg.risk_reward), 0.1)
    with c2:
        max_qty = st.number_input("Макс. позиция (в базовом активе)", 0.001, 10_000.0,
                                  float(cfg.max_qty_manual), 0.001)
        limit_qty_enabled = st.checkbox("Ограничивать максимальную позицию",
                                        value=bool(getattr(cfg, "limit_qty_enabled", True)))
    with c3:
        taker_fee = st.number_input("Комиссия taker (десятичная)", 0.0, 0.01, float(cfg.taker_fee_rate), 0.00005)
        price_for_logic = st.selectbox("Источник цены для логики", ["last", "mark"],
                                       index=0 if cfg.price_for_logic == "last" else 1)
    with c4:
        use_take_profit = st.checkbox("Использовать Take Profit", value=bool(cfg.use_take_profit))
        intrabar_tf = st.text_input("Интрабар TF (\"1\",\"3\",\"5\")", value=str(getattr(cfg, "intrabar_tf", "1")))

    # Интрабар: входы и трейлинг/обновления
    cIntra1, cIntra2 = st.columns(2)
    with cIntra1:
        use_intrabar = st.checkbox("Включить интрабар-трейл/обновления", value=bool(getattr(cfg, "use_intrabar", True)))
    with cIntra2:
        use_intrabar_entries = st.checkbox("Включить интрабар-входы", value=bool(getattr(cfg, "use_intrabar_entries", False)))

    # ───────────────────────────────  ФИЛЬТРЫ SFP  ───────────────────────────────
    st.subheader("🛡️ Фильтры SFP")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        sfp_len = st.number_input("SFP Length", 1, 10, int(getattr(cfg, "sfp_len", 2)))
    with s2:
        use_sfp_quality = st.checkbox("Фильтр качества SFP (wick+close-back)",
                                      value=bool(getattr(cfg, "use_sfp_quality", True)))
    with s3:
        wick_min_ticks = st.number_input("Мин. глубина фитиля (в тиках)", 0, 100,
                                         int(getattr(cfg, "wick_min_ticks", 7)))
    with s4:
        close_back_pct = st.number_input("Close-back (0.0 … 1.0)", 0.0, 1.0,
                                         float(getattr(cfg, "close_back_pct", 1.0)), 0.05)

    # ───────────────────────────── Stop-Loss Zone (Pine) ─────────────────────────
    st.subheader("📌 Stop-Loss Zone (Pine-like)")
    z1, z2, z3, z4, z5 = st.columns(5)
    with z1:
        use_swing_sl = st.checkbox("SL от свинга (pivot)", value=bool(getattr(cfg, "use_swing_sl", True)))
    with z2:
        use_prev_candle_sl = st.checkbox("SL от свечи [1]", value=bool(getattr(cfg, "use_prev_candle_sl", False)))
    with z3:
        sl_buf_ticks = st.number_input("Буфер к SL (ticks)", 0, 1000, int(getattr(cfg, "sl_buf_ticks", 40)))
    with z4:
        use_atr_buffer = st.checkbox("ATR-буфер", value=bool(getattr(cfg, "use_atr_buffer", False)))
    with z5:
        atr_mult = st.number_input("ATR Mult", 0.0, 10.0, float(getattr(cfg, "atr_mult", 0.0)), 0.1)

    tps = st.selectbox("Триггер стопа/тейка (биржа)", ["mark", "last"],
                       index=0 if str(getattr(cfg, "trigger_price_source", "mark")).lower() == "mark" else 1,
                       help="По какой цене срабатывает SL/TP на бирже.")

    # ───────────────────────────────  SMART TRAILING  ─────────────────────────────
    st.subheader("🎯 Smart Trailing")
    tr1, tr2, tr3, tr4 = st.columns(4)
    with tr1:
        enable_smart_trail = st.checkbox("Включить Smart Trailing", value=bool(getattr(cfg, "enable_smart_trail", True)))
    with tr2:
        use_arm_after_rr = st.checkbox("Арминг трейла после достижения RR",
                                       value=bool(getattr(cfg, "use_arm_after_rr", True)))
    with tr3:
        arm_rr = st.number_input("RR для арминга (R)", 0.1, 5.0, float(getattr(cfg, "arm_rr", 0.5)), 0.1)
    with tr4:
        arm_rr_basis = st.selectbox("База RR для арминга", ["extremum", "last"],
                                    index=0 if getattr(cfg, "arm_rr_basis", "extremum") == "extremum" else 1)

    tr5, tr6 = st.columns(2)
    with tr5:
        trailing_perc = st.number_input("Процент трейлинга (%)", 0.0, 5.0,
                                        float(getattr(cfg, "trailing_perc", 0.5)), 0.1)
    with tr6:
        trailing_offset_perc = st.number_input("Offset трейлинга (%)", 0.0, 5.0,
                                               float(getattr(cfg, "trailing_offset_perc", 0.4)), 0.1)

    # Баровый трейл / прочее
    st.subheader("📦 Баровый трейлинг / прочее")
    b1, b2 = st.columns(2)
    with b1:
        use_bar_trail = st.checkbox("Баровый трейлинг (lowest/highest N закрытых баров)",
                                    value=bool(getattr(cfg, "use_bar_trail", True)))
    with b2:
        trail_lookback = st.number_input("Trail lookback bars", 1, 300, int(getattr(cfg, "trail_lookback", 50)))
    trail_buf_ticks = st.number_input("Trail buffer (ticks)", 0, 500, int(getattr(cfg, "trail_buf_ticks", 40)))

    # ─────────────────────────────  КНОПКИ ДЕЙСТВИЯ  ─────────────────────────────
    st.markdown("---")
    cleft, cmid, cright = st.columns([1, 2, 1])

    with cmid:
        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Сохранить настройки", use_container_width=True, type="primary"):
                # записываем всё обратно в cfg и сохраняем
                cfg.risk_pct = float(risk_pct)
                cfg.risk_reward = float(risk_reward)
                cfg.max_qty_manual = float(max_qty)
                cfg.limit_qty_enabled = bool(limit_qty_enabled)

                cfg.taker_fee_rate = float(taker_fee)
                cfg.price_for_logic = str(price_for_logic).lower()

                cfg.use_take_profit = bool(use_take_profit)
                cfg.intrabar_tf = str(intrabar_tf)
                cfg.use_intrabar = bool(use_intrabar)
                cfg.use_intrabar_entries = bool(use_intrabar_entries)

                cfg.sfp_len = int(sfp_len)
                cfg.use_sfp_quality = bool(use_sfp_quality)
                cfg.wick_min_ticks = int(wick_min_ticks)
                cfg.close_back_pct = float(close_back_pct)

                cfg.use_swing_sl = bool(use_swing_sl)
                cfg.use_prev_candle_sl = bool(use_prev_candle_sl)
                cfg.sl_buf_ticks = int(sl_buf_ticks)
                cfg.use_atr_buffer = bool(use_atr_buffer)
                cfg.atr_mult = float(atr_mult)
                cfg.trigger_price_source = str(tps).lower()

                cfg.enable_smart_trail = bool(enable_smart_trail)
                cfg.use_arm_after_rr = bool(use_arm_after_rr)
                cfg.arm_rr = float(arm_rr)
                cfg.arm_rr_basis = str(arm_rr_basis)

                cfg.trailing_perc = float(trailing_perc)
                cfg.trailing_offset_perc = float(trailing_offset_perc)
                cfg.trailing_offset = float(trailing_offset_perc)

                cfg.use_bar_trail = bool(use_bar_trail)
                cfg.trail_lookback = int(trail_lookback)
                cfg.trail_buf_ticks = int(trail_buf_ticks)

                if cfg.validate():
                    cfg.save_config()
                    st.success("✅ Настройки сохранены.")
                else:
                    st.error("❌ Валидация не пройдена. Проверь значения.")

        with colB:
            if st.button("⭐ Применить пресет TradingView", use_container_width=True):
                # Эталонный набор из вашего списка
                cfg.use_intrabar_entries = False
                cfg.use_swing_sl = True
                cfg.use_prev_candle_sl = False
                cfg.sl_buf_ticks = 40
                cfg.use_atr_buffer = False
                cfg.atr_mult = 0.0
                cfg.trigger_price_source = "mark"

                cfg.use_sfp_quality = True
                cfg.wick_min_ticks = 7
                cfg.close_back_pct = 1.0

                cfg.use_take_profit = True
                cfg.risk_reward = 1.3

                cfg.use_arm_after_rr = True
                cfg.arm_rr = 0.5
                cfg.arm_rr_basis = "extremum"

                cfg.enable_smart_trail = True
                cfg.trailing_perc = 0.5
                cfg.trailing_offset_perc = 0.4
                cfg.trailing_offset = 0.4

                if cfg.validate():
                    cfg.save_config()
                    st.success("✅ Пресет применён и сохранён.")
                else:
                    st.error("❌ Пресет не прошёл валидацию (проверь значения).")

    # ─────────────────────────────  ПРОСМОТР ТЕКУЩЕГО  ────────────────────────────
    st.markdown("---")
    st.subheader("📋 Текущая конфигурация")
    with st.expander("Показать все параметры (config.json)"):
        st.json(cfg.to_dict())

if __name__ == "__main__":
    main()
    
