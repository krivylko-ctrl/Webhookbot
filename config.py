
from __future__ import annotations

from typing import Dict, Any
import json
import os

# -------------------- ENV helpers --------------------

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""

BYBIT_API_KEY       = env("BYBIT_API_KEY", "")
BYBIT_API_SECRET    = env("BYBIT_API_SECRET", "")
# Мы работаем с деривативами. Допускаем только допустимые категории,
# но по умолчанию фиксируемся на 'linear' (USDT-M фьючерсы).
BYBIT_ACCOUNT_TYPE  = env("BYBIT_ACCOUNT_TYPE", "linear").lower()
SYMBOL              = env("SYMBOL", "ETHUSDT").upper()
INTERVALS           = [i.strip() for i in env("INTERVALS", "1,15,60").split(",") if i.strip()]

def must_have():
    """Проверка критичных переменных окружения (актуально для live)."""
    missing = []
    if BYBIT_ACCOUNT_TYPE not in ("linear", "inverse", "option"):
        missing.append(f"BYBIT_ACCOUNT_TYPE (got '{BYBIT_ACCOUNT_TYPE}')")
    if missing:
        raise RuntimeError("Missing/invalid env: " + ", ".join(missing))


# =====================================================
#                    CONFIG CLASS
# =====================================================

class Config:
    """Конфигурация стратегии KWIN (эквивалент TV inputs, Pine v5)"""

    def __init__(self):
        # === ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
        self.symbol       = SYMBOL
        self.market_type  = BYBIT_ACCOUNT_TYPE or "linear"
        self.interval     = "15"    # базовый ТФ для сигналов

        # Риск/TP
        self.risk_reward  = float(env("RISK_REWARD", "1.3"))
        self.sfp_len      = 2
        self.risk_pct     = float(env("RISK_PCT", "3.0"))

        # === Управление TP ===
        # True — в бэктесте учитываем TP-выходы; False — игнорируем TP
        self.use_take_profit = env("USE_TAKE_PROFIT", "false").lower() not in ("0", "false", "no")

        # === SMART TRAILING ===
        self.enable_smart_trail   = env("ENABLE_SMART_TRAIL", "true").lower() not in ("0", "false", "no")
        self.smart_trail_mode     = env("SMART_TRAIL_MODE", "pine").lower()  # "pine"|"legacy" (на будущее)
        self.trailing_perc        = float(env("TRAILING_PERC", "0.5"))         # %
        self.trailing_offset_perc = float(env("TRAILING_OFFSET_PERC", "0.3"))  # %
        self.trailing_offset      = self.trailing_offset_perc                  # alias для совместимости

        # ARM (вооружение трейла после достижения RR)
        self.use_arm_after_rr = env("USE_ARM_AFTER_RR", "true").lower() not in ("0", "false", "no")
        self.arm_rr           = max(0.1, float(env("ARM_RR", "0.5")))          # в R, минимально 0.1
        self.arm_rr_basis     = env("ARM_RR_BASIS", "last").lower()        # "extremum"|"last"

        # Источники цены (по умолчанию триггеры по mark)
        self.price_for_logic      = env("PRICE_FOR_LOGIC", "last").lower()     # "last"|"mark"
        self.trigger_price_source = env("TRIGGER_PRICE_SOURCE", "mark").lower()# "last"|"mark"

        # === ЗОНАЛЬНЫЙ СТОП ===
# переключатели базы SL: свинговый pivot и/или экстремум SFP-свечи [0]
        self.use_swing_sl       = env("USE_SWING_SL", "false").lower() not in ("0","false","no")
        self.use_sfp_candle_sl = env("USE_SFP_CANDLE_SL", "false").lower() not in ("0", "false", "no")
        self.use_prev_candle_sl = env("USE_PREV_CANDLE_SL", "true").lower() not in ("0","false","no")
        self.sl_buf_ticks       = int(env("SL_BUF_TICKS", "0"))  # если нужен отступ — увеличь
        self.use_atr_buffer     = env("USE_ATR_BUFFER", "false").lower() not in ("0","false","no")
        self.atr_mult           = float(env("ATR_MULT", "0.0"))

# === ИНТРАБАР ===
        self.use_intrabar         = env("USE_INTRABAR", "true").lower() not in ("0","false","no")   # 1m только для трейлинга
        self.use_intrabar_entries = env("USE_INTRABAR_ENTRIES", "false").lower() not in ("0","false","no")  # ⛔ входы по M1
        self.intrabar_tf          = env("INTRABAR_TF", "1")
        self.intrabar_pull_limit  = int(env("INTRABAR_PULL_LIMIT", "1500"))
        self.smooth_intrabar      = env("SMOOTH_INTRABAR", "true").lower() not in ("0","false","no")
        self.intrabar_steps       = int(env("INTRABAR_STEPS", "6"))

        # === ОГРАНИЧЕНИЯ ПОЗИЦИИ ===
        self.limit_qty_enabled = env("LIMIT_QTY_ENABLED", "true").lower() not in ("0","false","no")
        self.max_qty_manual    = float(env("MAX_QTY_MANUAL", "50.0"))

        # === ФИЛЬТРЫ SFP ===
        self.use_sfp_quality = env("USE_SFP_QUALITY", "true").lower() not in ("0","false","no")
        self.wick_min_ticks  = int(env("WICK_MIN_TICKS", "7"))
        self.close_back_pct  = float(env("CLOSE_BACK_PCT", "1.0"))  # [0..1]

        # === БЭКТЕСТ/ЭФФЕКТЫ ИСПОЛНЕНИЯ ===
        self.period_choice = env("PERIOD_CHOICE", "30")  # "30"|"60"|"180"
        self.days_back     = int(env("DAYS_BACK", "30"))
        self.slippage_pct  = float(env("SLIPPAGE_PCT", "0.0"))
        self.latency_ms    = int(env("LATENCY_MS", "0"))

        # === МАРКЕТ ===
        self.taker_fee_rate = float(env("TAKER_FEE_RATE", "0.00055"))
        self.min_net_profit = float(env("MIN_NET_PROFIT", "1.2"))
        self.min_order_qty  = float(env("MIN_ORDER_QTY", "0.01"))
        self.qty_step       = float(env("QTY_STEP", "0.01"))
        self.tick_size      = float(env("TICK_SIZE", "0.01"))

        # Совместимость со старой логикой bar-trail (активно не меняем механику)
        self.use_bar_trail   = env("USE_BAR_TRAIL", "true").lower() not in ("0","false","no")
        self.trail_lookback  = int(env("TRAIL_LOOKBACK", "50"))
        self.trail_buf_ticks = int(env("TRAIL_BUF_TICKS", "40"))

        # Нормализация и загрузка config.json (если есть)
        self._update_days_back()
        self._normalize_derived()
        self.load_config()
        self._update_days_back()
        self._normalize_derived()

    # ---------- normalizers ----------

    def _update_days_back(self):
        pc = str(self.period_choice)
        if pc == "30":
            self.days_back = 30
        elif pc == "60":
            self.days_back = 60
        elif pc == "180":
            self.days_back = 180
        else:
            try:
                self.days_back = int(self.days_back or 30)
            except Exception:
                self.days_back = 30

    def _normalize_derived(self):
        # close_back_pct clamp -> [0..1]
        try:
            if self.close_back_pct is None:
                self.close_back_pct = 1.0
            if self.close_back_pct > 1.0:
                self.close_back_pct = float(self.close_back_pct) / 100.0
            if self.close_back_pct < 0.0:
                self.close_back_pct = 0.0
        except Exception:
            self.close_back_pct = 1.0

        # trailing_offset_perc sync c alias
        try:
            if self.trailing_offset is not None:
                self.trailing_offset_perc = float(self.trailing_offset)
        except Exception:
            pass

        # здравые шаги для ETH/BTC
        sym = (self.symbol or "").upper()
        if sym in ("ETHUSDT", "BTCUSDT"):
            # эти шаги потом будут уточнены через API (instrument info),
            # но для локального UI дадим разумные значения.
            self.qty_step = max(self.qty_step, 0.001)
            self.min_order_qty = max(self.min_order_qty, 0.001)
            self.tick_size = max(self.tick_size, 0.01)

        # строковые поля + вайтлисты
        self.price_for_logic = (self.price_for_logic or "last").lower()
        if self.price_for_logic not in ("last", "mark"):
            self.price_for_logic = "last"

        self.trigger_price_source = (self.trigger_price_source or "mark").lower()
        if self.trigger_price_source not in ("last", "mark"):
            self.trigger_price_source = "mark"

        self.arm_rr_basis = (self.arm_rr_basis or "extremum").lower()
        if self.arm_rr_basis not in ("extremum", "last"):
            self.arm_rr_basis = "extremum"

        # новые числовые — приводим к корректным диапазонам
        try:
            self.sl_buf_ticks = max(0, int(self.sl_buf_ticks))
        except Exception:
            self.sl_buf_ticks = 40

        # числа
        try:
            self.trailing_perc = max(0.0, float(self.trailing_perc))
        except Exception:
            self.trailing_perc = 0.5
        try:
            self.trailing_offset_perc = max(0.0, float(self.trailing_offset_perc))
        except Exception:
            self.trailing_offset_perc = 0.4

        # строки
        try:
            self.interval = str(self.interval)
        except Exception:
            self.interval = "15"
        try:
            self.intrabar_tf = str(self.intrabar_tf)
        except Exception:
            self.intrabar_tf = "1"

    # ---------- load/save ----------

    def load_config(self, filename: str = "config.json"):
        try:
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_config_data(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self, filename: str = "config.json"):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _apply_config_data(self, data: Dict[str, Any]):
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._update_days_back()
        self._normalize_derived()

    def update_from_dict(self, data: Dict[str, Any]):
        self._apply_config_data(data)
        self.save_config()

    # ---------- export ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            # базовые
            "symbol": self.symbol,
            "market_type": self.market_type,
            "interval": str(self.interval),
            "risk_reward": self.risk_reward,
            "sfp_len": self.sfp_len,
            "risk_pct": self.risk_pct,

            # TP
            "use_take_profit": self.use_take_profit,

            # Smart Trail
            "enable_smart_trail": self.enable_smart_trail,
            "smart_trail_mode": self.smart_trail_mode,
            "trailing_perc": self.trailing_perc,
            "trailing_offset_perc": self.trailing_offset_perc,
            "trailing_offset": self.trailing_offset,   # alias
            "use_arm_after_rr": self.use_arm_after_rr,
            "arm_rr": self.arm_rr,
            "arm_rr_basis": self.arm_rr_basis,

            # источники цены
            "price_for_logic": self.price_for_logic,
            "trigger_price_source": self.trigger_price_source,

            # зональный SL
            "use_prev_candle_sl": self.use_prev_candle_sl,
            "use_intrabar_entries": self.use_intrabar_entries,
            "use_swing_sl": self.use_swing_sl,
            "use_sfp_candle_sl": self.use_sfp_candle_sl,
            "use_sfp_candle_sl": self.use_sfp_candle_sl,
            "sl_buf_ticks": self.sl_buf_ticks,

            # интрабар
            "use_intrabar": self.use_intrabar,
            "intrabar_tf": str(self.intrabar_tf),
            "intrabar_pull_limit": self.intrabar_pull_limit,
            "smooth_intrabar": self.smooth_intrabar,
            "intrabar_steps": self.intrabar_steps,

            # фильтры
            "use_sfp_quality": self.use_sfp_quality,
            "wick_min_ticks": self.wick_min_ticks,
            "close_back_pct": self.close_back_pct,

            # бэктест/исполнение
            "period_choice": self.period_choice,
            "days_back": self.days_back,
            "slippage_pct": self.slippage_pct,
            "latency_ms": self.latency_ms,

            # маркет/ограничения
            "limit_qty_enabled": self.limit_qty_enabled,
            "max_qty_manual": self.max_qty_manual,
            "taker_fee_rate": self.taker_fee_rate,
            "min_net_profit": self.min_net_profit,
            "min_order_qty": self.min_order_qty,
            "qty_step": self.qty_step,
            "tick_size": self.tick_size,

            # совместимость (bar-trail)
            "use_bar_trail": self.use_bar_trail,
            "trail_lookback": self.trail_lookback,
            "trail_buf_ticks": self.trail_buf_ticks,
        }

    def validate(self) -> bool:
        try:
            if self.risk_reward <= 0:
                raise ValueError("risk_reward must be > 0")
            if not (0 < self.risk_pct <= 100):
                raise ValueError("risk_pct must be in (0..100]")
            if self.sfp_len < 1:
                raise ValueError("sfp_len >= 1")
            if self.max_qty_manual <= 0:
                raise ValueError("max_qty_manual must be > 0")
            if not (0.0 <= float(self.close_back_pct) <= 1.0):
                raise ValueError("close_back_pct must be in [0..1]")
            if self.arm_rr_basis not in ("extremum", "last"):
                raise ValueError("arm_rr_basis invalid")
            if self.price_for_logic not in ("last", "mark"):
                raise ValueError("price_for_logic invalid")
            if self.trigger_price_source not in ("last", "mark"):
                raise ValueError("trigger_price_source invalid")
            if self.sl_buf_ticks < 0:
                raise ValueError("sl_buf_ticks must be >= 0")
            return True
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
