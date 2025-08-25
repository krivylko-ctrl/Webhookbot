from __future__ import annotations

from typing import Dict, Any, Optional
import json
import os
from datetime import datetime, timezone

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
    """Конфигурация стратегии KWIN (эквивалент TV inputs, Pine v5).
       Содержит настройки для dual-SFP: Lux SFP + Classic 15m SFP (OR),
       15m gate, fee-filter(1R), once-per-swing, barPriority, dir-lock.
    """

    def __init__(self):
        # === ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
        self.symbol       = SYMBOL
        self.market_type  = BYBIT_ACCOUNT_TYPE or "linear"
        self.interval     = "15"    # базовый ТФ для сигналов

        # Риск/TP
        self.risk_reward  = float(env("RISK_REWARD", "1.3"))
        self.sfp_len      = 2                           # длина свинга для Classic 15m SFP
        self.risk_pct     = float(env("RISK_PCT", "3.0"))

        # === Управление TP ===
        self.use_take_profit = env("USE_TAKE_PROFIT", "false").lower() not in ("0", "false", "no")

        # === SMART TRAILING ===
        self.enable_smart_trail   = env("ENABLE_SMART_TRAIL", "true").lower() not in ("0", "false", "no")
        self.smart_trail_mode     = env("SMART_TRAIL_MODE", "pine").lower()  # "pine"|"legacy" (на будущее)
        self.trailing_perc        = float(env("TRAILING_PERC", "0.5"))         # %
        self.trailing_offset_perc = float(env("TRAILING_OFFSET_PERC", "0.4"))  # % (синхроним с alias ниже)
        self.trailing_offset      = self.trailing_offset_perc                  # alias для совместимости

        # ARM (вооружение трейла после достижения RR)
        self.use_arm_after_rr = env("USE_ARM_AFTER_RR", "true").lower() not in ("0", "false", "no")
        self.arm_rr           = max(0.1, float(env("ARM_RR", "0.5")))          # в R, минимально 0.1
        self.arm_rr_basis     = env("ARM_RR_BASIS", "extremum").lower()        # "extremum"|"last" (в стратегии база всё равно = close)

        # Источники цены — ДЕФОЛТ "close" для Pine-паритета!
        self.price_for_logic      = env("PRICE_FOR_LOGIC", "close").lower()     # "close"|"last"|"mark"
        self.trigger_price_source = env("TRIGGER_PRICE_SOURCE", "mark").lower() # "last"|"mark"

        # === ЗОНАЛЬНЫЙ СТОП ===
        self.use_swing_sl        = env("USE_SWING_SL", "false").lower() not in ("0","false","no")
        self.use_sfp_candle_sl   = env("USE_SFP_CANDLE_SL", "false").lower() not in ("0", "false", "no")
        self.use_prev_candle_sl  = env("USE_PREV_CANDLE_SL", "true").lower() not in ("0","false","no")
        self.sl_buf_ticks        = int(env("SL_BUF_TICKS", "0"))
        self.use_atr_buffer      = env("USE_ATR_BUFFER", "false").lower() not in ("0","false","no")
        self.atr_mult            = float(env("ATR_MULT", "0.0"))

        # === LUX SFP ===
        self.lux_mode                   = env("LUX_MODE", "true").lower() not in ("0","false","no")
        self.lux_swings                 = int(env("LUX_SWINGS", "2"))
        # В стратегии сейчас принудительно "none". Ставим такой же дефолт, чтобы не вводить в заблуждение.
        self.lux_volume_validation      = env("LUX_VOLUME_VALIDATION", "none").lower()  # "outside_gt" | "outside_lt" | "none"
        self.lux_volume_threshold_pct   = float(env("LUX_VOLUME_THRESHOLD_PCT", "10.0"))
        self.lux_auto                   = env("LUX_AUTO", "false").lower() not in ("0","false","no")
        self.lux_mlt                    = int(env("LUX_MLT", "10"))
        self.lux_ltf                    = env("LUX_LTF", "1")
        self.lux_premium                = env("LUX_PREMIUM", "false").lower() not in ("0","false","no")
        self.lux_expire_bars            = int(env("LUX_EXPIRE_BARS", "500"))

        # === ИНТРАБАР ===
        self.use_intrabar         = env("USE_INTRABAR", "true").lower() not in ("0","false","no")   # 1m для сигналов/трейлинга
        self.use_intrabar_entries = env("USE_INTRABAR_ENTRIES", "false").lower() not in ("0","false","no")  # сейчас стратегия смотрит только use_intrabar
        self.intrabar_tf          = env("INTRABAR_TF", "1")
        self.intrabar_pull_limit  = int(env("INTRABAR_PULL_LIMIT", "1500"))
        self.smooth_intrabar      = env("SMOOTH_INTRABAR", "true").lower() not in ("0","false","no")
        self.intrabar_steps       = int(env("INTRABAR_STEPS", "6"))

        # === ОГРАНИЧЕНИЯ ПОЗИЦИИ ===
        self.limit_qty_enabled = env("LIMIT_QTY_ENABLED", "true").lower() not in ("0","false","no")
        self.max_qty_manual    = float(env("MAX_QTY_MANUAL", "50.0"))

        # === ФИЛЬТРЫ SFP (для «классики», Lux может игнорить) ===
        self.use_sfp_quality = env("USE_SFP_QUALITY", "true").lower() not in ("0","false","no")
        self.wick_min_ticks  = int(env("WICK_MIN_TICKS", "7"))
        self.close_back_pct  = float(env("CLOSE_BACK_PCT", "1.0"))  # [0..1]

        # === БЭКТЕСТ/ОКНО ===
        self.period_choice = env("PERIOD_CHOICE", "30")  # "30"|"60"|"180"
        self.days_back     = int(env("DAYS_BACK", "30"))
        # фиксированный Cycle Start Time (UTC) — совместим с Pine:
        # приоритет: START_TIME_MS (unix ms) > START_TIME_ISO (YYYY-MM-DD HH:MM[:SS][Z])
        self.start_time_ms: Optional[int] = self._parse_start_time_ms(
            env("START_TIME_MS", ""), env("START_TIME_ISO", "")
        )

        # === Исполнение (на будущее) ===
        self.slippage_pct  = float(env("SLIPPAGE_PCT", "0.0"))
        self.latency_ms    = int(env("LATENCY_MS", "0"))

        # === МАРКЕТ ===
        self.taker_fee_rate = float(env("TAKER_FEE_RATE", "0.00055"))
        self.min_net_profit = float(env("MIN_NET_PROFIT", "2.0"))
        self.min_order_qty  = float(env("MIN_ORDER_QTY", "0.01"))
        self.qty_step       = float(env("QTY_STEP", "0.01"))
        self.tick_size      = float(env("TICK_SIZE", "0.01"))

        # === DUAL-SFP ДОП. НАСТРОЙКИ ===
        self.use_fee_filter     = env("USE_FEE_FILTER", "true").lower() not in ("0","false","no")
        self.use_once_per_swing = env("USE_ONCE_PER_SWING", "true").lower() not in ("0","false","no")
        # "Prefer Bear" | "Prefer Bull" | "Skip"
        self.bar_priority       = self._normalize_bar_priority(env("BAR_PRIORITY", "Skip"))
        self.use_dir_lock       = env("USE_DIR_LOCK", "true").lower() not in ("0","false","no")

        # Совместимость со старой логикой bar-trail
        self.use_bar_trail   = env("USE_BAR_TRAIL", "true").lower() not in ("0","false","no")
        self.trail_lookback  = int(env("TRAIL_LOOKBACK", "50"))
        self.trail_buf_ticks = int(env("TRAIL_BUF_TICKS", "40"))

        # === ДОП. ФЛАГИ ДЛЯ 1:1 С KWINStrategy ===
        self.initial_capital      = float(env("INITIAL_CAPITAL", "300.0"))      # используется если equity_source='local'
        self.equity_source        = env("EQUITY_SOURCE", "local").lower()       # 'local' | 'wallet'
        self.equity_mode          = env("EQUITY_MODE", "wallet_plus_upnl").lower()  # 'wallet' | 'wallet_plus_upnl'
        self.wallet_includes_upnl = env("WALLET_INCLUDES_UPNL", "false").lower() not in ("0","false","no")
        self.quantize_initial_sl  = env("QUANTIZE_INITIAL_SL", "false").lower() not in ("0","false","no")
        self.force_pine_min_qty   = env("FORCE_PINE_MIN_QTY", "false").lower() not in ("0","false","no")

        # Нормализация и загрузка config.json (если есть)
        self._update_days_back()
        self._normalize_derived()
        self.load_config()
        self._update_days_back()
        self._normalize_derived()

    # ---------- helpers ----------

    def _parse_start_time_ms(self, v_ms: str, v_iso: str) -> Optional[int]:
        """Парсинг START_TIME_MS (unix ms) либо START_TIME_ISO ('YYYY-MM-DD HH:MM[:SS][Z]')."""
        try:
            if v_ms and v_ms.strip().isdigit():
                return int(v_ms.strip())
        except Exception:
            pass
        v = (v_iso or "").strip()
        if not v:
            return None
        try:
            vv = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(vv)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)  # трактуем как UTC
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def _normalize_bar_priority(self, raw: str) -> str:
        r = (raw or "").strip().lower().replace("_", " ")
        if r in ("prefer bear", "bear"):
            return "Prefer Bear"
        if r in ("prefer bull", "bull"):
            return "Prefer Bull"
        return "Skip"

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
            self.qty_step = max(self.qty_step, 0.001)
            self.min_order_qty = max(self.min_order_qty, 0.001)
            self.tick_size = max(self.tick_size, 0.01)

        # строковые поля + вайтлисты
        self.price_for_logic = (self.price_for_logic or "close").lower()
        if self.price_for_logic not in ("close", "last", "mark"):
            self.price_for_logic = "close"

        self.trigger_price_source = (self.trigger_price_source or "mark").lower()
        if self.trigger_price_source not in ("last", "mark"):
            self.trigger_price_source = "mark"

        self.arm_rr_basis = (self.arm_rr_basis or "extremum").lower()
        if self.arm_rr_basis not in ("extremum", "last"):
            self.arm_rr_basis = "extremum"

        # числа
        try:
            self.sl_buf_ticks = max(0, int(self.sl_buf_ticks))
        except Exception:
            self.sl_buf_ticks = 0

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

        # ---- Lux поля ----
        self.lux_volume_validation = (self.lux_volume_validation or "none").lower()
        if self.lux_volume_validation not in ("outside_gt", "outside_lt", "none"):
            self.lux_volume_validation = "none"

        try:
            self.lux_volume_threshold_pct = max(0.0, min(100.0, float(self.lux_volume_threshold_pct)))
        except Exception:
            self.lux_volume_threshold_pct = 10.0

        try:
            self.lux_swings = max(1, int(self.lux_swings))
        except Exception:
            self.lux_swings = 2

        try:
            self.lux_expire_bars = max(1, int(self.lux_expire_bars))
        except Exception:
            self.lux_expire_bars = 500

        try:
            self.lux_ltf = str(self.lux_ltf or "1")
        except Exception:
            self.lux_ltf = "1"

        # ---- Dual-SFP поля ----
        self.bar_priority = self._normalize_bar_priority(self.bar_priority)

        # ---- Equity / исполнение ----
        self.equity_source = (self.equity_source or "local").lower()
        if self.equity_source not in ("local", "wallet"):
            self.equity_source = "local"
        self.equity_mode = (self.equity_mode or "wallet_plus_upnl").lower()
        if self.equity_mode not in ("wallet", "wallet_plus_upnl"):
            self.equity_mode = "wallet_plus_upnl"

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
            "use_swing_sl": self.use_swing_sl,
            "use_sfp_candle_sl": self.use_sfp_candle_sl,
            "sl_buf_ticks": self.sl_buf_ticks,
            "use_atr_buffer": self.use_atr_buffer,
            "atr_mult": self.atr_mult,

            # Lux SFP
            "lux_mode": self.lux_mode,
            "lux_swings": self.lux_swings,
            "lux_volume_validation": self.lux_volume_validation,   # "outside_gt" | "outside_lt" | "none"
            "lux_volume_threshold_pct": self.lux_volume_threshold_pct,
            "lux_auto": self.lux_auto,
            "lux_mlt": self.lux_mlt,
            "lux_ltf": self.lux_ltf,
            "lux_premium": self.lux_premium,
            "lux_expire_bars": self.lux_expire_bars,

            # интрабар
            "use_intrabar": self.use_intrabar,
            "use_intrabar_entries": self.use_intrabar_entries,
            "intrabar_tf": str(self.intrabar_tf),
            "intrabar_pull_limit": self.intrabar_pull_limit,
            "smooth_intrabar": self.smooth_intrabar,
            "intrabar_steps": self.intrabar_steps,

            # фильтры (для старой логики SFP)
            "use_sfp_quality": self.use_sfp_quality,
            "wick_min_ticks": self.wick_min_ticks,
            "close_back_pct": self.close_back_pct,

            # бэктест/исполнение
            "period_choice": self.period_choice,
            "days_back": self.days_back,
            "start_time_ms": self.start_time_ms,
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

            # dual-sfp
            "use_fee_filter": self.use_fee_filter,
            "use_once_per_swing": self.use_once_per_swing,
            "bar_priority": self.bar_priority,
            "use_dir_lock": self.use_dir_lock,

            # совместимость (bar-trail)
            "use_bar_trail": self.use_bar_trail,
            "trail_lookback": self.trail_lookback,
            "trail_buf_ticks": self.trail_buf_ticks,

            # --- KWIN parity / wallet ---
            "initial_capital": self.initial_capital,
            "equity_source": self.equity_source,
            "equity_mode": self.equity_mode,
            "wallet_includes_upnl": self.wallet_includes_upnl,
            "quantize_initial_sl": self.quantize_initial_sl,
            "force_pine_min_qty": self.force_pine_min_qty,
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
            if self.price_for_logic not in ("close", "last", "mark"):
                raise ValueError("price_for_logic invalid")
            if self.trigger_price_source not in ("last", "mark"):
                raise ValueError("trigger_price_source invalid")
            if self.sl_buf_ticks < 0:
                raise ValueError("sl_buf_ticks must be >= 0")

            # Lux checks
            if self.lux_volume_validation not in ("outside_gt", "outside_lt", "none"):
                raise ValueError("lux_volume_validation invalid")
            if not (0.0 <= self.lux_volume_threshold_pct <= 100.0):
                raise ValueError("lux_volume_threshold_pct must be in [0..100]")
            if self.lux_swings < 1:
                raise ValueError("lux_swings must be >= 1")
            if self.lux_expire_bars < 1:
                raise ValueError("lux_expire_bars must be >= 1")

            # Market
            if self.taker_fee_rate < 0:
                raise ValueError("taker_fee_rate must be >= 0")
            if self.min_net_profit < 0:
                raise ValueError("min_net_profit must be >= 0")

            # Dual-SFP
            if self.bar_priority not in ("Prefer Bear", "Prefer Bull", "Skip"):
                raise ValueError("bar_priority must be one of: Prefer Bear | Prefer Bull | Skip")
            if self.start_time_ms is not None and int(self.start_time_ms) < 0:
                raise ValueError("start_time_ms must be >= 0 (unix ms)")

            # Equity settings
            if self.equity_source not in ("local", "wallet"):
                raise ValueError("equity_source invalid")
            if self.equity_mode not in ("wallet", "wallet_plus_upnl"):
                raise ValueError("equity_mode invalid")
            if self.initial_capital <= 0:
                raise ValueError("initial_capital must be > 0")

            return True
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
