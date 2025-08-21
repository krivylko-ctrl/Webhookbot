from typing import Dict, Any
import json
import os

# -------------------- ENV helpers --------------------

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""

BYBIT_API_KEY      = env("BYBIT_API_KEY", "")
BYBIT_API_SECRET   = env("BYBIT_API_SECRET", "")
BYBIT_ACCOUNT_TYPE = env("BYBIT_ACCOUNT_TYPE", "linear").lower()
SYMBOL             = env("SYMBOL", "ETHUSDT").upper()
INTERVALS          = [i.strip() for i in env("INTERVALS", "1,15,60").split(",") if i.strip()]

def must_have():
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
        self.interval     = "15"    # базовой ТФ для сигналов

        # Риск/TP
        self.risk_reward  = float(env("RISK_REWARD", "1.3"))
        self.sfp_len      = 2
        self.risk_pct     = float(env("RISK_PCT", "3.0"))

        # === Управление TP ===
        self.use_take_profit = env("USE_TAKE_PROFIT", "false").lower() not in ("0","false","no")

        # === SMART TRAILING ===
        self.enable_smart_trail   = env("ENABLE_SMART_TRAIL", "true").lower() not in ("0","false","no")
        self.smart_trail_mode     = env("SMART_TRAIL_MODE", "pine").lower()  # "pine"|"legacy"
        self.trailing_perc        = float(env("TRAILING_PERC", "0.5"))       # %
        self.trailing_offset_perc = float(env("TRAILING_OFFSET_PERC", "0.4"))# %
        self.trailing_offset      = self.trailing_offset_perc                # alias

        # ARM (вооружение трейла после достижения RR)
        self.use_arm_after_rr = env("USE_ARM_AFTER_RR", "true").lower() not in ("0","false","no")
        self.arm_rr           = max(0.1, float(env("ARM_RR", "0.5")))  # ≥0.1
        self.arm_rr_basis     = (env("ARM_RR_BASIS", "extremum")).lower()

        # Источники цены
        self.price_for_logic      = "last"
        self.trigger_price_source = "last"

        # === ИНТРАБАР ===
        self.use_intrabar        = True
        self.intrabar_tf         = "1"
        self.intrabar_pull_limit = 1500
        self.smooth_intrabar     = True
        self.intrabar_steps      = 6

        # === ОГРАНИЧЕНИЯ ПОЗИЦИИ ===
        self.limit_qty_enabled = True
        self.max_qty_manual    = 50.0

        # === ФИЛЬТРЫ SFP ===
        self.use_sfp_quality = True
        self.wick_min_ticks  = 7
        self.close_back_pct  = 1.0

        # === БЭКТЕСТ ===
        self.period_choice = "30"
        self.days_back     = 30
        self.slippage_pct  = 0.0
        self.latency_ms    = 0

        # === МАРКЕТ ===
        self.taker_fee_rate = 0.00055
        self.min_net_profit = 1.2
        self.min_order_qty  = 0.01
        self.qty_step       = 0.01
        self.tick_size      = 0.01

        # Совместимость
        self.use_bar_trail  = True
        self.trail_lookback = 50
        self.trail_buf_ticks= 40

        # нормализация
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
            except:
                self.days_back = 30

    def _normalize_derived(self):
        # close_back_pct clamp
        try:
            if self.close_back_pct is None:
                self.close_back_pct = 1.0
            if self.close_back_pct > 1.0:
                self.close_back_pct = float(self.close_back_pct)/100.0
            if self.close_back_pct < 0.0:
                self.close_back_pct = 0.0
        except:
            self.close_back_pct = 1.0

        # trailing_offset_perc sync
        try:
            if self.trailing_offset is not None:
                self.trailing_offset_perc = float(self.trailing_offset)
        except:
            pass

        sym = (self.symbol or "").upper()
        if sym in ("ETHUSDT","BTCUSDT"):
            self.qty_step = 0.001
            self.min_order_qty = 0.001
            self.tick_size = 0.01

        self.price_for_logic      = (self.price_for_logic or "last").lower()
        if self.price_for_logic not in ("last","mark"):
            self.price_for_logic = "last"
        self.trigger_price_source = (self.trigger_price_source or "last").lower()
        if self.trigger_price_source not in ("last","mark"):
            self.trigger_price_source = "last"
        self.arm_rr_basis = (self.arm_rr_basis or "extremum").lower()
        if self.arm_rr_basis not in ("extremum","last"):
            self.arm_rr_basis = "extremum"

        try:
            self.trailing_perc = max(0.0, float(self.trailing_perc))
        except:
            self.trailing_perc = 0.5
        try:
            self.trailing_offset_perc = max(0.0, float(self.trailing_offset_perc))
        except:
            self.trailing_offset_perc = 0.4

        try:
            self.interval = str(self.interval)
        except:
            self.interval = "15"
        try:
            self.intrabar_tf = str(self.intrabar_tf)
        except:
            self.intrabar_tf = "1"

    # ---------- load/save ----------

    def load_config(self, filename: str = "config.json"):
        try:
            if os.path.exists(filename):
                with open(filename,"r") as f:
                    data=json.load(f)
                self._apply_config_data(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self, filename: str = "config.json"):
        try:
            with open(filename,"w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _apply_config_data(self, data: Dict[str,Any]):
        for k,v in data.items():
            if hasattr(self,k):
                setattr(self,k,v)
        self._update_days_back()
        self._normalize_derived()

    def update_from_dict(self,data:Dict[str,Any]):
        self._apply_config_data(data)
        self.save_config()

    # ---------- export ----------

    def to_dict(self)->Dict[str,Any]:
        return {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "interval": str(self.interval),
            "risk_reward": self.risk_reward,
            "sfp_len": self.sfp_len,
            "risk_pct": self.risk_pct,
            "use_take_profit": self.use_take_profit,

            # Smart Trail
            "enable_smart_trail": self.enable_smart_trail,
            "smart_trail_mode": self.smart_trail_mode,
            "trailing_perc": self.trailing_perc,
            "trailing_offset_perc": self.trailing_offset_perc,
            "trailing_offset": self.trailing_offset,
            "use_arm_after_rr": self.use_arm_after_rr,
            "arm_rr": self.arm_rr,
            "arm_rr_basis": self.arm_rr_basis,

            "price_for_logic": self.price_for_logic,
            "trigger_price_source": self.trigger_price_source,

            "use_intrabar": self.use_intrabar,
            "intrabar_tf": str(self.intrabar_tf),
            "intrabar_pull_limit": self.intrabar_pull_limit,
            "smooth_intrabar": self.smooth_intrabar,
            "intrabar_steps": self.intrabar_steps,

            "use_sfp_quality": self.use_sfp_quality,
            "wick_min_ticks": self.wick_min_ticks,
            "close_back_pct": self.close_back_pct,

            "period_choice": self.period_choice,
            "days_back": self.days_back,
            "slippage_pct": self.slippage_pct,
            "latency_ms": self.latency_ms,

            "limit_qty_enabled": self.limit_qty_enabled,
            "max_qty_manual": self.max_qty_manual,
            "taker_fee_rate": self.taker_fee_rate,
            "min_net_profit": self.min_net_profit,
            "min_order_qty": self.min_order_qty,
            "qty_step": self.qty_step,
            "tick_size": self.tick_size,

            "use_bar_trail": self.use_bar_trail,
            "trail_lookback": self.trail_lookback,
            "trail_buf_ticks": self.trail_buf_ticks,
        }

    def validate(self)->bool:
        try:
            if self.risk_reward <= 0: raise ValueError("risk_reward must be >0")
            if not (0 < self.risk_pct <= 100): raise ValueError("risk_pct must be 0..100")
            if self.sfp_len < 1: raise ValueError("sfp_len >=1")
            if self.max_qty_manual <= 0: raise ValueError("max_qty_manual >0")
            if not (0.0 <= float(self.close_back_pct) <= 1.0): raise ValueError("close_back_pct 0..1")
            if self.arm_rr_basis not in ("extremum","last"): raise ValueError("arm_rr_basis invalid")
            if self.price_for_logic not in ("last","mark"): raise ValueError("price_for_logic invalid")
            if self.trigger_price_source not in ("last","mark"): raise ValueError("trigger_price_source invalid")
            return True
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
