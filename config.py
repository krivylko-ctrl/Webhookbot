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
    # В live это полезно. Для backtest можно не вызывать must_have().
    missing = []
    if BYBIT_ACCOUNT_TYPE not in ("linear", "inverse", "option"):
        missing.append(f"BYBIT_ACCOUNT_TYPE (got '{BYBIT_ACCOUNT_TYPE}')")
    if missing:
        raise RuntimeError("Missing/invalid env: " + ", ".join(missing))


# =====================================================
#                    CONFIG CLASS
# =====================================================

class Config:
    """Конфигурация стратегии KWIN (эквивалент TV inputs)"""

    def __init__(self):
        # === ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
        self.symbol       = SYMBOL
        self.market_type  = BYBIT_ACCOUNT_TYPE or "linear"   # синхронизируем с env (по умолч. linear)
        self.interval     = "15"                              # добавлено: базовой ТФ для сигналов/графика

        # Риск/TP
        # допускаем переопределение через ENV
        self.risk_reward  = float(env("RISK_REWARD", "1.3"))
        self.sfp_len      = 2
        self.risk_pct     = float(env("RISK_PCT", "3.0"))

        # === SMART TRAILING ===
        self.enable_smart_trail     = env("ENABLE_SMART_TRAIL", "true").lower() not in ("0", "false", "no")
        self.trailing_perc          = float(env("TRAILING_PERC", "0.5"))    # в процентах
        self.trailing_offset_perc   = float(env("TRAILING_OFFSET_PERC", "0.4"))  # в процентах (ИСПОЛЬЗУЕМ ЭТО)
        # оставили trailing_offset для обратной совместимости, но не используем
        self.trailing_offset        = self.trailing_offset_perc

        # ARM (активация трейла после достижения RR)
        self.use_arm_after_rr = env("USE_ARM_AFTER_RR", "true").lower() not in ("0", "false", "no")
        self.arm_rr           = float(env("ARM_RR", "0.5"))      # в R
        self.arm_rr_basis     = "extremum"   # "extremum" | "last"

        # Источники цены (для логики и для триггера SL/TP в bt/live)
        self.price_for_logic        = "last"  # "last"|"mark"
        self.trigger_price_source   = "last"  # "last"|"mark"

        # === ИНТРАБАР (для bt и/или live обработчика минуток) ===
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
        self.close_back_pct  = 1.0  # [0..1], нормализуем ниже

        # === БЭКТЕСТ/ЭФФЕКТЫ ИСПОЛНЕНИЯ ===
        self.period_choice = "30"
        self.days_back     = 30
        self.slippage_pct  = 0.0     # доля цены при исполнении
        self.latency_ms    = 0       # заглушка для задержки

        # === ВНУТРЕННИЕ КОНСТАНТЫ/МАРКЕТ ===
        self.taker_fee_rate = 0.00055
        self.min_net_profit = 1.2
        self.min_order_qty  = 0.01
        self.qty_step       = 0.01
        self.tick_size      = 0.01

        # Старые поля — не используем, оставлены для совместимости UI
        self.use_bar_trail  = True
        self.trail_lookback = 50
        self.trail_buf_ticks= 40

        # Обновляем зависимые параметры
        self._update_days_back()
        self._normalize_derived()

        # Подгружаем из файла, если есть
        self.load_config()

        # финальная нормализация после загрузки
        self._update_days_back()
        self._normalize_derived()

    # ---------- нормализация и зависимости ----------

    def _update_days_back(self):
        pc = str(self.period_choice)
        if pc == "30":
            self.days_back = 30
        elif pc == "60":
            self.days_back = 60
        elif pc == "180":
            self.days_back = 180
        else:
            # дефолт
            try:
                self.days_back = int(self.days_back or 30)
            except Exception:
                self.days_back = 30

    def _normalize_derived(self):
        # close_back_pct в [0..1]
        try:
            if self.close_back_pct is None:
                self.close_back_pct = 1.0
            if self.close_back_pct > 1.0:
                self.close_back_pct = float(self.close_back_pct) / 100.0
            if self.close_back_pct < 0.0:
                self.close_back_pct = 0.0
        except Exception:
            self.close_back_pct = 1.0

        # trailing_offset_perc — берем из совместимого поля, если UI прислал старое имя
        try:
            if self.trailing_offset is not None:
                # если кто-то обновил только trailing_offset — синхронизируем
                self.trailing_offset_perc = float(self.trailing_offset)
        except Exception:
            pass

        # здравые дефолты шагов для ETH/BTC (чтобы не залипать на округлениях)
        sym = (self.symbol or "").upper()
        if sym in ("ETHUSDT", "BTCUSDT"):
            self.qty_step = 0.001
            self.min_order_qty = 0.001
            self.tick_size = 0.01

        # строковые поля к нижнему регистру + вайтлист
        self.price_for_logic      = (self.price_for_logic or "last").lower()
        if self.price_for_logic not in ("last", "mark"):
            self.price_for_logic = "last"
        self.trigger_price_source = (self.trigger_price_source or "last").lower()
        if self.trigger_price_source not in ("last", "mark"):
            self.trigger_price_source = "last"
        self.arm_rr_basis = (self.arm_rr_basis or "extremum").lower()
        if self.arm_rr_basis not in ("extremum", "last"):
            self.arm_rr_basis = "extremum"

        # числа (страхуемся от строк)
        try:
            self.trailing_perc = float(self.trailing_perc)
        except Exception:
            self.trailing_perc = 0.5
        try:
            self.trailing_offset_perc = float(self.trailing_offset_perc)
        except Exception:
            self.trailing_offset_perc = 0.4

        # interval — строка
        try:
            self.interval = str(self.interval)
        except Exception:
            self.interval = "15"

    # ---------- load/save ----------

    def load_config(self, filename: str = "config.json"):
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                self._apply_config_data(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self, filename: str = "config.json"):
        try:
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _apply_config_data(self, data: Dict[str, Any]):
        # аккуратно обновляем только известные атрибуты
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._update_days_back()
        self._normalize_derived()

    def update_from_dict(self, data: Dict[str, Any]):
        self._apply_config_data(data)
        self.save_config()

    # ---------- экспорт ----------

    def to_dict(self) -> Dict[str, Any]:
        """Экспорт полного набора настроек для UI/сохранения в JSON."""
        return {
            # базовые
            "symbol": self.symbol,
            "market_type": self.market_type,
            "interval": str(self.interval),
            "risk_reward": self.risk_reward,
            "sfp_len": self.sfp_len,
            "risk_pct": self.risk_pct,

            # трейлинг
            "enable_smart_trail": self.enable_smart_trail,
            "trailing_perc": self.trailing_perc,
            "trailing_offset_perc": self.trailing_offset_perc,
            "trailing_offset": self.trailing_offset,  # совместимость
            "use_arm_after_rr": self.use_arm_after_rr,
            "arm_rr": self.arm_rr,
            "arm_rr_basis": self.arm_rr_basis,

            # источники цены/триггера
            "price_for_logic": self.price_for_logic,
            "trigger_price_source": self.trigger_price_source,

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

            # backtest + исполнение
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

            # совместимость (не используются трейлом, оставлены для UI)
            "use_bar_trail": self.use_bar_trail,
            "trail_lookback": self.trail_lookback,
            "trail_buf_ticks": self.trail_buf_ticks,
        }

    # ---------- UI definitions (по желанию дополни в своём стеке) ----------

    def get_input_definitions(self) -> Dict[str, Dict]:
        return {
            "risk_reward": {"type":"float","label":"TP Risk/Reward Ratio","min":0.5,"max":5.0,"step":0.1,"value":self.risk_reward},
            "sfp_len":     {"type":"int","label":"Swing Length","min":1,"max":10,"step":1,"value":self.sfp_len},
            "risk_pct":    {"type":"float","label":"Risk % per trade","min":0.1,"max":10.0,"step":0.1,"value":self.risk_pct},

            "enable_smart_trail":{"type":"bool","label":"✅ Enable Smart Trailing TP","value":self.enable_smart_trail},
            "trailing_perc":     {"type":"float","label":"Trailing %","min":0.0,"max":5.0,"step":0.1,"value":self.trailing_perc},
            "trailing_offset_perc":{"type":"float","label":"Trailing Offset %","min":0.0,"max":5.0,"step":0.1,"value":self.trailing_offset_perc},
            "use_arm_after_rr":  {"type":"bool","label":"Enable Arm after RR≥X","value":self.use_arm_after_rr},
            "arm_rr":            {"type":"float","label":"Arm RR (R)","min":0.1,"max":5.0,"step":0.1,"value":self.arm_rr},
            "arm_rr_basis":      {"type":"select","label":"ARM basis","options":["extremum","last"],"value":self.arm_rr_basis},

            "price_for_logic":      {"type":"select","label":"Price for logic","options":["last","mark"],"value":self.price_for_logic},
            "trigger_price_source": {"type":"select","label":"Trigger price source","options":["last","mark"],"value":self.trigger_price_source},

            "use_intrabar":        {"type":"bool","label":"Use 1m intrabar trailing","value":self.use_intrabar},
            "intrabar_tf":         {"type":"select","label":"Intrabar TF","options":["1","3","5"],"value":str(self.intrabar_tf)},
            "intrabar_pull_limit": {"type":"int","label":"1m history limit","min":200,"max":2000,"step":100,"value":self.intrabar_pull_limit},
            "smooth_intrabar":     {"type":"bool","label":"Smooth intrabar (micro-steps)","value":self.smooth_intrabar},
            "intrabar_steps":      {"type":"int","label":"Micro-steps per 1m","min":1,"max":12,"step":1,"value":self.intrabar_steps},

            "use_sfp_quality": {"type":"bool","label":"Filter: SFP quality (wick+closeback)","value":self.use_sfp_quality},
            "wick_min_ticks":  {"type":"int","label":"Min wick ticks","min":0,"max":100,"step":1,"value":self.wick_min_ticks},
            "close_back_pct":  {"type":"float","label":"Close-back (0..1)","min":0.0,"max":1.0,"step":0.05,"value":self.close_back_pct},

            "period_choice": {"type":"select","label":"Backtest Period","options":["30","60","180"],"value":self.period_choice},
        }

    def validate(self) -> bool:
        try:
            if self.risk_reward <= 0: raise ValueError("Risk reward must be positive")
            if not (0 < self.risk_pct <= 100): raise ValueError("Risk percentage must be between 0 and 100")
            if self.sfp_len < 1: raise ValueError("SFP length must be at least 1")
            if self.max_qty_manual <= 0: raise ValueError("Max quantity must be positive")
            if not (0.0 <= float(self.close_back_pct) <= 1.0): raise ValueError("Close back must be 0..1")
            if self.arm_rr_basis not in ("extremum","last"): raise ValueError("arm_rr_basis must be extremum|last")
            if self.price_for_logic not in ("last","mark"): raise ValueError("price_for_logic must be last|mark")
            if self.trigger_price_source not in ("last","mark"): raise ValueError("trigger_price_source must be last|mark")
            return True
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
