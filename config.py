# config.py
from typing import Dict, Any, Optional
import json
import os


def env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""


# ===== ENV / Defaults =====
BYBIT_API_KEY      = env("BYBIT_API_KEY", "")
BYBIT_API_SECRET   = env("BYBIT_API_SECRET", "")
BYBIT_ACCOUNT_TYPE = env("BYBIT_ACCOUNT_TYPE", "linear")  # "linear" | "inverse" | "option" | "spot"
SYMBOL             = env("SYMBOL", "ETHUSDT").upper()
INTERVALS          = [i.strip() for i in env("INTERVALS", "1,15,60").split(",") if i.strip()]


def must_have():
    missing = []
    if not BYBIT_API_KEY:     missing.append("BYBIT_API_KEY")
    if not BYBIT_API_SECRET:  missing.append("BYBIT_API_SECRET")
    if not SYMBOL:            missing.append("SYMBOL")
    if not INTERVALS:         missing.append("INTERVALS")
    if BYBIT_ACCOUNT_TYPE not in ("linear", "inverse", "option", "spot"):
        missing.append(f"BYBIT_ACCOUNT_TYPE (got '{BYBIT_ACCOUNT_TYPE}')")
    if missing:
        raise RuntimeError("Missing/invalid env: " + ", ".join(missing))


class Config:
    """Конфигурация стратегии KWIN (эквивалент TV inputs)"""

    def __init__(self):
        # === ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
        self.risk_reward: float = 1.3     # TP Risk/Reward Ratio
        self.sfp_len: int = 2             # Swing Length
        self.risk_pct: float = 3.0        # Risk % per trade

        # === ЦЕНОВЫЕ ИСТОЧНИКИ / ТРИГГЕРЫ (1:1 с TV-концепцией) ===
        # чем пользоваться в логике (ARM/Trail/entry-расчёты): "last" | "mark"
        self.price_for_logic: str = "last"
        # чем триггерить stop/conditional ордера на Bybit: "last" | "mark"
        self.trigger_price_source: str = "last"
        # база для ARM RR: "extremum" (по high/low бара) или "last" (по текущей)
        self.arm_rr_basis: str = "extremum"

        # === КОМИССИИ / ФРИКЦИИ ===
        self.taker_fee_rate: float = 0.00055
        self.maker_fee_rate: float = 0.00020
        self.slippage_pct: float = 0.0     # 0.0005 => 5 bps
        self.latency_ms: int = 0           # "микро" задержка обработки SL/TP в симуляции

        # === SMART TRAILING ===
        self.enable_smart_trail: bool = True
        self.trailing_perc: float = 0.5         # Trailing % (от entry)
        self.trailing_offset: float = 0.4       # Backward-совместимость (если где-то используется)
        self.trailing_offset_perc: float = 0.4  # Offset % (от entry) — для процентного трейла
        self.use_arm_after_rr: bool = True      # ARM включение
        self.arm_rr: float = 0.5                # скольких "R" ждать до активации трейла
        self.use_bar_trail: bool = True         # если где-то включается баровый трейл
        self.trail_lookback: int = 50
        self.trail_buf_ticks: int = 40

        # === ИНТРАБАР (1m) ДЛЯ LIVE/BT ===
        self.use_intrabar: bool = True
        self.intrabar_tf: str = "1"
        self.intrabar_pull_limit: int = 1000
        self.intrabar_sim_two_phase: bool = True
        self.intrabar_sl_first: bool = True

        # === ОГРАНИЧЕНИЯ ПОЗИЦИИ ===
        self.limit_qty_enabled: bool = True
        self.max_qty_manual: float = 50.0

        # === ФИЛЬТРЫ SFP ===
        self.use_sfp_quality: bool = True
        self.wick_min_ticks: int = 7
        self.close_back_pct: float = 1.0  # 0..1

        # === БЭКТЕСТ ===
        self.period_choice: str = "30"
        self.days_back: int = 30  # будет синхронизировано _update_days_back()

        # === НАСТРОЙКИ РЫНКА ===
        self.market_type: str = BYBIT_ACCOUNT_TYPE or "linear"  # "linear" для деривативов, "spot" для спота…
        self.symbol: str = SYMBOL
        self.tick_size: float = 0.01
        self.min_order_qty: float = 0.01
        self.qty_step: float = 0.01

        # синхронизируем вычислимые поля
        self._update_days_back()

        # загрузка пользовательских переопределений из файла (если есть)
        self.load_config()

    # ------------------------
    # helpers: persist / ui
    # ------------------------
    def _update_days_back(self):
        """Обновление days_back на основе period_choice"""
        if self.period_choice == "30":
            self.days_back = 30
        elif self.period_choice == "60":
            self.days_back = 60
        elif self.period_choice == "180":
            self.days_back = 180
        else:
            # дефолт безопасный
            try:
                self.days_back = max(1, int(self.period_choice))
            except Exception:
                self.days_back = 30

    def load_config(self, filename: str = "config.json"):
        """Загрузка конфигурации из файла"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self._apply_config_data(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self, filename: str = "config.json"):
        """Сохранение конфигурации в файл"""
        try:
            config_data = self.to_dict()
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _apply_config_data(self, data: Dict[str, Any]):
        """Применение данных конфигурации"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # зависимые параметры
        self._update_days_back()

    def update_from_dict(self, data: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        self._apply_config_data(data)
        self.save_config()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь (всё, что важно сохранять)"""
        return {
            # core
            'risk_reward': self.risk_reward,
            'sfp_len': self.sfp_len,
            'risk_pct': self.risk_pct,

            # price/trigger/arm-basis
            'price_for_logic': self.price_for_logic,
            'trigger_price_source': self.trigger_price_source,
            'arm_rr_basis': self.arm_rr_basis,

            # fees & frictions
            'taker_fee_rate': self.taker_fee_rate,
            'maker_fee_rate': self.maker_fee_rate,
            'slippage_pct': self.slippage_pct,
            'latency_ms': self.latency_ms,

            # smart trail
            'enable_smart_trail': self.enable_smart_trail,
            'trailing_perc': self.trailing_perc,
            'trailing_offset': self.trailing_offset,
            'trailing_offset_perc': self.trailing_offset_perc,
            'use_arm_after_rr': self.use_arm_after_rr,
            'arm_rr': self.arm_rr,
            'use_bar_trail': self.use_bar_trail,
            'trail_lookback': self.trail_lookback,
            'trail_buf_ticks': self.trail_buf_ticks,

            # intrabar
            'use_intrabar': self.use_intrabar,
            'intrabar_tf': self.intrabar_tf,
            'intrabar_pull_limit': self.intrabar_pull_limit,
            'intrabar_sim_two_phase': self.intrabar_sim_two_phase,
            'intrabar_sl_first': self.intrabar_sl_first,

            # position limits
            'limit_qty_enabled': self.limit_qty_enabled,
            'max_qty_manual': self.max_qty_manual,

            # sfp quality
            'use_sfp_quality': self.use_sfp_quality,
            'wick_min_ticks': self.wick_min_ticks,
            'close_back_pct': self.close_back_pct,

            # backtest
            'period_choice': self.period_choice,
            'days_back': self.days_back,

            # market/instrument
            'market_type': self.market_type,
            'symbol': self.symbol,
            'tick_size': self.tick_size,
            'min_order_qty': self.min_order_qty,
            'qty_step': self.qty_step,
        }

    def get_input_definitions(self) -> Dict[str, Dict]:
        """Определения параметров для UI (аналог TV inputs).
        Вынесены ключевые настройки; price/trigger можно добавить при необходимости.
        """
        return {
            'risk_reward': {
                'type': 'float', 'label': 'TP Risk/Reward Ratio',
                'min': 0.5, 'max': 5.0, 'step': 0.1, 'value': self.risk_reward
            },
            'sfp_len': {
                'type': 'int', 'label': 'Swing Length',
                'min': 1, 'max': 10, 'step': 1, 'value': self.sfp_len
            },
            'risk_pct': {
                'type': 'float', 'label': 'Risk % per trade',
                'min': 0.1, 'max': 10.0, 'step': 0.1, 'value': self.risk_pct
            },
            'enable_smart_trail': {
                'type': 'bool', 'label': '✅ Enable Smart Trailing TP',
                'value': self.enable_smart_trail
            },
            'trailing_perc': {
                'type': 'float', 'label': 'Trailing %',
                'min': 0.1, 'max': 5.0, 'step': 0.1, 'value': self.trailing_perc
            },
            'trailing_offset_perc': {
                'type': 'float', 'label': 'Trailing Offset % (perc mode)',
                'min': 0.1, 'max': 5.0, 'step': 0.1, 'value': self.trailing_offset_perc
            },
            'use_arm_after_rr': {
                'type': 'bool', 'label': 'Enable Arm after RR≥X',
                'value': self.use_arm_after_rr
            },
            'arm_rr': {
                'type': 'float', 'label': 'Arm RR (R)',
                'min': 0.1, 'max': 2.0, 'step': 0.1, 'value': self.arm_rr
            },
            'use_bar_trail': {
                'type': 'bool', 'label': 'Use Bar-Low/High Smart Trail',
                'value': self.use_bar_trail
            },
            'trail_lookback': {
                'type': 'int', 'label': 'Trail lookback bars',
                'min': 1, 'max': 200, 'step': 1, 'value': self.trail_lookback
            },
            'trail_buf_ticks': {
                'type': 'int', 'label': 'Trail buffer (ticks)',
                'min': 0, 'max': 100, 'step': 1, 'value': self.trail_buf_ticks
            },
            # intrabar
            'use_intrabar': {
                'type': 'bool', 'label': '💚 Use intrabar (1m) processing',
                'value': self.use_intrabar
            },
            'intrabar_tf': {
                'type': 'select', 'label': 'Intrabar timeframe',
                'options': ['1', '3', '5'], 'value': self.intrabar_tf
            },
            'intrabar_pull_limit': {
                'type': 'int', 'label': 'Intrabar pull limit (bars)',
                'min': 100, 'max': 5000, 'step': 50, 'value': self.intrabar_pull_limit
            },
            'intrabar_sim_two_phase': {
                'type': 'bool', 'label': 'Intrabar two-phase sim (backtest)',
                'value': self.intrabar_sim_two_phase
            },
            'intrabar_sl_first': {
                'type': 'bool', 'label': 'SL priority over TP (intrabar)',
                'value': self.intrabar_sl_first
            },
            # limits
            'limit_qty_enabled': {
                'type': 'bool', 'label': 'Limit Max Position Qty',
                'value': self.limit_qty_enabled
            },
            'max_qty_manual': {
                'type': 'float', 'label': 'Max Qty (ETH)',
                'min': 0.01, 'max': 1000.0, 'step': 0.01, 'value': self.max_qty_manual
            },
            # sfp quality
            'use_sfp_quality': {
                'type': 'bool', 'label': 'Filter: SFP quality (wick+closeback)',
                'value': self.use_sfp_quality
            },
            'wick_min_ticks': {
                'type': 'int', 'label': 'SFP: min wick depth (ticks)',
                'min': 0, 'max': 100, 'step': 1, 'value': self.wick_min_ticks
            },
            'close_back_pct': {
                'type': 'float', 'label': 'SFP: min close-back % of wick',
                'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': self.close_back_pct
            },
            # backtest period
            'period_choice': {
                'type': 'select', 'label': 'Backtest Period',
                'options': ['30', '60', '180'], 'value': self.period_choice
            }
        }

    def validate(self) -> bool:
        """Валидация конфигурации"""
        try:
            if self.risk_reward <= 0:
                raise ValueError("Risk reward must be positive")

            if self.risk_pct <= 0 or self.risk_pct > 100:
                raise ValueError("Risk percentage must be between 0 and 100")

            if self.sfp_len < 1:
                raise ValueError("SFP length must be at least 1")

            if self.max_qty_manual <= 0:
                raise ValueError("Max quantity must be positive")

            if not (0.0 <= self.close_back_pct <= 1.0):
                raise ValueError("Close back percentage must be between 0 and 1")

            if self.intrabar_pull_limit < 100:
                raise ValueError("Intrabar pull limit must be >= 100")

            if self.price_for_logic not in ("last", "mark"):
                raise ValueError("price_for_logic must be 'last' or 'mark'")

            if self.trigger_price_source not in ("last", "mark"):
                raise ValueError("trigger_price_source must be 'last' or 'mark'")

            if self.arm_rr_basis not in ("extremum", "last"):
                raise ValueError("arm_rr_basis must be 'extremum' or 'last'")

            return True

        except Exception as e:
            print(f"Config validation error: {e}")
            return False
