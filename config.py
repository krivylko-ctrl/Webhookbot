from typing import Dict, Any
import json
import os

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else ""

BYBIT_API_KEY      = env("BYBIT_API_KEY", "")
BYBIT_API_SECRET   = env("BYBIT_API_SECRET", "")
BYBIT_ACCOUNT_TYPE = env("BYBIT_ACCOUNT_TYPE", "linear")
SYMBOL             = env("SYMBOL", "ETHUSDT").upper()
INTERVALS          = [i.strip() for i in env("INTERVALS", "1,15,60").split(",") if i.strip()]

def must_have():
    missing = []
    if not BYBIT_API_KEY:     missing.append("BYBIT_API_KEY")
    if not BYBIT_API_SECRET:  missing.append("BYBIT_API_SECRET")
    if not SYMBOL:            missing.append("SYMBOL")
    if not INTERVALS:         missing.append("INTERVALS")
    if BYBIT_ACCOUNT_TYPE not in ("linear", "inverse", "option"):
        missing.append(f"BYBIT_ACCOUNT_TYPE (got '{BYBIT_ACCOUNT_TYPE}')")
    if missing:
        raise RuntimeError("Missing/invalid env: " + ", ".join(missing))

class Config:
    """Конфигурация стратегии KWIN (эквивалент TV inputs)"""
    
    def __init__(self):
        # === ОСНОВНЫЕ ПАРАМЕТРЫ СТРАТЕГИИ ===
        self.risk_reward = 1.3          # TP Risk/Reward Ratio
        self.sfp_len = 2                # Swing Length
        self.risk_pct = 3.0             # Risk % per trade
        
        # === SMART TRAILING ===
        self.enable_smart_trail = True  # Enable Smart Trailing TP
        self.trailing_perc = 0.5        # Trailing %
        self.trailing_offset = 0.4      # Trailing Offset %
        self.trailing_offset_perc = 0.4 # Trailing Offset % для процентного трейла
        self.use_arm_after_rr = True    # Enable Arm after RR≥X
        self.arm_rr = 0.5               # Arm RR (R)
        self.use_bar_trail = True       # Use Bar-Low/High Smart Trail
        self.trail_lookback = 50        # Trail lookback bars
        self.trail_buf_ticks = 40       # Trail buffer (ticks)

        # === ИНТРАБАР (1m) ЛОГИКА ДЛЯ LIVE/BT ===
        self.use_intrabar = True        # Включить обработку внутри бара (эмуляция calc_on_every_tick)
        self.intrabar_tf = "1"          # Таймфрейм для интрабара (минуты как в Bybit/TV)
        self.intrabar_pull_limit = 1000 # Сколько 1m баров подкачивать/держать в памяти
        self.intrabar_sim_two_phase = True  # В бэктесте: двухфазный проход high/low
        self.intrabar_sl_first = True       # Приоритет: сначала SL, затем TP (как у strategy.exit)

        # === ОГРАНИЧЕНИЯ ПОЗИЦИИ ===
        self.limit_qty_enabled = True   # Limit Max Position Qty
        self.max_qty_manual = 50.0      # Max Qty (ETH)
        
        # === ФИЛЬТРЫ SFP ===
        self.use_sfp_quality = True     # Filter: SFP quality (wick+closeback)
        self.wick_min_ticks = 7         # SFP: min wick depth (ticks)
        self.close_back_pct = 1.0       # SFP: min close-back % of wick
        
        # === БЭКТЕСТ ===
        self.period_choice = "30"       # Backtest Period
        self.days_back = 30             # Расчитывается из period_choice
        
        # === ВНУТРЕННИЕ КОНСТАНТЫ ===
        self.taker_fee_rate = 0.00055   # Bybit taker fee
        self.min_net_profit = 1.2       # Минимальная чистая прибыль
        self.min_order_qty = 0.01       # Минимальный размер ордера
        self.qty_step = 0.01            # Шаг размера позиции
        
        # === НАСТРОЙКИ РЫНКА ===
        self.market_type = "linear"     # "linear" для деривативов, "spot" для спота
        self.symbol = SYMBOL            # Торгуемый символ
        self.tick_size = 0.01           # Тик-сайз для округления цен
        
        # Обновляем days_back на основе period_choice
        self._update_days_back()
        
        # Загружаем конфигурацию из файла если существует
        self.load_config()
    
    def _update_days_back(self):
        """Обновление days_back на основе period_choice"""
        if self.period_choice == "30":
            self.days_back = 30
        elif self.period_choice == "60":
            self.days_back = 60
        elif self.period_choice == "180":
            self.days_back = 180
    
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
        
        # Обновляем зависимые параметры
        self._update_days_back()
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        self._apply_config_data(data)
        self.save_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'risk_reward': self.risk_reward,
            'sfp_len': self.sfp_len,
            'risk_pct': self.risk_pct,
            'enable_smart_trail': self.enable_smart_trail,
            'trailing_perc': self.trailing_perc,
            'trailing_offset': self.trailing_offset,
            'trailing_offset_perc': self.trailing_offset_perc,
            'use_arm_after_rr': self.use_arm_after_rr,
            'arm_rr': self.arm_rr,
            'use_bar_trail': self.use_bar_trail,
            'trail_lookback': self.trail_lookback,
            'trail_buf_ticks': self.trail_buf_ticks,

            # --- интрабар ---
            'use_intrabar': self.use_intrabar,
            'intrabar_tf': self.intrabar_tf,
            'intrabar_pull_limit': self.intrabar_pull_limit,
            'intrabar_sim_two_phase': self.intrabar_sim_two_phase,
            'intrabar_sl_first': self.intrabar_sl_first,

            'limit_qty_enabled': self.limit_qty_enabled,
            'max_qty_manual': self.max_qty_manual,
            'use_sfp_quality': self.use_sfp_quality,
            'wick_min_ticks': self.wick_min_ticks,
            'close_back_pct': self.close_back_pct,
            'period_choice': self.period_choice,
            'days_back': self.days_back,
            'taker_fee_rate': self.taker_fee_rate,
            'min_net_profit': self.min_net_profit,
            'min_order_qty': self.min_order_qty,
            'qty_step': self.qty_step
        }
    
    def get_input_definitions(self) -> Dict[str, Dict]:
        """Получить определения параметров для UI (аналог TV inputs)"""
        return {
            'risk_reward': {
                'type': 'float',
                'label': 'TP Risk/Reward Ratio',
                'min': 0.5,
                'max': 5.0,
                'step': 0.1,
                'value': self.risk_reward
            },
            'sfp_len': {
                'type': 'int',
                'label': 'Swing Length',
                'min': 1,
                'max': 10,
                'step': 1,
                'value': self.sfp_len
            },
            'risk_pct': {
                'type': 'float',
                'label': 'Risk % per trade',
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'value': self.risk_pct
            },
            'enable_smart_trail': {
                'type': 'bool',
                'label': '✅ Enable Smart Trailing TP',
                'value': self.enable_smart_trail
            },
            'trailing_perc': {
                'type': 'float',
                'label': 'Trailing %',
                'min': 0.1,
                'max': 5.0,
                'step': 0.1,
                'value': self.trailing_perc
            },
            'trailing_offset': {
                'type': 'float',
                'label': 'Trailing Offset %',
                'min': 0.1,
                'max': 5.0,
                'step': 0.1,
                'value': self.trailing_offset
            },
            'trailing_offset_perc': {
                'type': 'float',
                'label': 'Trailing Offset % (perc mode)',
                'min': 0.1,
                'max': 5.0,
                'step': 0.1,
                'value': self.trailing_offset_perc
            },
            'use_arm_after_rr': {
                'type': 'bool',
                'label': 'Enable Arm after RR≥X',
                'value': self.use_arm_after_rr
            },
            'arm_rr': {
                'type': 'float',
                'label': 'Arm RR (R)',
                'min': 0.1,
                'max': 2.0,
                'step': 0.1,
                'value': self.arm_rr
            },
            'use_bar_trail': {
                'type': 'bool',
                'label': 'Use Bar-Low/High Smart Trail',
                'value': self.use_bar_trail
            },
            'trail_lookback': {
                'type': 'int',
                'label': 'Trail lookback bars',
                'min': 1,
                'max': 200,
                'step': 1,
                'value': self.trail_lookback
            },
            'trail_buf_ticks': {
                'type': 'int',
                'label': 'Trail buffer (ticks)',
                'min': 0,
                'max': 100,
                'step': 1,
                'value': self.trail_buf_ticks
            },

            # --- интрабар UI ---
            'use_intrabar': {
                'type': 'bool',
                'label': '💚 Use intrabar (1m) processing',
                'value': self.use_intrabar
            },
            'intrabar_tf': {
                'type': 'select',
                'label': 'Intrabar timeframe',
                'options': ['1', '3', '5'],
                'value': self.intrabar_tf
            },
            'intrabar_pull_limit': {
                'type': 'int',
                'label': 'Intrabar pull limit (bars)',
                'min': 100,
                'max': 5000,
                'step': 50,
                'value': self.intrabar_pull_limit
            },
            'intrabar_sim_two_phase': {
                'type': 'bool',
                'label': 'Intrabar two-phase sim (backtest)',
                'value': self.intrabar_sim_two_phase
            },
            'intrabar_sl_first': {
                'type': 'bool',
                'label': 'SL priority over TP (intrabar)',
                'value': self.intrabar_sl_first
            },

            'limit_qty_enabled': {
                'type': 'bool',
                'label': 'Limit Max Position Qty',
                'value': self.limit_qty_enabled
            },
            'max_qty_manual': {
                'type': 'float',
                'label': 'Max Qty (ETH)',
                'min': 0.01,
                'max': 1000.0,
                'step': 0.01,
                'value': self.max_qty_manual
            },
            'use_sfp_quality': {
                'type': 'bool',
                'label': 'Filter: SFP quality (wick+closeback)',
                'value': self.use_sfp_quality
            },
            'wick_min_ticks': {
                'type': 'int',
                'label': 'SFP: min wick depth (ticks)',
                'min': 0,
                'max': 100,
                'step': 1,
                'value': self.wick_min_ticks
            },
            'close_back_pct': {
                'type': 'float',
                'label': 'SFP: min close-back % of wick',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'value': self.close_back_pct
            },
            'period_choice': {
                'type': 'select',
                'label': 'Backtest Period',
                'options': ['30', '60', '180'],
                'value': self.period_choice
            }
        }
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        try:
            # Проверяем основные параметры
            if self.risk_reward <= 0:
                raise ValueError("Risk reward must be positive")
            
            if self.risk_pct <= 0 or self.risk_pct > 100:
                raise ValueError("Risk percentage must be between 0 and 100")
            
            if self.sfp_len < 1:
                raise ValueError("SFP length must be at least 1")
            
            if self.max_qty_manual <= 0:
                raise ValueError("Max quantity must be positive")
            
            if self.close_back_pct < 0 or self.close_back_pct > 1:
                raise ValueError("Close back percentage must be between 0 and 1")

            if self.intrabar_pull_limit < 100:
                raise ValueError("Intrabar pull limit must be >= 100")
            
            return True
        
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
