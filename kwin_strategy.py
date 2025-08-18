import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import math

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from utils import price_round, qty_round
from database import Database


class KWINStrategy:
    """Основная логика стратегии KWIN"""

    def __init__(self, config: Config, bybit_api, state_manager: StateManager, db: Database):
        self.config = config
        self.api = bybit_api
        self.state = state_manager
        self.db = db
        self.trail_engine = TrailEngine(config, state_manager, bybit_api)
        self.analytics = TradingAnalytics()

        # Внутренние данные (crash-safe state)
        self.candles_15m = []
        self.candles_1m = []
        self.candles_1h = []  # ← добавлено
        self.last_processed_time = None
        self.last_processed_bar_ts = 0  # Для восстановления после crash

        # Trade state (зафиксированные значения для RR расчета)
        self.entry_price = None
        self.entry_sl = None
        self.trade_id = None
        self.armed = False  # ArmRR статус

        # Версионирование стратегии
        self.strategy_version = "2.0.1"

        # Состояние входов
        self.can_enter_long = True
        self.can_enter_short = True
        self.last_candle_close_15m = None

        # Инструмент (используем из конфига)
        self.symbol = self.config.symbol
        self.tick_size = 0.01
        self.qty_step = 0.01
        self.min_order_qty = 0.01

        # Получаем информацию об инструменте
        self._init_instrument_info()

        # Критичный патч: нормализация close_back_pct к диапазону [0..1]
        if self.config.close_back_pct > 1.0:
            self.config.close_back_pct = self.config.close_back_pct / 100.0
        elif self.config.close_back_pct < 0.0:
            self.config.close_back_pct = 0.0

    # ====== утилиты выравнивания и порядка ======
    def _align_15m_ms(self, ts_ms: int) -> int:
        """Выравнивание метки времени к границе 15 минут (мс)."""
        return (int(ts_ms) // 900_000) * 900_000

    def _ensure_15m_desc(self):
        """Новейшая свеча первой (индекс 0)."""
        if self.candles_15m and self.candles_15m[0].get("timestamp") is not None:
            self.candles_15m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    def _current_bar_ts_ms(self) -> int:
        """Штамп времени ТЕКУЩЕГО закрытого 15m бара (мс)."""
        return int(self.last_processed_bar_ts or 0)
    # ============================================

    def _init_instrument_info(self):
        """Инициализация информации об инструменте"""
        try:
            if self.api:
                if hasattr(self.api, 'set_market_type') and hasattr(self.config, 'market_type'):
                    self.api.set_market_type(self.config.market_type)
                if hasattr(self.api, 'get_instruments_info'):
                    info = self.api.get_instruments_info(self.symbol)
                    if info:
                        if 'priceFilter' in info:
                            self.tick_size = float(info['priceFilter']['tickSize'])
                        if 'lotSizeFilter' in info:
                            self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                            self.min_order_qty = float(info['lotSizeFilter']['minOrderQty'])
        except Exception as e:
            print(f"Error initializing instrument info: {e}")

        # Если биржа дала фильтры - синхронизируем в конфиг
        if self.qty_step and self.qty_step > 0:
            self.config.qty_step = self.qty_step
        if self.min_order_qty and self.min_order_qty > 0:
            self.config.min_order_qty = self.min_order_qty

        # Fallback
        if not self.tick_size or self.tick_size <= 0:
            self.tick_size = 0.01
        if not self.qty_step or self.qty_step <= 0:
            self.qty_step = 0.01
        if not self.min_order_qty or self.min_order_qty <= 0:
            self.min_order_qty = 0.01

    def on_bar_close_15m(self, candle: Dict):
        """ТОЧНАЯ синхронизация с Pine Script: обработка только закрытых 15м баров"""
        try:
            # Новейший в начало (как в Pine)
            self.candles_15m.insert(0, candle)
            max_history = 200
            if len(self.candles_15m) > max_history:
                self.candles_15m = self.candles_15m[:max_history]

            # строгий порядок
            self._ensure_15m_desc()

            # выровненная метка бара и реентранси-гард
            current_bar_time = candle.get('start') or candle.get('open_time') or candle.get('timestamp')
            if not current_bar_time:
                return
            aligned_ts = self._align_15m_ms(int(current_bar_time))
            if self.last_processed_bar_ts == aligned_ts:
                return  # этот бар уже обрабатывали

            # сброс флагов входа на новом закрытом баре
            self.can_enter_long = True
            self.can_enter_short = True
            self.last_candle_close_15m = aligned_ts
            self.last_processed_bar_ts = aligned_ts
            try:
                print(f"[STRATEGY] New 15m bar: {float(candle['close']):.2f} at {aligned_ts}")
            except Exception:
                pass

            self.run_cycle()
        except Exception as e:
            print(f"Error in on_bar_close_15m: {e}")

    def on_bar_close_60m(self, candle: Dict):
        """Обработка закрытых 1ч баров для дополнительного анализа"""
        try:
            self.candles_1h.insert(0, candle)
            if len(self.candles_1h) > 100:
                self.candles_1h = self.candles_1h[:100]
        except Exception as e:
            print(f"Error in on_bar_close_60m: {e}")

    def on_bar_close_1m(self, candle: Dict):
        """Обработка 1м баров для дополнительного мониторинга"""
        pass

    def update_candles(self):
        """Обновление свечей с биржи"""
        try:
            if not self.api:
                return

            # 15m
            klines_15m = self.api.get_klines(self.symbol, "15", 100)
            if klines_15m:
                for k in klines_15m:
                    ts = k.get("timestamp")
                    if ts is not None and ts < 1_000_000_000_000:  # секунды -> мс
                        k["timestamp"] = int(ts * 1000)
                # строгий порядок
                klines_15m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                self.candles_15m = klines_15m

            # 1m
            klines_1m = self.api.get_klines(self.symbol, "1", 10)
            if klines_1m:
                self.candles_1m = klines_1m

            if self.candles_15m:
                current_candle = self.candles_15m[0]
                current_timestamp = int(current_candle.get('timestamp', 0))
                if current_timestamp and current_timestamp < 1_000_000_000_000:
                    current_timestamp *= 1000
                aligned_timestamp = self._align_15m_ms(current_timestamp)

                # ГАРД: один вызов на закрытый бар
                if self.last_processed_bar_ts == aligned_timestamp:
                    return

                self.last_processed_bar_ts = aligned_timestamp
                self.can_enter_long = True
                self.can_enter_short = True
                self.on_bar_close()
        except Exception as e:
            print(f"Error updating candles: {str(e) if e else 'Unknown error'}")

    def on_bar_close(self):
        """Обработка новой 15-минутной свечи"""
        if len(self.candles_15m) < self.config.sfp_len + 2:
            return

        bull_sfp = self._detect_bull_sfp()
        bear_sfp = self._detect_bear_sfp()

        current_ts = self.candles_15m[0]['timestamp']  # ms
        if not self._is_in_backtest_window_utc(current_ts):
            return

        current_position = self.state.get_current_position()
        if current_position and current_position.get('status') == 'open':
            return

        if bull_sfp and self.can_enter_long:
            self._process_long_entry()
        elif bear_sfp and self.can_enter_short:
            self._process_short_entry()

    # ======== PINE-EXACT SFP DETECTION (pivotlow/pivothigh) ========

    def _is_prev_pivot_low(self, left: int, right: int = 1) -> bool:
        """
        ta.pivotlow(left,right) на текущем баре, где сам pivot на предыдущем (index=1).
        Требует окна [0 .. 1+left].
        """
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        window_lows = [float(self.candles_15m[i]['low']) for i in range(0, 1 + left + 1)]
        pivot_val = float(self.candles_15m[1]['low'])
        return pivot_val == min(window_lows)

    def _is_prev_pivot_high(self, left: int, right: int = 1) -> bool:
        """ta.pivothigh(left,right) для pivot на предыдущем баре (index=1)."""
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        window_highs = [float(self.candles_15m[i]['high']) for i in range(0, 1 + left + 1)]
        pivot_val = float(self.candles_15m[1]['high'])
        return pivot_val == max(window_highs)

    def _detect_bull_sfp(self) -> bool:
        """
        isBullSFP_15m = pivotlow(sfpLen,1)
                        и low < low[sfpLen] и open > low[sfpLen] и close > low[sfpLen]
        """
        sfpLen = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (sfpLen + 2):
            return False

        curr = self.candles_15m[0]
        ref_low = float(self.candles_15m[sfpLen]['low'])  # low[sfpLen]

        cond_pivot = self._is_prev_pivot_low(sfpLen, right=1)
        cond_break = float(curr['low']) < ref_low
        cond_close = float(curr['open']) > ref_low and float(curr['close']) > ref_low

        if cond_pivot and cond_break and cond_close:
            if getattr(self.config, "use_sfp_quality", True):
                return self._check_bull_sfp_quality_new(curr, {"low": ref_low})
            return True
        return False

    def _detect_bear_sfp(self) -> bool:
        """
        isBearSFP_15m = pivothigh(sfpLen,1)
                        и high > high[sfpLen] и open < high[sfpLen] и close < high[sfpLen]
        """
        sfpLen = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (sfpLen + 2):
            return False

        curr = self.candles_15m[0]
        ref_high = float(self.candles_15m[sfpLen]['high'])  # high[sfpLen]

        cond_pivot = self._is_prev_pivot_high(sfpLen, right=1)
        cond_break = float(curr['high']) > ref_high
        cond_close = float(curr['open']) < ref_high and float(curr['close']) < ref_high

        if cond_pivot and cond_break and cond_close:
            if getattr(self.config, "use_sfp_quality", True):
                return self._check_bear_sfp_quality_new(curr, {"high": ref_high})
            return True
        return False

    def _check_bull_sfp_quality_new(self, current: dict, pivot: dict) -> bool:
        """
        bullWickDepth   = max(ref_low - low, 0)
        bullCloseBackOK = (close - low) >= bullWickDepth * closeBackPct
        + порог по тикам: wickMinTicks * mTick
        """
        ref_low = float(pivot['low'])
        low     = float(current['low'])
        close   = float(current['close'])

        wick_depth = max(ref_low - low, 0.0)
        m_tick = float(getattr(self, "tick_size", 0.01)) or 0.01
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False

        close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
        required_close_back = wick_depth * close_back_pct
        return (close - low) >= required_close_back

    def _check_bear_sfp_quality_new(self, current: dict, pivot: dict) -> bool:
        """
        bearWickDepth   = max(high - ref_high, 0)
        bearCloseBackOK = (high - close) >= bearWickDepth * closeBackPct
        + порог по тикам.
        """
        ref_high = float(pivot['high'])
        high     = float(current['high'])
        close    = float(current['close'])

        wick_depth = max(high - ref_high, 0.0)
        m_tick = float(getattr(self, "tick_size", 0.01)) or 0.01
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False

        close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
        required_close_back = wick_depth * close_back_pct
        return (high - close) >= required_close_back

    # ======== /PINE-EXACT SFP DETECTION ========

    # ---------- MARKET-ордер ----------
    def _place_market_order(self, direction: str, quantity: float, stop_loss: Optional[float] = None):
        if not self.api or not hasattr(self.api, 'place_order'):
            print("API not available for placing order")
            return None
        side_up = "Buy" if direction == "long" else "Sell"
        side_lo = "buy" if direction == "long" else "sell"
        qty = float(quantity)
        # пробуем стиль v5: orderType
        try:
            return self.api.place_order(
                symbol=self.symbol,
                side=side_up,
                orderType="Market",
                qty=qty,
                stop_loss=stop_loss
            )
    ...
