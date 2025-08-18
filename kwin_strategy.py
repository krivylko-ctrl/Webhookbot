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

    # ====== ДОБАВЛЕНО: утилиты выравнивания и порядка ======
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
    # =======================================================

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
    Точная копия ta.pivotlow(left, right), оценённая на ТЕКУЩЕМ баре,
    где сам pivot стоит на предыдущем баре (index=1 в self.candles_15m).
    Требует окно [0 .. 1+left] (right=1 → включает текущий бар 0).
    """
    need = left + right + 1  # кол-во баров вокруг пивота (включительно)
    # нам нужен ещё 1 бар, т.к. пивот на index=1 → всего need+1 баров (0..1+left)
    if len(self.candles_15m) < (need + 1):
        return False
    # окно индексов: 0 .. (1+left) включительно
    window_lows = [float(self.candles_15m[i]['low']) for i in range(0, 1 + left + 1)]
    pivot_val = float(self.candles_15m[1]['low'])
    return pivot_val == min(window_lows)

def _is_prev_pivot_high(self, left: int, right: int = 1) -> bool:
    """Точная копия ta.pivothigh(left, right) для пивота на предыдущем баре (index=1)."""
    need = left + right + 1
    if len(self.candles_15m) < (need + 1):
        return False
    window_highs = [float(self.candles_15m[i]['high']) for i in range(0, 1 + left + 1)]
    pivot_val = float(self.candles_15m[1]['high'])
    return pivot_val == max(window_highs)

def _detect_bull_sfp(self) -> bool:
    """
    Pine эквивалент:
    isBullSFP_15m = pivotlow(sfpLen, 1)
                    and low < low[sfpLen] and open > low[sfpLen] and close > low[sfpLen]
    """
    sfpLen = int(getattr(self.config, "sfp_len", 2))
    # нужно как минимум sfpLen-сдвиг + текущий + предыдущий → sfpLen+1+1 баров
    if len(self.candles_15m) < (sfpLen + 2):
        return False

    curr = self.candles_15m[0]
    ref_low = float(self.candles_15m[sfpLen]['low'])  # low[sfpLen] в терминах Pine

    cond_pivot = self._is_prev_pivot_low(sfpLen, right=1)
    cond_break = float(curr['low'])   < ref_low
    cond_close = float(curr['open'])  > ref_low and float(curr['close']) > ref_low

    if cond_pivot and cond_break and cond_close:
        if getattr(self.config, "use_sfp_quality", True):
            # качество — ровно как в Pine: тень в тиках и close-back
            return self._check_bull_sfp_quality_new(curr, {"low": ref_low})
        return True
    return False

def _detect_bear_sfp(self) -> bool:
    """
    Pine эквивалент:
    isBearSFP_15m = pivothigh(sfpLen, 1)
                    and high > high[sfpLen] and open < high[sfpLen] and close < high[sfpLen]
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
    Pine эквивалент quality-фильтра:
    bullWickDepth   = (low < ref_low) ? (ref_low - low) : 0
    bullCloseBackOK = bullWickDepth > 0 and (close - low) >= bullWickDepth * closeBackPct
    плюс порог по тик-ам: wickMinTicks * mTick
    """
    ref_low = float(pivot['low'])
    low     = float(current['low'])
    close   = float(current['close'])

    wick_depth = max(ref_low - low, 0.0)
    m_tick = float(getattr(self, "tick_size", 0.01))
    if m_tick <= 0:
        m_tick = 0.01
    wick_ticks = wick_depth / m_tick
    if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
        return False

    close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
    required_close_back = wick_depth * close_back_pct
    return (close - low) >= required_close_back

def _check_bear_sfp_quality_new(self, current: dict, pivot: dict) -> bool:
    """
    Pine эквивалент quality-фильтра для шорта:
    bearWickDepth   = (high > ref_high) ? (high - ref_high) : 0
    bearCloseBackOK = bearWickDepth > 0 and (high - close) >= bearWickDepth * closeBackPct
    и порог по тик-ам.
    """
    ref_high = float(pivot['high'])
    high     = float(current['high'])
    close    = float(current['close'])

    wick_depth = max(high - ref_high, 0.0)
    m_tick = float(getattr(self, "tick_size", 0.01))
    if m_tick <= 0:
        m_tick = 0.01
    wick_ticks = wick_depth / m_tick
    if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
        return False

    close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
    required_close_back = wick_depth * close_back_pct
    return (high - close) >= required_close_back
    
# ======== /PINE-EXACT SFP DETECTION ========
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
        except TypeError:
            # пробуем snake_case стиль
            return self.api.place_order(
                symbol=self.symbol,
                side=side_lo,
                order_type="market",
                qty=qty,
                stop_loss=stop_loss
            )

    def _process_long_entry(self):
        """Обработка входа в лонг"""
        try:
            current_price = self._get_current_price()
            if not current_price:
                return
            if len(self.candles_15m) < 2:
                return
            stop_loss = float(self.candles_15m[1]['low'])
            entry_price = float(current_price)
            quantity = self._calculate_position_size(entry_price, stop_loss, "long")
            if not quantity:
                return
            stop_size = entry_price - stop_loss
            take_profit = entry_price + stop_size * self.config.risk_reward
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return

            order_result = self._place_market_order("long", quantity, stop_loss=stop_loss)
            if order_result is None:
                return

            # ===== ТОЧЕЧНАЯ ПРАВКА: время входа = время текущего закрытого бара =====
            bar_ts_ms = self._current_bar_ts_ms()

            trade_data = {
                'symbol': self.symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': float(quantity),
                'entry_time': datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                'status': 'open'
            }
            self.db.save_trade(trade_data)

            self.state.set_position({
                'symbol': self.symbol,
                'direction': 'long',
                'size': float(quantity),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open',
                'armed': not self.config.use_arm_after_rr,
                'entry_time_ts': bar_ts_ms  # ← сохраняем штамп бара
            })
            # важно: исключаем второй вход на том же баре
            self.can_enter_long = False
            self.can_enter_short = False
            print(f"Long entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            print(f"Error processing long entry: {str(e) if e else 'Unknown error'}")

    def _process_short_entry(self):
        """Обработка входа в шорт"""
        try:
            current_price = self._get_current_price()
            if not current_price:
                return
            if len(self.candles_15m) < 2:
                return
            stop_loss = float(self.candles_15m[1]['high'])
            entry_price = float(current_price)
            quantity = self._calculate_position_size(entry_price, stop_loss, "short")
            if not quantity:
                return
            stop_size = stop_loss - entry_price
            take_profit = entry_price - stop_size * self.config.risk_reward
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return

            order_result = self._place_market_order("short", quantity, stop_loss=stop_loss)
            if order_result is None:
                return

            # ===== ТОЧЕЧНАЯ ПРАВКА: время входа = время текущего закрытого бара =====
            bar_ts_ms = self._current_bar_ts_ms()

            trade_data = {
                'symbol': self.symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': float(quantity),
                'entry_time': datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                'status': 'open'
            }
            self.db.save_trade(trade_data)

            self.state.set_position({
                'symbol': self.symbol,
                'direction': 'short',
                'size': float(quantity),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open',
                'armed': not self.config.use_arm_after_rr,
                'entry_time_ts': bar_ts_ms  # ← сохраняем штамп бара
            })
            # важно: исключаем второй вход на том же баре
            self.can_enter_short = False
            self.can_enter_long = False
            print(f"Short entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            print(f"Error processing short entry: {str(e) if e else 'Unknown error'}")

    def _get_current_price(self) -> Optional[float]:
        """Получить текущую цену"""
        try:
            if not self.api:
                return None
            ticker = self.api.get_ticker(self.symbol) or {}
            # используем mark_price, если доступен, иначе last_price
            price = ticker.get('mark_price') or ticker.get('last_price')
            if price is not None:
                return float(price)
        except Exception as e:
            print(f"Error getting current price: {str(e) if e else 'Unknown error'}")
        return None

    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        """Расчет размера позиции (qty в базовом активе, напр. ETH)"""
        try:
            equity = self.state.get_equity()
            if equity is None or equity <= 0:
                return None
            risk_amount = equity * (self.config.risk_pct / 100.0)
            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None
            quantity = risk_amount / stop_size  # ← qty в ETH
            quantity = qty_round(quantity, self.qty_step)
            if self.config.limit_qty_enabled:
                quantity = min(quantity, self.config.max_qty_manual)
            if quantity < self.min_order_qty:
                return None
            return float(quantity)
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float,
                                    take_profit: float, quantity: float) -> bool:
    """
    Pine-эквивалент okTrade:
      okTrade = qty > 0
                and qty >= minOrderQty
                and expNetPnL >= minNetProfit
    """
    try:
        if quantity is None:
            return False

        qty = float(quantity)
        if qty <= 0:
            return False

        min_order_qty = float(getattr(self.config, "min_order_qty", 0.01))
        if qty < min_order_qty:
            return False

        taker = float(getattr(self.config, "taker_fee_rate", 0.00055))
        gross = abs(float(take_profit) - float(entry_price)) * qty
        fees  = float(entry_price) * qty * taker * 2.0  # entry + exit
        net   = gross - fees

        min_net_profit = float(getattr(self.config, "min_net_profit", 1.2))
        return net >= min_net_profit
    except Exception as e:
        print(f"Error validating position: {e}")
        return False
        
    def _is_in_backtest_window(self, current_time: datetime) -> bool:
        print("WARNING: Используется устаревший метод _is_in_backtest_window, нужен UTC вариант")
        start_date = current_time - timedelta(days=self.config.days_back)
        return current_time >= start_date

    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        """UTC-полночь как в Pine Script"""
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=self.config.days_back)
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

    def _update_smart_trailing(self, position: Dict):
        """НОВЫЙ smart trailing с Bar High/Low и Arm механизмом"""
        try:
            if not self.config.enable_smart_trail:
                return
            direction = position.get('direction')
            entry_price = position.get('entry_price')
            current_sl = position.get('stop_loss')
            if not all([direction, entry_price, current_sl]):
                return
            current_price = self._get_current_price()
            if not current_price:
                return
            armed = position.get('armed', not self.config.use_arm_after_rr)
            if not armed and self.config.use_arm_after_rr:
                if direction == 'long':
                    unrealized_profit = current_price - entry_price
                    required_move = (entry_price - current_sl) * getattr(self.config, 'arm_rr', 0.5)
                    armed = unrealized_profit >= required_move
                else:
                    unrealized_profit = entry_price - current_price
                    required_move = (current_sl - entry_price) * getattr(self.config, 'arm_rr', 0.5)
                    armed = unrealized_profit >= required_move
                if armed:
                    position['armed'] = True
                    self.state.set_position(position)
                    print(f"Position ARMED at {self.config.arm_rr}R")
            if not armed:
                return
            if getattr(self.config, 'use_bar_trail', False):
                new_sl = self._calculate_bar_trailing_stop(str(direction), float(current_sl))
            else:
                new_sl = self._calculate_percentage_trailing_stop(str(direction), current_price, float(current_sl))
            if new_sl and new_sl != current_sl:
                if direction == 'long' and new_sl > current_sl:
                    self._update_stop_loss(position, new_sl)
                elif direction == 'short' and new_sl < current_sl:
                    self._update_stop_loss(position, new_sl)
        except Exception as e:
            print(f"Error in smart trailing: {e}")

    def _calculate_bar_trailing_stop(self, direction: str, current_sl: float) -> Optional[float]:
        try:
            lookback = getattr(self.config, 'trail_lookback', 50) or 50
            if len(self.candles_15m) < lookback:
                return current_sl
            history_bars = self.candles_15m[1:lookback+1]
            if direction == 'long':
                min_low = min(bar['low'] for bar in history_bars)
                new_sl = max(min_low, current_sl)
            else:
                max_high = max(bar['high'] for bar in history_bars)
                new_sl = min(max_high, current_sl)
            return new_sl
        except Exception as e:
            print(f"Error calculating bar trailing stop: {e}")
            return current_sl

    def _calculate_percentage_trailing_stop(self, direction: str, current_price: float, current_sl: float) -> Optional[float]:
        try:
            trail_pct = getattr(self.config, 'trailing_perc', 0.5) / 100.0
            if direction == 'long':
                trail_distance = current_price * trail_pct
                new_sl = current_price - trail_distance
                return max(new_sl, current_sl)
            else:
                trail_distance = current_price * trail_pct
                new_sl = current_price + trail_distance
                return min(new_sl, current_sl)
        except Exception as e:
            print(f"Error calculating percentage trailing stop: {e}")
            return current_sl

    def _update_stop_loss(self, position: Dict, new_sl: float):
        """Обновление стоп-лосса"""
        try:
            if not self.api or not hasattr(self.api, "modify_order"):
                print("API not available for updating stop loss")
                return
            result = self.api.modify_order(
                symbol=position['symbol'],
                stop_loss=new_sl
            )
            if result:
                position['stop_loss'] = new_sl
                self.state.set_position(position)
                print(f"Trailing SL updated: {new_sl:.4f}")
        except Exception as e:
            print(f"Error updating stop loss: {e}")

    def process_trailing(self):
        """LEGACY метод для обратной совместимости"""
        try:
            current_position = self.state.get_current_position()
            if current_position and current_position.get('status') == 'open':
                self._update_smart_trailing(current_position)
        except Exception as e:
            print(f"Error processing trailing: {e}")

    def run_cycle(self):
        """Основной цикл обработки с НОВОЙ Pine Script логикой"""
        try:
            if not self.candles_15m:
                return
            current_position = self.state.get_current_position()
            if current_position and current_position.get('status') == 'open':
                self._update_smart_trailing(current_position)
            else:
                self.on_bar_close()
        except Exception as e:
            print(f"Error in run_cycle: {str(e) if e else 'Unknown error'}")

    def _update_equity(self):
        """Обновление equity"""
        try:
            if not self.api:
                return
            wallet = self.api.get_wallet_balance()
            if wallet and wallet.get("list"):
                for account in wallet["list"]:
                    if account.get("accountType") == "SPOT":
                        for coin in account.get("coin", []):
                            if coin.get("coin") == "USDT":
                                equity = float(coin.get("equity", 0))
                                self.state.set_equity(equity)
                                self.db.save_equity_snapshot(equity)
                                break
        except Exception as e:
            print(f"Error updating equity: {str(e) if e else 'Unknown error'}")
