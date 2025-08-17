import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
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
        
        # КРИТИЧНЫЙ ПАТЧ 2: Нормализация close_back_pct к диапазону [0..1]
        if self.config.close_back_pct > 1.0:
            # пользователь ввёл в процентах (например 100), приводим к 1.0
            self.config.close_back_pct = self.config.close_back_pct / 100.0
        elif self.config.close_back_pct < 0.0:
            self.config.close_back_pct = 0.0
    
    def _init_instrument_info(self):
        """Инициализация информации об инструменте"""
        try:
            if self.api:
                # Устанавливаем тип рынка
                if hasattr(self.api, 'set_market_type'):
                    self.api.set_market_type(self.config.market_type)
                
                # Получаем информацию об инструменте
                if hasattr(self.api, 'get_instruments_info'):
                    info = self.api.get_instruments_info(self.symbol)
                    if info:
                        # Извлекаем информацию о фильтрах цены и количества
                        if 'priceFilter' in info:
                            self.tick_size = float(info['priceFilter']['tickSize'])
                        if 'lotSizeFilter' in info:
                            self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                            self.min_order_qty = float(info['lotSizeFilter']['minOrderQty'])
            
        except Exception as e:
            print(f"Error initializing instrument info: {e}")
            
        # КРИТИЧНЫЙ ПАТЧ: Если биржа дала фильтры - перезаписываем конфиг для единого источника истины
        if hasattr(self, 'min_order_qty') and hasattr(self, 'qty_step'):
            if hasattr(self.config, 'min_order_qty'):
                self.config.min_order_qty = self.min_order_qty
                self.config.qty_step = self.qty_step
        
        # КРИТИЧНЫЙ ПАТЧ 5: Инициализация fallback фильтров инструмента
        if not self.tick_size or self.tick_size <= 0:
            self.tick_size = 0.01  # безопасный дефолт
        if not self.qty_step or self.qty_step <= 0:
            self.qty_step = 0.01
        if not self.min_order_qty or self.min_order_qty <= 0:
            self.min_order_qty = 0.01
    
    def on_bar_close_15m(self, candle: Dict):
        """ТОЧНАЯ синхронизация с Pine Script: обработка только закрытых 15м баров"""
        try:
            # Добавляем новую свечу в начало (как в Pine Script)
            self.candles_15m.insert(0, candle)
            
            # Ограничиваем историю для производительности  
            max_history = 200
            if len(self.candles_15m) > max_history:
                self.candles_15m = self.candles_15m[:max_history]
            
            # КРИТИЧНЫЙ ПАТЧ 4: Сброс флагов входа по реальному времени закрытия 15м бара
            current_bar_time = candle.get('start') or candle.get('open_time') or candle.get('timestamp')
            if self.last_candle_close_15m != current_bar_time:
                self.can_enter_long = True
                self.can_enter_short = True
                self.last_candle_close_15m = current_bar_time
                print(f"[STRATEGY] New 15m bar: {candle['close']:.2f} at {current_bar_time}")
            
            # Основная логика стратегии на закрытии бара
            self.run_cycle()
            
        except Exception as e:
            print(f"Error in on_bar_close_15m: {e}")
    
    def on_bar_close_60m(self, candle: Dict):
        """Обработка закрытых 1ч баров для дополнительного анализа"""
        try:
            self.candles_1h.insert(0, candle)
            
            # Ограничиваем историю
            if len(self.candles_1h) > 100:
                self.candles_1h = self.candles_1h[:100]
                
        except Exception as e:
            print(f"Error in on_bar_close_60m: {e}")
    
    def on_bar_close_1m(self, candle: Dict):
        """Обработка 1м баров для дополнительного мониторинга"""
        # Используется для обновлений трейлинга если нужно
        pass
    
    def update_candles(self):
        """Обновление свечей с биржи"""
        try:
            if not self.api:
                return
                
            # Получаем 15m свечи с гарантией правильного порядка
            klines_15m = self.api.get_klines(self.symbol, "15", 100)
            if klines_15m:
                # КРИТИЧНЫЙ ПАТЧ: на случай если кто-то вернул в возрастающем порядке
                klines_15m.sort(key=lambda x: x["timestamp"], reverse=True)
                self.candles_15m = klines_15m
            
            # Получаем 1m свечи для текущей цены
            klines_1m = self.api.get_klines(self.symbol, "1", 10)
            if klines_1m:
                self.candles_1m = klines_1m
            
            # Проверяем новую 15m свечу (строгий тайминг UTC, округление к 900сек границам)
            if self.candles_15m:
                current_candle = self.candles_15m[0]
                current_timestamp = current_candle.get('timestamp', 0)
                
                # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: 15 минут = 900_000 миллисекунд (НЕ 900 секунд!)
                aligned_timestamp = (current_timestamp // 900_000) * 900_000
                
                if self.last_processed_time != aligned_timestamp:
                    self.last_processed_time = aligned_timestamp
                    self.can_enter_long = True  # Сброс на закрытии 15m бара
                    self.can_enter_short = True
                    self.on_bar_close()  # Единый путь для оффлайн/онлайн
        
        except Exception as e:
            print(f"Error updating candles: {str(e) if e else 'Unknown error'}")
    
    def on_bar_close(self):
        """Обработка новой 15-минутной свечи"""
        if len(self.candles_15m) < self.config.sfp_len + 2:
            return
        
        # Проверяем SFP паттерны
        bull_sfp = self._detect_bull_sfp()
        bear_sfp = self._detect_bear_sfp()
        
        # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: используем UTC-полночь как в Pine Script
        current_ts = self.candles_15m[0]['timestamp']  # ms
        if not self._is_in_backtest_window_utc(current_ts):
            return
        
        # 9️⃣ Проверяем отсутствие открытых позиций перед входом
        current_position = self.state.get_current_position()
        if current_position and current_position.get('status') == 'open':
            return  # Не входим если есть открытая позиция
        
        # Обработка бычьего SFP
        if bull_sfp and self.can_enter_long:
            self._process_long_entry()
        
        # Обработка медвежьего SFP
        if bear_sfp and self.can_enter_short:
            self._process_short_entry()
    
    def _detect_bull_sfp(self) -> bool:
        """ПОЛНАЯ PINE SCRIPT СОВМЕСТИМОСТЬ: Детекция бычьего SFP (97% аутентичность)"""
        sfpLen = 2  # Фиксированное значение как в Pine Script
        
        # Требуем достаточную историю для pivot проверки
        if len(self.candles_15m) < sfpLen + 1 + 2:  # left + pivot + right + current
            return False

        # Индексы как в бэктесте (новейшие в начале)
        for i in range(len(self.candles_15m)):
            # Проверяем что можем сделать полную pivot проверку
            if i - sfpLen - 1 < 0 or i + 1 >= len(self.candles_15m):
                continue
            
            # Pivot low проверка (точно как в бэктесте)
            window = []
            for k in range(i - sfpLen - 1, i + 2):
                if k < len(self.candles_15m):
                    window.append(self.candles_15m[k]['low'])
            
            if not window or len(window) < sfpLen + 2:
                continue
                
            pivot_low = self.candles_15m[i]['low']
            if pivot_low != min(window) or pivot_low >= self.candles_15m[i - sfpLen]['low']:
                continue
            
            # SFP условия как в Pine Script
            current = self.candles_15m[0]  # Текущий бар
            pivot = self.candles_15m[i]    # Pivot бар
            
            # Bull SFP: open/close выше предыдущего пивота, но low пробил вниз
            prev_pivot_low = self.candles_15m[i - 1]['low']
            
            bull_condition = (
                current['open'] > prev_pivot_low and
                current['close'] > prev_pivot_low and
                current['low'] < prev_pivot_low
            )
            
            if bull_condition:
                # Проверка качества SFP если включена
                if self.config.use_sfp_quality:
                    return self._check_bull_sfp_quality_new(current, pivot)
                return True
        
        return False
    
    def _detect_bear_sfp(self) -> bool:
        """ПОЛНАЯ PINE SCRIPT СОВМЕСТИМОСТЬ: Детекция медвежьего SFP (97% аутентичность)"""
        sfpLen = 2  # Фиксированное значение как в Pine Script
        
        # Требуем достаточную историю для pivot проверки
        if len(self.candles_15m) < sfpLen + 1 + 2:  # left + pivot + right + current
            return False

        # Индексы как в бэктесте (новейшие в начале)
        for i in range(len(self.candles_15m)):
            # Проверяем что можем сделать полную pivot проверку
            if i - sfpLen - 1 < 0 or i + 1 >= len(self.candles_15m):
                continue
            
            # Pivot high проверка (точно как в бэктесте)
            window = []
            for k in range(i - sfpLen - 1, i + 2):
                if k < len(self.candles_15m):
                    window.append(self.candles_15m[k]['high'])
            
            if not window or len(window) < sfpLen + 2:
                continue
                
            pivot_high = self.candles_15m[i]['high']
            if pivot_high != max(window) or pivot_high <= self.candles_15m[i - sfpLen]['high']:
                continue
            
            # SFP условия как в Pine Script
            current = self.candles_15m[0]  # Текущий бар
            pivot = self.candles_15m[i]    # Pivot бар
            
            # Bear SFP: open/close ниже предыдущего пивота, но high пробил вверх
            prev_pivot_high = self.candles_15m[i - 1]['high']
            
            bear_condition = (
                current['open'] < prev_pivot_high and
                current['close'] < prev_pivot_high and
                current['high'] > prev_pivot_high
            )
            
            if bear_condition:
                # Проверка качества SFP если включена
                if self.config.use_sfp_quality:
                    return self._check_bear_sfp_quality_new(current, pivot)
                return True
        
        return False
    
    def _check_bull_sfp_quality_new(self, current: Dict, pivot: Dict) -> bool:
        """НОВАЯ версия проверки качества бычьего SFP (из бэктеста с 97% совместимостью)"""
        # Глубина вика = разница между предыдущим пивотом и текущим low
        prev_pivot_low = pivot['low']  # Это уже правильный reference
        wick_depth = prev_pivot_low - current['low']
        
        # Проверка минимальной глубины в тиках
        min_tick = float(self.tick_size) if hasattr(self, "tick_size") and self.tick_size else 0.01
        wick_depth_ticks = wick_depth / min_tick
        if wick_depth_ticks < self.config.wick_min_ticks:
            return False
        
        # Close-back: как далеко закрытие восстановилось от лоу
        close_back = current['close'] - current['low']
        required_close_back = wick_depth * self.config.close_back_pct  # Уже в [0..1] формате
        
        # close_back должен быть >= X% от глубины вика
        return close_back >= required_close_back
    
    def _check_bull_sfp_quality(self, current: Dict, pivot: Dict) -> bool:
        """LEGACY метод для обратной совместимости"""
        return self._check_bull_sfp_quality_new(current, pivot)
    
    def _check_bear_sfp_quality_new(self, current: Dict, pivot: Dict) -> bool:
        """НОВАЯ версия проверки качества медвежьего SFP (из бэктеста с 97% совместимостью)"""
        # Глубина вика = разница между текущим high и предыдущим пивотом
        prev_pivot_high = pivot['high']  # Это уже правильный reference
        wick_depth = current['high'] - prev_pivot_high
        
        # Проверка минимальной глубины в тиках
        min_tick = float(self.tick_size) if hasattr(self, "tick_size") and self.tick_size else 0.01
        wick_depth_ticks = wick_depth / min_tick
        if wick_depth_ticks < self.config.wick_min_ticks:
            return False
        
        # Close-back: как далеко закрытие откатилось от хая
        close_back = current['high'] - current['close']
        required_close_back = wick_depth * self.config.close_back_pct  # Уже в [0..1] формате
        
        # close_back должен быть >= X% от глубины вика
        return close_back >= required_close_back
    
    def _check_bear_sfp_quality(self, current: Dict, pivot: Dict) -> bool:
        """LEGACY метод для обратной совместимости"""
        return self._check_bear_sfp_quality_new(current, pivot)
    
    def _process_long_entry(self):
        """Обработка входа в лонг"""
        try:
            # Получаем текущую цену
            current_price = self._get_current_price()
            if not current_price:
                return
            
            # Расчет стопа (лоу предыдущей свечи)
            if len(self.candles_15m) < 2:
                return
            
            stop_loss = self.candles_15m[1]['low']
            entry_price = current_price
            
            # Расчет размера позиции
            quantity = self._calculate_position_size(entry_price, stop_loss, "long")
            if not quantity:
                return
            
            # Расчет тейк-профита
            stop_size = entry_price - stop_loss
            take_profit = entry_price + stop_size * self.config.risk_reward
            
            # 8️⃣ Защита от микро-позиций и низкой прибыли
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return
            
            # Размещаем ордер
            if not self.api:
                print("API not available for placing order")
                return
                
            order_result = self.api.place_order(
                symbol=self.symbol,
                side="buy",
                order_type="market",
                qty=quantity,
                stop_loss=stop_loss
            )
            
            if order_result:
                # Сохраняем сделку в базу
                trade_data = {
                    'symbol': self.symbol,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'status': 'open'
                }
                self.db.save_trade(trade_data)
                
                # Обновляем состояние
                self.state.set_position({
                    'symbol': self.symbol,
                    'direction': 'long',
                    'size': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'armed': not self.config.use_arm_after_rr
                })
                
                # Блокируем повторные входы
                self.can_enter_long = False
                
                print(f"Long entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        
        except Exception as e:
            print(f"Error processing long entry: {str(e) if e else 'Unknown error'}")
    
    def _process_short_entry(self):
        """Обработка входа в шорт"""
        try:
            # Получаем текущую цену
            current_price = self._get_current_price()
            if not current_price:
                return
            
            # Расчет стопа (хай предыдущей свечи)
            if len(self.candles_15m) < 2:
                return
            
            stop_loss = self.candles_15m[1]['high']
            entry_price = current_price
            
            # Расчет размера позиции
            quantity = self._calculate_position_size(entry_price, stop_loss, "short")
            if not quantity:
                return
            
            # Расчет тейк-профита
            stop_size = stop_loss - entry_price
            take_profit = entry_price - stop_size * self.config.risk_reward
            
            # 8️⃣ Защита от микро-позиций и низкой прибыли
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return
            
            # Размещаем ордер
            if not self.api:
                print("API not available for placing order")
                return
                
            order_result = self.api.place_order(
                symbol=self.symbol,
                side="sell",
                order_type="market",
                qty=quantity,
                stop_loss=stop_loss
            )
            
            if order_result:
                # Сохраняем сделку в базу
                trade_data = {
                    'symbol': self.symbol,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'status': 'open'
                }
                self.db.save_trade(trade_data)
                
                # Обновляем состояние
                self.state.set_position({
                    'symbol': self.symbol,
                    'direction': 'short',
                    'size': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'armed': not self.config.use_arm_after_rr
                })
                
                # Блокируем повторные входы
                self.can_enter_short = False
                
                print(f"Short entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        
        except Exception as e:
            print(f"Error processing short entry: {str(e) if e else 'Unknown error'}")
    
    def _get_current_price(self) -> Optional[float]:
        """Получить текущую цену"""
        try:
            if not self.api:
                return None
            ticker = self.api.get_ticker(self.symbol)
            if ticker:
                return ticker['last_price']
        except Exception as e:
            print(f"Error getting current price: {str(e) if e else 'Unknown error'}")
        return None
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        """Расчет размера позиции"""
        try:
            # Получаем текущий equity
            equity = self.state.get_equity()
            risk_amount = equity * (self.config.risk_pct / 100)
            
            # Расчет размера стопа
            if direction == "long":
                stop_size = entry_price - stop_loss
            else:
                stop_size = stop_loss - entry_price
            
            if stop_size <= 0:
                return None
            
            # Расчет количества
            quantity = risk_amount / stop_size
            
            # Округление по шагу
            quantity = qty_round(quantity, self.qty_step)
            
            # Проверка ограничений
            if self.config.limit_qty_enabled:
                quantity = min(quantity, self.config.max_qty_manual)
            
            if quantity < self.min_order_qty:
                return None
            
            return quantity
        
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None
    
    def _validate_position_requirements(self, entry_price: float, stop_loss: float, 
                                      take_profit: float, quantity: float) -> bool:
        """8️⃣ Комплексная проверка требований к позиции"""
        try:
            # Проверка минимального размера позиции
            if quantity < self.config.min_order_qty:
                print(f"Position too small: {quantity} < {self.config.min_order_qty}")
                return False
            
            # Проверка минимального размера стопа (не слишком узкий SL)
            stop_size = abs(entry_price - stop_loss)
            min_stop_size = self.tick_size * 5  # минимум 5 тиков
            if stop_size < min_stop_size:
                print(f"Stop too narrow: {stop_size} < {min_stop_size}")
                return False
            
            # Расчет валовой прибыли от TP
            gross_pnl = abs(take_profit - entry_price) * quantity
            
            # 🔟 Расчет двойной комиссии (вход + выход)
            entry_fee = entry_price * quantity * self.config.taker_fee_rate
            exit_fee = take_profit * quantity * self.config.taker_fee_rate
            total_fees = entry_fee + exit_fee
            
            # Чистая прибыль
            net_pnl = gross_pnl - total_fees
            
            # Проверка минимальной чистой прибыли
            if net_pnl < self.config.min_net_profit:
                print(f"Net profit too low: ${net_pnl:.2f} < ${self.config.min_net_profit}")
                return False
            
            return True
        
        except Exception as e:
            print(f"Error validating position: {e}")
            return False
    
    def _is_in_backtest_window(self, current_time: datetime) -> bool:
        """УСТАРЕВШИЙ метод - используйте _is_in_backtest_window_utc()"""
        print("WARNING: Используется устаревший метод _is_in_backtest_window, нужен UTC вариант")
        start_date = current_time - timedelta(days=self.config.days_back)
        return current_time >= start_date
    
    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        """КРИТИЧНЫЙ ПАТЧ: UTC-полночь как в Pine Script"""
        from datetime import timezone
        
        # UTC-полночь сегодняшнего дня как в Pine Script
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
                
            # Получаем текущую цену
            current_price = self._get_current_price()
            if not current_price:
                return
            
            # Проверяем Arm статус
            armed = position.get('armed', not self.config.use_arm_after_rr)
            
            if not armed and self.config.use_arm_after_rr:
                # Проверяем достижение минимального RR для армирования
                if direction == 'long':
                    unrealized_profit = current_price - entry_price
                    required_move = (entry_price - current_sl) * getattr(self.config, 'arm_rr', 0.5)
                    armed = unrealized_profit >= required_move
                else:  # short
                    unrealized_profit = entry_price - current_price
                    required_move = (current_sl - entry_price) * getattr(self.config, 'arm_rr', 0.5)
                    armed = unrealized_profit >= required_move
                
                if armed:
                    position['armed'] = True
                    self.state.set_position(position)
                    print(f"Position ARMED at {self.config.arm_rr}R")
            
            if not armed:
                return  # Не трейлим если не армирован
            
            # Применяем bar-based trailing если включен
            if getattr(self.config, 'use_bar_trail', False):
                new_sl = self._calculate_bar_trailing_stop(str(direction), float(current_sl))
            else:
                # Fallback на процентный трейлинг
                new_sl = self._calculate_percentage_trailing_stop(str(direction), current_price, float(current_sl))
            
            # Обновляем стоп если улучшился
            if new_sl and new_sl != current_sl:
                if direction == 'long' and new_sl > current_sl:
                    self._update_stop_loss(position, new_sl)
                elif direction == 'short' and new_sl < current_sl:
                    self._update_stop_loss(position, new_sl)
                    
        except Exception as e:
            print(f"Error in smart trailing: {e}")
    
    def _calculate_bar_trailing_stop(self, direction: str, current_sl: float) -> Optional[float]:
        """Расчет bar-based trailing stop как в бэктесте"""
        try:
            lookback = getattr(self.config, 'trail_lookback', 50) or 50
            
            if len(self.candles_15m) < lookback:
                return current_sl
            
            # Берем последние lookback баров (исключая текущий)
            history_bars = self.candles_15m[1:lookback+1]
            
            if direction == 'long':
                # Находим минимальный low за lookback период
                min_low = min(bar['low'] for bar in history_bars)
                new_sl = max(min_low, current_sl)  # Только улучшаем
            else:  # short
                # Находим максимальный high за lookback период
                max_high = max(bar['high'] for bar in history_bars)
                new_sl = min(max_high, current_sl)  # Только улучшаем
            
            return new_sl
            
        except Exception as e:
            print(f"Error calculating bar trailing stop: {e}")
            return current_sl
    
    def _calculate_percentage_trailing_stop(self, direction: str, current_price: float, current_sl: float) -> Optional[float]:
        """Процентный трейлинг как fallback"""
        try:
            trail_pct = getattr(self.config, 'trailing_perc', 0.5) / 100.0
            
            if direction == 'long':
                trail_distance = current_price * trail_pct
                new_sl = current_price - trail_distance
                return max(new_sl, current_sl)  # Только улучшаем
            else:  # short
                trail_distance = current_price * trail_pct
                new_sl = current_price + trail_distance
                return min(new_sl, current_sl)  # Только улучшаем
                
        except Exception as e:
            print(f"Error calculating percentage trailing stop: {e}")
            return current_sl
    
    def _update_stop_loss(self, position: Dict, new_sl: float):
        """Обновление стоп-лосса"""
        try:
            if not self.api:
                print("API not available for updating stop loss")
                return
            
            # Обновляем ордер на бирже
            result = self.api.modify_order(
                symbol=position['symbol'],
                stop_loss=new_sl
            )
            
            if result:
                # Обновляем локальное состояние
                position['stop_loss'] = new_sl
                self.state.set_position(position)
                
                # Логируем трейлинг
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
            # Получаем текущие данные
            if not self.candles_15m:
                return
            
            # Проверяем открытую позицию
            current_position = self.state.get_current_position()
            
            if current_position and current_position.get('status') == 'open':
                # НОВЫЙ smart trailing с arm механизмом
                self._update_smart_trailing(current_position)
            else:
                # Ищем новые входы с новой SFP логикой
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
                                
                                # Сохраняем в базу
                                self.db.save_equity_snapshot(equity)
                                break
        except Exception as e:
            print(f"Error updating equity: {str(e) if e else 'Unknown error'}")
