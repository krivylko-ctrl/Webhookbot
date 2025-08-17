import math
from typing import Dict, Optional
from datetime import datetime

from config import Config
from state_manager import StateManager
from utils import price_round
from analytics import TrailingLogger

class TrailEngine:
    """Движок Smart Trailing для управления стоп-лоссами"""
    
    def __init__(self, config: Config, state_manager: StateManager, bybit_api):
        self.config = config
        self.state = state_manager
        self.api = bybit_api
        
        # 4️⃣ Логгер трейлинга
        self.trail_logger = TrailingLogger()
        
        # Состояние трейлинга
        self.last_trail_price_long = None
        self.last_trail_price_short = None
        
    def process_trailing(self, position: Dict, current_price: float):
        """Основная логика обработки трейлинга"""
        if not position or not current_price:
            return
        
        direction = position.get('direction')
        
        if direction == 'long':
            self._process_long_trailing(position, current_price)
        elif direction == 'short':
            self._process_short_trailing(position, current_price)
    
    def _process_long_trailing(self, position: Dict, current_price: float):
        """Обработка трейлинга для лонг позиции"""
        try:
            entry_price = position.get('entry_price')
            current_sl = position.get('stop_loss')
            armed = position.get('armed', False)
            
            if not entry_price or not current_sl:
                return
            
            # Проверяем арминг после достижения RR
            if self.config.use_arm_after_rr and not armed:
                moved = current_price - entry_price
                need = (entry_price - current_sl) * self.config.arm_rr
                
                if moved >= need:
                    position['armed'] = True
                    armed = True
                    self.state.update_position_armed(True)
                    print(f"Long position armed at RR: {moved / (entry_price - current_sl):.2f}")
            
            if not armed:
                return
            
            new_sl = current_sl
            
            # Bar Trail логика
            if self.config.use_bar_trail:
                new_sl = self._calculate_bar_trail_long(current_sl)
            
            # Процентный Trail логика с offset
            if getattr(self.config, 'trailing_perc', 0) > 0:
                pct_sl = self._calc_pct_offset_long(current_price, entry_price, current_sl)
                new_sl = max(new_sl, pct_sl)
            
            # Offset Trail логика (обычный)
            if self.config.trailing_offset > 0:
                offset_sl = self._calculate_offset_trail_long(current_price, entry_price, current_sl)
                new_sl = max(new_sl, offset_sl)
            
            # Обновляем стоп-лосс если он изменился
            if new_sl > current_sl:
                # 4️⃣ Логируем движение трейла
                self.trail_logger.log_trail_movement(
                    position, current_sl, new_sl, current_price, 
                    "Long Trail Update", 
                    lookback_value=self.config.trail_lookback if self.config.use_bar_trail else 0,
                    buffer_ticks=self.config.trail_buf_ticks
                )
                self._update_stop_loss(position, new_sl)
        
        except Exception as e:
            print(f"Error in long trailing: {e}")
    
    def _process_short_trailing(self, position: Dict, current_price: float):
        """Обработка трейлинга для шорт позиции"""
        try:
            entry_price = position.get('entry_price')
            current_sl = position.get('stop_loss')
            armed = position.get('armed', False)
            
            if not entry_price or not current_sl:
                return
            
            # Проверяем арминг после достижения RR
            if self.config.use_arm_after_rr and not armed:
                moved = entry_price - current_price
                need = (current_sl - entry_price) * self.config.arm_rr
                
                if moved >= need:
                    position['armed'] = True
                    armed = True
                    self.state.update_position_armed(True)
                    print(f"Short position armed at RR: {moved / (current_sl - entry_price):.2f}")
            
            if not armed:
                return
            
            new_sl = current_sl
            
            # Bar Trail логика
            if self.config.use_bar_trail:
                new_sl = self._calculate_bar_trail_short(current_sl)
            
            # Процентный Trail логика с offset
            if getattr(self.config, 'trailing_perc', 0) > 0:
                pct_sl = self._calc_pct_offset_short(current_price, entry_price, current_sl)
                new_sl = min(new_sl, pct_sl)
            
            # Offset Trail логика (обычный)
            if self.config.trailing_offset > 0:
                offset_sl = self._calculate_offset_trail_short(current_price, entry_price, current_sl)
                new_sl = min(new_sl, offset_sl)
            
            # Обновляем стоп-лосс если он изменился
            if new_sl < current_sl:
                # 4️⃣ Логируем движение трейла
                self.trail_logger.log_trail_movement(
                    position, current_sl, new_sl, current_price, 
                    "Short Trail Update",
                    lookback_value=self.config.trail_lookback if self.config.use_bar_trail else 0,
                    buffer_ticks=self.config.trail_buf_ticks
                )
                self._update_stop_loss(position, new_sl)
        
        except Exception as e:
            print(f"Error in short trailing: {e}")
    
    def _calculate_bar_trail_long(self, current_sl: float) -> float:
        """BarTrail по "закрытым" барам: lowest(low, N)[1] - не включай текущий бар"""
        try:
            symbol = self.config.symbol
            klines = self.api.get_klines(symbol, "15", self.config.trail_lookback + 2)  # +2 для исключения текущего
            if not klines or len(klines) < self.config.trail_lookback + 1:
                return current_sl
            
            # BarTrail по "закрытым" барам: используем свечи klines[1:N+1] (исключаем текущий [0])
            lookback_lows = [candle['low'] for candle in klines[1:self.config.trail_lookback + 1]]
            min_low = min(lookback_lows) if lookback_lows else current_sl
            
            # Добавляем буфер с тик-сайзом из конфигурации (единый источник)
            tick_size = self._tick_size()  # Единый источник истины
            buffer = self.config.trail_buf_ticks * tick_size
            bar_trail_sl = min_low - buffer
            
            # Стоп не может двигаться вниз
            return max(bar_trail_sl, current_sl)
        
        except Exception as e:
            print(f"Error calculating bar trail long: {e}")
            return current_sl
    
    def _calculate_bar_trail_short(self, current_sl: float) -> float:
        """BarTrail по "закрытым" барам: highest(high, N)[1] - не включай текущий бар"""
        try:
            symbol = self.config.symbol
            klines = self.api.get_klines(symbol, "15", self.config.trail_lookback + 2)  # +2 для исключения текущего
            if not klines or len(klines) < self.config.trail_lookback + 1:
                return current_sl
            
            # BarTrail по "закрытым" барам: используем свечи klines[1:N+1] (исключаем текущий [0])
            lookback_highs = [candle['high'] for candle in klines[1:self.config.trail_lookback + 1]]
            max_high = max(lookback_highs) if lookback_highs else current_sl
            
            # Добавляем буфер с тик-сайзом из конфигурации (единый источник)
            tick_size = self._tick_size()  # Единый источник истины
            buffer = self.config.trail_buf_ticks * tick_size
            bar_trail_sl = max_high + buffer
            
            # Стоп не может двигаться вверх
            return min(bar_trail_sl, current_sl)
        
        except Exception as e:
            print(f"Error calculating bar trail short: {e}")
            return current_sl
    
    def _calculate_offset_trail_long(self, current_price: float, entry_price: float, current_sl: float) -> float:
        """Расчет Offset Trail для лонг позиции"""
        try:
            # Расчет трейлинга по проценту
            trailing_distance = entry_price * (self.config.trailing_offset / 100)
            offset_sl = current_price - trailing_distance
            
            # Стоп не может двигаться вниз
            return max(offset_sl, current_sl)
        
        except Exception as e:
            print(f"Error calculating offset trail long: {e}")
            return current_sl
    
    def _calculate_offset_trail_short(self, current_price: float, entry_price: float, current_sl: float) -> float:
        """Расчет Offset Trail для шорт позиции"""
        try:
            # Расчет трейлинга по проценту
            trailing_distance = entry_price * (self.config.trailing_offset / 100)
            offset_sl = current_price + trailing_distance
            
            # Стоп не может двигаться вверх
            return min(offset_sl, current_sl)
        
        except Exception as e:
            print(f"Error calculating offset trail short: {e}")
            return current_sl
    
    def _update_stop_loss(self, position: Dict, new_sl: float):
        """Правильный апдейт стопа (derivatives) согласно техническим требованиям"""
        try:
            from utils import price_round
            import time
            
            # Получаем символ и тик-сайз из конфигурации (единый источник)
            symbol = position.get('symbol', self.config.symbol)
            tick_size = self._tick_size()  # Единый источник
            
            # Округляем цену правильным тик-сайзом
            new_sl = price_round(new_sl, tick_size)
            
            # Правильное обновление SL для деривативов - POST /v5/position/trading-stop
            if hasattr(self.api, 'market_type') and self.api.market_type == 'linear':
                result = self.api.update_position_stop_loss(symbol, new_sl)
            else:
                # Для спота создаем условный STOP-ордер с reduceOnly=true
                direction = position.get('direction')
                size = position.get('size')
                
                side = 'sell' if direction == 'long' else 'buy'
                
                result = self.api.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="stop",
                    qty=size,
                    price=new_sl,
                    reduce_only=True,
                    order_link_id=f"trail_{int(time.time())}"
                )
            
            if result:
                # Обновляем состояние
                position['stop_loss'] = new_sl
                self.state.update_position_stop_loss(new_sl)
                
                print(f"Stop loss updated: {new_sl}")
            else:
                print(f"Failed to update stop loss on exchange, updated locally: {new_sl}")
                position['stop_loss'] = new_sl
                self.state.update_position_stop_loss(new_sl)
                
        except Exception as e:
            print(f"Error updating stop loss: {e}")

    def _tick_size(self) -> float:
        """Получение размера тика из конфигурации"""
        # Единый источник тик-сайза из конфигурации
        return getattr(self.config, 'tick_size', 0.01)

    def _calc_pct_offset_long(self, current_price: float, entry_price: float, current_sl: float) -> float:
        """Расчет процентного трейла с offset для long позиций"""
        try:
            trailing_perc = getattr(self.config, 'trailing_perc', 0.02)  # 2%
            trailing_offset_perc = getattr(self.config, 'trailing_offset_perc', 0.004)  # 0.4%
            
            # Процентный трейл от цены входа
            pct_trail = entry_price * (1 + trailing_perc)
            
            # Ограничение: не ближе offset к текущей цене
            min_distance = current_price * trailing_offset_perc
            pct_trail_with_offset = current_price - min_distance
            
            # Берем максимум (самый высокий SL для лонга)
            new_sl = max(pct_trail, pct_trail_with_offset, current_sl)
            
            return new_sl
            
        except Exception as e:
            print(f"Error in _calc_pct_offset_long: {e}")
            return current_sl

    def _calc_pct_offset_short(self, current_price: float, entry_price: float, current_sl: float) -> float:
        """Расчет процентного трейла с offset для short позиций"""
        try:
            trailing_perc = getattr(self.config, 'trailing_perc', 0.02)  # 2%
            trailing_offset_perc = getattr(self.config, 'trailing_offset_perc', 0.004)  # 0.4%
            
            # Процентный трейл от цены входа
            pct_trail = entry_price * (1 - trailing_perc)
            
            # Ограничение: не ближе offset к текущей цене
            min_distance = current_price * trailing_offset_perc
            pct_trail_with_offset = current_price + min_distance
            
            # Берем минимум (самый низкий SL для шорта)
            new_sl = min(pct_trail, pct_trail_with_offset, current_sl)
            
            return new_sl
            
        except Exception as e:
            print(f"Error in _calc_pct_offset_short: {e}")
            return current_sl
            print(f"Error updating stop loss: {e}")
    
    def _log_trailing_update(self, position: Dict, new_sl: float):
        """Логирование обновления трейлинга"""
        try:
            log_data = {
                'timestamp': datetime.now(),
                'symbol': position.get('symbol'),
                'direction': position.get('direction'),
                'old_sl': position.get('stop_loss'),
                'new_sl': new_sl,
                'action': 'trailing_update'
            }
            # Здесь можно добавить сохранение в базу данных или файл лога
            print(f"Trailing log: {log_data}")
        
        except Exception as e:
            print(f"Error logging trailing update: {e}")
