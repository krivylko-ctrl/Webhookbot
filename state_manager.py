import json
from typing import Dict, Optional
from datetime import datetime
from database import Database

class StateManager:
    """Управление состоянием торгового бота"""
    
    def __init__(self, db: Database):
        self.db = db
        self._current_position = None
        self._equity = 100.0  # Начальный капитал
        self._bot_status = "stopped"
        
        # Загружаем сохраненное состояние
        self._load_state()
    
    def _load_state(self):
        """Загрузка сохраненного состояния"""
        try:
            state = self.db.get_bot_state()
            if state:
                self._current_position = state.get('position')
                self._equity = state.get('equity', 100.0)
                self._bot_status = state.get('status', 'stopped')
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def _save_state(self):
        """Сохранение текущего состояния"""
        try:
            state = {
                'position': self._current_position,
                'equity': self._equity,
                'status': self._bot_status,
                'updated_at': datetime.now().isoformat()
            }
            self.db.save_bot_state(state)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def get_current_position(self) -> Optional[Dict]:
        """Получить текущую позицию"""
        return self._current_position
    
    def set_position(self, position: Dict):
        """Установить новую позицию"""
        self._current_position = position
        self._save_state()
    
    def update_position_stop_loss(self, new_sl: float):
        """Обновить стоп-лосс текущей позиции"""
        if self._current_position:
            self._current_position['stop_loss'] = new_sl
            self._save_state()
    
    def update_position_armed(self, armed: bool):
        """Обновить статус арминга позиции"""
        if self._current_position:
            self._current_position['armed'] = armed
            self._save_state()
    
    def close_position(self, exit_price: float, exit_reason: str = "manual"):
        """Закрыть текущую позицию"""
        if self._current_position:
            # Рассчитываем PnL
            entry_price = self._current_position.get('entry_price')
            quantity = self._current_position.get('size')
            direction = self._current_position.get('direction')
            
            if entry_price and quantity:
                if direction == 'long':
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity
                
                # Учитываем комиссии
                commission = (entry_price + exit_price) * quantity * 0.00055
                net_pnl = pnl - commission
                
                # Обновляем equity
                self._equity += net_pnl
                
                # Сохраняем закрытую сделку
                trade_data = {
                    'symbol': self._current_position.get('symbol'),
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'pnl': net_pnl,
                    'rr': self._calculate_rr(self._current_position, exit_price),
                    'exit_reason': exit_reason,
                    'exit_time': datetime.now(),
                    'status': 'closed'
                }
                self.db.update_trade_exit(trade_data)
            
            # Очищаем позицию
            self._current_position = None
            self._save_state()
    
    def clear_position(self):
        """Очистка текущей позиции"""
        self._current_position = None
        self._save_state()
    
    def _calculate_rr(self, position: Dict, exit_price: float) -> float:
        """Расчет Risk/Reward ratio"""
        try:
            entry_price = position.get('entry_price')
            stop_loss = position.get('stop_loss')
            direction = position.get('direction')
            
            if not all([entry_price, stop_loss]):
                return 0.0
            
            if direction == 'long':
                risk = float(entry_price) - float(stop_loss)
                reward = float(exit_price) - float(entry_price)
            else:
                risk = float(stop_loss) - float(entry_price)
                reward = float(entry_price) - float(exit_price)
            
            if risk <= 0:
                return 0.0
            
            return reward / risk
        except:
            return 0.0
    
    def get_equity(self) -> float:
        """Получить текущий equity"""
        return self._equity
    
    def set_equity(self, equity: float):
        """Установить equity"""
        self._equity = equity
        self._save_state()
    
    def get_bot_status(self) -> str:
        """Получить статус бота"""
        return self._bot_status
    
    def set_bot_status(self, status: str):
        """Установить статус бота"""
        self._bot_status = status
        self._save_state()
    
    def is_position_open(self) -> bool:
        """Проверить открыта ли позиция"""
        return self._current_position is not None
    
    def get_position_direction(self) -> Optional[str]:
        """Получить направление позиции"""
        if self._current_position:
            return self._current_position.get('direction')
        return None
    
    def get_position_entry_price(self) -> Optional[float]:
        """Получить цену входа в позицию"""
        if self._current_position:
            return self._current_position.get('entry_price')
        return None
    
    def get_position_stop_loss(self) -> Optional[float]:
        """Получить стоп-лосс позиции"""
        if self._current_position:
            return self._current_position.get('stop_loss')
        return None
    
    def get_position_take_profit(self) -> Optional[float]:
        """Получить тейк-профит позиции"""
        if self._current_position:
            return self._current_position.get('take_profit')
        return None
    
    def get_position_size(self) -> Optional[float]:
        """Получить размер позиции"""
        if self._current_position:
            return self._current_position.get('size')
        return None
    
    def is_position_armed(self) -> bool:
        """Проверить заармлена ли позиция"""
        if self._current_position:
            return self._current_position.get('armed', False)
        return False
    
    def get_state_summary(self) -> Dict:
        """Получить сводку текущего состояния"""
        return {
            'equity': self._equity,
            'bot_status': self._bot_status,
            'position_open': self.is_position_open(),
            'position': self._current_position,
            'timestamp': datetime.now().isoformat()
        }
