"""
Демо-режим для тестирования бота без реального API
Генерирует тестовые данные свечей и симулирует торговлю
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class DemoAPI:
    """Демо API для тестирования без реального подключения к бирже"""
    
    def __init__(self):
        self.base_price = 2500.0  # Базовая цена ETH
        self.current_price = self.base_price
        self.last_update = time.time()
        
    def get_server_time(self) -> int:
        """Возвращает текущее время"""
        return int(time.time())
    
    def get_ticker(self, symbol: str) -> Dict:
        """Возвращает текущую цену с небольшими случайными изменениями"""
        # Обновляем цену каждые несколько секунд
        now = time.time()
        if now - self.last_update > 5:
            change_pct = random.uniform(-0.002, 0.002)  # ±0.2%
            self.current_price *= (1 + change_pct)
            self.last_update = now
        
        return {
            'symbol': symbol,
            'last_price': round(self.current_price, 2),
            'bid': round(self.current_price * 0.9995, 2),
            'ask': round(self.current_price * 1.0005, 2),
            'volume': random.uniform(10000, 50000)
        }
    
    def get_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        """Генерирует тестовые свечи"""
        klines = []
        current_time = int(time.time() * 1000)
        
        # Интервал в миллисекундах
        interval_ms = {
            '1': 60000,       # 1 минута
            '15': 900000,     # 15 минут
            '1h': 3600000,    # 1 час
            '4h': 14400000,   # 4 часа
            '1d': 86400000    # 1 день
        }.get(interval, 900000)
        
        for i in range(limit):
            timestamp = current_time - (i * interval_ms)
            
            # Генерируем OHLC данные
            base_price = self.current_price * (1 + random.uniform(-0.01, 0.01))
            high = base_price * (1 + random.uniform(0, 0.005))
            low = base_price * (1 - random.uniform(0, 0.005))
            open_price = base_price * (1 + random.uniform(-0.002, 0.002))
            close_price = base_price * (1 + random.uniform(-0.002, 0.002))
            
            # Убеждаемся что high >= max(open, close) и low <= min(open, close)
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            kline = {
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': random.uniform(100, 1000)
            }
            klines.append(kline)
        
        # Возвращаем в обратном порядке (новые свечи первыми)
        return list(reversed(klines))
    
    def get_wallet_balance(self) -> Dict:
        """Возвращает тестовый баланс"""
        return {
            "list": [{
                "accountType": "SPOT",
                "coin": [{
                    "coin": "USDT",
                    "equity": "1000.00",
                    "walletBalance": "1000.00",
                    "free": "1000.00",
                    "locked": "0.00"
                }]
            }]
        }
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None) -> Dict:
        """Симулирует размещение ордера"""
        order_id = f"demo_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "price": str(price) if price else str(self.current_price),
            "status": "Filled",
            "timeInForce": "GTC"
        }
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Симулирует отмену ордера"""
        return True
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Возвращает пустой список открытых ордеров"""
        return []
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Возвращает пустую историю ордеров"""
        return []
    
    def get_instrument_info(self, symbol: str) -> Dict:
        """Возвращает информацию об инструменте"""
        return {
            "priceFilter": {"tickSize": "0.01"},
            "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01"}
        }

def create_demo_api() -> DemoAPI:
    """Создает экземпляр демо API"""
    return DemoAPI()