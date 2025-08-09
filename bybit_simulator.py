import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class BybitSimulator:
    """Симулятор торговли для демонстрации работы без реального API"""
    
    def __init__(self):
        self.positions = {}
        self.orders = {}
        self.balance = 10000.0  # Симуляция начального баланса в USDT
        
    def test_connection(self):
        """Тест соединения (симуляция)"""
        return {
            "simulation": True,
            "status": "connected",
            "message": "СИМУЛЯЦИЯ: API недоступен, используется симулятор",
            "balance": self.balance
        }
    
    def place_order(self, symbol, side, amount, leverage=10, sl_percent=1.0, tp_percent=3.0):
        """Симуляция размещения ордера"""
        
        # Симулируем реальную цену ETH
        base_price = 2650.0  # Примерная цена ETH
        price_variation = random.uniform(-50, 50)  # Колебания цены
        current_price = base_price + price_variation
        
        # Расчет стоп-лосс и тейк-профит
        if side.lower() == 'long':
            sl_price = current_price * (1 - sl_percent / 100)
            tp_price = current_price * (1 + tp_percent / 100)
        else:  # short
            sl_price = current_price * (1 + sl_percent / 100)
            tp_price = current_price * (1 - tp_percent / 100)
        
        # Расчет необходимой маржи
        required_margin = (current_price * amount) / leverage
        
        order_id = f"SIM_{int(datetime.now().timestamp())}"
        
        # Сохранение позиции в симуляторе
        self.positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": current_price,
            "leverage": leverage,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "required_margin": required_margin,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"🎯 СИМУЛЯЦИЯ: {side.upper()} {amount} {symbol}")
        logger.info(f"   Цена входа: ${current_price:.2f}")
        logger.info(f"   Плечо: {leverage}x")
        logger.info(f"   Маржа: ${required_margin:.2f}")
        logger.info(f"   SL: ${sl_price:.2f} ({sl_percent}%)")
        logger.info(f"   TP: ${tp_price:.2f} ({tp_percent}%)")
        
        return {
            "simulation": True,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": current_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "leverage": leverage,
            "required_margin": required_margin,
            "status": "filled"
        }
    
    def close_position(self, symbol):
        """Симуляция закрытия позиции"""
        
        if symbol not in self.positions:
            return {
                "simulation": True,
                "error": "Позиция не найдена в симуляторе"
            }
        
        position = self.positions[symbol]
        
        # Симулируем текущую цену (с небольшим профитом)
        entry_price = position["entry_price"]
        profit_variation = random.uniform(10, 50)  # Симулируем прибыль
        
        if position["side"].lower() == "long":
            exit_price = entry_price + profit_variation
        else:
            exit_price = entry_price - profit_variation
        
        # Расчет P&L
        if position["side"].lower() == "long":
            pnl = (exit_price - entry_price) * position["amount"] * position["leverage"]
        else:
            pnl = (entry_price - exit_price) * position["amount"] * position["leverage"]
        
        logger.info(f"🎯 СИМУЛЯЦИЯ: Закрытие {position['side'].upper()} {symbol}")
        logger.info(f"   Цена входа: ${entry_price:.2f}")
        logger.info(f"   Цена выхода: ${exit_price:.2f}")
        logger.info(f"   P&L: ${pnl:.2f}")
        
        # Удаляем позицию
        del self.positions[symbol]
        
        # Обновляем баланс
        self.balance += pnl
        
        return {
            "simulation": True,
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "new_balance": self.balance,
            "status": "closed"
        }
    
    def update_stop_loss(self, symbol, direction, stop_price, trail_amount=None):
        """Симуляция обновления стоп-лосс"""
        
        if symbol not in self.positions:
            return {
                "simulation": True,
                "error": "Позиция не найдена для обновления SL"
            }
        
        # Обновляем SL в симуляторе
        self.positions[symbol]["sl_price"] = stop_price
        
        logger.info(f"🎯 СИМУЛЯЦИЯ: Обновление SL для {symbol}")
        logger.info(f"   Новый стоп-лосс: ${stop_price:.2f}")
        if trail_amount:
            logger.info(f"   Трейлинг: ${trail_amount:.2f}")
        
        return {
            "simulation": True,
            "symbol": symbol,
            "new_sl_price": stop_price,
            "trail_amount": trail_amount,
            "status": "updated"
        }
    
    def get_positions(self):
        """Получение текущих позиций (симуляция)"""
        return {
            "simulation": True,
            "positions": list(self.positions.values()),
            "total_count": len(self.positions)
        }
    
    def get_balance(self):
        """Получение баланса (симуляция)"""
        return {
            "simulation": True,
            "balance": self.balance,
            "currency": "USDT"
        }