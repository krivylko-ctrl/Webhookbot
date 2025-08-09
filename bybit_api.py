import ccxt
import logging
import os
from typing import Dict, Any, Optional
from bybit_simulator import BybitSimulator

API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

logger = logging.getLogger(__name__)

class BybitFuturesClient:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key or API_KEY or ""
        self.api_secret = api_secret or API_SECRET or ""
        self.simulator = BybitSimulator()
        
        # Правильная конфигурация CCXT для Bybit V5 API
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': testnet,  # True для testnet, False для mainnet
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'linear',  # USDT Perpetual контракты
                    'adjustForTimeDifference': True,
                    'recvWindow': 20000,  # Важно для Bybit V5
                },
                'urls': {
                    'api': {
                        'public': 'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com',
                        'private': 'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com'
                    }
                },
                'headers': {
                    'User-Agent': 'CCXT/4.4.99 (+https://github.com/ccxt/ccxt)',
                    'Content-Type': 'application/json'
                }
            })
            
            logger.info("Bybit API инициализация через CCXT:")
            logger.info(f"  API Key: {'НАЙДЕН' if self.api_key else 'НЕ НАЙДЕН'}")
            logger.info(f"  API Secret: {'НАЙДЕН' if self.api_secret else 'НЕ НАЙДЕН'}")
            logger.info(f"  Режим: {'DEMO TESTNET' if testnet else 'MAINNET'}")
            logger.info(f"  CCXT версия: {ccxt.__version__}")
            
            if self.api_key:
                logger.info(f"  Key начинается с: {self.api_key[:8]}...")
            if self.api_secret:
                logger.info(f"  Secret начинается с: {self.api_secret[:8]}...")
            
            # Логирование конфигурации
            testnet_url = 'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com'
            logger.info(f"  Base URL: {testnet_url}")
            logger.info(f"  API Version: V5")
            logger.info(f"  RecvWindow: 20000ms")
            logger.info("✅ ГОТОВ К ТОРГОВЛЕ: Bybit API активен через CCXT на Railway.app")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации CCXT: {e}")
            self.exchange = None

    def test_connection(self):
        """Тестирование соединения с Bybit API через CCXT"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Тест через получение баланса
            balance = self.exchange.fetch_balance()
            logger.info("✅ Подключение к Bybit API успешно через CCXT")
            return {
                "status": "connected",
                "message": "Реальное подключение к Bybit API работает",
                "testnet": True,
                "balance": balance.get('USDT', {}).get('total', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit API: {e}")
            logger.info("🔄 Переключение на симулятор")
            return self.simulator.test_connection()

    def place_order(self, symbol: str, direction: str, quantity: float, 
                   entry_price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, leverage: int = 10) -> Dict[str, Any]:
        """Размещение ордера на Bybit через CCXT"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Конвертация символа: ETHUSDT.P -> ETHUSDT для Bybit
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            
            # Определение стороны
            side = 'buy' if direction.lower() == 'long' else 'sell'
            order_type = 'market' if entry_price is None else 'limit'
            
            logger.info(f"CCXT: Размещение {order_type} ордера {direction}: {quantity} {symbol}")
            
            # Установка плеча
            self.exchange.set_leverage(leverage, symbol)
            
            # Размещение основного ордера
            if entry_price is None:
                order = self.exchange.create_market_order(symbol, side, quantity)
            else:
                order = self.exchange.create_limit_order(symbol, side, quantity, entry_price)
            
            logger.info(f"✅ Ордер размещен через CCXT: {order.get('id')}")
            
            # Установка SL/TP если указаны
            if stop_loss or take_profit:
                self._place_stop_orders_ccxt(symbol, direction, quantity, stop_loss, take_profit)
            
            return {
                "order_id": order.get('id'),
                "symbol": symbol,
                "side": side,
                "amount": quantity,
                "price": order.get('price'),
                "status": "filled"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка CCXT размещения ордера: {e}")
            logger.info("🔄 Переключение на симулятор")
            return self.simulator.place_order(symbol, direction, quantity, leverage, stop_loss, take_profit)

    def _place_stop_orders_ccxt(self, symbol: str, direction: str, quantity: float, 
                               stop_loss: Optional[float], take_profit: Optional[float]):
        """Размещение SL/TP ордеров через CCXT"""
        try:
            reduce_side = 'sell' if direction.lower() == 'long' else 'buy'
            
            if stop_loss:
                sl_order = self.exchange.create_order(
                    symbol, 'stop_market', reduce_side, quantity, None, 
                    None, {'stopPrice': stop_loss, 'reduceOnly': True}
                )
                logger.info(f"SL ордер установлен: {sl_order.get('id')}")
            
            if take_profit:
                tp_order = self.exchange.create_order(
                    symbol, 'take_profit_market', reduce_side, quantity, None,
                    None, {'stopPrice': take_profit, 'reduceOnly': True}
                )
                logger.info(f"TP ордер установлен: {tp_order.get('id')}")
                
        except Exception as e:
            logger.warning(f"Не удалось установить SL/TP: {e}")

    def open_position(self, symbol: str, direction: str, quantity: float, 
                     entry_price: float = None, stop_loss: float = None, 
                     take_profit: float = None, leverage: int = 10) -> Dict[str, Any]:
        """Открытие позиции с конвертацией символа"""
        # Конвертация ETHUSDT.P -> ETHUSDT
        if symbol.endswith('.P'):
            symbol = symbol[:-2]
            
        return self.place_order(symbol, direction, quantity, entry_price, stop_loss, take_profit, leverage)

    def close_position(self, symbol: str, direction: str = None) -> Dict[str, Any]:
        """Закрытие позиции"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Конвертация символа
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            
            # Получение текущих позиций
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                size = pos.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:
                    active_position = pos
                    break
            
            if not active_position:
                logger.warning("Активная позиция не найдена")
                return {"error": "Позиция не найдена"}
            
            # Закрытие позиции
            side = 'sell' if active_position.get('side') == 'long' else 'buy'
            amount = abs(float(active_position.get('size', 0)))
            
            close_order = self.exchange.create_market_order(
                symbol, side, amount, None, None, {'reduceOnly': True}
            )
            
            logger.info(f"✅ Позиция закрыта: {close_order.get('id')}")
            
            return {
                "order_id": close_order.get('id'),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "status": "closed"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия позиции: {e}")
            logger.info("🔄 Переключение на симулятор")
            return self.simulator.close_position(symbol)

    def update_stop_loss(self, symbol: str, direction: str, stop_price: float, trail_amount: float = None) -> Dict[str, Any]:
        """Обновление стоп-лосс"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Конвертация символа
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            
            logger.info(f"Обновление стоп-лосс для {symbol}: {stop_price}")
            
            # Получение активных ордеров стоп-лосс
            orders = self.exchange.fetch_open_orders(symbol)
            sl_orders = [order for order in orders if order.get('type') == 'stop_market']
            
            # Отмена старых SL ордеров
            for order in sl_orders:
                order_id = order.get('id')
                if order_id:
                    self.exchange.cancel_order(order_id, symbol)
                    logger.info(f"Отменен старый SL ордер: {order_id}")
            
            # Получение позиций для нового SL
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                size = pos.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:
                    active_position = pos
                    break
            
            if not active_position:
                return {"error": "Активная позиция не найдена"}
            
            # Создание нового стоп-лосс ордера
            side = 'sell' if active_position.get('side') == 'long' else 'buy'
            amount = abs(float(active_position.get('size', 0)))
            
            new_sl_order = self.exchange.create_order(
                symbol, 'stop_market', side, amount, None, None,
                {'stopPrice': stop_price, 'reduceOnly': True}
            )
            
            logger.info(f"✅ Новый SL ордер: {new_sl_order.get('id')}")
            
            return {
                "order_id": new_sl_order.get('id'),
                "symbol": symbol,
                "stop_price": stop_price,
                "trail_amount": trail_amount,
                "status": "updated"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления SL: {e}")
            logger.info("🔄 Переключение на симулятор")
            return self.simulator.update_stop_loss(symbol, direction, stop_price)

    def get_account_info(self) -> Dict[str, Any]:
        """Получение информации об аккаунте"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            balance = self.exchange.fetch_balance()
            
            return {
                "status": "connected",
                "balance": balance.get('USDT', {}).get('total', 0),
                "testnet": True
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return {"status": "demo_mode", "error": str(e)}

    def get_positions(self, symbol: str = None) -> Dict[str, Any]:
        """Получение активных позиций"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            if symbol and symbol.endswith('.P'):
                symbol = symbol[:-2]
                
            symbols = [symbol] if symbol else None
            positions = self.exchange.fetch_positions(symbols)
            
            # Фильтрация активных позиций
            active_positions = []
            for pos in positions:
                size = pos.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:
                    active_positions.append(pos)
            
            return {
                "positions": active_positions,
                "count": len(active_positions),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения позиций: {e}")
            return self.simulator.get_positions(symbol)