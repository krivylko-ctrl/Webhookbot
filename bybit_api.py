import ccxt
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BybitFuturesClient:
    def __init__(self, testnet=False):
        """
        Инициализация Bybit PRODUCTION клиента через CCXT
        ТОЛЬКО MAINNET - БЕЗ ТЕСТНЕТА
        """
        self.testnet = False  # Принудительно только mainnet
        
        # Получение API ключей из переменных окружения
        self.api_key = os.environ.get('BYBIT_API_KEY')
        self.api_secret = os.environ.get('BYBIT_API_SECRET')
        
        logger.info("Bybit PRODUCTION API инициализация через CCXT:")
        logger.info(f"  API Key: {'НАЙДЕН' if self.api_key else 'НЕ НАЙДЕН'}")
        logger.info(f"  API Secret: {'НАЙДЕН' if self.api_secret else 'НЕ НАЙДЕН'}")
        logger.info(f"  Режим: PRODUCTION MAINNET")
        logger.info(f"  CCXT версия: {ccxt.__version__}")
        
        if self.api_key and self.api_secret:
            logger.info(f"  Key начинается с: {self.api_key[:8]}...")
            logger.info(f"  Secret начинается с: {self.api_secret[:8]}...")
        
        # Инициализация CCXT клиента для PRODUCTION mainnet
        try:
            if self.api_key and self.api_secret:
                self.exchange = ccxt.bybit({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'sandbox': False,  # PRODUCTION MAINNET
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'linear',  # USDT Perpetual контракты
                        'adjustForTimeDifference': True,
                        'recvWindow': 20000,  # Важно для Bybit V5
                    },
                    'urls': {
                        'api': {
                            'public': 'https://api.bybit.com',
                            'private': 'https://api.bybit.com'
                        }
                    },
                })
            else:
                logger.error("API ключи не найдены")
                self.exchange = None
                raise Exception("API ключи не настроены")
            
            # Логирование конфигурации
            logger.info(f"  Base URL: https://api.bybit.com")
            logger.info(f"  API Version: V5")
            logger.info(f"  RecvWindow: 20000ms")
            logger.info("✅ ГОТОВ К PRODUCTION ТОРГОВЛЕ: Bybit mainnet API активен через CCXT")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации CCXT: {e}")
            self.exchange = None
            raise e

    def test_connection(self) -> Dict[str, Any]:
        """Тестирование соединения с Bybit PRODUCTION"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            balance = self.exchange.fetch_balance()
            logger.info("✅ Подключение к Bybit PRODUCTION API успешно")
            
            return {
                "status": "connected",
                "message": "Подключение к Bybit PRODUCTION API работает",
                "testnet": False,
                "balance": balance.get('USDT', {}).get('total', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit PRODUCTION API: {e}")
            return {
                "status": "error",
                "message": f"Ошибка подключения к Bybit PRODUCTION: {str(e)}",
                "testnet": False
            }

    def place_order(self, symbol: str, direction: str, quantity: float, 
                   entry_price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, leverage: int = 1) -> Dict[str, Any]:
        """Размещение ордера в Bybit PRODUCTION - РЕАЛЬНЫЕ ДЕНЬГИ"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Конвертация символа ETHUSDT.P → ETHUSDT
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            
            # Установка кредитного плеча
            if leverage > 1:
                self.exchange.set_leverage(leverage, symbol)
                logger.info(f"PRODUCTION: Установлено кредитное плечо {leverage}x для {symbol}")
            
            # Размещение рыночного ордера
            side = 'buy' if direction.lower() == 'long' else 'sell'
            order = self.exchange.create_market_order(symbol, side, quantity)
            
            logger.info(f"✅ PRODUCTION ОРДЕР: {order.get('id')} - {direction} {quantity} {symbol}")
            
            # Размещение стоп-ордеров если указаны
            if stop_loss or take_profit:
                self._place_stop_orders_ccxt(symbol, direction, quantity, stop_loss, take_profit)
            
            return {
                "order_id": order.get('id'),
                "symbol": symbol,
                "side": side,
                "amount": quantity,
                "price": order.get('price'),
                "status": "filled",
                "mode": "production"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения PRODUCTION ордера: {e}")
            return {
                "error": f"Ошибка размещения ордера: {str(e)}",
                "symbol": symbol,
                "direction": direction,
                "mode": "production"
            }

    def _place_stop_orders_ccxt(self, symbol: str, direction: str, quantity: float, 
                               stop_loss: Optional[float], take_profit: Optional[float]):
        """Размещение стоп-ордеров через CCXT в PRODUCTION"""
        try:
            position_side = 'long' if direction.lower() == 'long' else 'short'
            
            if stop_loss:
                sl_side = 'sell' if direction.lower() == 'long' else 'buy'
                sl_order = self.exchange.create_order(
                    symbol, 'stop_market', sl_side, quantity, None, None,
                    {'stopPrice': stop_loss, 'reduceOnly': True}
                )
                logger.info(f"✅ PRODUCTION Stop Loss: {stop_loss} (ID: {sl_order.get('id')})")
            
            if take_profit:
                tp_side = 'sell' if direction.lower() == 'long' else 'buy'
                tp_order = self.exchange.create_order(
                    symbol, 'take_profit_market', tp_side, quantity, None, None,
                    {'stopPrice': take_profit, 'reduceOnly': True}
                )
                logger.info(f"✅ PRODUCTION Take Profit: {take_profit} (ID: {tp_order.get('id')})")
                
        except Exception as e:
            logger.error(f"❌ Ошибка размещения PRODUCTION стоп-ордеров: {e}")

    def open_position(self, symbol: str, direction: str, quantity: float, 
                     entry_price: Optional[float] = None, stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None, leverage: int = 1) -> Dict[str, Any]:
        """Открытие позиции в PRODUCTION - РЕАЛЬНЫЕ ДЕНЬГИ"""
        return self.place_order(symbol=symbol, direction=direction, quantity=quantity, 
                               entry_price=entry_price, stop_loss=stop_loss, 
                               take_profit=take_profit, leverage=leverage)
    
    def close_position(self, symbol: str, direction: str = None) -> Dict[str, Any]:
        """Закрытие позиции в PRODUCTION - РЕАЛЬНЫЕ ДЕНЬГИ"""
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
                logger.warning("PRODUCTION: Активная позиция не найдена")
                return {"error": "Позиция не найдена"}
            
            # Закрытие позиции
            side = 'sell' if active_position.get('side') == 'long' else 'buy'
            amount = abs(float(active_position.get('size', 0)))
            
            close_order = self.exchange.create_market_order(
                symbol, side, amount, None, None, {'reduceOnly': True}
            )
            
            logger.info(f"✅ PRODUCTION ПОЗИЦИЯ ЗАКРЫТА: {close_order.get('id')}")
            
            return {
                "order_id": close_order.get('id'),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "status": "closed",
                "mode": "production"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия PRODUCTION позиции: {e}")
            return {
                "error": f"Ошибка закрытия позиции: {str(e)}",
                "symbol": symbol,
                "mode": "production"
            }

    def update_stop_loss(self, symbol: str, direction: str, stop_price: float, trail_amount: float = None) -> Dict[str, Any]:
        """Обновление стоп-лосс в PRODUCTION - РЕАЛЬНЫЕ ДЕНЬГИ"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Конвертация символа
            if symbol.endswith('.P'):
                symbol = symbol[:-2]
            
            logger.info(f"PRODUCTION: Обновление стоп-лосс для {symbol}: {stop_price}")
            
            # Получение активных ордеров стоп-лосс
            orders = self.exchange.fetch_open_orders(symbol)
            sl_orders = [order for order in orders if order.get('type') == 'stop_market']
            
            # Отмена старых SL ордеров
            for order in sl_orders:
                order_id = order.get('id')
                if order_id:
                    self.exchange.cancel_order(order_id, symbol)
                    logger.info(f"PRODUCTION: Отменен старый SL ордер: {order_id}")
            
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
            
            logger.info(f"✅ PRODUCTION НОВЫЙ SL: {new_sl_order.get('id')}")
            
            return {
                "order_id": new_sl_order.get('id'),
                "symbol": symbol,
                "stop_price": stop_price,
                "trail_amount": trail_amount,
                "status": "updated",
                "mode": "production"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления PRODUCTION SL: {e}")
            return {
                "error": f"Ошибка обновления стоп-лосс: {str(e)}",
                "symbol": symbol,
                "mode": "production"
            }

    def get_account_info(self) -> Dict[str, Any]:
        """Получение информации об аккаунте PRODUCTION - РЕАЛЬНЫЙ БАЛАНС"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            balance = self.exchange.fetch_balance()
            
            return {
                "status": "connected",
                "balance": balance.get('USDT', {}).get('total', 0),
                "testnet": False,
                "mode": "production"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения PRODUCTION баланса: {e}")
            return {"error": str(e), "testnet": False, "mode": "production"}

    def get_positions(self, symbol: str = None) -> Dict[str, Any]:
        """Получение активных позиций PRODUCTION - РЕАЛЬНЫЕ ПОЗИЦИИ"""
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
                "status": "success",
                "mode": "production"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения PRODUCTION позиций: {e}")
            return {"error": str(e), "mode": "production"}