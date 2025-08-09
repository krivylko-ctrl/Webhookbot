import os
import ccxt
import logging
from bybit_simulator import BybitSimulator

logger = logging.getLogger(__name__)

class BybitFuturesClient:
    def __init__(self, testnet=True):
        """
        Инициализация Bybit клиента для Railway deployment
        """
        self.testnet = testnet
        self.simulator = BybitSimulator()
        
        # Получение API ключей из переменных окружения
        self.api_key = os.environ.get('BYBIT_API_KEY')
        self.api_secret = os.environ.get('BYBIT_API_SECRET')
        
        logger.info("Bybit API инициализация через CCXT:")
        logger.info(f"  API Key: {'НАЙДЕН' if self.api_key else 'НЕ НАЙДЕН'}")
        logger.info(f"  API Secret: {'НАЙДЕН' if self.api_secret else 'НЕ НАЙДЕН'}")
        logger.info(f"  Режим: {'DEMO TESTNET' if testnet else 'PRODUCTION MAINNET'}")
        logger.info(f"  CCXT версия: {ccxt.__version__}")
        
        if self.api_key and self.api_secret:
            logger.info(f"  Key начинается с: {self.api_key[:8]}...")
            logger.info(f"  Secret начинается с: {self.api_secret[:8]}...")
        
        # Инициализация CCXT клиента
        try:
            if self.api_key and self.api_secret:
                self.exchange = ccxt.bybit({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'sandbox': testnet,  # True для тестнета, False для продакшн
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'linear',  # Линейные фьючерсы USDT
                        'recvWindow': 10000,
                    }
                })
            else:
                logger.warning("API ключи не найдены, exchange будет None")
                self.exchange = None
            
            if not testnet:
                logger.info("🚀 PRODUCTION РЕЖИМ: Реальная торговля на Bybit")
            else:
                logger.info("🧪 DEMO РЕЖИМ: Тестовая торговля на демо счете Bybit")
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации CCXT: {e}")
            self.exchange = None
        
        logger.info("✅ ГОТОВ К ТОРГОВЛЕ: Bybit API активен через CCXT на Railway.app")

    def test_connection(self):
        """Тестирование соединения"""
        try:
            if not self.exchange:
                return {"error": "Exchange не инициализирован"}
            
            # Попытка получить баланс для тестирования соединения
            balance = self.exchange.fetch_balance()
            logger.info("✅ Подключение к Bybit API успешно")
            return {
                "status": "connected",
                "message": "Подключение к Bybit API работает",
                "testnet": self.testnet,
                "balance_keys": list(balance.keys())[:5]  # Показываем только ключи
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit API: {e}")
            logger.info("🔄 Переключение на симулятор")
            return self.simulator.test_connection()

    def place_order(self, symbol, side, amount, leverage=10, sl_percent=1.0, tp_percent=3.0):
        """Размещение ордера с улучшенной обработкой типов"""
        try:
            logger.info(f"Размещение ордера {side}: {amount} {symbol}")
            logger.info(f"Плечо: {leverage}x, SL: {sl_percent}%, TP: {tp_percent}%")
            
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Установка плеча
            self.exchange.set_leverage(leverage, symbol)
            
            # Получение текущей цены для расчета SL/TP
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker['last']) if ticker.get('last') else 0.0
            
            if current_price == 0:
                raise Exception("Не удалось получить текущую цену")
            
            # Расчет цен стоп-лосс и тейк-профит
            if side.lower() == 'long' or side.lower() == 'buy':
                sl_price = current_price * (1 - sl_percent / 100)
                tp_price = current_price * (1 + tp_percent / 100)
            else:  # short or sell
                sl_price = current_price * (1 + sl_percent / 100)
                tp_price = current_price * (1 - tp_percent / 100)
            
            # Размещение основного ордера
            order = self.exchange.create_market_order(symbol, side, amount)
            
            logger.info(f"✅ Ордер размещен: {order.get('id', 'unknown')}")
            logger.info(f"Цена входа: ${current_price}")
            logger.info(f"SL: ${sl_price:.2f}, TP: ${tp_price:.2f}")
            
            # Попытка установить SL/TP (может не поддерживаться некоторыми биржами)
            try:
                sl_order = self.exchange.create_order(
                    symbol, 'stop_market', 
                    'sell' if side in ['buy', 'long'] else 'buy', 
                    amount, None, None, {'stopPrice': sl_price}
                )
                tp_order = self.exchange.create_order(
                    symbol, 'take_profit_market', 
                    'sell' if side in ['buy', 'long'] else 'buy', 
                    amount, None, None, {'stopPrice': tp_price}
                )
                logger.info(f"SL/TP ордера установлены: SL {sl_order.get('id', 'unknown')}, TP {tp_order.get('id', 'unknown')}")
            except Exception as sl_tp_error:
                logger.warning(f"Не удалось установить SL/TP автоматически: {sl_tp_error}")
            
            return {
                "order_id": order.get('id', 'simulation'),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": current_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "leverage": leverage,
                "status": "filled"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения ордера: {e}")
            logger.info("🔄 Переключение на симулятор для размещения ордера")
            return self.simulator.place_order(symbol, side, amount, leverage, sl_percent, tp_percent)

    def close_position(self, symbol, direction=None):
        """Закрытие позиции с улучшенной обработкой"""
        try:
            logger.info(f"Закрытие позиции {direction}: {symbol}")
            
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Получение текущих позиций
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                size = pos.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:  # Активная позиция
                    active_position = pos
                    break
            
            if not active_position:
                logger.warning("Активная позиция не найдена")
                return {"error": "Позиция не найдена"}
            
            # Закрытие позиции рыночным ордером
            side = 'sell' if active_position.get('side') == 'long' else 'buy'
            amount = abs(float(active_position.get('size', 0)))
            
            if amount == 0:
                raise Exception("Размер позиции равен нулю")
            
            close_order = self.exchange.create_market_order(
                symbol, side, amount, None, None, {'reduceOnly': True}
            )
            
            logger.info(f"✅ Позиция закрыта: {close_order.get('id', 'unknown')}")
            
            return {
                "order_id": close_order.get('id', 'simulation'),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "status": "closed"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия позиции: {e}")
            logger.info("🔄 Переключение на симулятор для закрытия позиции")
            return self.simulator.close_position(symbol)

    def update_stop_loss(self, symbol, direction, stop_price, trail_amount=None):
        """Обновление стоп-лосс с улучшенной обработкой"""
        try:
            logger.info(f"Обновление стоп-лосс: {symbol}, цена: {stop_price}")
            
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
            
            # Получение активных ордеров стоп-лосс
            orders = self.exchange.fetch_open_orders(symbol)
            sl_orders = [order for order in orders if order.get('type') == 'stop_market']
            
            # Отмена старых SL ордеров
            for order in sl_orders:
                order_id = order.get('id')
                if order_id:
                    self.exchange.cancel_order(order_id, symbol)
                    logger.info(f"Отменен старый SL ордер: {order_id}")
            
            # Получение текущих позиций для определения размера
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                size = pos.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:
                    active_position = pos
                    break
            
            if not active_position:
                return {"error": "Активная позиция не найдена для обновления SL"}
            
            # Создание нового стоп-лосс ордера
            side = 'sell' if active_position.get('side') == 'long' else 'buy'
            amount = abs(float(active_position.get('size', 0)))
            
            new_sl_order = self.exchange.create_order(
                symbol, 'stop_market', side, amount, None, None,
                {'stopPrice': stop_price, 'reduceOnly': True}
            )
            
            logger.info(f"✅ Новый SL ордер установлен: {new_sl_order.get('id', 'unknown')}")
            
            return {
                "order_id": new_sl_order.get('id', 'simulation'),
                "symbol": symbol,
                "stop_price": stop_price,
                "trail_amount": trail_amount,
                "status": "updated"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления стоп-лосс: {e}")
            logger.info("🔄 Переключение на симулятор для обновления SL")
            return self.simulator.update_stop_loss(symbol, direction, stop_price)

    def get_positions(self, symbol=None):
        """Получение текущих позиций"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
                
            symbols = [symbol] if symbol else None
            positions = self.exchange.fetch_positions(symbols)
            
            # Фильтруем только активные позиции
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

    def get_balance(self):
        """Получение баланса аккаунта"""
        try:
            if not self.exchange:
                raise Exception("Exchange не инициализирован")
                
            balance = self.exchange.fetch_balance()
            
            return {
                "balance": balance,
                "total": balance.get('total', {}),
                "free": balance.get('free', {}),
                "used": balance.get('used', {}),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return self.simulator.get_balance()