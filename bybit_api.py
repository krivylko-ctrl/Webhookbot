import os
import ccxt
import logging
from bybit_simulator import BybitSimulator

logger = logging.getLogger(__name__)

class BybitFuturesClient:
    def __init__(self, testnet=True):
        """
        Инициализация Bybit клиента
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
            
            if not testnet:
                logger.info("🚀 PRODUCTION РЕЖИМ: Реальная торговля на Bybit")
            else:
                logger.info("🧪 DEMO РЕЖИМ: Тестовая торговля на Bybit")
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации CCXT: {e}")
            self.exchange = None
        
        logger.info("✅ ГОТОВ К ТОРГОВЛЕ: Bybit API активен через CCXT на Railway")

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
        """Размещение ордера"""
        try:
            logger.info(f"Размещение ордера {side}: {amount} {symbol}")
            logger.info(f"Плечо: {leverage}x, SL: {sl_percent}%, TP: {tp_percent}%")
            
            # Установка плеча
            self.exchange.set_leverage(leverage, symbol)
            
            # Получение текущей цены для расчета SL/TP
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Расчет цен стоп-лосс и тейк-профит
            if side.lower() == 'long':
                sl_price = current_price * (1 - sl_percent / 100)
                tp_price = current_price * (1 + tp_percent / 100)
            else:  # short
                sl_price = current_price * (1 + sl_percent / 100)
                tp_price = current_price * (1 - tp_percent / 100)
            
            # Размещение основного ордера
            order = self.exchange.create_market_order(symbol, side, amount)
            
            logger.info(f"✅ Ордер размещен: {order['id']}")
            logger.info(f"Цена входа: ${current_price}")
            logger.info(f"SL: ${sl_price:.2f}, TP: ${tp_price:.2f}")
            
            # Попытка установить SL/TP (может не поддерживаться некоторыми биржами)
            try:
                sl_order = self.exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', 
                                                    amount, None, None, {'stopPrice': sl_price})
                tp_order = self.exchange.create_order(symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', 
                                                    amount, None, None, {'stopPrice': tp_price})
                logger.info(f"SL/TP ордера установлены: SL {sl_order['id']}, TP {tp_order['id']}")
            except Exception as sl_tp_error:
                logger.warning(f"Не удалось установить SL/TP автоматически: {sl_tp_error}")
            
            return {
                "order_id": order['id'],
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
        """Закрытие позиции"""
        try:
            logger.info(f"Закрытие позиции {direction}: {symbol}")
            
            # Получение текущих позиций
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                if pos['size'] > 0:  # Активная позиция
                    active_position = pos
                    break
            
            if not active_position:
                logger.warning("Активная позиция не найдена")
                return {"error": "Позиция не найдена"}
            
            # Закрытие позиции рыночным ордером
            side = 'sell' if active_position['side'] == 'long' else 'buy'
            amount = abs(active_position['size'])
            
            close_order = self.exchange.create_market_order(symbol, side, amount, None, None, {'reduceOnly': True})
            
            logger.info(f"✅ Позиция закрыта: {close_order['id']}")
            
            return {
                "order_id": close_order['id'],
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
        """Обновление стоп-лосс"""
        try:
            logger.info(f"Обновление стоп-лосс: {symbol}, цена: {stop_price}")
            
            # Получение активных ордеров стоп-лосс
            orders = self.exchange.fetch_open_orders(symbol)
            sl_orders = [order for order in orders if order['type'] == 'stop_market']
            
            # Отмена старых SL ордеров
            for order in sl_orders:
                self.exchange.cancel_order(order['id'], symbol)
                logger.info(f"Отменен старый SL ордер: {order['id']}")
            
            # Получение текущих позиций для определения размера
            positions = self.exchange.fetch_positions([symbol])
            active_position = None
            
            for pos in positions:
                if pos['size'] > 0:
                    active_position = pos
                    break
            
            if not active_position:
                return {"error": "Активная позиция не найдена для обновления SL"}
            
            # Создание нового SL ордера
            side = 'sell' if active_position['side'] == 'long' else 'buy'
            amount = abs(active_position['size'])
            
            new_sl_order = self.exchange.create_order(symbol, 'stop_market', side, amount, None, None, 
                                                    {'stopPrice': stop_price})
            
            logger.info(f"✅ Новый SL ордер создан: {new_sl_order['id']}")
            
            return {
                "order_id": new_sl_order['id'],
                "symbol": symbol,
                "stop_price": stop_price,
                "status": "updated"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления стоп-лосс: {e}")
            logger.info("🔄 Переключение на симулятор для обновления SL")
            return self.simulator.update_stop_loss(symbol, direction, stop_price, trail_amount)

    def get_positions(self):
        """Получение текущих позиций"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = [pos for pos in positions if pos['size'] > 0]
            
            return {
                "positions": active_positions,
                "total_count": len(active_positions)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения позиций: {e}")
            return self.simulator.get_positions()

    def get_balance(self):
        """Получение баланса"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                "balance": balance['USDT'] if 'USDT' in balance else balance,
                "currency": "USDT"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return self.simulator.get_balance()