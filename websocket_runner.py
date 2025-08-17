#!/usr/bin/env python3
"""
WebSocket Runner для KWIN Trading Bot
Запускает WebSocket подключение для получения реальных данных с Bybit
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BybitWebSocket:
    def __init__(self):
        self.url = "wss://stream.bybit.com/v5/public/linear"
        self.subscriptions = [
            "kline.15.BTCUSDT",  # 15-минутные свечи BTC/USDT
            "kline.15.ETHUSDT",  # 15-минутные свечи ETH/USDT
            "publicTrade.BTCUSDT",  # Публичные сделки
            "publicTrade.ETHUSDT"
        ]
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Подключение к WebSocket"""
        try:
            logger.info(f"Подключение к {self.url}")
            self.websocket = await websockets.connect(self.url)
            logger.info("WebSocket подключен успешно")
            
            # Подписываемся на каналы
            await self.subscribe()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка подключения WebSocket: {e}")
            return False
    
    async def subscribe(self):
        """Подписка на каналы данных"""
        if not self.websocket:
            return
            
        subscription_msg = {
            "op": "subscribe",
            "args": self.subscriptions
        }
        
        await self.websocket.send(json.dumps(subscription_msg))
        logger.info(f"Подписались на каналы: {self.subscriptions}")
    
    async def listen(self):
        """Основной цикл получения данных"""
        self.running = True
        logger.info("Начинаем прослушивание WebSocket данных...")
        
        try:
            while self.running:
                if not self.websocket:
                    break
                    
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Обрабатываем различные типы сообщений
                if 'topic' in data:
                    await self.handle_data(data)
                elif 'success' in data:
                    logger.info(f"Подписка успешна: {data}")
                else:
                    logger.debug(f"Неизвестное сообщение: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket соединение закрыто")
        except Exception as e:
            logger.error(f"Ошибка в цикле WebSocket: {e}")
        finally:
            self.running = False
    
    async def handle_data(self, data):
        """Обработка входящих данных"""
        topic = data.get('topic', '')
        
        if 'kline' in topic:
            await self.handle_kline_data(data)
        elif 'publicTrade' in topic:
            await self.handle_trade_data(data)
    
    async def handle_kline_data(self, data):
        """Обработка данных свечей"""
        try:
            if 'data' not in data or not data['data']:
                return
                
            kline_data = data['data'][0]
            # Исправлена структура данных Bybit WebSocket
            symbol = kline_data.get('symbol', 'UNKNOWN')
            if not symbol or symbol == 'UNKNOWN':
                # Попробуем извлечь из topic
                topic = data.get('topic', '')
                if 'kline' in topic:
                    # Формат: kline.15.BTCUSDT
                    parts = topic.split('.')
                    if len(parts) >= 3:
                        symbol = parts[2]
            
            timestamp = datetime.fromtimestamp(int(kline_data.get('start', 0)) / 1000)
            
            logger.info(f"Свеча {symbol}: "
                       f"O={kline_data.get('open', 0)}, "
                       f"H={kline_data.get('high', 0)}, "
                       f"L={kline_data.get('low', 0)}, "
                       f"C={kline_data.get('close', 0)}, "
                       f"V={kline_data.get('volume', 0)}, "
                       f"Time={timestamp}")
                       
        except Exception as e:
            logger.error(f"Ошибка обработки kline данных: {e}")
            logger.debug(f"Данные для отладки: {data}")
    
    async def handle_trade_data(self, data):
        """Обработка данных сделок"""
        try:
            for trade in data['data']:
                symbol = trade['s']
                price = trade['p']
                size = trade['v']
                side = trade['S']
                timestamp = datetime.fromtimestamp(int(trade['T']) / 1000)
                
                logger.debug(f"Сделка {symbol}: {side} {size} @ {price} ({timestamp})")
                
        except Exception as e:
            logger.error(f"Ошибка обработки trade данных: {e}")
    
    async def close(self):
        """Закрытие соединения"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket соединение закрыто")

async def main():
    """Главная функция"""
    logger.info("Запуск WebSocket Runner для KWIN Trading Bot")
    
    ws = BybitWebSocket()
    
    try:
        # Подключаемся
        if await ws.connect():
            # Слушаем данные
            await ws.listen()
        else:
            logger.error("Не удалось подключиться к WebSocket")
            
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        await ws.close()
        logger.info("WebSocket Runner завершён")

if __name__ == "__main__":
    # Проверяем доступность интернета
    try:
        import requests
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=5)
        if response.status_code == 403:
            logger.warning("⚠️ Географические ограничения Bybit API. WebSocket может не работать.")
        elif response.status_code == 200:
            logger.info("✅ Доступ к Bybit API подтвержден")
    except:
        logger.warning("⚠️ Проблемы с доступом к интернету или Bybit API")
    
    # Запускаем WebSocket
    asyncio.run(main())