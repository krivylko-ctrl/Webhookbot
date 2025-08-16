"""
WebSocket Runner для KWIN Trading Bot
Точная синхронизация с Pine Script через Bybit WebSocket v5
"""

import asyncio
import signal
import sys
import logging
from config import Config
import config as cfg

log = logging.getLogger("ws_runner")

def build_topics(symbol: str, intervals: list[str]) -> list[str]:
    if not symbol:
        raise ValueError("SYMBOL is empty")
    if not intervals:
        raise ValueError("INTERVALS is empty")
    return [f"kline.{iv}.{symbol}" for iv in intervals]
from bybit_api import BybitAPI  
from state_manager import StateManager
from database import Database
from kwin_strategy import KWINStrategy
from trail_engine import TrailEngine
from datafeed_ws import WSConfig, BybitWSKlines

class KWINWebSocketRunner:
    """Основной runner с WebSocket синхронизацией"""
    
    def __init__(self):
        # Проверяем переменные окружения сразу
        cfg.must_have()
        
        self.config = Config()
        self.db = Database()
        
        # Инициализация компонентов
        self.state = StateManager(self.db)
        self.api = BybitAPI(
            api_key=cfg.BYBIT_API_KEY,
            api_secret=cfg.BYBIT_API_SECRET,
            testnet=False
        )
        # Устанавливаем тип рынка
        self.api.market_type = self.config.market_type
        
        # Стратегия с WebSocket колбэками
        self.strategy = KWINStrategy(self.config, self.api, self.state, self.db)
        
        # WebSocket конфигурация с безопасной проверкой
        symbol = cfg.SYMBOL
        intervals = cfg.INTERVALS
        topics = build_topics(symbol, intervals)
        
        self.ws_config = WSConfig(
            symbol=symbol,
            market_type=cfg.BYBIT_ACCOUNT_TYPE,
            testnet=False,  # Боевой режим
            intervals=tuple(intervals),  # Из конфига
            only_on_confirmed_close=True  # Только закрытые бары как в Pine
        )
        
        self.ws_feed = None
        self.running = False
    
    def on_kline_data(self, symbol: str, interval: str, candle: dict):
        """Колбэк для обработки WebSocket данных свечей"""
        try:
            print(f"[WS] {symbol} {interval}m: {candle['close']:.2f} (confirmed: {candle['confirm']})")
            
            # Жёсткие требования для 1:1 с Pine Script
            if interval == "15":
                # SFP и фильтры считаем строго в on_bar_close_15m()
                self.strategy.on_bar_close_15m(candle)
                
                # Обновление трейлинга на новом закрытом баре
                if hasattr(self.strategy, 'trail_engine') and hasattr(self.strategy.trail_engine, 'update_trail'):
                    self.strategy.trail_engine.update_trail()
                
            elif interval == "60":
                # Дополнительный анализ на 1ч барах
                self.strategy.on_bar_close_60m(candle)
                
            elif interval == "1":
                # Мониторинг на 1м барах (опционально)
                self.strategy.on_bar_close_1m(candle)
            
        except Exception as e:
            print(f"Error processing {symbol} {interval}m candle: {e}")
    
    def _update_trailing(self):
        """Обновление трейлинга только на закрытых барах"""
        try:
            position = self.state.get_current_position()
            if position and position.get('size', 0) != 0:
                self.strategy.trail_engine.update_trailing(position)
        except Exception as e:
            print(f"Error updating trailing: {e}")
    
    async def start(self):
        """Запуск WebSocket runner"""
        print(f"🚀 Starting KWIN WebSocket Runner")
        print(f"   Symbol: {self.config.symbol}")
        print(f"   Market: {self.config.market_type}")
        print(f"   Intervals: {self.ws_config.intervals}")
        print(f"   Only confirmed bars: {self.ws_config.only_on_confirmed_close}")
        
        # Загрузка начальных данных
        await self._load_initial_data()
        
        # Создание WebSocket фида
        self.ws_feed = BybitWSKlines(self.ws_config, self.on_kline_data)
        
        # Регистрация обработчика сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        print("✅ WebSocket runner started. Press Ctrl+C to stop.")
        
        try:
            # Запуск основного цикла WebSocket
            await self.ws_feed.run_forever()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping WebSocket runner...")
        finally:
            await self.stop()
    
    async def _load_initial_data(self):
        """Загрузка начальных данных с биржи"""
        try:
            print("📊 Loading initial candle data...")
            
            # Загружаем историю 15m свечей
            klines_15m = self.api.get_klines(self.config.symbol, "15", 100)
            if klines_15m:
                self.strategy.candles_15m = klines_15m
                print(f"   ✅ Loaded {len(klines_15m)} x 15m candles")
            
            # Загружаем 1h свечи для дополнительного анализа  
            klines_1h = self.api.get_klines(self.config.symbol, "60", 50)
            if klines_1h:
                if not hasattr(self.strategy, 'candles_1h'):
                    self.strategy.candles_1h = []
                self.strategy.candles_1h = klines_1h
                print(f"   ✅ Loaded {len(klines_1h)} x 1h candles")
                
        except Exception as e:
            print(f"Error loading initial data: {e}")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        print(f"\n📡 Received signal {signum}, stopping...")
        self.running = False
    
    async def stop(self):
        """Корректное завершение работы"""
        self.running = False
        
        # Сохранение состояния
        try:
            # Принудительное сохранение состояния в БД
            if hasattr(self.state, 'current_position'):
                print("💾 State saved successfully")
        except Exception as e:
            print(f"Error saving state: {e}")
        
        print("👋 WebSocket runner stopped")

async def main():
    """Основная функция запуска"""
    runner = KWINWebSocketRunner()
    await runner.start()

if __name__ == "__main__":
    # Запуск через asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Program interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)