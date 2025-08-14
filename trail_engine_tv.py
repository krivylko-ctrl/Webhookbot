# TradingView Enhanced Trail Engine - Production Ready
from dataclasses import dataclass, field
from typing import Optional, Dict
import time
import logging
import os
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    symbol: str
    side: str                 # "long" | "short"
    entry_price: Optional[float] = None
    stop: Optional[float] = None
    last_exchange_stop: Optional[float] = None
    last_update_ts: float = field(default_factory=lambda: 0.0)

# Хранилище состояний по символам
STATE: Dict[str, PositionState] = {}

# Настройки антиспама и политики приёма уровня TV
ANTI_SPAM_SECONDS = float(os.getenv("TRAIL_ANTI_SPAM_SEC", "2.0"))
MIN_STEP_TICKS    = int(os.getenv("TRAIL_MIN_STEP_TICKS", "1"))
TICK_SIZE         = float(os.getenv("TRAIL_TICK_SIZE", "0.01"))

# ВАЖНО: на отладке ставим True — сервер НЕ пересчитывает уровень
ACCEPT_TV_LEVEL_AS_SOURCE_OF_TRUTH = True

def round_to_tick(px: float) -> float:
    """Округление цены к tick size"""
    return round(px / TICK_SIZE) * TICK_SIZE

def is_stricter(side: str, new_stop: float, old_stop: Optional[float]) -> bool:
    """Проверка, что новый стоп строже старого"""
    if old_stop is None:
        return True
    if side == "long":
        return new_stop > old_stop   # для long стоп только вверх
    else:
        return new_stop < old_stop   # для short стоп только вниз

def step_ok(new_stop: float, old_stop: Optional[float]) -> bool:
    """Проверка минимального шага"""
    if old_stop is None:
        return True
    return abs(new_stop - old_stop) >= MIN_STEP_TICKS * TICK_SIZE

class ExchangeClient:
    """Абстракция над биржей с интеграцией Bybit V5"""
    
    def __init__(self):
        try:
            from bybit_v5_fixed import get_client
            self.client = get_client()
        except ImportError:
            logger.error("❌ Cannot import bybit_v5_fixed.get_client()")
            self.client = None
    
    def cancel_all_take_profits(self, symbol: str):
        """Отмена всех take profit ордеров"""
        logger.info(f"[EX] Cancel all TPs: {symbol}")
        if not self.client:
            return
        
        try:
            # Получаем активные ордера
            orders_resp = self.client.make_request("GET", "/v5/order/realtime", {
                "category": "linear",
                "symbol": symbol,
                "orderFilter": "Order"
            })
            
            if orders_resp.get("retCode") == 0:
                orders = orders_resp.get("result", {}).get("list", [])
                for order in orders:
                    if order.get("orderType") == "Limit" and order.get("reduceOnly"):
                        # Отменяем TP ордер
                        cancel_resp = self.client.make_request("POST", "/v5/order/cancel", {
                            "category": "linear",
                            "symbol": symbol,
                            "orderId": order.get("orderId")
                        })
                        logger.info(f"[EX] Cancelled TP order: {order.get('orderId')}")
        except Exception as e:
            logger.error(f"❌ Failed to cancel TPs: {e}")

    def place_or_replace_stop(self, symbol: str, side: str, stop_price: float, reduce_only: bool = True):
        """Установка/обновление стоп-лосса"""
        logger.info(f"[EX] Set STOP {side} {symbol} @ {stop_price} (reduce_only={reduce_only})")
        if not self.client:
            return
        
        try:
            # Используем trading-stop для установки SL
            payload = {
                "category": "linear",
                "symbol": symbol,
                "stopLoss": str(stop_price),
                "slTriggerBy": "LastPrice"
            }
            result = self.client.make_request("POST", "/v5/position/trading-stop", payload)
            
            if result.get("retCode") == 0:
                logger.info(f"✅ Stop loss set: {symbol} @ {stop_price}")
            else:
                logger.warning(f"⚠️ Failed to set stop: {result.get('retMsg')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to set stop: {e}")

# Глобальный клиент биржи
EX = ExchangeClient()

def ensure_state(symbol: str) -> PositionState:
    """Обеспечение существования состояния для символа"""
    if symbol not in STATE:
        STATE[symbol] = PositionState(symbol=symbol, side="long")
    return STATE[symbol]

def handle_entry(symbol: str, direction: str, entry_price: float, stop_loss: float,
                 qty: float, cancel_take_profit: bool = False):
    """Обработка входа в позицию"""
    st = ensure_state(symbol)
    st.side = direction
    st.entry_price = entry_price
    st.stop = round_to_tick(stop_loss)

    if cancel_take_profit:
        EX.cancel_all_take_profits(symbol)

    # Ставим первичный стоп
    EX.place_or_replace_stop(symbol, direction, st.stop, reduce_only=True)
    st.last_exchange_stop = st.stop
    st.last_update_ts = time.time()
    logger.info(f"[ENTRY] {symbol} {direction} qty={qty} ep={entry_price} SL={st.stop} TP cancelled={cancel_take_profit}")

def handle_trail_init(symbol: str, direction: str, hint_price: Optional[float],
                      trail_points: Optional[float], trail_offset: Optional[float],
                      force: bool = False):
    """Инициализация трейлинга"""
    st = ensure_state(symbol)
    st.side = direction
    
    if hint_price is not None:
        new_stop = round_to_tick(hint_price)
        
        if is_stricter(st.side, new_stop, st.stop):
            now = time.time()
            if force or (now - st.last_update_ts >= ANTI_SPAM_SECONDS and step_ok(new_stop, st.stop)):
                st.stop = new_stop
                EX.place_or_replace_stop(symbol, st.side, st.stop, reduce_only=True)
                st.last_exchange_stop = st.stop
                st.last_update_ts = now
                logger.info(f"[TRAIL_INIT] {symbol} {st.side} SL={st.stop} (force={force})")

def handle_trail_update(symbol: str, direction: str, new_trail_stop: float,
                        force: bool = False):
    """Обновление трейлинг стопа"""
    st = ensure_state(symbol)
    st.side = direction
    new_stop = round_to_tick(new_trail_stop)

    # Проверка монотонности
    if not is_stricter(st.side, new_stop, st.stop):
        logger.info(f"[TRAIL_SKIP] not stricter: old={st.stop} new={new_stop}")
        return

    now = time.time()
    if force:
        # Принудительное обновление
        st.stop = new_stop
        EX.place_or_replace_stop(symbol, st.side, st.stop, reduce_only=True)
        st.last_exchange_stop = st.stop
        st.last_update_ts = now
        logger.info(f"[TRAIL_FORCE] {symbol} {st.side} SL={st.stop}")
        return

    # Обычный режим с проверками
    if (now - st.last_update_ts) < ANTI_SPAM_SECONDS:
        logger.info(f"[TRAIL_SPAM] skipped by time gate {symbol}")
        return
    if not step_ok(new_stop, st.stop):
        logger.info(f"[TRAIL_STEP] skipped by step gate {symbol}")
        return

    st.stop = new_stop
    EX.place_or_replace_stop(symbol, st.side, st.stop, reduce_only=True)
    st.last_exchange_stop = st.stop
    st.last_update_ts = now
    logger.info(f"[TRAIL] {symbol} {st.side} SL={st.stop}")

def handle_exit(symbol: str, direction: str, reason: str = ""):
    """Обработка выхода из позиции"""
    logger.info(f"[EXIT] {symbol} {direction} reason={reason}")
    
    # Очищаем состояние
    if symbol in STATE:
        del STATE[symbol]
        logger.info(f"[EXIT] Cleared state for {symbol}")

def get_position_state(symbol: str) -> Optional[PositionState]:
    """Получение текущего состояния позиции"""
    return STATE.get(symbol)

def clear_all_states():
    """Очистка всех состояний"""
    STATE.clear()
    logger.info("[CLEAR] All position states cleared")