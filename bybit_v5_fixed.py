# bybit_v5_fixed.py - Точная реализация Bybit V5 API по спецификации
import os
import json
import time
import hmac
import hashlib
import requests
import logging
from typing import Optional, Dict, Any
from math import floor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ===== УЛУЧШЕННАЯ МЕТА-ИНФОРМАЦИЯ И КЕШИРОВАНИЕ =====
@dataclass
class InstInfo:
    tick: float
    lot: float
    min_qty: float

class InstCache:
    def __init__(self, client):
        self.client = client
        self.cache = {}      # symbol -> (InstInfo, ts)
        self.ttl = 300       # 5 минут

    def _fetch(self, symbol: str) -> InstInfo:
        # v5 instruments info
        params = {"category": "linear", "symbol": symbol}
        r = self.client.make_request("GET", "/v5/market/instruments-info", params)
        if r.get("retCode") != 0 or not r.get("result", {}).get("list"):
            # fallback values
            return InstInfo(tick=0.01, lot=0.001, min_qty=0.001)
        
        data = r["result"]["list"][0]
        tick = float(data["priceFilter"]["tickSize"])
        lot = float(data["lotSizeFilter"]["qtyStep"])
        minq = float(data["lotSizeFilter"]["minOrderQty"])
        return InstInfo(tick=tick, lot=lot, min_qty=minq)

    def get(self, symbol: str) -> InstInfo:
        rec = self.cache.get(symbol)
        if not rec or time.time() - rec[1] > self.ttl:
            info = self._fetch(symbol)
            self.cache[symbol] = (info, time.time())
            return info
        return rec[0]

# ===== АНТИ-ДУБЛИКАТЫ ДЛЯ TRAIL/EXIT =====
LAST_ACTIONS = {
    "trail_key": None,     # e.g. "ETHUSDT:long:4554.40"
    "trail_ts": 0,
    "exit_key": None,      # e.g. "ETHUSDT:short:sl"
    "exit_ts": 0,
}
DEBOUNCE_SEC = 10  # окно гашения дублей

def norm_symbol(s: str) -> str:
    """Нормализация символа: ETHUSDT.P -> ETHUSDT"""
    return s.replace(".P", "").replace(":PERP", "").upper()

def is_dup_trail(symbol, direction, new_stop):
    key = f"{symbol}:{direction}:{new_stop:.2f}"
    now = time.time()
    if LAST_ACTIONS["trail_key"] == key and (now - LAST_ACTIONS["trail_ts"]) < DEBOUNCE_SEC:
        return True
    LAST_ACTIONS["trail_key"] = key
    LAST_ACTIONS["trail_ts"] = now
    return False

def is_dup_exit(symbol, direction, reason):
    key = f"{symbol}:{direction}:{reason}"
    now = time.time()
    if LAST_ACTIONS["exit_key"] == key and (now - LAST_ACTIONS["exit_ts"]) < DEBOUNCE_SEC:
        return True
    LAST_ACTIONS["exit_key"] = key
    LAST_ACTIONS["exit_ts"] = now
    return False

class BybitV5API:
    """
    Прямая реализация Bybit V5 API с точной синхронизацией времени
    Соответствует спецификации V5: timestamp + api_key + recv_window + body
    """
    def __init__(self, api_key: str, api_secret: str):
        # Определение базового URL с поддержкой testnet
        use_testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        self.BASE_URL = os.getenv("BYBIT_BASE") or ("https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com")
        self.BASE_URL = self.BASE_URL.rstrip("/")
    
        self.api_key = api_key
        self.api_secret = api_secret
        self.category = (os.getenv("BYBIT_CATEGORY") or "linear").lower()  # Принудительно linear для USDT-перпетуалов
        self.server_time_offset = 0
        self.last_sync_time = 0
        self._filters_cache = {}  # кэш биржевых фильтров
        
        # Инициализация улучшенного кеша инструментов
        self.inst_cache = InstCache(self)
        
        logger.info("🚀 Bybit V5 API инициализация по спецификации:")
        logger.info(f"  API Key: {'✅ НАЙДЕН' if api_key else '❌ НЕ НАЙДЕН'}")
        logger.info(f"  API Secret: {'✅ НАЙДЕН' if api_secret else '❌ НЕ НАЙДЕН'}")
        logger.info(f"  Base URL: {self.BASE_URL}")
        logger.info(f"  Mode: {'TESTNET' if 'testnet' in self.BASE_URL else 'MAINNET'}")
        
        # Первичная синхронизация времени
        self.sync_server_time()
        
    def sync_server_time(self) -> bool:
        """Синхронизация времени с сервером Bybit по /v5/market/time"""
        try:
            response = requests.get(f"{self.BASE_URL}/v5/market/time", timeout=5)
            if response.status_code == 200:
                server_time_seconds = int(response.json()["result"]["timeSecond"])
                server_time_ms = server_time_seconds * 1000
                local_time_ms = int(time.time() * 1000)
                
                self.server_time_offset = server_time_ms - local_time_ms
                self.last_sync_time = local_time_ms
                
                logger.info(f"⏰ Время синхронизировано с Bybit сервером")
                logger.info(f"   Смещение: {self.server_time_offset}ms")
                return True
            else:
                logger.error(f"❌ Ошибка синхронизации времени: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Критическая ошибка синхронизации времени: {e}")
            return False
    
    def get_server_timestamp(self) -> int:
        """Получить точный timestamp сервера с автосинхронизацией каждые 60 секунд"""
        current_time = int(time.time() * 1000)
        
        # Автосинхронизация каждые 60 секунд для дельты ≤1с
        if current_time - self.last_sync_time > 60000:
            self.sync_server_time()
        
        return current_time + self.server_time_offset
    
    def generate_signature(self, timestamp: str, recv_window: str, body: str = "") -> str:
        """
        Генерация HMAC-SHA256 подписи по спецификации Bybit V5
        Строка подписи: timestamp + api_key + recv_window + body
        """
        pre_sign_string = timestamp + self.api_key + recv_window + body
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            pre_sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"🔐 Подпись V5 для: {pre_sign_string[:50]}...")
        return signature
    
    def _compact(self, d):
        """Компактный JSON для V5 API"""
        return json.dumps(d, separators=(",", ":")) if d is not None else ""

    def _query(self, params: Optional[Dict]) -> str:
        """Генерация query string для GET запросов V5"""
        if not params:
            return ""
        # ВАЖНО: сортируем ключи и не кодируем лишнего — Bybit подписывает raw query
        parts = [f"{k}={params[k]}" for k in sorted(params)]
        return "&".join(parts)

    def make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Выполнение HTTP запроса к Bybit V5 API
        GET: подписываем queryString
        POST: подписываем body
        """
        url = f"{self.BASE_URL}{endpoint}"
        ts = str(self.get_server_timestamp())
        recv = "5000"

        method = method.upper()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv,
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                # Подписываем ИМЕННО queryString для GET запросов
                query = self._query(params)
                pre_sign = ts + self.api_key + recv + query
                sign = hmac.new(self.api_secret.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
                headers["X-BAPI-SIGN"] = sign

                full_url = url + (f"?{query}" if query else "")
                r = requests.get(full_url, headers=headers, timeout=10)

            else:  # POST/PUT
                body = self._compact(params or {})
                pre_sign = ts + self.api_key + recv + body
                sign = hmac.new(self.api_secret.encode(), pre_sign.encode(), hashlib.sha256).hexdigest()
                headers["X-BAPI-SIGN"] = sign

                r = requests.post(url, headers=headers, data=body, timeout=10)

            logger.info(f"📡 V5 {method} {endpoint} → HTTP {r.status_code}")
            j = r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"http": r.text}
            
            # Полное логирование ответа Bybit для отладки
            if "retCode" in j:
                logger.info(f"📊 Bybit Response: retCode={j.get('retCode')}, retMsg={j.get('retMsg')}, retExtInfo={j.get('retExtInfo')}")
            
            return j if r.status_code == 200 else {"retCode": -1, "retMsg": f"HTTP {r.status_code}", "raw": j}

        except Exception as e:
            logger.error(f"❌ V5 request error: {e}")
            return {"retCode": -2, "retMsg": str(e)}
    
    def get_instrument(self, symbol: str, category: str = "linear"):
        q = {"category": category, "symbol": symbol}
        info = self.make_request("GET", "/v5/market/instruments-info", q)
        if info.get("retCode") == 0 and info.get("result", {}).get("list"):
            return info["result"]["list"][0]
        return None

    def round_qty(self, symbol: str, qty: float, category: str = "linear") -> str:
        inst = self.get_instrument(symbol, category) or {}
        lot = (inst.get("lotSizeFilter") or {})
        step = float(lot.get("qtyStep", "0.001"))
        min_qty = float(lot.get("minOrderQty", "0.001"))
        # округляем вниз к шагу и убеждаемся, что не меньше минимума
        stepped = max(min_qty, (int(qty / step)) * step)
        # строка без лишних нулей
        return f"{stepped:.10f}".rstrip("0").rstrip(".")
        
    def get_position_size(self, symbol: str, side_hint: str = None, category: str = "linear") -> float:
        # берем список позиций и ищем ненулевую
        res = self.make_request("GET", "/v5/position/list", {"category": category, "symbol": symbol})
        if res.get("retCode") != 0:
            return 0.0
        for p in res.get("result", {}).get("list", []):
            # side = "Buy" (long) или "Sell" (short)
            size = float(p.get("size", "0"))
            if size > 0:
                if side_hint:
                    if (side_hint.lower() == "long" and p.get("side") == "Buy") or \
                       (side_hint.lower() == "short" and p.get("side") == "Sell"):
                        return size
                else:
                    return size
        return 0.0

    def _get_symbol_filters(self, symbol: str) -> dict:
        """Кэшируем биржевые фильтры для символа (step, min, tick)."""
        cache_key = f"filters:{symbol}"
        if cache_key in self._filters_cache:
            return self._filters_cache[cache_key]

        params = {"category": "linear", "symbol": symbol}
        info = self.make_request("GET", "/v5/market/instruments-info", params)
        if info.get("retCode") == 0 and info.get("result", {}).get("list"):
            item = info["result"]["list"][0]
            lot = item.get("lotSizeFilter", {}) or {}
            price = item.get("priceFilter", {}) or {}
            filters = {
                "qtyStep": float(lot.get("qtyStep", "0.001")),
                "minOrderQty": float(lot.get("minOrderQty", "0.001")),
                "tickSize": float(price.get("tickSize", "0.01")),
            }
            self._filters_cache[cache_key] = filters
            return filters
        # дефолты на всякий
        return {"qtyStep": 0.001, "minOrderQty": 0.001, "tickSize": 0.01}

    def _round_step_down(self, value: float, step: float) -> float:
        return (int(value / step) * step)

    def round_price_enhanced(self, symbol: str, price: float) -> float:
        """Улучшенное округление цены с кешированием"""
        info = self.inst_cache.get(symbol)
        return floor(price / info.tick) * info.tick

    def round_qty_enhanced(self, symbol: str, qty: float) -> float:
        """Улучшенное округление объема с кешированием"""
        info = self.inst_cache.get(symbol)
        return floor(qty / info.lot) * info.lot

    def validate_qty_enhanced(self, symbol: str, qty: float) -> bool:
        """Проверка минимального объема"""
        info = self.inst_cache.get(symbol)
        return qty >= info.min_qty

    def _get_qty_rules(self, symbol: str):
        """Берём шаг лота и минималку из /v5/market/instruments-info."""
        params = {"category": "linear", "symbol": symbol}
        r = self.make_request("GET", "/v5/market/instruments-info", params)
        if r.get("retCode") != 0 or not r.get("result", {}).get("list"):
            return 0.001, 0.001  # дефолтные значения
        data = r["result"]["list"][0]
        lot = data["lotSizeFilter"]
        qty_step = float(lot["qtyStep"])
        min_qty = float(lot["minOrderQty"])
        return qty_step, min_qty

    def _get_open_position(self, symbol: str):
        """Читаем открытую позицию из /v5/position/list. Возвращаем (size, side, positionIdx)."""
        params = {"category": "linear", "symbol": symbol}
        r = self.make_request("GET", "/v5/position/list", params)
        if r.get("retCode") != 0:
            return 0.0, None, 0
        items = r.get("result", {}).get("list", [])
        # one-way: positionIdx == 0. hedge: 1=Buy/Long, 2=Sell/Short
        pos = max(items, key=lambda x: abs(float(x.get("size", "0"))), default=None)
        if not pos:
            return 0.0, None, 0
        size = float(pos.get("size", "0"))
        side = pos.get("side")  # "Buy" или "Sell"
        idx = int(pos.get("positionIdx", 0))
        return size, side, idx

    def _floor_to_step(self, qty: float, step: float) -> float:
        return floor(qty / step) * step

    def _cancel_all_tp_sl(self, symbol: str):
        """Мягко убираем активные ордера на символе (TP/SL/прочие), чтобы не мешали закрытию."""
        try:
            payload = {"category": "linear", "symbol": symbol}
            self.make_request("POST", "/v5/order/cancel-all", payload)
        except Exception as e:
            logger.warning(f"cancel-all failed: {e}")

    def close_position_market_hard(self, symbol: str, prefer_cancel_all: bool = True):
        """
        Закрыть всю позицию по рынку:
        - игнорирует qty из вебхука
        - берёт фактический posSize
        - округляет под qtyStep
        - reduceOnly + IOC
        """
        # 1) Читаем позицию
        pos_size, pos_side, pos_idx = self._get_open_position(symbol)
        if pos_size <= 0:
            logger.info(f"[exit] Position already flat on {symbol} — nothing to close.")
            return {"status": "info", "message": "Position already closed"}

        # 2) Правила количества
        qty_step, min_qty = self._get_qty_rules(symbol)
        qty = self._floor_to_step(pos_size, qty_step)
        if qty <= 0:
            logger.info(f"[exit] Computed qty {qty} <= 0 after step rounding — skip.")
            return {"status": "info", "message": "Qty too small after rounding"}
        
        if qty < min_qty:
            # Если шаг округлил слишком мало, но позиция по факту ≥ min_qty — подтянем до min_qty в сетке шага
            if pos_size >= min_qty:
                k = max(1, floor(min_qty / qty_step))
                qty = k * qty_step
            else:
                logger.info(f"[exit] pos_size {pos_size} < min_qty {min_qty} — похоже, уже закрыта.")
                return {"status": "info", "message": "Position size below minimum"}

        # 3) Рекомендуется убрать хвосты (TP/SL), чтобы не конфликтовали
        if prefer_cancel_all:
            self._cancel_all_tp_sl(symbol)

        # 4) Противоположная сторона для закрытия
        side_to_close = "Sell" if pos_side == "Buy" else "Buy"

        # 5) Маркет-закрытие reduceOnly + IOC
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side_to_close,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
            "timeInForce": "IOC",
            "positionIdx": pos_idx,  # 0/1/2 — оставляем как на бирже
        }
        
        logger.info(f"[exit] Closing {qty} {symbol} by Market (reduceOnly, IOC).")
        result = self.make_request("POST", "/v5/order/create", payload)
        
        if result.get("retCode") != 0:
            error_msg = result.get("retMsg", "Unknown error")
            logger.error(f"Bybit close-market failed: {error_msg}")
            return {"status": "error", "message": f"Close failed: {error_msg}"}
        
        order_id = result.get("result", {}).get("orderId", "Unknown")
        return {"status": "success", "message": f"Position closed with qty {qty}", "order_id": order_id}

    def place_entry_two_step(self, symbol: str, side: str, qty: str, 
                            take_profit: Optional[str] = None, stop_loss: Optional[str] = None,
                            leverage: Optional[int] = None) -> Dict[str, Any]:
        """
        Двухшаговый вход: 1) Market ордер, 2) TP/SL через trading-stop
        Решает проблему 'Price invalid' при одновременной отправке
        """
        # Нормализация символа
        symbol = norm_symbol(symbol)
        
        # Установка плеча если указано
        if leverage:
            self.set_leverage(symbol, leverage)
        
        # Округление объема по биржевым правилам
        qty_float = float(qty)
        qty_rounded = self.round_qty_enhanced(symbol, qty_float)
        
        if not self.validate_qty_enhanced(symbol, qty_rounded):
            return {"status": "error", "message": f"Qty {qty_rounded} below minimum"}
        
        # Шаг 1: Чистый Market ордер без TP/SL
        order_params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": f"{qty_rounded:.3f}",
            "reduceOnly": False,
            "timeInForce": "IOC"
        }
        
        logger.info(f"📊 V5 Two-Step Entry: {symbol} {side} {qty_rounded}")
        result = self.make_request("POST", "/v5/order/create", order_params)
        
        if result.get("retCode") != 0:
            error_msg = result.get("retMsg", "Market order failed")
            return {"status": "error", "message": f"Entry failed: {error_msg}"}
        
        order_id = result.get("result", {}).get("orderId", "Unknown")
        
        # Шаг 2: Установка TP/SL через trading-stop (если указаны)
        if take_profit or stop_loss:
            time.sleep(0.1)  # Микро-задержка для обработки market ордера
            
            tp_params = {"category": "linear", "symbol": symbol}
            
            if take_profit:
                tp_rounded = self.round_price_enhanced(symbol, float(take_profit))
                tp_params["takeProfit"] = f"{tp_rounded:.2f}"
                tp_params["tpTriggerBy"] = "LastPrice"
            
            if stop_loss:
                sl_rounded = self.round_price_enhanced(symbol, float(stop_loss))
                tp_params["stopLoss"] = f"{sl_rounded:.2f}"
                tp_params["slTriggerBy"] = "LastPrice"
            
            tp_result = self.make_request("POST", "/v5/position/trading-stop", tp_params)
            
            if tp_result.get("retCode") != 0:
                logger.warning(f"TP/SL setup failed: {tp_result.get('retMsg')}")
            else:
                logger.info(f"🎯 TP/SL установлены: TP={take_profit}, SL={stop_loss}")
        
        return {
            "status": "success", 
            "message": "Two-step entry completed", 
            "order_id": order_id
        }

    def trail_update_enhanced(self, symbol: str, direction: str, new_stop: str) -> Dict[str, Any]:
        """
        Корректное обновление трейлинг-стопа:
        - нормализуем символ
        - округляем цену по tickSize
        - гасим дубликаты на уровне (symbol, side, price)
        - шлём /v5/position/trading-stop (stopLoss, LastPrice)
        """
        symbol = norm_symbol(symbol)
        new_stop_float = float(new_stop)
        new_stop_rounded = self.round_price_enhanced(symbol, new_stop_float)

        # антидубль
        if is_dup_trail(symbol, direction, new_stop_rounded):
            return {"status": "skipped", "message": "Duplicate trail update ignored"}

        # проверим, что позиция есть
        pos_size, pos_side, pos_idx = self._get_open_position(symbol)
        if pos_size <= 0:
            return {"status": "info", "message": "No position to trail"}

        payload = {
            "category": "linear",
            "symbol": symbol,
            "stopLoss": f"{new_stop_rounded:.2f}",
            "slTriggerBy": "LastPrice"
        }
        res = self.make_request("POST", "/v5/position/trading-stop", payload)
        if res.get("retCode") == 0:
            return {"status": "success", "message": f"Trail SL -> {new_stop_rounded:.2f}"}
        else:
            return {"status": "error", "message": f"Trail update failed: {res.get('retMsg', 'Unknown error')}"}

    def exit_enhanced(self, symbol: str, direction: str, reason: str = "", qty: str = None) -> Dict[str, Any]:
        """
        Улучшенный exit с защитой от дублей и неубиваемым закрытием
        """
        symbol = norm_symbol(symbol)
        if is_dup_exit(symbol, direction, reason):
            return {"status": "skipped", "message": "Duplicate exit ignored"}
        return self.close_position_market_hard(symbol, prefer_cancel_all=True)
    
    def test_connection(self) -> Dict[str, Any]:
        """Корректный тест соединения через wallet-balance для проверки прав/UTA"""
        try:
            # 1) Публичное время — проверяем сеть
            t = self.make_request("GET", "/v5/market/time")
            if t.get("retCode", 0) != 0:
                return {"status": "error", "message": f"market/time failed: {t}"}

            # 2) Приватный баланс — проверяем подпись/права/UTA
            res = self.make_request("GET", "/v5/account/wallet-balance", {"accountType":"UNIFIED"})
            rc = res.get("retCode", -1)
            
            if rc == 0:
                return {"status":"connected","message":"V5 API OK","accountType":"UNIFIED"}
            elif rc in (10001,10002):
                return {"status":"error","message":"Invalid signature/timestamp (проверь recvWindow, подпись, время, IP allowlist)", "detail":res}
            elif rc == 10006:
                return {"status":"error","message":"Insufficient permissions (ключ без Trade/Read или IP не разрешён)", "detail":res}
            elif str(rc).startswith("1312"):
                return {"status":"error","message":"Account type / UTA mismatch. Включи UNIFIED или укажи корректный accountType.", "detail":res}
            else:
                return {"status":"error","message":f"retCode {rc}", "detail":res}
                
        except Exception as e:
            return {"status": "error", "message": f"V5 connection test failed: {str(e)}"}
    
    def set_leverage(self, symbol: str, leverage: int = 30, category: str = "linear") -> Dict[str, Any]:
        """Установка плеча для символа"""
        payload = {
            "category": category, 
            "symbol": symbol, 
            "buyLeverage": str(leverage), 
            "sellLeverage": str(leverage)
        }
        logger.info(f"🎯 Setting leverage {leverage}x for {symbol}")
        result = self.make_request("POST", "/v5/position/set-leverage", payload)
        
        rc = result.get("retCode")
        if rc in (0, 110043):
            # 110043 = уже установлено такое же плечо — это OK
            if rc == 110043:
                logger.info(f"✅ Leverage already {leverage}x for {symbol}")
            else:
                logger.info(f"✅ Leverage {leverage}x set for {symbol}")
            return {"status": "success", "leverage": leverage}
        else:
            logger.warning(f"⚠️ Failed to set leverage: {result.get('retMsg', 'Unknown error')}")
            return {"status": "warning", "message": result.get('retMsg', 'Leverage setting failed')}
    
    def close_position(self, symbol: str, direction: str, qty: str = None, category: str = "linear") -> Dict[str, Any]:
        """
        Закрытие позиции. Если qty is None — закрываем ВЕСЬ текущий объём.
        """
        # противоположная сторона для ордера
        side = "Sell" if direction.lower() == "long" else "Buy"

        # если qty не передан – берём точный размер открытой позиции
        if qty is None:
            pos_size = self.get_position_size(symbol, side_hint=("long" if side == "Sell" else "short"), category=category)
            if pos_size <= 0:
                return {"status": "error", "message": "No open position to close"}
            qty = self.round_qty(symbol, pos_size, category)
        else:
            # на всякий случай округлим любой входящий qty под шаг
            qty = self.round_qty(symbol, float(qty), category)

        payload = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
            "timeInForce": "IOC"
        }

        logger.info(f"🔻 Closing position: {symbol} {side} qty={qty}")
        result = self.make_request("POST", "/v5/order/create", payload)

        if result.get("retCode") == 0:
            order_id = result.get("result", {}).get("orderId", "Unknown")
            return {"status": "success", "message": "Position closed", "order_id": order_id}
        else:
            return {"status": "error", "message": f"Failed to close position: {result.get('retMsg', 'Unknown error')}"}
    
    def update_stop_loss(self, symbol: str, stop_loss: str, category: str = "linear") -> Dict[str, Any]:
        """Обновление stop-loss через trading-stop"""
        payload = {
            "category": category,
            "symbol": symbol,
            "stopLoss": str(stop_loss)
        }
        
        logger.info(f"🛡️ Updating stop-loss for {symbol} to {stop_loss}")
        result = self.make_request("POST", "/v5/position/trading-stop", payload)
        
        if result.get("retCode") == 0:
            return {"status": "success", "message": f"Stop-loss updated to {stop_loss}"}
        else:
            error_msg = result.get("retMsg", "Unknown error")
            return {"status": "error", "message": f"Failed to update stop-loss: {error_msg}"}
    
    def place_order(self, symbol: str, side: str, qty: str, price: Optional[str] = None, 
                   take_profit: Optional[str] = None, stop_loss: Optional[str] = None, 
                   leverage: Optional[int] = None) -> Dict[str, Any]:
        """
        Размещение ордера через V5 API
        category: всегда linear для USDT-перпетуалов
        symbol: без суффикса .P (например, ETHUSDT)
        """
        # Нормализация символа: удаляем .P суффикс
        if symbol.endswith(".P"):
            symbol = symbol[:-2]
            logger.info(f"📈 Нормализация символа: {symbol}.P → {symbol}")
        
        # Установка плеча если указано
        if leverage:
            self.set_leverage(symbol, leverage)
        
        # Получаем фильтры символа и округляем параметры
        filters = self._get_symbol_filters(symbol)
        qty_raw = float(qty)
        qty_norm = self._round_step_down(qty_raw, filters["qtyStep"])

        if qty_norm < filters["minOrderQty"]:
            return {"status":"error",
                    "message": f"Qty {qty_raw} < min {filters['minOrderQty']} после округления шагом {filters['qtyStep']}"}

        qty = f"{qty_norm:.2f}"  # округляем до 2 знаков

        if price is not None and price != "market":
            tick = filters["tickSize"]
            p_norm = self._round_step_down(float(price), tick)
            price = f"{p_norm:.2f}"
        
        # Подготовка параметров ордера
        order_params = {
            "category": "linear",  # Всегда linear для USDT-перпетуалов
            "symbol": symbol,
            "side": side,  # "Buy" или "Sell"
            "orderType": "Market" if price is None else "Limit",
            "qty": qty,  # Округленный объем
            "timeInForce": "IOC" if price is None else "GTC"
        }
        
        # Добавляем цену только для лимитных ордеров
        if price is not None and price != "market":
            order_params["price"] = str(price)
        
        # Добавляем SL/TP если указаны
        if take_profit:
            order_params["takeProfit"] = str(take_profit)
            logger.info(f"🎯 Take Profit set: {take_profit}")
        
        if stop_loss:
            order_params["stopLoss"] = str(stop_loss)
            logger.info(f"🛡️ Stop Loss set: {stop_loss}")
        
        logger.info(f"📊 V5 Order: {symbol} {side} {qty} @ {price or 'MARKET'}")
        
        result = self.make_request("POST", "/v5/order/create", order_params)
        
        if result.get("retCode") == 0:
            order_id = result.get("result", {}).get("orderId", "Unknown")
            return {
                "status": "success",
                "message": f"Ордер размещен через V5 API",
                "order_id": order_id,
                "symbol": symbol
            }
        else:
            error_msg = result.get("retMsg", "Unknown V5 error")
            return {
                "status": "error", 
                "message": f"V5 ордер ошибка: {error_msg}"
            }

# Глобальная инициализация V5 API клиента
def initialize_v5_client():
    """Инициализация V5 API клиента с поддержкой разных имен переменных"""
    api_key = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ BYBIT_API_KEY/SECRET (или BYBIT_KEY/SECRET) не найдены в env")
        return None
    
    logger.info(f"✅ V5 Client init with key: {api_key[:8]}...")
    return BybitV5API(api_key, api_secret)

# Создание глобального клиента V5
v5_client = initialize_v5_client()

def normalize_symbol(symbol: str) -> str:
    """Нормализация символа для Bybit V5"""
    return symbol.replace(".P", "") if symbol.endswith(".P") else symbol

def mask(text: str) -> str:
    """Маскировка конфиденциальных данных"""
    if not text or len(text) < 8:
        return "***"
    return text[:4] + "***" + text[-2:]

def execute_trade_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Выполнение торгового сигнала через V5 API"""
    if not v5_client:
        return {"status": "error", "message": "V5 API клиент не инициализирован"}
    
    try:
        signal_type = signal.get("type", "entry").lower()
        symbol = signal.get("symbol", "ETHUSDT")
        direction = signal.get("direction", "long").lower()
        qty = str(signal.get("qty", "0.001"))
        
        logger.info(f"🎯 Выполнение сигнала V5: {signal_type} {symbol} {direction} {qty}")
        
        if signal_type == "entry":
            # двухшаговый MARKET-вход (исключает Price invalid)
            take_profit = signal.get("take_profit")
            stop_loss   = signal.get("stop_loss")
            leverage    = signal.get("leverage", 30)

            side = "Buy" if direction == "long" else "Sell"

            # если в сигнале явно дан order_type, уважаем его
            order_type = str(signal.get("order_type", "market")).lower()
            if order_type == "limit":
                # редкий случай: пользователь хочет лимит — оставляем текущий путь
                entry_price = signal.get("entry_price")
                return v5_client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=entry_price,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    leverage=leverage
                )
            # по умолчанию — двухшаговый маркет
            return v5_client.place_entry_two_step(
                symbol=symbol,
                side=side,
                qty=qty,
                take_profit=take_profit,
                stop_loss=stop_loss,
                leverage=leverage
            )

        elif signal_type == "exit":
            # надёжный выход
            return v5_client.exit_enhanced(symbol, direction, reason=signal.get("reason",""))

        elif signal_type == "trail_update":
            # принимаем разные ключи: new_trail_stop / new_stop_loss / new_stop
            new_stop = (signal.get("new_trail_stop") or
                        signal.get("new_stop_loss") or
                        signal.get("new_stop"))
            if not new_stop:
                return {"status": "error", "message": "new_trail_stop/new_stop_loss/new_stop required"}
            return v5_client.trail_update_enhanced(symbol, direction, str(new_stop))
        
        else:
            return {"status": "error", "message": f"Unknown signal type: {signal_type}"}
        
    except Exception as e:
        error_msg = f"Ошибка выполнения сигнала V5: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"status": "error", "message": error_msg}

def close_position_market(symbol: str, side: str, qty: float) -> dict:
    """Закрытие позиции market ордером без SL/TP"""
    try:
        client = get_client()
        if not client:
            return {"status": "error", "message": "Client not initialized"}
        
        # Определяем противоположную сторону для закрытия
        close_side = "Sell" if side.lower() in ["long", "buy"] else "Buy"
        
        payload = {
            "category": client.category,
            "symbol": symbol,
            "side": close_side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "reduceOnly": True
        }
        
        response = client.place_order(**payload)
        return {"status": "success", "data": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_client() -> Optional[BybitV5API]:
    """Ленивая инициализация клиента для trail_engine"""
    global v5_client
    if v5_client is None:
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        if api_key and api_secret:
            v5_client = BybitV5API(api_key, api_secret)
    return v5_client

def test_connection() -> Dict[str, Any]:
    """Тестирование соединения с V5 API"""
    if not v5_client:
        return {"status": "error", "message": "V5 API клиент не инициализирован"}
    
    return v5_client.test_connection()