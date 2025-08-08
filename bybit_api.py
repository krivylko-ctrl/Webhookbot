import requests
import time
import hmac
import hashlib
import uuid
import logging
import os
from typing import Dict, Any, Optional

API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

logger = logging.getLogger(__name__)

class BybitFuturesClient:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key or API_KEY or ""
        self.api_secret = api_secret or API_SECRET or ""
        
        # Используем DEMO URL для тестирования
        if testnet:
            self.base_url = "https://api-demo.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        logger.info("Bybit API инициализация:")
        logger.info(f"  API Key: {'НАЙДЕН' if self.api_key else 'НЕ НАЙДЕН'}")
        logger.info(f"  API Secret: {'НАЙДЕН' if self.api_secret else 'НЕ НАЙДЕН'}")
        logger.info(f"  Режим: {'DEMO TESTNET' if testnet else 'MAINNET'}")
        
        if self.api_key:
            logger.info(f"  Key начинается с: {self.api_key[:8]}...")
        if self.api_secret:
            logger.info(f"  Secret начинается с: {self.api_secret[:8]}...")
        
        logger.info("✅ ГОТОВ К ТОРГОВЛЕ: Bybit DEMO режим активен")

    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Генерация подписи для Bybit API"""
        param_str = timestamp + self.api_key + "5000" + params  # recv_window = 5000
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, endpoint: str, params: Dict[str, Any], method: str = "GET") -> Dict[str, Any]:
        """Выполнение запроса к Bybit API"""
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
        
        if method == "POST":
            import json
            params_str = json.dumps(params) if params else ""
            signature = self._generate_signature(timestamp, params_str)
            headers["X-BAPI-SIGN"] = signature
            
            try:
                response = requests.post(url, json=params, headers=headers, timeout=10)
                logger.info(f"POST {endpoint}: {response.status_code}")
                return response.json()
            except Exception as e:
                logger.error(f"POST запрос к {endpoint} ошибка: {e}")
                return {"error": str(e)}
        else:
            # GET запрос
            params_str = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
            signature = self._generate_signature(timestamp, params_str)
            headers["X-BAPI-SIGN"] = signature
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                logger.info(f"GET {endpoint}: {response.status_code}")
                return response.json()
            except Exception as e:
                logger.error(f"GET запрос к {endpoint} ошибка: {e}")
                return {"error": str(e)}

    def place_order(self, symbol: str, direction: str, quantity: float, 
                   entry_price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None) -> Dict[str, Any]:
        """Размещение ордера на Bybit"""
        try:
            # Определение стороны для Bybit
            side = "Buy" if direction.lower() == "long" else "Sell"
            
            # Параметры основного ордера
            params = {
                'category': 'linear',  # USDT Perpetual
                'symbol': symbol,
                'side': side,
                'orderType': 'Market' if entry_price is None else 'Limit',
                'qty': str(quantity),
                'timeInForce': 'GTC'
            }
            
            # Если указана цена входа - лимитный ордер
            if entry_price is not None:
                params['price'] = str(entry_price)
            
            logger.info(f"Размещение ордера {direction}: {quantity} {symbol}")
            result = self._make_request("/v5/order/create", params, "POST")
            
            if result.get("retCode") == 0:
                logger.info(f"✅ Ордер размещен: {result.get('result', {}).get('orderId')}")
                
                # Размещение Stop Loss и Take Profit если указаны
                if stop_loss or take_profit:
                    self._place_stop_orders(symbol, direction, stop_loss, take_profit)
                
                return result
            else:
                logger.error(f"❌ Ошибка размещения ордера: {result.get('retMsg')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Исключение при размещении ордера: {e}")
            return {"error": str(e)}

    def _place_stop_orders(self, symbol: str, direction: str, stop_loss: Optional[float], 
                          take_profit: Optional[float]) -> None:
        """Размещение стоп-ордеров"""
        try:
            if stop_loss:
                # Stop Loss ордер
                sl_side = "Sell" if direction.lower() == "long" else "Buy"
                sl_params = {
                    'category': 'linear',
                    'symbol': symbol,
                    'side': sl_side,
                    'orderType': 'Market',
                    'qty': '0',  # Закрыть всю позицию
                    'triggerPrice': str(stop_loss),
                    'reduceOnly': True
                }
                
                sl_result = self._make_request("/v5/order/create", sl_params, "POST")
                if sl_result.get("retCode") == 0:
                    logger.info(f"✅ Stop Loss размещен: {stop_loss}")
                else:
                    logger.error(f"❌ Ошибка Stop Loss: {sl_result.get('retMsg')}")
            
            if take_profit:
                # Take Profit ордер
                tp_side = "Sell" if direction.lower() == "long" else "Buy"
                tp_params = {
                    'category': 'linear',
                    'symbol': symbol,
                    'side': tp_side,
                    'orderType': 'Market',
                    'qty': '0',  # Закрыть всю позицию
                    'triggerPrice': str(take_profit),
                    'reduceOnly': True
                }
                
                tp_result = self._make_request("/v5/order/create", tp_params, "POST")
                if tp_result.get("retCode") == 0:
                    logger.info(f"✅ Take Profit размещен: {take_profit}")
                else:
                    logger.error(f"❌ Ошибка Take Profit: {tp_result.get('retMsg')}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка размещения стоп-ордеров: {e}")

    def update_trailing_stop(self, symbol: str, direction: str, new_stop_price: float) -> Dict[str, Any]:
        """Обновление трейлинг стопа"""
        try:
            # Сначала отменяем старые стоп-ордера
            self.cancel_open_orders(symbol, order_type="conditional")
            
            # Размещаем новый стоп
            sl_side = "Sell" if direction.lower() == "long" else "Buy"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': sl_side,
                'orderType': 'Market',
                'qty': '0',  # Закрыть всю позицию
                'triggerPrice': str(new_stop_price),
                'reduceOnly': True
            }
            
            result = self._make_request("/v5/order/create", params, "POST")
            if result.get("retCode") == 0:
                logger.info(f"✅ Трейлинг стоп обновлен: {new_stop_price}")
            else:
                logger.error(f"❌ Ошибка обновления трейлинг стопа: {result.get('retMsg')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления трейлинг стопа: {e}")
            return {"error": str(e)}

    def close_position(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Закрытие позиции"""
        try:
            # Определяем сторону для закрытия (противоположную текущей позиции)
            side = "Sell" if direction.lower() == "long" else "Buy"
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': 'Market',
                'qty': '0',  # Закрыть всю позицию
                'reduceOnly': True
            }
            
            logger.info(f"Закрытие позиции {direction}: {symbol}")
            result = self._make_request("/v5/order/create", params, "POST")
            
            if result.get("retCode") == 0:
                logger.info(f"✅ Позиция закрыта: {symbol}")
            else:
                logger.error(f"❌ Ошибка закрытия позиции: {result.get('retMsg')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия позиции: {e}")
            return {"error": str(e)}

    def cancel_open_orders(self, symbol: str, order_type: str = "all") -> Dict[str, Any]:
        """Отмена открытых ордеров"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol
            }
            
            result = self._make_request("/v5/order/cancel-all", params, "POST")
            
            if result.get("retCode") == 0:
                logger.info(f"✅ Ордера отменены для {symbol}")
            else:
                logger.error(f"❌ Ошибка отмены ордеров: {result.get('retMsg')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка отмены ордеров: {e}")
            return {"error": str(e)}

    def get_positions(self) -> Dict[str, Any]:
        """Получение открытых позиций"""
        params = {
            'category': 'linear',
            'settleCoin': 'USDT'
        }
        return self._make_request("/v5/position/list", params, "GET")

    def get_account_info(self) -> Dict[str, Any]:
        """Получение информации об аккаунте"""
        params = {
            'accountType': 'UNIFIED'
        }
        return self._make_request("/v5/account/wallet-balance", params, "GET")

    def ping(self) -> Dict[str, Any]:
        """Проверка соединения с Bybit"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/v5/market/time", timeout=10)
            end_time = time.time()
            latency = round((end_time - start_time) * 1000, 2)
            
            if response.status_code == 200:
                data = response.json()
                server_time = data.get("result", {}).get("timeSecond", "N/A")
                logger.info(f"✅ Bybit ping успешен: {latency}ms, время сервера: {server_time}")
                return {
                    "status": "success",
                    "latency_ms": latency,
                    "server_time": server_time,
                    "testnet": "DEMO" in self.base_url
                }
            else:
                logger.error(f"❌ Bybit ping неудачен: {response.status_code}")
                return {"status": "failed", "code": response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Ошибка ping: {e}")
            return {"status": "error", "message": str(e)}