import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode
import websocket
import threading
from typing import Dict, List, Optional, Callable
from utils_round import round_price, round_qty

class BybitAPI:
    """Класс для работы с Bybit API v5"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # По умолчанию linear (деривативы), можно изменить через set_market_type
        self.market_type = "linear"
        
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        self.session = requests.Session()
        self.ws = None
        self.ws_callbacks = {}
    
    def set_market_type(self, market_type: str):
        """Установить тип рынка: 'linear' или 'spot'"""
        self.market_type = market_type
        if self.testnet:
            self.ws_url = f"wss://stream-testnet.bybit.com/v5/public/{market_type}"
        else:
            self.ws_url = f"wss://stream.bybit.com/v5/public/{market_type}"

    import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode
import websocket
import threading
from typing import Dict, List, Optional, Callable

class BybitAPI:
    """Класс для работы с Bybit API v5"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # По умолчанию linear (деривативы), можно изменить через set_market_type
        self.market_type = "linear"
        
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        self.session = requests.Session()
        self.ws = None
        self.ws_callbacks = {}
    
    def set_market_type(self, market_type: str):
        """Установить тип рынка: 'linear' или 'spot'"""
        self.market_type = market_type
        if self.testnet:
            self.ws_url = f"wss://stream-testnet.bybit.com/v5/public/{market_type}"
        else:
            self.ws_url = f"wss://stream.bybit.com/v5/public/{market_type}"
        
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Генерация подписи для запроса"""
        param_str = timestamp + self.api_key + "5000" + params
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _send_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Отправка HTTP запроса"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        if method == "GET":
            param_str = urlencode(params) if params else ""
        else:
            param_str = json.dumps(params) if params else ""
        
        signature = self._generate_signature(param_str, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(url, data=param_str, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    def get_server_time(self) -> Optional[int]:
        """Получить время сервера"""
        response = self._send_request("GET", "/v5/market/time")
        if response.get("retCode") == 0:
            return int(response["result"]["timeSecond"])
        else:
            print(f"Server time error: {response}")
        return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict]]:
        """Получить исторические свечи"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = self._send_request("GET", "/v5/market/kline", params)
        if response.get("retCode") == 0:
            klines = []
            for item in response["result"]["list"]:
                klines.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })
            # КРИТИЧНЫЙ ПАТЧ: TV-эквивалент - последний элемент = самая свежая закрытая свеча
            klines.sort(key=lambda x: x["timestamp"], reverse=True)
            return klines
        return None
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Получить текущую цену"""
        params = {
            "category": self.market_type,
            "symbol": symbol
        }
        
        response = self._send_request("GET", "/v5/market/tickers", params)
        if response.get("retCode") == 0 and response["result"]["list"]:
            ticker = response["result"]["list"][0]
            return {
                "symbol": ticker["symbol"],
                "last_price": float(ticker["lastPrice"]),
                "bid": float(ticker["bid1Price"]),
                "ask": float(ticker["ask1Price"]),
                "volume": float(ticker["volume24h"])
            }
        return None
    
    def get_wallet_balance(self) -> Optional[Dict]:
        """Получить баланс кошелька"""
        params = {"accountType": "SPOT"}
        
        response = self._send_request("GET", "/v5/account/wallet-balance", params)
        if response.get("retCode") == 0:
            return response["result"]
        return None
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, order_link_id: Optional[str] = None,
                   reduce_only: bool = False) -> Optional[Dict]:
        """Разместить ордер"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "side": side.title(),
            "orderType": order_type.title(),
            "qty": str(qty)
        }
        
        # Дополнительные параметры для безопасности
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["orderLinkId"] = order_link_id
            
        # triggerBy: на Bybit задай triggerBy="LastPrice" для консистентности с TV
        if order_type.lower() in ["stop", "conditional"]:
            params["triggerBy"] = "LastPrice"
        
        if price:
            params["price"] = str(price)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        
        response = self._send_request("POST", "/v5/order/create", params)
        if response.get("retCode") == 0:
            return response["result"]
        return None
    

    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Отменить ордер"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "orderId": order_id
        }
        
        response = self._send_request("POST", "/v5/order/cancel", params)
        return response.get("retCode") == 0
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List[Dict]]:
        """Получить открытые ордера"""
        params = {"category": self.market_type}
        if symbol:
            params["symbol"] = symbol
        
        response = self._send_request("GET", "/v5/order/realtime", params)
        if response.get("retCode") == 0:
            return response["result"]["list"]
        return None
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> Optional[List[Dict]]:
        """Получить историю ордеров"""
        params = {
            "category": self.market_type,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol
        
        response = self._send_request("GET", "/v5/order/history", params)
        if response.get("retCode") == 0:
            return response["result"]["list"]
        return None
    
    def update_position_stop_loss(self, symbol: str, stop_loss: float) -> Optional[Dict]:
        """Обновить стоп-лосс позиции (для деривативов)"""
        if self.market_type == "linear":
            params = {
                "category": "linear", 
                "symbol": symbol, 
                "stopLoss": str(stop_loss)
            }
            response = self._send_request("POST", "/v5/position/trading-stop", params)
            if response.get("retCode") == 0:
                return response["result"]
        return None
    
    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        """Получить информацию об инструменте"""
        params = {
            "category": self.market_type,
            "symbol": symbol
        }
        
        response = self._send_request("GET", "/v5/market/instruments-info", params)
        if response.get("retCode") == 0 and response["result"]["list"]:
            return response["result"]["list"][0]
        return None
    
    def start_websocket(self, on_message: Optional[Callable] = None):
        """Запустить WebSocket соединение"""
        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                
                # Обработка подписок
                if "topic" in data:
                    topic = data["topic"]
                    if topic in self.ws_callbacks:
                        self.ws_callbacks[topic](data)
            except Exception as e:
                print(f"WebSocket message error: {e}")
        
        def on_ws_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_ws_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_ws_open(ws):
            print("WebSocket connection opened")
        
        try:
            from websocket import WebSocketApp  # type: ignore
            self.ws = WebSocketApp(
            self.ws_url,
            on_open=on_ws_open,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close
            )
            
            # Запуск в отдельном потоке
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
        except ImportError:
            print("WebSocket library not available")
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Подписаться на свечи через WebSocket"""
        if not self.ws:
            self.start_websocket()
        
        topic = f"kline.{interval}.{symbol}"
        self.ws_callbacks[topic] = callback
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [topic]
        }
        
        if self.ws:
            self.ws.send(json.dumps(subscribe_msg))
    


        
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Генерация подписи для запроса"""
        param_str = timestamp + self.api_key + "5000" + params
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _send_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Отправка HTTP запроса"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        if method == "GET":
            param_str = urlencode(params) if params else ""
        else:
            param_str = json.dumps(params) if params else ""
        
        signature = self._generate_signature(param_str, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method == "POST":
                response = self.session.post(url, data=param_str, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    def get_server_time(self) -> Optional[int]:
        """Получить время сервера"""
        response = self._send_request("GET", "/v5/market/time")
        if response.get("retCode") == 0:
            return int(response["result"]["timeSecond"])
        else:
            print(f"Server time error: {response}")
        return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict]]:
        """Получить исторические свечи"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = self._send_request("GET", "/v5/market/kline", params)
        if response.get("retCode") == 0:
            klines = []
            for item in response["result"]["list"]:
                klines.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })
            # КРИТИЧНЫЙ ПАТЧ: TV-эквивалент - последний элемент = самая свежая закрытая свеча
            klines.sort(key=lambda x: x["timestamp"], reverse=True)
            return klines
        return None
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Получить текущую цену"""
        params = {
            "category": self.market_type,
            "symbol": symbol
        }
        
        response = self._send_request("GET", "/v5/market/tickers", params)
        if response.get("retCode") == 0 and response["result"]["list"]:
            ticker = response["result"]["list"][0]
            return {
                "symbol": ticker["symbol"],
                "last_price": float(ticker["lastPrice"]),
                "bid": float(ticker["bid1Price"]),
                "ask": float(ticker["ask1Price"]),
                "volume": float(ticker["volume24h"])
            }
        return None
    
    def get_wallet_balance(self) -> Optional[Dict]:
        """Получить баланс кошелька"""
        params = {"accountType": "SPOT"}
        
        response = self._send_request("GET", "/v5/account/wallet-balance", params)
        if response.get("retCode") == 0:
            return response["result"]
        return None
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, order_link_id: Optional[str] = None,
                   reduce_only: bool = False) -> Optional[Dict]:
        """Разместить ордер"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "side": side.title(),
            "orderType": order_type.title(),
            "qty": str(qty)
        }
        
        # Дополнительные параметры для безопасности
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["orderLinkId"] = order_link_id
            
        # triggerBy: на Bybit задай triggerBy="LastPrice" для консистентности с TV
        if order_type.lower() in ["stop", "conditional"]:
            params["triggerBy"] = "LastPrice"
        
        if price:
            params["price"] = str(price)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        
        response = self._send_request("POST", "/v5/order/create", params)
        if response.get("retCode") == 0:
            return response["result"]
        return None
    

    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Отменить ордер"""
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "orderId": order_id
        }
        
        response = self._send_request("POST", "/v5/order/cancel", params)
        return response.get("retCode") == 0
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List[Dict]]:
        """Получить открытые ордера"""
        params = {"category": self.market_type}
        if symbol:
            params["symbol"] = symbol
        
        response = self._send_request("GET", "/v5/order/realtime", params)
        if response.get("retCode") == 0:
            return response["result"]["list"]
        return None
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> Optional[List[Dict]]:
        """Получить историю ордеров"""
        params = {
            "category": self.market_type,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol
        
        response = self._send_request("GET", "/v5/order/history", params)
        if response.get("retCode") == 0:
            return response["result"]["list"]
        return None
    
    def update_position_stop_loss(self, symbol: str, stop_loss: float) -> Optional[Dict]:
        """Обновить стоп-лосс позиции (для деривативов)"""
        if self.market_type == "linear":
            params = {
                "category": "linear", 
                "symbol": symbol, 
                "stopLoss": str(stop_loss)
            }
            response = self._send_request("POST", "/v5/position/trading-stop", params)
            if response.get("retCode") == 0:
                return response["result"]
        return None
    
    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        """Получить информацию об инструменте"""
        params = {
            "category": self.market_type,
            "symbol": symbol
        }
        
        response = self._send_request("GET", "/v5/market/instruments-info", params)
        if response.get("retCode") == 0 and response["result"]["list"]:
            return response["result"]["list"][0]
        return None
    
    def start_websocket(self, on_message: Optional[Callable] = None):
        """Запустить WebSocket соединение"""
        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                
                # Обработка подписок
                if "topic" in data:
                    topic = data["topic"]
                    if topic in self.ws_callbacks:
                        self.ws_callbacks[topic](data)
            except Exception as e:
                print(f"WebSocket message error: {e}")
        
        def on_ws_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_ws_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_ws_open(ws):
            print("WebSocket connection opened")
        
        try:
            from websocket import WebSocketApp  # type: ignore
            self.ws = WebSocketApp(
            self.ws_url,
            on_open=on_ws_open,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close
            )
            
            # Запуск в отдельном потоке
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
        except ImportError:
            print("WebSocket library not available")
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Подписаться на свечи через WebSocket"""
        if not self.ws:
            self.start_websocket()
        
        topic = f"kline.{interval}.{symbol}"
        self.ws_callbacks[topic] = callback
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [topic]
        }
        
        if self.ws:
            self.ws.send(json.dumps(subscribe_msg))
    

