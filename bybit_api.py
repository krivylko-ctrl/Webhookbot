import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode
import threading
from typing import Dict, List, Optional, Callable

# Необязательные хелперы округления по тик-сайзу/шагу
try:
    from utils_round import round_price, round_qty  # noqa: F401
except Exception:
    def round_price(x, *_args, **_kw): return x
    def round_qty(x, *_args, **_kw): return x


class BybitAPI:
    """Лёгкая обёртка над Bybit v5 (HTTP + паблик WS). Совместима с KWINStrategy."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # По умолчанию деривативы ("linear"); можно сменить set_market_type("spot")
        self.market_type = "linear"

        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        self.session = requests.Session()
        self.ws = None
        self.ws_callbacks: Dict[str, Callable] = {}

    # -------------------- общие утилиты --------------------

    def set_market_type(self, market_type: str):
        """Установить тип рынка: 'linear' | 'inverse' | 'option' | 'spot'."""
        self.market_type = market_type
        if self.testnet:
            self.ws_url = f"wss://stream-testnet.bybit.com/v5/public/{market_type}"
        else:
            self.ws_url = f"wss://stream.bybit.com/v5/public/{market_type}"

    def _generate_signature(self, payload: str, timestamp: str) -> str:
        """
        v5 HMAC: sign = HMAC_SHA256(secret, timestamp + apiKey + recvWindow + payload)
        где payload = urlencode(params) для GET, json.dumps(params) для POST.
        """
        prehash = timestamp + self.api_key + "5000" + payload
        return hmac.new(
            self.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _send_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Отправка HTTP-запроса. Подписываем ВСЁ (GET/POST) — безопасно и стабильно."""
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time() * 1000))

        if method.upper() == "GET":
            payload_str = urlencode(params) if params else ""
        else:
            payload_str = json.dumps(params) if params else ""

        signature = self._generate_signature(payload_str, timestamp)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json",
        }

        try:
            if method.upper() == "GET":
                r = self.session.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == "POST":
                r = self.session.post(url, data=payload_str, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return {"retCode": -1, "retMsg": str(e)}

    @staticmethod
    def _map_trigger_by(source: str) -> str:
        """'last'|'mark' -> Bybit triggerBy значение."""
        return "MarkPrice" if str(source).lower() == "mark" else "LastPrice"

    # -------------------- маркет-данные --------------------

    def get_server_time(self) -> Optional[int]:
        """Время сервера (секунды)."""
        resp = self._send_request("GET", "/v5/market/time")
        if resp.get("retCode") == 0:
            result = resp.get("result") or {}
            # timeSecond (сек), timeNano (нс) — встречаются оба варианта
            if "timeSecond" in result:
                return int(result["timeSecond"])
            if "timeNano" in result:
                return int(int(result["timeNano"]) / 1_000_000_000)
        return None

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict]]:
        """
        Исторические свечи.
        Возвращает список в возрастающем порядке времени: out[0] — самая старая,
        out[-1] — САМАЯ СВЕЖАЯ ЗАКРЫТАЯ (эквивалент TV).
        """
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        resp = self._send_request("GET", "/v5/market/kline", params)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            out: List[Dict] = []
            for it in lst:
                out.append({
                    "timestamp": int(it[0]),
                    "open": float(it[1]),
                    "high": float(it[2]),
                    "low": float(it[3]),
                    "close": float(it[4]),
                    "volume": float(it[5]) if it[5] is not None else 0.0,
                })
            out.sort(key=lambda x: x["timestamp"])  # возрастающе
            return out
        return None

    def get_ticker(self, symbol: str) -> Dict:
        """Тикер по символу (last/mark + best bid/ask)."""
        params = {"category": self.market_type, "symbol": symbol}
        resp = self._send_request("GET", "/v5/market/tickers", params)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            if lst:
                t = lst[0]
                return {
                    "symbol": t.get("symbol"),
                    "lastPrice": float(t.get("lastPrice", 0) or 0),
                    "markPrice": float(t.get("markPrice", 0) or 0),
                    "bid1Price": float(t.get("bid1Price", 0) or 0),
                    "ask1Price": float(t.get("ask1Price", 0) or 0),
                    "volume24h": float(t.get("volume24h", 0) or 0),
                }
        return {}

    def get_price(self, symbol: str, source: str = "last") -> float:
        """
        Унифицированный доступ к цене для логики/триггеров.
        source: 'last' | 'mark'
        """
        t = self.get_ticker(symbol)
        last = float(t.get("lastPrice", 0) or 0)
        mark = float(t.get("markPrice", 0) or 0)
        return mark if (str(source).lower() == "mark" and mark > 0) else last

    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        """Информация об инструменте (тик-сайз, шаг количества и т.д.)."""
        params = {"category": self.market_type, "symbol": symbol}
        resp = self._send_request("GET", "/v5/market/instruments-info", params)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            return lst[0] if lst else None
        return None

    # -------------------- аккаунт / ордера --------------------

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[Dict]:
        """Баланс кошелька (по умолчанию UNIFIED для деривативов)."""
        params = {"accountType": account_type}
        resp = self._send_request("GET", "/v5/account/wallet-balance", params)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_link_id: Optional[str] = None,
        reduce_only: bool = False,
        trigger_by_source: str = "last",   # 'last' | 'mark' — важно для стопов/условных
        time_in_force: Optional[str] = None,
        position_idx: Optional[int] = None,
    ) -> Optional[Dict]:
        """Создать ордер (Market/Limit/Conditional/Stop)."""
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": symbol,
            "side": str(side).title(),            # Buy/Sell
            "orderType": str(order_type).title(), # Market/Limit/Conditional/Stop
            "qty": str(qty),
        }

        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if time_in_force:
            params["timeInForce"] = time_in_force
        if position_idx is not None:
            params["positionIdx"] = str(position_idx)

        if price is not None:
            params["price"] = str(price)
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)

        # Для условных/стопов — источник триггера
        if order_type.lower() in {"stop", "conditional"}:
            params["triggerBy"] = self._map_trigger_by(trigger_by_source)

        resp = self._send_request("POST", "/v5/order/create", params)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    def modify_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        price: Optional[float] = None,
        qty: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trigger_by_source: str = "last",
    ) -> Optional[Dict]:
        """
        Унифицированная правка:
        - Если меняем SL/TP активной ПОЗИЦИИ (деривативы) -> /v5/position/trading-stop
        - Если правим параметры конкретного ОРДЕРА (price/qty) -> /v5/order/amend
        """
        # 1) апдейт стоп-лосса/тейк-профита позиции (чаще всего используется в трейлинге)
        if stop_loss is not None or take_profit is not None:
            if self.market_type != "linear":
                # для spot тут обычно не используется; можно расширить при необходимости
                pass
            params_ts: Dict[str, str] = {
                "category": "linear",
                "symbol": symbol,
            }
            if stop_loss is not None:
                params_ts["stopLoss"] = str(stop_loss)
                # можно указать источник триггера для SL
                params_ts["slTriggerBy"] = self._map_trigger_by(trigger_by_source)
            if take_profit is not None:
                params_ts["takeProfit"] = str(take_profit)
                params_ts["tpTriggerBy"] = self._map_trigger_by(trigger_by_source)

            resp = self._send_request("POST", "/v5/position/trading-stop", params_ts)
            if resp.get("retCode") == 0:
                return resp.get("result")
            return None

        # 2) апдейт существующего ОРДЕРА (лимитники и т.п.)
        if order_id is None and order_link_id is None:
            # нечего менять — не знаем какой ордер
            return None

        params_amend: Dict[str, str] = {
            "category": self.market_type,
            "symbol": symbol,
        }
        if order_id:
            params_amend["orderId"] = order_id
        if order_link_id:
            params_amend["orderLinkId"] = order_link_id
        if price is not None:
            params_amend["price"] = str(price)
        if qty is not None:
            params_amend["qty"] = str(qty)

        resp = self._send_request("POST", "/v5/order/amend", params_amend)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    def cancel_order(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> bool:
        """Отменить ордер по orderId или orderLinkId."""
        params: Dict[str, str] = {"category": self.market_type, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        resp = self._send_request("POST", "/v5/order/cancel", params)
        return resp.get("retCode") == 0

    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List[Dict]]:
        """Открытые ордера."""
        params: Dict[str, str] = {"category": self.market_type}
        if symbol:
            params["symbol"] = symbol
        resp = self._send_request("GET", "/v5/order/realtime", params)
        if resp.get("retCode") == 0:
            return (resp.get("result") or {}).get("list")
        return None

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> Optional[List[Dict]]:
        """История ордеров."""
        params: Dict[str, str] = {"category": self.market_type, "limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        resp = self._send_request("GET", "/v5/order/history", params)
        if resp.get("retCode") == 0:
            return (resp.get("result") or {}).get("list")
        return None

    def update_position_stop_loss(self, symbol: str, stop_loss: float) -> Optional[Dict]:
        """Старый интерфейс (деривативы): прокся на position/trading-stop."""
        if self.market_type != "linear":
            return None
        params = {"category": "linear", "symbol": symbol, "stopLoss": str(stop_loss)}
        resp = self._send_request("POST", "/v5/position/trading-stop", params)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    # -------------------- WebSocket (паблик) --------------------

    def start_websocket(self, on_message: Optional[Callable] = None):
        """Запуск паблик WebSocket-соединения."""
        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                topic = data.get("topic")
                if topic and topic in self.ws_callbacks:
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
                on_close=on_ws_close,
            )
            th = threading.Thread(target=self.ws.run_forever, daemon=True)
            th.start()
        except ImportError:
            print("WebSocket library not available")

    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Подписка на kline через паблик WebSocket."""
        if not self.ws:
            self.start_websocket()
        topic = f"kline.{interval}.{symbol}"
        self.ws_callbacks[topic] = callback
        msg = {"op": "subscribe", "args": [topic]}
        if self.ws:
            self.ws.send(json.dumps(msg))
