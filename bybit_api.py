import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode
import threading
from typing import Dict, List, Optional, Callable, Tuple, Any

# Необязательные хелперы округления по тик-сайзу/шагу
try:
    from utils_round import round_price, round_qty  # noqa: F401
except Exception:
    def round_price(x, *_args, **_kw): return x
    def round_qty(x, *_args, **_kw): return x


class BybitAPI:
    """Лёгкая обёртка над Bybit v5 (HTTP + паблик WS). Совместима с KWINStrategy/TrailEngine."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.testnet = bool(testnet)

        # По умолчанию — деривативы ("linear"); меняется через set_market_type()
        self.market_type = "linear"

        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        self.session = requests.Session()
        self.ws = None
        self.ws_callbacks: Dict[str, Callable] = {}

    # ==================== внутренняя утилита подписи ====================

    @staticmethod
    def _sorted_qs(params: Dict[str, Any]) -> str:
        """Стабильный querystring: ключи в алф. порядке (требование v5 для подписи)."""
        if not params:
            return ""
        # Преобразуем все значения к строкам
        items = [(k, "" if v is None else str(v)) for k, v in params.items()]
        items.sort(key=lambda kv: kv[0])
        return urlencode(items)

    def _generate_signature(self, payload: str, timestamp_ms: str) -> str:
        """
        v5 HMAC:
        sign = HMAC_SHA256(secret, timestamp + apiKey + recvWindow + payload)
        payload:
          - GET: querystring (без '?'), ключи отсортированы
          - POST: json.dumps(...) без лишних пробелов
        """
        prehash = timestamp_ms + self.api_key + "5000" + payload
        return hmac.new(
            self.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _send_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        *,
        requires_auth: bool = False,
        timeout: int = 30
    ) -> Dict:
        """
        Отправка HTTP-запроса к Bybit v5.
        - Публичные эндпоинты (market/*) — без подписи (устойчивее).
        - Приватные (account/order/position) — с HMAC-подписью v5.
        """
        params = params or {}
        url = f"{self.base_url}{endpoint}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        try:
            if requires_auth:
                timestamp = str(int(time.time() * 1000))
                if method.upper() == "GET":
                    payload_str = self._sorted_qs(params)
                else:
                    payload_str = json.dumps(params, separators=(',', ':'), ensure_ascii=False)
                signature = self._generate_signature(payload_str, timestamp)
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": "5000",
                })

            if method.upper() == "GET":
                # Для GET публичных: не подписываем, просто передаём params
                r = self.session.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                # Для POST подписываем (когда requires_auth=True)
                data = json.dumps(params, separators=(',', ':'), ensure_ascii=False) if params else None
                r = self.session.post(url, data=data, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Error [{method} {endpoint}]: {e}")
            return {"retCode": -1, "retMsg": str(e)}
        except Exception as e:
            print(f"API Unexpected Error [{method} {endpoint}]: {e}")
            return {"retCode": -1, "retMsg": f"unexpected: {e}"}

    @staticmethod
    def _map_trigger_by(source: str) -> str:
        """'last'|'mark' -> Bybit triggerBy/..TriggerBy значение."""
        return "MarkPrice" if str(source).lower() == "mark" else "LastPrice"

    # ==================== конфиг WS/market type ====================

    def set_market_type(self, market_type: str):
        """Установить тип рынка: 'linear' | 'inverse' | 'option' | 'spot'."""
        m = (market_type or "linear").lower()
        self.market_type = m
        base = "stream-testnet" if self.testnet else "stream"
        self.ws_url = f"wss://{base}.bybit.com/v5/public/{m}"

    # ==================== публичные маркет-данные ====================

    def get_server_time(self) -> Optional[int]:
        """Время сервера (секунды)."""
        resp = self._send_request("GET", "/v5/market/time", requires_auth=False)
        if resp.get("retCode") == 0:
            result = resp.get("result") or {}
            if "timeSecond" in result:
                return int(result["timeSecond"])
            if "timeNano" in result:
                return int(int(result["timeNano"]) / 1_000_000_000)
        return None

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict]]:
        """
        Исторические свечи.
        Возвращает список в возрастающем порядке времени:
          out[0] — самая старая, out[-1] — последняя ЗАКРЫТАЯ.
        Поля: timestamp(ms), open, high, low, close, volume.
        """
        params = {
            "category": self.market_type,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        resp = self._send_request("GET", "/v5/market/kline", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            out: List[Dict] = []
            for it in lst:
                # Bybit v5: [start, open, high, low, close, volume, turnover]
                out.append({
                    "timestamp": int(it[0]),  # уже в миллисекундах
                    "open": float(it[1]),
                    "high": float(it[2]),
                    "low": float(it[3]),
                    "close": float(it[4]),
                    "volume": float(it[5]) if it[5] is not None else 0.0,
                })
            out.sort(key=lambda x: x["timestamp"])
            return out
        return None

    def get_ticker(self, symbol: str) -> Dict:
        """Тикер по символу (last/mark + best bid/ask). Совместимо со стратегией."""
        params = {"category": self.market_type, "symbol": symbol}
        resp = self._send_request("GET", "/v5/market/tickers", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            if lst:
                t = lst[0]
                # Возвращаем ключи в стиле TV/бота (lastPrice/markPrice и пр.)
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
        """Унифицированный доступ к цене: source: 'last' | 'mark'."""
        t = self.get_ticker(symbol)
        last = float(t.get("lastPrice", 0) or 0)
        mark = float(t.get("markPrice", 0) or 0)
        return mark if (str(source).lower() == "mark" and mark > 0) else last

    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        """
        Информация об инструменте, включая priceFilter/lotSizeFilter.
        Совместимо с логикой стратегии (tickSize/qtyStep/minOrderQty).
        """
        params = {"category": self.market_type, "symbol": symbol}
        resp = self._send_request("GET", "/v5/market/instruments-info", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            return lst[0] if lst else None
        return None

    # ==================== приватные: аккаунт/ордера/позиции ====================

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[Dict]:
        """Баланс кошелька (по умолчанию UNIFIED для деривативов)."""
        params = {"accountType": account_type}
        resp = self._send_request("GET", "/v5/account/wallet-balance", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        orderType: str,
        qty: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_link_id: Optional[str] = None,
        reduce_only: bool = False,
        trigger_by_source: str = "last",   # 'last' | 'mark' — важно для стопов/условных
        time_in_force: Optional[str] = None,
        position_idx: Optional[int] = None,
        tpsl_mode: Optional[str] = None,   # "Full"|"Partial" (опц.)
    ) -> Optional[Dict]:
        """
        Создать ордер (Market/Limit/Conditional/Stop).
        Примечание: у Bybit v5 для прикрепления SL/TP к позиции можно передавать stopLoss/takeProfit.
        """
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": symbol,
            "side": str(side).title(),             # Buy/Sell
            "orderType": str(orderType).title(),   # Market/Limit/Conditional/Stop
            "qty": str(qty),
        }

        if price is not None:
            params["price"] = str(price)
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if time_in_force:
            params["timeInForce"] = time_in_force
        if position_idx is not None:
            params["positionIdx"] = str(position_idx)
        if tpsl_mode:
            params["tpslMode"] = str(tpsl_mode)

        # Встроенные SL/TP
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            # В create-ордере можно указать slTriggerBy (необяз.)
            params["slTriggerBy"] = self._map_trigger_by(trigger_by_source)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            params["tpTriggerBy"] = self._map_trigger_by(trigger_by_source)

        # Для условных/стоп-ордеров — основное поле 'triggerBy'
        if orderType.lower() in {"stop", "conditional"}:
            params["triggerBy"] = self._map_trigger_by(trigger_by_source)

        resp = self._send_request("POST", "/v5/order/create", params, requires_auth=True)
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
        position_idx: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Унифицированная правка:
        - SL/TP активной ПОЗИЦИИ -> /v5/position/trading-stop
        - параметры конкретного ОРДЕРА (price/qty) -> /v5/order/amend
        """
        # 1) апдейт SL/TP позиции (чаще используется в трейлинге)
        if stop_loss is not None or take_profit is not None:
            cat = self.market_type if self.market_type in ("linear", "inverse") else "linear"
            params_ts: Dict[str, str] = {
                "category": cat,
                "symbol": symbol,
            }
            if position_idx is not None:
                params_ts["positionIdx"] = str(position_idx)

            if stop_loss is not None:
                params_ts["stopLoss"] = str(stop_loss)
                params_ts["slTriggerBy"] = self._map_trigger_by(trigger_by_source)
            if take_profit is not None:
                params_ts["takeProfit"] = str(take_profit)
                params_ts["tpTriggerBy"] = self._map_trigger_by(trigger_by_source)

            resp = self._send_request("POST", "/v5/position/trading-stop", params_ts, requires_auth=True)
            if resp.get("retCode") == 0:
                return resp.get("result")
            return None

        # 2) апдейт существующего ОРДЕРА
        if order_id is None and order_link_id is None and price is None and qty is None:
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

        resp = self._send_request("POST", "/v5/order/amend", params_amend, requires_auth=True)
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
        resp = self._send_request("POST", "/v5/order/cancel", params, requires_auth=True)
        return resp.get("retCode") == 0

    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List[Dict]]:
        """Открытые ордера."""
        params: Dict[str, str] = {"category": self.market_type}
        if symbol:
            params["symbol"] = symbol
        resp = self._send_request("GET", "/v5/order/realtime", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return (resp.get("result") or {}).get("list")
        return None

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> Optional[List[Dict]]:
        """История ордеров."""
        params: Dict[str, str] = {"category": self.market_type, "limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        resp = self._send_request("GET", "/v5/order/history", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return (resp.get("result") or {}).get("list")
        return None

    def update_position_stop_loss(self, symbol: str, stop_loss: float, trigger_by_source: str = "last",
                                  position_idx: Optional[int] = None) -> Optional[Dict]:
        """
        Старый интерфейс из стратегии: прокся на /v5/position/trading-stop.
        Совместимо с KWINStrategy._update_stop_loss().
        """
        cat = self.market_type if self.market_type in ("linear", "inverse") else "linear"
        params = {
            "category": cat,
            "symbol": symbol,
            "stopLoss": str(stop_loss),
            "slTriggerBy": self._map_trigger_by(trigger_by_source),
        }
        if position_idx is not None:
            params["positionIdx"] = str(position_idx)

        resp = self._send_request("POST", "/v5/position/trading-stop", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return resp.get("result")
        return None

    # ==================== WebSocket (паблик) ====================

    def start_websocket(self, on_message: Optional[Callable] = None):
        """Запуск паблик WebSocket-соединения (market streams)."""
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
        try:
            if self.ws:
                self.ws.send(json.dumps(msg))
        except Exception as e:
            print(f"WebSocket send error: {e}")
