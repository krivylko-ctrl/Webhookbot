# bybit_api.py — совместимая обёртка для KWINStrategy (HTTP + WS v5)

import requests
import json
import time
import hashlib
import hmac
from urllib.parse import urlencode
import threading
from typing import Dict, List, Optional, Callable, Any

# Необязательные хелперы округления по тик-сайзу/шагу
try:
    from utils_round import round_price, round_qty  # noqa: F401
except Exception:
    def round_price(x, *_args, **_kw): return x
    def round_qty(x, *_args, **_kw): return x


class BybitAPI:
    """Лёгкая обёртка над Bybit v5 (HTTP + паблик WS).
       По умолчанию — категория linear (USDT-M)."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.testnet = bool(testnet)

        # По умолчанию — деривативы ("linear")
        self.market_type = "linear"

        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KWINBot/1.0 (+https://example.local)",
            "Accept": "application/json",
        })

        self.ws = None
        self.ws_callbacks: Dict[str, Callable] = {}

    # ==================== внутренняя утилита подписи ====================

    @staticmethod
    def _sorted_qs(params: Dict[str, Any]) -> str:
        if not params:
            return ""
        items = []
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                v = ",".join(map(str, v))
            items.append((k, "" if v is None else str(v)))
        items.sort(key=lambda kv: kv[0])
        return urlencode(items)

    def _generate_signature(self, payload: str, timestamp_ms: str) -> str:
        # формула подписи v5: ts + api_key + recv_window + payload
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
                r = self.session.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                data = json.dumps(params, separators=(',', ':'), ensure_ascii=False) if params else None
                r = self.session.post(url, data=data, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            r.raise_for_status()
            resp = r.json()
            if isinstance(resp, dict) and resp.get("retCode") not in (0, "0", None):
                try:
                    print(f"[BybitAPI] {method} {endpoint} retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
                except Exception:
                    pass
            return resp
        except requests.exceptions.RequestException as e:
            print(f"API Request Error [{method} {endpoint}]: {e}")
            return {"retCode": -1, "retMsg": str(e)}
        except Exception as e:
            print(f"API Unexpected Error [{method} {endpoint}]: {e}")
            return {"retCode": -1, "retMsg": f"unexpected: {e}"}

    @staticmethod
    def _map_trigger_by(source: str) -> str:
        s = str(source).lower()
        if s.startswith("mark"):
            return "MarkPrice"
        if s.startswith("index"):
            return "IndexPrice"
        return "LastPrice"

    @staticmethod
    def _to_float(x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    # ==================== конфиг ====================

    def set_market_type(self, market_type: str):
        """Жёстко фиксируем категорию (linear/inverse/spot/option). По умолчанию — linear."""
        m = (market_type or "linear").lower()
        self.market_type = m
        base = "stream-testnet" if self.testnet else "stream"
        # Эндпоинт WS зависит от категории
        self.ws_url = f"wss://{base}.bybit.com/v5/public/{m}"

    def force_linear(self):
        """Удобный хелпер: насильно фиксирует работу только с фьючерсами (linear)."""
        self.set_market_type("linear")

    # ==================== публичные ====================

    def get_server_time(self) -> Optional[int]:
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
        Последние N свечей по категории self.market_type.
        interval: строка минут ("1","3","5","15","60",...).
        Возвращает список по возрастанию timestamp (мс).
        """
        params = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "interval": str(interval),
            "limit": int(limit),
        }
        resp = self._send_request("GET", "/v5/market/kline", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = ((resp.get("result") or {}).get("list") or [])
            if not lst:
                print(f"[BybitAPI] get_klines returned empty list for {params}")
                return []
            out: List[Dict] = []
            for it in lst:
                try:
                    out.append({
                        "timestamp": int(it[0]),
                        "open":  self._to_float(it[1]),
                        "high":  self._to_float(it[2]),
                        "low":   self._to_float(it[3]),
                        "close": self._to_float(it[4]),
                        "volume": self._to_float(it[5]),
                    })
                except Exception:
                    continue
            out.sort(key=lambda x: x["timestamp"])
            return out
        print(f"[BybitAPI] get_klines error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    def get_klines_window(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Чтение свечей с окнами времени (start/end в мс). Возвращает по возрастанию timestamp.
        """
        params = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "interval": str(interval),
            "limit": int(limit),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        resp = self._send_request("GET", "/v5/market/kline", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = ((resp.get("result") or {}).get("list") or [])
            out: List[Dict] = []
            for it in lst:
                try:
                    out.append({
                        "timestamp": int(it[0]),
                        "open":  self._to_float(it[1]),
                        "high":  self._to_float(it[2]),
                        "low":   self._to_float(it[3]),
                        "close": self._to_float(it[4]),
                        "volume": self._to_float(it[5]),
                    })
                except Exception:
                    continue
            out.sort(key=lambda x: x["timestamp"])
            return out
        print(f"[BybitAPI] get_klines_window error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    # Алиас
    def get_klines_between(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[Dict]:
        rows = self.get_klines_window(symbol, interval, start_ms=start_ms, end_ms=end_ms, limit=limit) or []
        return rows

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        params = {"category": self.market_type, "symbol": (symbol or "").upper()}
        resp = self._send_request("GET", "/v5/market/tickers", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            if lst:
                t = lst[0]
                _f = self._to_float
                return {
                    "symbol": t.get("symbol"),
                    # camelCase — подхватится стратегией
                    "lastPrice": _f(t.get("lastPrice", 0)),
                    "markPrice": _f(t.get("markPrice", 0)),
                    # дубли snake для совместимости со сторонним кодом (не помешают)
                    "last_price": _f(t.get("lastPrice", 0)),
                    "mark_price": _f(t.get("markPrice", 0)),
                    "bid1Price": _f(t.get("bid1Price", 0)),
                    "ask1Price": _f(t.get("ask1Price", 0)),
                    "volume24h": _f(t.get("volume24h", 0)),
                }
            print(f"[BybitAPI] get_ticker empty list for {params}")
        else:
            print(f"[BybitAPI] get_ticker error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    def get_price(self, symbol: str, source: str = "last") -> float:
        t = self.get_ticker(symbol) or {}
        last = self._to_float(t.get("lastPrice", 0))
        mark = self._to_float(t.get("markPrice", 0))
        return mark if (str(source).lower() == "mark" and mark > 0) else last

    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        params = {"category": self.market_type, "symbol": (symbol or "").upper()}
        resp = self._send_request("GET", "/v5/market/instruments-info", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            if not lst:
                print(f"[BybitAPI] instruments-info empty for {params}")
            # структура с priceFilter/lotSizeFilter — как ожидает стратегия
            return lst[0] if lst else None
        print(f"[BybitAPI] instruments-info error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    # ==================== приватные (trade/account) ====================

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[Dict]:
        params = {"accountType": account_type}
        resp = self._send_request("GET", "/v5/account/wallet-balance", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return resp.get("result")
        print(f"[BybitAPI] wallet-balance error: {resp.get('retCode')} {resp.get('retMsg')}")
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
        trigger_by_source: str = "mark",
        time_in_force: Optional[str] = None,
        position_idx: Optional[int] = None,
        tpsl_mode: Optional[str] = None,
    ) -> Optional[Dict]:
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "side": str(side).title(),
            "orderType": str(orderType).title(),
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

        # SL/TP + источник триггера (MarkPrice/LastPrice)
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            params["slTriggerBy"] = self._map_trigger_by(trigger_by_source)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            params["tpTriggerBy"] = self._map_trigger_by(trigger_by_source)

        # для stop/conditional-ордеров указываем общий triggerBy
        if orderType.lower() in {"stop", "conditional"}:
            params["triggerBy"] = self._map_trigger_by(trigger_by_source)

        resp = self._send_request("POST", "/v5/order/create", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return resp.get("result")
        print(f"[BybitAPI] place_order error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    def update_position_stop_loss(
        self,
        symbol: str,
        new_sl: float,
        *,
        trigger_by_source: str = "mark",
        position_idx: Optional[int] = None
    ) -> bool:
        """
        Обновляет SL по открытой позиции (используется стратегией/трейлом).
        По v5 это /v5/position/trading-stop (категория linear/inverse).
        """
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "stopLoss": str(new_sl),
            "slTriggerBy": self._map_trigger_by(trigger_by_source),
        }
        if position_idx is not None:
            params["positionIdx"] = str(position_idx)

        resp = self._send_request("POST", "/v5/position/trading-stop", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return True
        try:
            print(f"[BybitAPI] update_position_stop_loss failed: {resp.get('retMsg')}")
        except Exception:
            pass
        return False

    def modify_order(
        self,
        symbol: str,
        *,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        price: Optional[float] = None,
        qty: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trigger_by_source: str = "mark",
    ) -> Optional[Dict]:
        """
        Изменение ордера. Если ID/LinkID не передан, а хотим поменять только SL/TP,
        прозрачно фолбэкнемся на обновление SL/TP по позиции (trading-stop).
        """
        # Фолбэк — только SL/TP без id
        if (order_id is None and order_link_id is None) and (stop_loss is not None or take_profit is not None) and price is None and qty is None:
            ok_sl = True
            ok_tp = True
            if stop_loss is not None:
                ok_sl = self.update_position_stop_loss(symbol, float(stop_loss), trigger_by_source=trigger_by_source)
            if take_profit is not None:
                params_tp = {
                    "category": self.market_type,
                    "symbol": (symbol or "").upper(),
                    "takeProfit": str(take_profit),
                    "tpTriggerBy": self._map_trigger_by(trigger_by_source),
                }
                resp_tp = self._send_request("POST", "/v5/position/trading-stop", params_tp, requires_auth=True)
                ok_tp = (resp_tp.get("retCode") == 0)
                if not ok_tp:
                    try:
                        print(f"[BybitAPI] modify_order TP fallback failed: {resp_tp.get('retMsg')}")
                    except Exception:
                        pass
            return {"ok": (ok_sl and ok_tp)}

        # Иначе — обычный amend
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
        }
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if price is not None:
            params["price"] = str(price)
        if qty is not None:
            params["qty"] = str(qty)

        resp = self._send_request("POST", "/v5/order/amend", params, requires_auth=True)
        if resp.get("retCode") != 0:
            print(f"[BybitAPI] order/amend error: {resp.get('retCode')} {resp.get('retMsg')}")
            return None

        # Доп. шаг: SL/TP
        if stop_loss is not None or take_profit is not None:
            ok = True
            if stop_loss is not None:
                ok = ok and self.update_position_stop_loss(symbol, float(stop_loss), trigger_by_source=trigger_by_source)
            if take_profit is not None:
                params_tp = {
                    "category": self.market_type,
                    "symbol": (symbol or "").upper(),
                    "takeProfit": str(take_profit),
                    "tpTriggerBy": self._map_trigger_by(trigger_by_source),
                }
                resp_tp = self._send_request("POST", "/v5/position/trading-stop", params_tp, requires_auth=True)
                ok = ok and (resp_tp.get("retCode") == 0)
            return {"ok": ok, "result": resp.get("result")}

        return resp.get("result")

    # ==================== WebSocket ====================

    def start_websocket(self, on_message: Optional[Callable] = None):
        """Запуск WS с авто-переподключением (публичный стрим категории market_type)."""
        from websocket import WebSocketApp

        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                topic = data.get("topic")
                if topic and topic in self.ws_callbacks:
                    self.ws_callbacks[topic](data)
            except Exception as e:
                print(f"WS message error: {e}")

        def on_ws_error(ws, error):
            print(f"WS error: {error}")

        def on_ws_close(ws, *_):
            print("WS closed — reconnecting...")
            time.sleep(2)
            self.start_websocket(on_message)

        def on_ws_open(ws):
            print("WS connection opened")

        self.ws = WebSocketApp(
            self.ws_url,
            on_open=on_ws_open,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def _convert_kline_item(self, it: Dict) -> Optional[Dict]:
        """Конвертация WS-kline item -> candle dict (как ждёт KWINStrategy)."""
        try:
            ts = int(it.get("start") or it.get("startTime") or it.get("t") or 0)
            if ts == 0:
                return None
            return {
                "timestamp": ts,  # уже мс у Bybit
                "open":  self._to_float(it.get("open")  or it.get("o")),
                "high":  self._to_float(it.get("high")  or it.get("h")),
                "low":   self._to_float(it.get("low")   or it.get("l")),
                "close": self._to_float(it.get("close") or it.get("c")),
                "volume": self._to_float(it.get("volume") or it.get("v")),
            }
        except Exception:
            return None

    def subscribe_kline(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """
        Подписка на kline. В callback отдаём УЖЕ ГОТОВЫЙ candle-словарь
        {'timestamp','open','high','low','close','volume'} ТОЛЬКО по закрытию бара.
        """
        if not self.ws:
            self.start_websocket()
        topic = f"kline.{interval}.{(symbol or '').upper()}"

        def _cb(data: Dict):
            try:
                items = data.get("data") or []
                for it in items:
                    if bool(it.get("confirm", False)):
                        candle = self._convert_kline_item(it)
                        if candle:
                            callback(candle)
            except Exception as e:
                print(f"kline callback error: {e}")

        self.ws_callbacks[topic] = _cb
        msg = {"op": "subscribe", "args": [topic]}
        try:
            if self.ws:
                self.ws.send(json.dumps(msg))
        except Exception as e:
            print(f"WS send error: {e}")
