# bybit_api.py — REST-обёртка для KWINStrategy (HTTP v5, БЕЗ WebSocket)
import requests
import json
import time
import hashlib
import hmac
import random
from urllib.parse import urlencode
from typing import Dict, List, Optional, Callable, Any

# Необязательные хелперы округления по тик-сайзу/шагу (стратегия сама округляет, но дублируем защитно)
try:
    from utils_round import round_price, round_qty, floor_to_tick, ceil_to_tick  # noqa: F401
except Exception:
    def round_price(x, *_args, **_kw): return x
    def round_qty(x, *_args, **_kw): return x
    def floor_to_tick(x, *_): return x
    def ceil_to_tick(x, *_): return x


class BybitAPI:
    """Лёгкая REST-обёртка над Bybit v5 (без WS).
       По умолчанию — категория linear (USDT-M).
       Совместимо с KWINStrategy.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        *,
        market_type: str = "linear",
        account_type: str = "UNIFIED",
        timeout: int = 15,
        retries: int = 2,
        backoff: float = 0.5,
        recv_window: int = 5000,
        proxies: Optional[Dict[str, str]] = None,
        hedge_mode: bool = True,
        default_position_idx: Optional[int] = None,
        **_kwargs
    ):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.testnet = bool(testnet)

        # Категория рынка для v5: "linear"|"inverse"|"spot"|"option"
        self.market_type = (market_type or "linear").lower()
        if self.market_type not in ("linear", "inverse", "spot", "option"):
            self.market_type = "linear"

        # Тип аккаунта для /wallet-balance
        self.account_type = (account_type or "UNIFIED").upper()
        self.hedge_mode = bool(hedge_mode)
        self.default_position_idx = default_position_idx  # 1=One-Way / 3=Hedge-Short / 5=Hedge-Long (зависит от категории)

        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        # HTTP с повторными попытками
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.recv_window = int(recv_window)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KWINBot/1.0",
            "Accept": "application/json",
        })
        if proxies:
            try:
                self.session.proxies.update(proxies)
            except Exception:
                pass

        # ----- MUST-HAVE: кэш инструментов и внешний провайдер цены -----
        self._instr_cache: Dict[str, Dict] = {}
        self._instr_cache_ts: Dict[str, float] = {}
        self._instr_ttl_sec: int = 300

        self._price_provider: Optional[Callable[[str, str], Optional[float]]] = None

    # ==================== утилиты подписи/сетевые ====================

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
        prehash = timestamp_ms + self.api_key + str(self.recv_window) + payload
        return hmac.new(
            self.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _http_call(self, method: str, url: str, **kw) -> requests.Response:
        # Простые ретраи на сетевые ошибки/5xx с экспон.бэк-оффом + джиттер
        last_exc = None
        for i in range(self.retries + 1):
            try:
                return self.session.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                last_exc = e
                delay = self.backoff * (2 ** i)
                delay *= (0.8 + 0.4 * random.random())
                time.sleep(delay)
        raise last_exc

    def _send_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        *,
        requires_auth: bool = False,
        timeout: Optional[int] = None
    ) -> Dict:
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        # Коды, при которых имеет смысл повторить (rate-limit/сервис)
        RETRYABLE_CODES = {10006, 10007, 30005, 30033}  # too many requests / timeout / system busy (примерный набор)

        try:
            # внутренний цикл ретраев по retCode (помимо сетевых)
            for attempt in range(self.retries + 1):
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
                        "X-BAPI-RECV-WINDOW": str(self.recv_window),
                    })

                if method.upper() == "GET":
                    r = self._http_call("GET", url, params=params, headers=headers)
                elif method.upper() == "POST":
                    data = json.dumps(params, separators=(',', ':'), ensure_ascii=False) if params else None
                    r = self._http_call("POST", url, data=data, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                r.raise_for_status()
                resp = r.json()

                # успех = retCode 0 или отсутствие retCode
                ret_code = resp.get("retCode")
                if ret_code in (0, "0", None):
                    return resp

                # неуспех — возможно повторим
                if isinstance(ret_code, int) and ret_code in RETRYABLE_CODES and attempt < self.retries:
                    delay = self.backoff * (2 ** attempt)
                    delay *= (0.8 + 0.4 * random.random())
                    time.sleep(delay)
                    continue

                try:
                    print(f"[BybitAPI] {method} {endpoint} retCode={ret_code} retMsg={resp.get('retMsg')}")
                except Exception:
                    pass
                return resp  # вернём как есть

            # если вылетели по ретраям
            return {"retCode": -1, "retMsg": "retry_exceeded"}
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
        """Фиксируем категорию (linear/inverse/spot/option). По умолчанию — linear."""
        m = (market_type or "linear").lower()
        if m not in ("linear", "inverse", "spot", "option"):
            m = "linear"
        self.market_type = m

    def force_linear(self):
        """Насильно фиксирует работу только с фьючерсами (linear)."""
        self.set_market_type("linear")

    # ==================== кэш инструментов / квантование ====================

    def _get_instr_cached(self, symbol: str) -> Optional[Dict]:
        key = (symbol or "").upper()
        now = time.time()
        if key in self._instr_cache and (now - self._instr_cache_ts.get(key, 0)) < self._instr_ttl_sec:
            return self._instr_cache[key]
        info = self._get_instruments_info_uncached(key)
        if info:
            self._instr_cache[key] = info
            self._instr_cache_ts[key] = now
        return info

    def quantize_price(self, symbol: str, price: float) -> float:
        info = self._get_instr_cached(symbol) or {}
        tick = float(((info.get("priceFilter") or {}).get("tickSize") or 0.0) or 0.0)
        if tick <= 0:
            return float(price)
        # округление к ближайшему тика (как в стратегии для pine-like)
        return round(float(price) / tick) * tick

    def quantize_qty(self, symbol: str, qty: float) -> float:
        info = self._get_instr_cached(symbol) or {}
        step = float(((info.get("lotSizeFilter") or {}).get("qtyStep") or 0.0) or 0.0)
        if step <= 0:
            return float(qty)
        return round(float(qty) / step) * step

    def enforce_min_qty(self, symbol: str, qty: float) -> float:
        info = self._get_instr_cached(symbol) or {}
        m = float(((info.get("lotSizeFilter") or {}).get("minOrderQty") or 0.0) or 0.0)
        return float(qty) if (m <= 0 or float(qty) >= m) else 0.0

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

    def _dedup_and_sort_kl(self, rows: List[Dict]) -> List[Dict]:
        """Дедуп по timestamp и сортировка по возрастанию — детерминизм для бэктеста."""
        uniq: Dict[int, Dict] = {}
        for r in rows or []:
            try:
                ts = int(r.get("timestamp") or 0)
                if ts:
                    uniq[ts] = {
                        "timestamp": ts,
                        "open":  self._to_float(r.get("open")),
                        "high":  self._to_float(r.get("high")),
                        "low":   self._to_float(r.get("low")),
                        "close": self._to_float(r.get("close")),
                        "volume": self._to_float(r.get("volume")),
                    }
            except Exception:
                continue
        return sorted(uniq.values(), key=lambda x: x["timestamp"])

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        """
        Последние N свечей по категории self.market_type.
        interval: строка минут ("1","3","5","15","60",...).
        Возвращает список по **убыванию** timestamp (newest-first).
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
            out: List[Dict] = []
            for it in lst:
                try:
                    ts = int(it[0])
                    if ts < 1_000_000_000_000:
                        ts *= 1000
                    out.append({
                        "timestamp": ts,
                        "open":  self._to_float(it[1]),
                        "high":  self._to_float(it[2]),
                        "low":   self._to_float(it[3]),
                        "close": self._to_float(it[4]),
                        "volume": self._to_float(it[5]),
                    })
                except Exception:
                    continue
            out.sort(key=lambda x: x["timestamp"], reverse=True)
            return out
        print(f"[BybitAPI] get_klines error: {resp.get('retCode')} {resp.get('retMsg')}")
        return []

    def get_klines_window(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Чтение свечей с окнами времени (start/end в мс). Возвращает по возрастанию timestamp.
        Реализована пагинация по cursor до полного покрытия окна.
        """
        params = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "interval": str(interval),
            "limit": int(min(max(1, limit), 1000)),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        out: List[Dict] = []
        safety_pages = 0
        cursor: Optional[str] = None

        while True:
            q = dict(params)
            if cursor:
                q["cursor"] = cursor
            resp = self._send_request("GET", "/v5/market/kline", q, requires_auth=False)
            if resp.get("retCode") != 0:
                print(f"[BybitAPI] get_klines_window error: {resp.get('retCode')} {resp.get('retMsg')}")
                break
            result = resp.get("result") or {}
            lst = result.get("list") or []
            for it in lst:
                try:
                    ts = int(it[0])
                    if ts < 1_000_000_000_000:
                        ts *= 1000
                    out.append({
                        "timestamp": ts,
                        "open":  self._to_float(it[1]),
                        "high":  self._to_float(it[2]),
                        "low":   self._to_float(it[3]),
                        "close": self._to_float(it[4]),
                        "volume": self._to_float(it[5]),
                    })
                except Exception:
                    continue

            cursor = result.get("nextPageCursor")
            safety_pages += 1
            if not cursor or safety_pages > 200:
                break

        return self._dedup_and_sort_kl(out)

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
                # Вернём поля и в camel, и в snake — стратегия умеет оба варианта.
                return {
                    "symbol": t.get("symbol"),
                    "lastPrice": _f(t.get("lastPrice", 0)),
                    "markPrice": _f(t.get("markPrice", 0)),
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

    # ----- внешний провайдер цены -----
    def set_price_provider(self, fn: Callable[[str, str], Optional[float]]) -> None:
        """fn(symbol, source['last'|'mark']) -> price|None"""
        self._price_provider = fn

    def get_price(self, symbol: str, source: str = "last") -> float:
        # сперва — внешний провайдер (WS), затем REST
        if self._price_provider:
            try:
                p = self._price_provider(symbol, source)
                if isinstance(p, (int, float)) and p > 0:
                    return float(p)
            except Exception:
                pass
        t = self.get_ticker(symbol) or {}
        last = self._to_float(t.get("lastPrice", 0))
        mark = self._to_float(t.get("markPrice", 0))
        return mark if (str(source).lower() == "mark" and mark > 0) else last

    # ----- instruments-info + кэш -----
    def _get_instruments_info_uncached(self, symbol: str) -> Optional[Dict]:
        params = {"category": self.market_type, "symbol": (symbol or "").upper()}
        resp = self._send_request("GET", "/v5/market/instruments-info", params, requires_auth=False)
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            if not lst:
                print(f"[BybitAPI] instruments-info empty for {params}")
                return None
            info = dict(lst[0])
            # Нормализация: приводим к float-типам, если они строковые
            try:
                pf = info.get("priceFilter") or {}
                ls = info.get("lotSizeFilter") or {}
                for k in ("tickSize",):
                    if k in pf:
                        pf[k] = float(pf[k])
                for k in ("qtyStep", "minOrderQty"):
                    if k in ls:
                        ls[k] = float(ls[k])
                info["priceFilter"] = pf
                info["lotSizeFilter"] = ls
            except Exception:
                pass
            return info
        print(f"[BybitAPI] instruments-info error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    def get_instruments_info(self, symbol: str) -> Optional[Dict]:
        """Публичный метод: отдаём из кэша с TTL."""
        return self._get_instr_cached(symbol)

    # ==================== приватные (trade/account) ====================

    def get_wallet_balance(self, account_type: Optional[str] = None) -> Optional[Dict]:
        """
        Стратегия может вызывать это, когда equity_source='wallet'.
        По умолчанию берём self.account_type (обычно 'UNIFIED').
        """
        params = {"accountType": (account_type or self.account_type)}
        resp = self._send_request("GET", "/v5/account/wallet-balance", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return resp.get("result")
        print(f"[BybitAPI] wallet-balance error: {resp.get('retCode')} {resp.get('retMsg')}")
        return None

    def _fetch_position_size(self, symbol: str) -> Dict[str, Any]:
        """
        Возвращает агрегированную позицию по символу:
        { 'size': float, 'side': 'long'|'short'|None }
        """
        params = {"category": self.market_type, "symbol": (symbol or "").upper()}
        resp = self._send_request("GET", "/v5/position/list", params, requires_auth=True)
        size = 0.0
        side = None
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            for p in lst:
                sz = self._to_float(p.get("size", 0))
                sd = (p.get("side") or "").lower()
                if sz <= 0:
                    continue
                if side is None:
                    side = "long" if sd == "buy" else "short"
                size += sz if sd == "buy" else -sz
            if size > 0:
                side = "long"
            elif size < 0:
                side = "short"
                size = abs(size)
            else:
                side = None
        else:
            print(f"[BybitAPI] position/list error: {resp.get('retCode')} {resp.get('retMsg')}")
        return {"size": float(size), "side": side}

    def close_position(self, symbol: str) -> bool:
        """
        Закрыть текущую позицию рыночным ордером (reduceOnly).
        Именно это дергает KWINStrategy при flip/close.
        """
        pos = self._fetch_position_size(symbol)
        size = float(pos.get("size") or 0.0)
        side = pos.get("side")
        if size <= 0 or side not in ("long", "short"):
            return True  # уже нет позиции — считаем успехом

        close_side = "Sell" if side == "long" else "Buy"
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "side": close_side,
            "orderType": "Market",
            "qty": str(size),
            "reduceOnly": "true",
            # идемпотентность
            "orderLinkId": f"kwin-cls-{(symbol or '').upper()}-{int(time.time()*1000)}",
        }
        # в hedge-режиме укажем positionIdx, если задан по умолчанию
        if self.default_position_idx is not None:
            params["positionIdx"] = str(self.default_position_idx)

        resp = self._send_request("POST", "/v5/order/create", params, requires_auth=True)
        if resp.get("retCode") == 0:
            return True
        print(f"[BybitAPI] close_position error: {resp.get('retCode')} {resp.get('retMsg')}")
        return False

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
        """
        KWINStrategy отправляет Market + stop_loss, trigger_by_source = 'last'|'mark'.
        Здесь же делаем корректную мапу для v5 + квантование и minQty.
        """
        # квантование / minQty
        q = self.quantize_qty(symbol, float(qty))
        q = self.enforce_min_qty(symbol, q)
        if q <= 0:
            print("[BybitAPI] place_order: qty < minOrderQty")
            return None
        p = None
        if price is not None:
            p = self.quantize_price(symbol, float(price))

        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "side": str(side).title(),            # Buy/Sell
            "orderType": str(orderType).title(),  # Market/Limit/...
            "qty": str(q),
        }
        if p is not None:
            params["price"] = str(p)
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_link_id:
            params["orderLinkId"] = order_link_id
        else:
            params["orderLinkId"] = f"kwin-{(symbol or '').upper()}-{int(time.time()*1000)}"
        if time_in_force:
            params["timeInForce"] = time_in_force

        # positionIdx: если явно передали — используем, иначе дефолт, если задан
        eff_pos_idx = position_idx if position_idx is not None else self.default_position_idx
        if eff_pos_idx is not None:
            params["positionIdx"] = str(eff_pos_idx)
        if tpsl_mode:
            params["tpslMode"] = str(tpsl_mode)

        # SL/TP + источник триггера (MarkPrice/LastPrice)
        if stop_loss is not None:
            sl_q = self.quantize_price(symbol, float(stop_loss))
            params["stopLoss"] = str(sl_q)
            params["slTriggerBy"] = self._map_trigger_by(trigger_by_source)
        if take_profit is not None:
            tp_q = self.quantize_price(symbol, float(take_profit))
            params["takeProfit"] = str(tp_q)
            params["tpTriggerBy"] = self._map_trigger_by(trigger_by_source)

        # Если это stop/conditional — общий triggerBy
        if str(orderType).lower() in {"stop", "conditional"}:
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
        По v5 это /v5/position/trading-stop.
        """
        sl_q = self.quantize_price(symbol, float(new_sl))
        params: Dict[str, str] = {
            "category": self.market_type,
            "symbol": (symbol or "").upper(),
            "stopLoss": str(sl_q),
            "slTriggerBy": self._map_trigger_by(trigger_by_source),
        }
        eff_pos_idx = position_idx if position_idx is not None else self.default_position_idx
        if eff_pos_idx is not None:
            params["positionIdx"] = str(eff_pos_idx)

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
                tp_q = self.quantize_price(symbol, float(take_profit))
                params_tp = {
                    "category": self.market_type,
                    "symbol": (symbol or "").upper(),
                    "takeProfit": str(tp_q),
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
            p = self.quantize_price(symbol, float(price))
            params["price"] = str(p)
        if qty is not None:
            q = self.quantize_qty(symbol, float(qty))
            q = self.enforce_min_qty(symbol, q)
            if q <= 0:
                print("[BybitAPI] modify_order: qty < minOrderQty")
                return None
            params["qty"] = str(q)

        resp = self._send_request("POST", "/v5/order/amend", params, requires_auth=True)
        if resp.get("retCode") != 0:
            print(f"[BybitAPI] order/amend error: {resp.get('retCode')} {resp.get('retMsg')}")
            return None

        # Доп. шаг: SL/TP через trading-stop
        if stop_loss is not None or take_profit is not None:
            ok = True
            if stop_loss is not None:
                ok = ok and self.update_position_stop_loss(symbol, float(stop_loss), trigger_by_source=trigger_by_source)
            if take_profit is not None:
                tp_q = self.quantize_price(symbol, float(take_profit))
                params_tp = {
                    "category": self.market_type,
                    "symbol": (symbol or "").upper(),
                    "takeProfit": str(tp_q),
                    "tpTriggerBy": self._map_trigger_by(trigger_by_source),
                }
                resp_tp = self._send_request("POST", "/v5/position/trading-stop", params_tp, requires_auth=True)
                ok = ok and (resp_tp.get("retCode") == 0)
            return {"ok": ok, "result": resp.get("result")}

        return resp.get("result")
