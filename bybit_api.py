import time
import hmac
import json
import hashlib
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlencode

class BybitHTTPError(Exception):
    pass

class BybitREST:
    """
    Минимальная v5-обёртка с ретраями (429/5xx), таймаутами и унифицированной ошибкой.
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str, timeout: int = 20):
        self.api_key = api_key or ""
        self.api_secret = (api_secret or "").encode()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})

    def _sign(self, ts: int, method: str, path: str, query: str, body: str) -> (str, str):
        recv_window = "5000"
    # v5: sign = ts + apiKey + recvWindow + body  (без method/path/query)
        payload = f"{ts}{self.api_key}{recv_window}{body}"
        sign = hmac.new(self.api_secret, payload.encode(), hashlib.sha256).hexdigest()
        return sign, recv_window

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        params = params or {}
        ts = int(time.time() * 1000)

        if method.upper() == "GET":
            query = "?" + urlencode(params, doseq=True) if params else ""
            body = ""
        else:
            query = ""
            body = json.dumps(params, separators=(",", ":"))

        sign, recv = self._sign(ts, method, path, query, body)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(ts),
            "X-BAPI-SIGN": sign,
            "X-BAPI-RECV-WINDOW": recv,
        }

        # Простые ретраи на 429/5xx
        for attempt in range(4):
            try:
                if method.upper() == "GET":
                    r = self.s.get(url, headers=headers, params=params, timeout=self.timeout)
                else:
                    r = self.s.post(url, headers=headers, data=body, timeout=self.timeout)

                if r.status_code >= 500 or r.status_code == 429:
                    time.sleep(0.4 * (attempt + 1))
                    continue

                if r.status_code >= 400:
                    raise BybitHTTPError(f"HTTP {r.status_code}: {r.text}")

                data = r.json()
                if data.get("retCode") != 0:
                    raise BybitHTTPError(f"retCode {data.get('retCode')} {data.get('retMsg')} | {data}")
                return data

            except requests.RequestException as e:
                # сетевой глитч — ретрай
                time.sleep(0.4 * (attempt + 1))
                if attempt == 3:
                    raise BybitHTTPError(f"network error: {e}")

        # не должны сюда дойти
        raise BybitHTTPError("unreachable after retries")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params)

    def post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, params)
