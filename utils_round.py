import math
import time
from typing import Dict, Any

# Bybit формат шага цены/кол-ва берём из /v5/market/instruments-info
# result.list[*].lotSizeFilter.stepSize, priceFilter.tickSize

class InstrumentCache:
    def __init__(self, bybit_client, ttl_sec: int = 300):
        self.cli = bybit_client
        self.ttl = ttl_sec
        self.cache: Dict[str, Dict[str, Any]] = {}  # key: f"{category}:{symbol}" -> {"ts":..., "spec":...}

    def get(self, symbol: str, category: str) -> Dict[str, Any]:
        key = f"{category}:{symbol}"
        now = time.time()
        item = self.cache.get(key)
        if item and now - item["ts"] < self.ttl:
            return item["spec"]

        data = self.cli.get("/v5/market/instruments-info", {"category": category, "symbol": symbol})
        lst = (data.get("result", {}) or {}).get("list", []) or []
        if not lst:
            # запасной дефолт — обычно ETHUSDT: tick=0.05, lot=0.001 — но лучше не полагаться
            spec = {"tickSize": "0.01", "stepSize": "0.001"}
        else:
            info = lst[0]
            tick = (info.get("priceFilter") or {}).get("tickSize") or "0.01"
            step = (info.get("lotSizeFilter") or {}).get("stepSize") or "0.001"
            spec = {"tickSize": str(tick), "stepSize": str(step)}

        self.cache[key] = {"ts": now, "spec": spec}
        return spec

def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor((value + 1e-12) / step) * step

def _quantize_str(x: float, step_str: str) -> str:
    # корректно форматируем строку в нужной точности
    if "." in step_str:
        dec = len(step_str.split(".")[1].rstrip("0"))
    else:
        dec = 0
    fmt = f"{{:.{dec}f}}"
    return fmt.format(x)

def normalize_price_qty(price: str, qty: str, tp: str, sl: str, spec: Dict[str, str]):
    tick = float(spec["tickSize"])
    step = float(spec["stepSize"])

    p  = _round_to_step(float(price), tick)
    tpp = _round_to_step(float(tp),    tick)
    sll = _round_to_step(float(sl),    tick)
    q  = _round_to_step(float(qty),  step)

    return (
        _quantize_str(p,   spec["tickSize"]),
        _quantize_str(q,   spec["stepSize"]),
        _quantize_str(tpp, spec["tickSize"]),
        _quantize_str(sll, spec["tickSize"]),
    )
