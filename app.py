import os
import json
import logging
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from bybit_api import BybitREST, BybitHTTPError
from utils_round import InstrumentCache, normalize_price_qty

load_dotenv()

# ================== ENV / CONFIG ==================
PORT           = int(os.getenv("PORT", "8000"))
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO").upper()
WEBHOOK_SECRET = (os.getenv("WEBHOOK_SECRET") or "").strip()
REQUIRE_SECRET = os.getenv("REQUIRE_SECRET", "false").lower() in ("1","true","yes")
BYBIT_BASE_URL = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").rstrip("/")

BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Если ключей нет — предупреждаем (но не падаем, чтобы /health работал)
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    print("WARNING: BYBIT_API_KEY / BYBIT_API_SECRET не заданы — запросы к Bybit упадут.")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("tv-bybit-proxy")

app = Flask(__name__)

bybit = BybitREST(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, base_url=BYBIT_BASE_URL)
instr_cache = InstrumentCache(bybit_client=bybit)

# ================== SECURITY ==================
def verify_secret(data: Dict[str, Any]) -> bool:
    """
    TradingView не умеет слать кастомные заголовки. Поэтому допускаем секрет в payload: data["secret"].
    Если REQUIRE_SECRET=false, проверка не обязательна.
    """
    if not REQUIRE_SECRET:
        return True
    sent = (data.get("secret") or "").strip()
    return WEBHOOK_SECRET and sent and (sent == WEBHOOK_SECRET)

# ================== VALIDATION ==================
def validate_payload(data: Dict[str, Any]) -> Tuple[bool, str]:
    for k in ("event", "exchange", "category", "symbol", "tv_symbol"):
        if k not in data:
            return False, f"Missing field: {k}"

    if data["category"] != "linear":
        return False, "Only category=linear supported"

    if data["event"] == "open_block":
        for k in ("side", "legs", "oid_prefix"):
            if k not in data:
                return False, f"Missing field in open_block: {k}"
        if data["side"] not in ("Buy", "Sell"):
            return False, "side must be Buy or Sell"
        legs = data["legs"]
        if not isinstance(legs, list) or not legs:
            return False, "legs must be non-empty array"
        for i, leg in enumerate(legs):
            for lk in ("id","orderLinkId","price","qty","lev","tp","sl"):
                if lk not in leg:
                    return False, f"Missing leg field `{lk}` at index {i}"
    elif data["event"] == "cancel_block":
        if "oid_prefix" not in data:
            return False, "Missing oid_prefix for cancel_block"
    else:
        return False, "Unsupported event"

    return True, ""

# ================== BYBIT HELPERS ==================
def ensure_leverage(symbol: str, category: str, lev: str):
    """
    Ставим одинаковое плечо для buy/sell. Если уже стоит — Bybit вернёт OK/noop.
    """
    try:
        bybit.post("/v5/position/set-leverage", {
            "category": category,
            "symbol": symbol,
            "buyLeverage": lev,
            "sellLeverage": lev
        })
        log.info(f"[lev] {symbol} -> {lev}x ok")
    except BybitHTTPError as e:
        # Часто при совпадении значений будет retCode 0 OK — а при несовпадении тоже OK.
        log.warning(f"[lev] warn: {e}")

def list_active_orders(symbol: str, category: str) -> List[Dict[str, Any]]:
    """
    Возвращает активные ордера по символу.
    """
    out = []
    cursor = None
    while True:
        params = {"category": category, "symbol": symbol}
        if cursor:
            params["cursor"] = cursor
        data = bybit.get("/v5/order/realtime", params)
        lst = (data.get("result", {}) or {}).get("list", []) or []
        out.extend(lst)
        cursor = (data.get("result", {}) or {}).get("nextPageCursor")
        if not cursor:
            break
    return out

def already_exists(order_link_id: str, active_orders: List[Dict[str, Any]]) -> bool:
    for o in active_orders:
        if (o.get("orderLinkId") or "") == order_link_id:
            return True
    return False

def place_block_orders(side: str, symbol: str, category: str, legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1) Выставляем плечо по макс. lev среди ног
    2) Нормализуем price/qty/tp/sl по шагам инструмента
    3) Не создаём ордера с уже существующим orderLinkId
    4) Создаём лимитные ордера с tpslMode=Full
    """
    # 1) max lev
    max_lev = max([int(float(leg["lev"])) for leg in legs])
    ensure_leverage(symbol, category, str(max_lev))

    # 2) кэш инструментов (tickSize/lotSize)
    spec = instr_cache.get(symbol=symbol, category=category)

    # 3) список активных для идемпотентности
    actives = list_active_orders(symbol, category)

    results = []
    for leg in legs:
        olid_raw = str(leg["orderLinkId"])
        olid = olid_raw[:36]  # лимит Bybit v5
        price, qty, tp, sl = normalize_price_qty(
            price=str(leg["price"]),
            qty=str(leg["qty"]),
            tp=str(leg["tp"]),
            sl=str(leg["sl"]),
            spec=spec
        )
        if already_exists(olid, actives):
            log.info(f"[order] skip (exists): {olid}")
            results.append({"orderLinkId": olid, "status": "exists"})
            continue

        body = {
            "category": category,
            "symbol": symbol,
            "side": side,                 # "Buy"/"Sell"
            "orderType": "Limit",
            "qty": qty,
            "price": price,
            "timeInForce": "GTC",
            "orderLinkId": olid,
            "takeProfit": tp,
            "stopLoss": sl,
            "tpslMode": "Full"
        }
        try:
            resp = bybit.post("/v5/order/create", body)
            results.append({"orderLinkId": olid, "status": "created", "resp": resp.get("result")})
            log.info(f"[order] created: {olid} {side} {symbol} p={price} q={qty} tp={tp} sl={sl}")
        except BybitHTTPError as e:
            results.append({"orderLinkId": olid, "status": "error", "error": str(e)})
            log.error(f"[order] error {olid}: {e}")

    return {"placed": results}

def cancel_block_orders(symbol: str, category: str, oid_prefix: str) -> Dict[str, Any]:
    """
    Отменяем все активные ордера, чей orderLinkId начинается с oid_prefix
    """
    cancelled = []
    actives = list_active_orders(symbol, category)
    for o in actives:
        olid = (o.get("orderLinkId") or "")
        if olid.startswith(oid_prefix):
            try:
                bybit.post("/v5/order/cancel", {
                    "category": category,
                    "symbol": symbol,
                    "orderLinkId": olid
                })
                cancelled.append(olid)
                log.info(f"[cancel] {olid}")
            except BybitHTTPError as e:
                log.error(f"[cancel] error {olid}: {e}")
    return {"cancelled": cancelled}

# ================== ROUTES ==================
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/webhook")
def webhook():
    # 1) разбор JSON
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            raise ValueError("payload is not an object")
    except Exception:
        return jsonify({"ok": False, "error": "invalid json"}), 400

    # 2) секрет (опционально)
    if not verify_secret(data):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    # 3) валидация
    ok, msg = validate_payload(data)
    if not ok:
        return jsonify({"ok": False, "error": f"validation: {msg}"}), 400

    event    = data["event"]
    category = data["category"]
    symbol   = data["symbol"]

    try:
        if event == "open_block":
            side = data["side"]
            legs = data["legs"]
            result = place_block_orders(side, symbol, category, legs)
            return jsonify({"ok": True, **result})

        elif event == "cancel_block":
            oid_prefix = data["oid_prefix"]
            result = cancel_block_orders(symbol, category, oid_prefix)
            return jsonify({"ok": True, **result})

        else:
            return jsonify({"ok": False, "error": "unsupported event"}), 400

    except BybitHTTPError as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    except Exception as e:
        log.exception("webhook unhandled")
        return jsonify({"ok": False, "error": f"internal: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
