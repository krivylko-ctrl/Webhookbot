# utils_round.py
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Optional

# -----------------------------
# БАЗОВЫЕ ФУНКЦИИ
# -----------------------------
def round_price(price: float, tick: float, mode: str = "down") -> float:
    """
    Округление цены к шагу tick.
    mode="down" -> floor (вниз, для SL long)
    mode="up"   -> ceil  (вверх, для SL short)
    """
    if tick <= 0:
        return float(price)
    q = Decimal(str(tick))
    p = Decimal(str(price))
    if mode.lower() == "up":
        return float((p / q).to_integral_value(rounding=ROUND_UP) * q)
    return float((p / q).to_integral_value(rounding=ROUND_DOWN) * q)


def round_qty(qty: float, step: float, mode: str = "down") -> float:
    """
    Округление количества к шагу step.
    mode="down" -> floor (для ордеров по рынку, Binance/MEXC/Bybit)
    mode="up"   -> ceil  (для защитных ордеров, если нужно гарантировать исполнение)
    """
    if step <= 0:
        return float(qty)
    q = Decimal(str(step))
    v = Decimal(str(qty))
    if mode.lower() == "up":
        return float((v / q).to_integral_value(rounding=ROUND_UP) * q)
    return float((v / q).to_integral_value(rounding=ROUND_DOWN) * q)


# -----------------------------------------
# ДОП. ХЕЛПЕРЫ 1:1 с PineScript
# -----------------------------------------
def floor_to_tick(price: float, tick: float) -> float:
    """Жёстко вниз к ближайшему тиковому шагу (для SL long)."""
    return round_price(price, tick, mode="down")


def ceil_to_tick(price: float, tick: float) -> float:
    """Жёстко вверх к ближайшему тиковому шагу (для SL short)."""
    return round_price(price, tick, mode="up")


def round_to_tick(price: float, tick: float) -> float:
    """
    Округление к ближайшему тиковому шагу (как Pine `round()`).
    Используется для entry/TP, где не важна предвзятость вверх/вниз.
    """
    if tick <= 0:
        return float(price)
    q = Decimal(str(tick))
    p = Decimal(str(price))
    return float((p / q).to_integral_value(rounding=ROUND_HALF_UP) * q)


# -----------------------------
# АЛИАСЫ ДЛЯ СОВМЕСТИМОСТИ
# -----------------------------
# В проекте могут использовать старые вызовы:
#   from utils_round import round_price, round_qty
#   from utils_round import price_round, qty_round
price_round = round_to_tick   # ближе всего к PineScript `round()`
qty_round = round_qty
