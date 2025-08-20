# utils_round.py
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Optional


# -----------------------------
# БАЗОВЫЕ ФУНКЦИИ (как у тебя)
# -----------------------------
def round_price(price: float, tick: float, mode: str = "down") -> float:
    """
    Округление цены к шагу tick.
    mode="down"  -> округление вниз (floor)
    mode="up"    -> округление вверх (ceil)
    """
    if tick <= 0:
        return float(price)
    q = Decimal(str(tick))
    p = Decimal(str(price))
    if mode.lower() == "up":
        return float((p / q).to_integral_value(rounding=ROUND_UP) * q)
    # по умолчанию вниз
    return float((p / q).to_integral_value(rounding=ROUND_DOWN) * q)


def round_qty(qty: float, step: float, mode: str = "down") -> float:
    """
    Округление количества к шагу step.
    mode="down"  -> округление вниз (floor)
    mode="up"    -> округление вверх (ceil)
    """
    if step <= 0:
        return float(qty)
    q = Decimal(str(step))
    v = Decimal(str(qty))
    if mode.lower() == "up":
        return float((v / q).to_integral_value(rounding=ROUND_UP) * q)
    # по умолчанию вниз
    return float((v / q).to_integral_value(rounding=ROUND_DOWN) * q)


# -----------------------------------------
# ДОП. ХЕЛПЕРЫ ДЛЯ 1:1 С PINE-ЛОГИКОЙ ТИКОВ
# -----------------------------------------
def floor_to_tick(price: float, tick: float) -> float:
    """Жёстко вниз к ближайшему тиковому шагу (используй для SL long)."""
    return round_price(price, tick, mode="down")


def ceil_to_tick(price: float, tick: float) -> float:
    """Жёстко вверх к ближайшему тиковому шагу (используй для SL short)."""
    return round_price(price, tick, mode="up")


def round_to_tick(price: float, tick: float) -> float:
    """
    Округление к ближайшему тиковому шагу (как 'nearest').
    Полезно для entry/TP, когда не требуется предвзятость вниз/вверх.
    """
    if tick <= 0:
        return float(price)
    q = Decimal(str(tick))
    p = Decimal(str(price))
    # банковское округление нам не нужно — берём HALF_UP
    return float((p / q).to_integral_value(rounding=ROUND_HALF_UP) * q)


# -----------------------------
# АЛИАСЫ ДЛЯ ОБРАТСОВМЕСТИМОСТИ
# -----------------------------
# В проекте могли использовать:
#   from utils_round import round_price, round_qty
#   from utils_round import price_round, qty_round
# Оставляем те же имена.
price_round = round_to_tick   # "по-умолчанию" — к ближайшему тиковому шагу
qty_round = round_qty         # количество традиционно режем вниз
