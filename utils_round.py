# utils_round.py
from decimal import Decimal, ROUND_DOWN, ROUND_UP

def round_price(price: float, tick: float, mode: str = "down") -> float:
    if tick <= 0:
        return float(price)
    q = Decimal(str(tick))
    p = Decimal(str(price))
    if mode == "up":
        return float((p / q).to_integral_value(rounding=ROUND_UP) * q)
    return float((p / q).to_integral_value(rounding=ROUND_DOWN) * q)

def round_qty(qty: float, step: float, mode: str = "down") -> float:
    if step <= 0:
        return float(qty)
    q = Decimal(str(step))
    v = Decimal(str(qty))
    if mode == "up":
        return float((v / q).to_integral_value(rounding=ROUND_UP) * q)
    return float((v / q).to_integral_value(rounding=ROUND_DOWN) * q)
