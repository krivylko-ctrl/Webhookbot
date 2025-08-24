# utils_round.py
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, localcontext, getcontext
from typing import Optional, Literal, Union

# Высокая точность для внутренних расчётов, чтобы исключить артефакты (0.30000000000004 и т.п.)
getcontext().prec = 40

# Микро-эпсилон для «прилипания» к ближайшему тику
_EPS = Decimal("1e-18")


NumberLike = Union[float, int, str, Decimal]


def _D(x: NumberLike) -> Decimal:
    """Безопасная конвертация в Decimal через str(x) — гасит двоичную погрешность."""
    # Decimal принимает Decimal как есть; иначе — через str
    return x if isinstance(x, Decimal) else Decimal(str(x))


def _q_step(step: NumberLike) -> Decimal:
    """Нормализация шага (tick/qty_step) в Decimal."""
    d = _D(step)
    if d <= 0:
        raise ValueError("Step must be > 0")
    return d


# -----------------------------
# БАЗОВЫЕ ФУНКЦИИ
# -----------------------------
def round_price(price: NumberLike, tick: NumberLike, mode: Literal["down", "up"] = "down") -> float:
    """
    Округление цены к шагу tick.
      mode="down" -> floor (вниз, для SL long)
      mode="up"   -> ceil  (вверх, для SL short)
    """
    if float(tick) <= 0:
        return float(price)

    with localcontext() as ctx:
        ctx.prec = 40
        q = _q_step(tick)
        p = _D(price)

        r = p / q
        frac = r - r.to_integral_value(rounding=ROUND_DOWN)

        # «Прилипание» к ближайшему тику, если уже почти на нём
        if frac.copy_abs() <= _EPS or (1 - frac).copy_abs() <= _EPS:
            r = r.to_integral_value(rounding=ROUND_HALF_UP)
        else:
            r = r.to_integral_value(rounding=ROUND_UP if mode.lower() == "up" else ROUND_DOWN)

        return float(r * q)


def round_qty(qty: NumberLike, step: NumberLike, mode: Literal["down", "up"] = "down") -> float:
    """
    Округление количества к шагу step.
      mode="down" -> floor (обычно для рыночных/лимитных ордеров)
      mode="up"   -> ceil  (если нужно гарантированно не недобрать, например при min_order_qty)
    """
    if float(step) <= 0:
        return float(qty)

    with localcontext() as ctx:
        ctx.prec = 40
        q = _q_step(step)
        v = _D(qty)

        r = v / q
        frac = r - r.to_integral_value(rounding=ROUND_DOWN)
        if frac.copy_abs() <= _EPS or (1 - frac).copy_abs() <= _EPS:
            r = r.to_integral_value(rounding=ROUND_HALF_UP)
        else:
            r = r.to_integral_value(rounding=ROUND_UP if mode.lower() == "up" else ROUND_DOWN)

        return float(r * q)


# -----------------------------------------
# ДОП. ХЕЛПЕРЫ 1:1 с PineScript
# -----------------------------------------
def floor_to_tick(price: NumberLike, tick: NumberLike) -> float:
    """Жёстко вниз к ближайшему тиковому шагу (для SL long)."""
    return round_price(price, tick, mode="down")


def ceil_to_tick(price: NumberLike, tick: NumberLike) -> float:
    """Жёстко вверх к ближайшему тиковому шагу (для SL short)."""
    return round_price(price, tick, mode="up")


def round_to_tick(price: NumberLike, tick: NumberLike) -> float:
    """
    Округление к ближайшему тиковому шагу (как Pine `round()`).
    Используется для entry/TP, где не важна предвзятость вверх/вниз.
    """
    if float(tick) <= 0:
        return float(price)

    with localcontext() as ctx:
        ctx.prec = 40
        q = _q_step(tick)
        p = _D(price)
        r = (p / q).to_integral_value(rounding=ROUND_HALF_UP)
        return float(r * q)


# -----------------------------
# УДОБНЫЕ ДОПОЛНЕНИЯ (без ломания API)
# -----------------------------
def is_on_tick(price: NumberLike, tick: NumberLike) -> bool:
    """Проверка, лежит ли цена точно на сетке тик-сайза (с учётом ε)."""
    if float(tick) <= 0:
        return True
    with localcontext() as ctx:
        ctx.prec = 40
        q = _q_step(tick)
        p = _D(price)
        r = p / q
        return (r - r.to_integral_value(rounding=ROUND_HALF_UP)).copy_abs() <= _EPS


def snap_to_tick_if_close(price: NumberLike, tick: NumberLike) -> float:
    """
    «Прилипание» к ближайшему тику, если расстояние меньше ε.
    Полезно перед отправкой на биржу, чтобы избежать reject из-за 1e-15.
    """
    if float(tick) <= 0:
        return float(price)
    with localcontext() as ctx:
        ctx.prec = 40
        q = _q_step(tick)
        p = _D(price)
        r = p / q
        nearest = r.to_integral_value(rounding=ROUND_HALF_UP)
        if (r - nearest).copy_abs() <= _EPS:
            return float(nearest * q)
    return float(price)


def ensure_min_qty(qty: NumberLike, min_qty: NumberLike, step: NumberLike) -> float:
    """
    Гарантируем минимальный объём: если после округления вниз он < min_qty,
    дожимаем вверх к ближайшему валидному шагу ≥ min_qty.
    """
    if float(step) <= 0:
        return max(float(qty), float(min_qty))
    qd = round_qty(qty, step, mode="down")
    if qd + 0.0 < float(min_qty):  # "+ 0.0" — явная конверсия в float
        return round_qty(min_qty, step, mode="up")
    return qd


# -----------------------------
# АЛИАСЫ ДЛЯ СОВМЕСТИМОСТИ
# -----------------------------
# Старые вызовы в проекте:
#   from utils_round import round_price, round_qty
#   from utils_round import price_round, qty_round
price_round = round_to_tick   # ближе всего к PineScript `round()`
qty_round = round_qty
