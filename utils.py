import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Union

# Устанавливаем контекст для Decimal (исключаем плавающую ошибку)
getcontext().prec = 28

def price_round(price: float, tick_size: float = 0.01) -> float:
    """Округление цены до tick size через Decimal"""
    if tick_size <= 0:
        return price
    
    price_decimal = Decimal(str(price))
    tick_decimal = Decimal(str(tick_size))
    return float((price_decimal / tick_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_decimal)

def qty_round(quantity: float, qty_step: float = 0.01) -> float:
    """Округление количества до qty step через Decimal"""
    if qty_step <= 0:
        return quantity
    
    qty_decimal = Decimal(str(quantity))
    step_decimal = Decimal(str(qty_step))
    return float((qty_decimal / step_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * step_decimal)

def calculate_fees(price: float, quantity: float, fee_rate: float = 0.00055, both_sides: bool = True) -> float:
    """Расчет комиссий = 2× taker (entry_fee + exit_fee)"""
    single_fee = price * quantity * fee_rate
    return single_fee * 2 if both_sides else single_fee

def calculate_pnl(entry_price: float, exit_price: float, quantity: float, direction: str, 
                 include_fees: bool = True, fee_rate: float = 0.00055) -> float:
    """Расчет PnL"""
    if direction.lower() == 'long':
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity
    
    if include_fees:
        fees = calculate_fees(entry_price, quantity, fee_rate) + calculate_fees(exit_price, quantity, fee_rate, False)
        return gross_pnl - fees
    
    return gross_pnl

def calculate_rr(entry_price: float, exit_price: float, stop_loss: float, direction: str) -> float:
    """Расчет Risk/Reward ratio"""
    try:
        if direction.lower() == 'long':
            risk = entry_price - stop_loss
            reward = exit_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - exit_price
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    except:
        return 0.0

def format_currency(amount: float, decimals: int = 2) -> str:
    """Форматирование валютных значений"""
    return f"${amount:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Форматирование процентных значений"""
    return f"{value:.{decimals}f}%"

def validate_price(price: Union[float, str]) -> bool:
    """Валидация цены"""
    try:
        price_float = float(price)
        return price_float > 0
    except:
        return False

def validate_quantity(quantity: Union[float, str], min_qty: float = 0.01) -> bool:
    """Валидация количества"""
    try:
        qty_float = float(quantity)
        return qty_float >= min_qty
    except:
        return False

def timestamp_to_string(timestamp, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Преобразование timestamp в строку"""
    try:
        if isinstance(timestamp, str):
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        return dt.strftime(format_str)
    except:
        return str(timestamp)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Безопасное деление"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Ограничение значения в диапазоне"""
    return max(min_val, min(max_val, value))

def format_time_duration(hours: float) -> str:
    """Форматирование длительности времени"""
    try:
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            return f"{days}d {remaining_hours}h"
    except:
        return "0m"
