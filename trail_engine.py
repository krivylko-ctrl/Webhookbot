# trail_engine.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class _CfgView:
    # Ожидаем, что в config есть эти поля; ставим безопасные дефолты на случай отсутствия
    trailing_perc: float = 0.5          # проценты, например 0.5 -> 0.5%
    trailing_offset_perc: float = 0.4   # проценты, например 0.4 -> 0.4%
    use_arm_after_rr: bool = True
    arm_rr: float = 0.5                 # в R
    # ниже просто для совместимости, они не используются здесь:
    # (наличие атрибутов в конфиге не обязательно)

class SmartTrailEngine:
    """
    Процентный Smart Trail.
    - Никакого bar-trail: только trailing как % от цены входа (+ опциональный offset).
    - Активация (arming) по RR, если включено use_arm_after_rr.
    """
    def __init__(self, config, *_, **__):
        # берём только то, что нужно; остальное игнорируем (совместимость сигнатуры)
        # обеспечим наличие нужных полей
        self.config = _CfgView(
            trailing_perc=getattr(config, "trailing_perc", 0.5),
            trailing_offset_perc=getattr(config, "trailing_offset_perc", 0.4),
            use_arm_after_rr=getattr(config, "use_arm_after_rr", True),
            arm_rr=getattr(config, "arm_rr", 0.5),
        )
        self.reset()

    # ---- публичный API ----
    def on_entry(self, entry_price: float, stop_loss: float, direction: str):
        """Вызывай при открытии позиции: фиксируем базовые значения."""
        self.entry_price = float(entry_price)
        self.stop_loss = float(stop_loss)
        self.direction = str(direction)
        # активировать сразу или ждать RR — по настройке
        self.active = not self.config.use_arm_after_rr
        return self.stop_loss

    def update(self, current_price: float):
        """
        Обновление стопа на закрытии бара.
        Возвращает новое значение SL (или старое, если не изменился).
        """
        if self.entry_price is None or self.stop_loss is None or self.direction is None:
            return self.stop_loss

        current_price = float(current_price)

        # 1) Активация по RR (если требуется)
        if not self.active and self.config.use_arm_after_rr:
            risk = abs(self.entry_price - self.stop_loss)
            if risk > 0:
                if self.direction == "long":
                    rr = (current_price - self.entry_price) / risk
                else:
                    rr = (self.entry_price - current_price) / risk
                if rr >= float(self.config.arm_rr):
                    self.active = True

        if not self.active:
            return self.stop_loss

        # 2) Процентный трейл (проценты заданы как 0.5 -> 0.5%)
        trail_dist = self.entry_price * (float(self.config.trailing_perc) / 100.0)
        offset     = self.entry_price * (float(self.config.trailing_offset_perc) / 100.0)

        if self.direction == "long":
            candidate = current_price - trail_dist - offset
            if candidate > self.stop_loss:
                self.stop_loss = candidate
        else:  # short
            candidate = current_price + trail_dist + offset
            if candidate < self.stop_loss:
                self.stop_loss = candidate

        return self.stop_loss

    def reset(self):
        """Вызывай при закрытии позиции."""
        self.active = False
        self.entry_price = None
        self.stop_loss = None
        self.direction = None


# --- Совместимость со старым импортом ---
# В проекте используется: from trail_engine import TrailEngine
# Делает новый движок полностью совместимым без правок других файлов.
class TrailEngine(SmartTrailEngine):
    pass
