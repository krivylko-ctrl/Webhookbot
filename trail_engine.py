# trail_engine.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class _CfgView:
    """
    Слой-конфиг с безопасными дефолтами.
    Ожидаем, что реальные поля приедут из config (но не требуем обязательно).
    """
    trailing_perc: float = 0.5            # 0.5 -> 0.5% от цены входа
    trailing_offset_perc: float = 0.4     # 0.4 -> 0.4% от цены входа
    use_arm_after_rr: bool = True         # включать ARM по RR
    arm_rr: float = 0.5                   # целевой RR для ARM
    arm_rr_basis: str = "extremum"        # "extremum" | "last" — база для RR при ARM
    epsilon: float = 1e-9                 # защита от дрожания сравнения


class SmartTrailEngine:
    """
    Процентный Smart Trail:
      • дистанция трейла = % от цены входа (+ опциональный offset),
      • активация (ARM) по RR (если включено),
      • без «bar-trail» — движение стопа зависит от поданной цены/экстремума.
    Совместим с прежним импортом (см. класс-обёртку TrailEngine внизу).
    """

    def __init__(self, config, *_, **__):
        # Берём только нужные поля из внешнего config. Остальное игнорируем.
        self.config = _CfgView(
            trailing_perc=float(getattr(config, "trailing_perc", 0.5)),
            trailing_offset_perc=float(getattr(config, "trailing_offset_perc", 0.4)),
            use_arm_after_rr=bool(getattr(config, "use_arm_after_rr", True)),
            arm_rr=float(getattr(config, "arm_rr", 0.5)),
            arm_rr_basis=str(getattr(config, "arm_rr_basis", "extremum")).lower(),
            epsilon=float(getattr(config, "epsilon", 1e-9)),
        )
        self.reset()

    # ---------------- публичный API ----------------

    def on_entry(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """
        Вызывается при открытии позиции.
        """
        self.entry_price = float(entry_price)
        self.stop_loss = float(stop_loss)
        self.direction = "long" if str(direction).lower().startswith("l") else "short"
        # активировать сразу или ждать ARM по RR — по настройке
        self.active = not self.config.use_arm_after_rr
        self.was_armed = not self.config.use_arm_after_rr
        return self.stop_loss

    def update(self, current_price: float) -> float:
        """
        Обновление на закрытии бара/тика одной ценой.
        Возвращает текущее значение SL (возможно обновлённое).
        """
        p = float(current_price)
        return self._update_internal(current_price=p, hi=p, lo=p)

    def update_intrabar(self, high: float, low: float, last: float) -> float:
        """
        Интрабар-обновление: передай high/low/last — движок сам выберет базу
        для ARM (extremum/last) и новую точку для кандидата стопа.
        """
        high = float(high)
        low  = float(low)
        last = float(last)

        if self.config.arm_rr_basis == "extremum":
            rr_px = (high if self.direction == "long" else low)
        else:
            rr_px = last

        return self._update_internal(current_price=last, hi=high, lo=low, rr_basis_price=rr_px)

    # ---------------- внутренняя логика ----------------

    def _update_internal(
        self,
        current_price: float,
        hi: float,
        lo: float,
        rr_basis_price: float | None = None
    ) -> float:
        """
        Единая точка обновления: ARM по RR + пересчёт кандидата SL.
        """
        if self.entry_price is None or self.stop_loss is None or self.direction is None:
            return self.stop_loss

        # 1) ARM по RR (если требуется)
        if not self.active and self.config.use_arm_after_rr:
            base = rr_basis_price if rr_basis_price is not None else current_price
            risk = abs(self.entry_price - self.stop_loss)
            if risk > self.config.epsilon:
                if self.direction == "long":
                    rr = (float(base) - self.entry_price) / risk
                else:
                    rr = (self.entry_price - float(base)) / risk
                if rr + self.config.epsilon >= float(self.config.arm_rr):
                    self.active = True
                    self.was_armed = True

        if not self.active:
            return self.stop_loss

        # 2) Процентный трейл от цены входа
        trail_dist = self.entry_price * (self.config.trailing_perc / 100.0)
        offset     = self.entry_price * (self.config.trailing_offset_perc / 100.0)

        if self.direction == "long":
            candidate = float(current_price) - trail_dist - offset
            # Улучшаем только в сторону профита (монотонность)
            if candidate > self.stop_loss + self.config.epsilon:
                # важно: разрешаем проходить выше entry (брейк-ивен/лок прибыли)
                self.stop_loss = candidate
        else:  # short
            candidate = float(current_price) + trail_dist + offset
            if candidate < self.stop_loss - self.config.epsilon:
                self.stop_loss = candidate

        return self.stop_loss

    def reset(self):
        """Сброс при закрытии позиции."""
        self.active: bool = False
        self.was_armed: bool = False
        self.entry_price: float | None = None
        self.stop_loss: float | None = None
        self.direction: str | None = None

    # --------- удобные свойства/флаги для логов/отладки ---------

    @property
    def is_active(self) -> bool:
        return bool(self.active)

    @property
    def armed_once(self) -> bool:
        """Был ли хотя бы раз заармлен по RR."""
        return bool(self.was_armed)


# --- Совместимость со старым импортом ---
# В проекте используется: from trail_engine import TrailEngine
class TrailEngine(SmartTrailEngine):
    pass
