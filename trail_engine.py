# trail_engine.py
from datetime import datetime

class SmartTrailEngine:
    def __init__(self, config):
        self.config = config
        self.active = False
        self.entry_price = None
        self.stop_loss = None
        self.direction = None

    def on_entry(self, entry_price, stop_loss, direction):
        """Инициализация трейла при входе в сделку"""
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.direction = direction
        self.active = False  # активируется только после достижения armRR
        return self.stop_loss

    def update(self, current_price):
        """Обновление стопа на закрытии бара"""
        if not self.entry_price or not self.stop_loss:
            return self.stop_loss

        # === Проверка активации трейла (armRR) ===
        rr = None
        if self.direction == "long":
            risk = self.entry_price - self.stop_loss
            rr = (current_price - self.entry_price) / risk if risk > 0 else 0
        elif self.direction == "short":
            risk = self.stop_loss - self.entry_price
            rr = (self.entry_price - current_price) / risk if risk > 0 else 0

        if rr is not None and rr >= self.config.arm_rr:
            self.active = True

        if not self.active:
            return self.stop_loss

        # === Основная логика процентного трейла ===
        trail_dist = self.entry_price * self.config.trailing_perc
        offset = self.entry_price * self.config.trailing_offset_perc

        if self.direction == "long":
            candidate_sl = current_price - trail_dist - offset
            if candidate_sl > self.stop_loss:
                self.stop_loss = candidate_sl

        elif self.direction == "short":
            candidate_sl = current_price + trail_dist + offset
            if candidate_sl < self.stop_loss:
                self.stop_loss = candidate_sl

        return self.stop_loss

    def reset(self):
        """Сброс после выхода из сделки"""
        self.active = False
        self.entry_price = None
        self.stop_loss = None
        self.direction = None
