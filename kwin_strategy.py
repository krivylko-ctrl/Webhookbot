from __future__ import annotations

import math
import time
import pandas as pd
import numpy as np

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from database import Database

from utils_round import round_price, round_qty

class KWINStrategy:
    """
    Основная стратегия KWIN (с поддержкой смарт-трейла).
    """

    def __init__(self, config: Config, api, state_manager: StateManager, db: Database, **kwargs):
    # обратная совместимость с вызовами вида bybit_api=...
        if api is None and "bybit_api" in kwargs:
            api = kwargs.get("bybit_api")

        self.config = config
        self.api = api
        self.state = state_manager
        self.db = db

        # тик для округлений
        self.tick_size = getattr(config, "tick_size", 0.1)

        # трейл-движок (смарт)
        self.trail_engine = TrailEngine(config)

        # для логики
        self.symbol = getattr(config, "symbol", "ETHUSDT")
        self.interval = str(getattr(config, "interval", "15"))
        self.intrabar_tf = str(getattr(config, "intrabar_tf", "1"))

    # ============ ВСПОМОГАТЕЛЬНОЕ ============

    def _get_current_price(self) -> Optional[float]:
        """Текущая цена через API"""
        try:
            t = self.api.get_ticker(self.symbol)
            if not t:
                return None
            return float(t.get("last_price") or t.get("price"))
        except Exception:
            return None

    def _get_bar_extremes_for_trailing(self, fallback_price: float) -> (float, float):
        """
        Берём экстремумы бара для трейлинга.
        Если нет данных — fallback_price как high=low.
        """
        try:
            kl = self.api.get_klines(self.symbol, self.intrabar_tf, 1)
            if kl and isinstance(kl, list) and len(kl) > 0:
                b = kl[-1]
                return float(b["high"]), float(b["low"])
        except Exception:
            pass
        return fallback_price, fallback_price

    # ============ CORE-API (вызывается извне) ============

    def on_bar_close_15m(self, bar: Dict):
        """
        Основная логика входа по закрытию 15m бара.
        Здесь остаётся твоя ICT/фильтрация сигналов.
        """
        # Заглушка: просто логируем бар
        self.db.log_event("15m bar", f"{bar}")

        # Пример: проверяем наличие позиции
        pos = self.state.get_current_position()
        if not pos:
            # если позиции нет — можно генерировать вход
            # >>> сюда вставляешь условия (SMT/FVG/OB и т.д.)
            pass
        else:
            # если позиция есть — обновляем trailing
            self.process_trailing()

    # ================== TRAILING ==================

    def process_trailing(self):
        """
        Смарт-трейлинг: обновляет стоп-лосс, если условия выполнены.
        Этот метод нужно дёргать часто (в потоке _bg_bot_loop).
        """
        pos = self.state.get_current_position()
        if not pos or float(pos.get("size", 0)) == 0:
            return

        side = pos.get("side")
        entry = float(pos.get("entry_price", 0))
        sl = float(pos.get("stop_loss", 0) or 0.0)

        current_price = self._get_current_price()
        if not current_price:
            return

        # Win/loss в R-множителях
        rr = (current_price - entry) / (entry - sl) if side == "long" else (entry - current_price) / (sl - entry)

        # Проверка ARM (включаем трейл после достижения arm_rr)
        if self.config.use_arm_after_rr and rr < self.config.arm_rr:
            return

        # Берём экстремумы бара (для динамики трейла)
        hi, lo = self._get_bar_extremes_for_trailing(current_price)

        new_sl = self.trail_engine.calc_new_stop(
            side=side,
            entry_price=entry,
            current_price=current_price,
            prev_sl=sl,
            bar_high=hi,
            bar_low=lo,
        )

        if new_sl and abs(new_sl - sl) >= self.tick_size:
            # обновляем стоп в базе + бирже
            self._update_stop(pos, new_sl)

    # ================== STOP MANAGEMENT ==================

    def _update_stop(self, pos: Dict, new_sl: float):
        """
        Обновление стоп-ордера: и в стейте, и на бирже.
        """
        side = pos.get("side")
        size = float(pos.get("size", 0))

        if size <= 0:
            return

        try:
            # Сохраняем в state/db
            self.state.update_stop_loss(new_sl)
            self.db.log_event("STOP UPDATE", f"side={side} new_sl={new_sl}")

            # Отправляем в Bybit
            if self.api:
                self.api.update_stop_order(
                    symbol=self.config.symbol,
                    side=side,
                    size=size,
                    stop_price=new_sl
                )
        except Exception as e:
            self.db.log_event("STOP ERROR", str(e))

    # ================== PRICE HELPERS ==================

    def _get_current_price(self) -> Optional[float]:
        """
        Унифицированный способ получить последнюю цену.
        """
        try:
            if hasattr(self.api, "get_ticker"):
                ticker = self.api.get_ticker(self.config.symbol)
                if ticker and "last_price" in ticker:
                    return float(ticker["last_price"])
        except Exception:
            return None
        return None

    def _get_bar_extremes_for_trailing(self, current_price: float) -> Tuple[float, float]:
        """
        Вспомогательный метод: берём high/low для текущего трейла.
        Если API не даёт данные — fallback на ±0.5% от цены.
        """
        try:
            kl = self.api.get_klines(self.config.symbol, "15", 1)
            if kl:
                bar = kl[-1]
                return float(bar["high"]), float(bar["low"])
        except Exception:
            pass

        # fallback
        return current_price * 1.005, current_price * 0.995    # ================== STOP MANAGEMENT ==================

    def _update_stop(self, pos: Dict, new_sl: float):
        """
        Обновление стоп-ордера: и в стейте, и на бирже.
        """
        side = pos.get("side")
        size = float(pos.get("size", 0))

        if size <= 0:
            return

        try:
            # Сохраняем в state/db
            self.state.update_stop_loss(new_sl)
            self.db.log_event("STOP UPDATE", f"side={side} new_sl={new_sl}")

            # Отправляем в Bybit
            if self.api:
                self.api.update_stop_order(
                    symbol=self.config.symbol,
                    side=side,
                    size=size,
                    stop_price=new_sl
                )
        except Exception as e:
            self.db.log_event("STOP ERROR", str(e))

    # ================== PRICE HELPERS ==================

    def _get_current_price(self) -> Optional[float]:
        """
        Унифицированный способ получить последнюю цену.
        """
        try:
            if hasattr(self.api, "get_ticker"):
                ticker = self.api.get_ticker(self.config.symbol)
                if ticker and "last_price" in ticker:
                    return float(ticker["last_price"])
        except Exception:
            return None
        return None

    def _get_bar_extremes_for_trailing(self, current_price: float) -> Tuple[float, float]:
        """
        Вспомогательный метод: берём high/low для текущего трейла.
        Если API не даёт данные — fallback на ±0.5% от цены.
        """
        try:
            kl = self.api.get_klines(self.config.symbol, "15", 1)
            if kl:
                bar = kl[-1]
                return float(bar["high"]), float(bar["low"])
        except Exception:
            pass

        # fallback
        return current_price * 1.005, current_price * 0.995
