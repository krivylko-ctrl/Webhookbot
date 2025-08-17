import math
from typing import Dict, Optional, List
from datetime import datetime, timezone

from config import Config
from state_manager import StateManager
from utils import price_round
from analytics import TrailingLogger


class TrailEngine:
    """Движок Smart Trailing для управления стоп-лоссами"""

    def __init__(self, config: Config, state_manager: StateManager, bybit_api):
        self.config = config
        self.state = state_manager
        self.api = bybit_api

        # Логгер трейлинга (совместим с аналитической страницей)
        self.trail_logger = TrailingLogger()

        # Служебные поля
        self.last_trail_price_long: Optional[float] = None
        self.last_trail_price_short: Optional[float] = None

    # ============== Публичная точка входа ==============

    def process_trailing(self, position: Dict, current_price: float):
        """Основная логика обработки трейлинга"""
        if not position or not current_price:
            return

        direction = position.get("direction")
        if direction == "long":
            self._process_long_trailing(position, current_price)
        elif direction == "short":
            self._process_short_trailing(position, current_price)

    # ============== ЛОНГ ==============

    def _process_long_trailing(self, position: Dict, current_price: float):
        """Обработка трейлинга для лонг позиции"""
        try:
            entry_price = float(position.get("entry_price") or 0)
            current_sl = float(position.get("stop_loss") or 0)
            armed = bool(position.get("armed", False))
            if entry_price <= 0 or current_sl <= 0:
                return

            # 1) Арм по RR (как в стратегии)
            if self.config.use_arm_after_rr and not armed:
                moved = current_price - entry_price
                need = (entry_price - current_sl) * float(getattr(self.config, "arm_rr", 0.5))
                if moved >= need:
                    position["armed"] = True
                    armed = True
                    if hasattr(self.state, "update_position_armed"):
                        self.state.update_position_armed(True)
                    print(f"Long position armed at RR: {moved / max(entry_price - current_sl, 1e-9):.2f}")

            if not armed:
                return

            new_sl = current_sl

            # 2) Баровый трейл (по закрытым барам)
            if self.config.use_bar_trail:
                new_sl = self._calculate_bar_trail_long(current_sl)

            # 3) Процент+offset трейл (термины Pine: trail_points = entry * pct, trail_offset = entry * offset)
            per = float(getattr(self.config, "trailing_perc", 0.0)) / 100.0
            off = float(getattr(self.config, "trailing_offset_perc", 0.0)) / 100.0
            if per > 0.0:
                cand = self._calc_pct_offset_long(entry_price=entry_price,
                                                  current_price=current_price,
                                                  current_sl=current_sl,
                                                  pct=per, offset=off)
                new_sl = max(new_sl, cand)

            # 4) Применяем только улучшение SL
            if new_sl > current_sl:
                self._log_trail(position, current_sl, new_sl, current_price, trigger="Long Trail Update")
                self._update_stop_loss(position, new_sl)

        except Exception as e:
            print(f"Error in long trailing: {e}")

    # ============== ШОРТ ==============

    def _process_short_trailing(self, position: Dict, current_price: float):
        """Обработка трейлинга для шорт позиции"""
        try:
            entry_price = float(position.get("entry_price") or 0)
            current_sl = float(position.get("stop_loss") or 0)
            armed = bool(position.get("armed", False))
            if entry_price <= 0 or current_sl <= 0:
                return

            # 1) Арм по RR
            if self.config.use_arm_after_rr and not armed:
                moved = entry_price - current_price
                need = (current_sl - entry_price) * float(getattr(self.config, "arm_rr", 0.5))
                if moved >= need:
                    position["armed"] = True
                    armed = True
                    if hasattr(self.state, "update_position_armed"):
                        self.state.update_position_armed(True)
                    print(f"Short position armed at RR: {moved / max(current_sl - entry_price, 1e-9):.2f}")

            if not armed:
                return

            new_sl = current_sl

            # 2) Баровый трейл
            if self.config.use_bar_trail:
                new_sl = self._calculate_bar_trail_short(current_sl)

            # 3) Процент+offset трейл
            per = float(getattr(self.config, "trailing_perc", 0.0)) / 100.0
            off = float(getattr(self.config, "trailing_offset_perc", 0.0)) / 100.0
            if per > 0.0:
                cand = self._calc_pct_offset_short(entry_price=entry_price,
                                                   current_price=current_price,
                                                   current_sl=current_sl,
                                                   pct=per, offset=off)
                new_sl = min(new_sl, cand)

            # 4) Применяем только улучшение SL
            if new_sl < current_sl:
                self._log_trail(position, current_sl, new_sl, current_price, trigger="Short Trail Update")
                self._update_stop_loss(position, new_sl)

        except Exception as e:
            print(f"Error in short trailing: {e}")

    # ============== Баровый трейл (15m, закрытые бары) ==============

    def _get_closed_15m_window(self, lookback: int) -> List[Dict]:
        """
        Возвращает список последних ЗАКРЫТЫХ 15m свечей длиной lookback.
        Гарантированно исключаем текущий незакрытый бар.
        """
        if not self.api or lookback <= 0:
            return []

        # Берём с запасом: lookback + 2, отсортируем по timestamp убыванию (newest-first)
        kl = self.api.get_klines(self.config.symbol, "15", lookback + 2) or []
        if not kl:
            return []

        def _ts(c):
            return c.get("timestamp") or c.get("start") or c.get("open_time") or c.get("t") or 0

        # Нормализация и сортировка newest-first
        norm = []
        for c in kl:
            try:
                ts = int(_ts(c))
                if ts and ts < 10_000_000_000:  # сек -> мс
                    ts *= 1000
                norm.append({
                    "timestamp": ts,
                    "open": float(c.get("open")),
                    "high": float(c.get("high")),
                    "low": float(c.get("low")),
                    "close": float(c.get("close")),
                })
            except Exception:
                continue

        norm.sort(key=lambda x: x["timestamp"], reverse=True)
        if len(norm) < lookback + 1:
            return []

        # Исключаем текущий бар [0], берём закрытые [1 : lookback+1]
        return norm[1:lookback + 1]

    def _calculate_bar_trail_long(self, current_sl: float) -> float:
        """BarTrail по закрытым барам: lowest(low, N)[1] − buffer"""
        try:
            lookback = int(getattr(self.config, "trail_lookback", 50) or 50)
            if lookback <= 0:
                return current_sl

            closed = self._get_closed_15m_window(lookback)
            if not closed:
                return current_sl

            lb_low = min(float(c["low"]) for c in closed)
            tick = self._tick_size()
            buf = float(getattr(self.config, "trail_buf_ticks", 0)) * tick
            bar_trail_sl = lb_low - buf
            return max(bar_trail_sl, current_sl)
        except Exception as e:
            print(f"Error calculating bar trail long: {e}")
            return current_sl

    def _calculate_bar_trail_short(self, current_sl: float) -> float:
        """BarTrail по закрытым барам: highest(high, N)[1] + buffer"""
        try:
            lookback = int(getattr(self.config, "trail_lookback", 50) or 50)
            if lookback <= 0:
                return current_sl

            closed = self._get_closed_15m_window(lookback)
            if not closed:
                return current_sl

            lb_high = max(float(c["high"]) for c in closed)
            tick = self._tick_size()
            buf = float(getattr(self.config, "trail_buf_ticks", 0)) * tick
            bar_trail_sl = lb_high + buf
            return min(bar_trail_sl, current_sl)
        except Exception as e:
            print(f"Error calculating bar trail short: {e}")
            return current_sl

    # ============== Процент + offset (в терминах Pine) ==============

    def _calc_pct_offset_long(self, entry_price: float, current_price: float,
                              current_sl: float, pct: float, offset: float) -> float:
        """
        Эмуляция strategy.exit(trail_points=entry*pct, trail_offset=entry*offset) для лонга.
        Используем «пик» со входа из state, если он есть.
        """
        try:
            pos = self.state.get_current_position() if hasattr(self.state, "get_current_position") else None
            peak = float(pos.get("peak", entry_price)) if pos else entry_price
            trail_points = entry_price * pct
            offset_pts = entry_price * offset
            cand = peak - trail_points
            # не ближе к цене, чем offset
            cand = min(cand, current_price - offset_pts)
            return max(cand, current_sl)
        except Exception:
            return current_sl

    def _calc_pct_offset_short(self, entry_price: float, current_price: float,
                               current_sl: float, pct: float, offset: float) -> float:
        """Эмуляция для шорта (используем «дно» со входа)."""
        try:
            pos = self.state.get_current_position() if hasattr(self.state, "get_current_position") else None
            trough = float(pos.get("trough", entry_price)) if pos else entry_price
            trail_points = entry_price * pct
            offset_pts = entry_price * offset
            cand = trough + trail_points
            cand = max(cand, current_price + offset_pts)
            return min(cand, current_sl)
        except Exception:
            return current_sl

    # ============== Обновление стоп-лосса на бирже/эмуляция ==============

    def _update_stop_loss(self, position: Dict, new_sl: float):
        """Правильный апдейт стопа: округляем по тик-сайзу и пытаемся обновить позицию/ордер."""
        try:
            symbol = position.get("symbol", self.config.symbol)
            tick = self._tick_size()
            new_sl_rounded = price_round(float(new_sl), tick)

            result = None

            # 1) Универсальный маршрут — если есть modify_order у API-обвязки
            if hasattr(self.api, "modify_order"):
                try:
                    result = self.api.modify_order(symbol=symbol, stop_loss=new_sl_rounded)
                except TypeError:
                    # альтернативные сигнатуры
                    result = self.api.modify_order(symbol=symbol, sl=new_sl_rounded)

            # 2) Деривативный маршрут
            if not result and hasattr(self.api, "update_position_stop_loss"):
                result = self.api.update_position_stop_loss(symbol, new_sl_rounded)

            # 3) Фолбэк — условный стоп-ордер reduce-only
            if not result and hasattr(self.api, "place_order"):
                direction = position.get("direction")
                size = float(position.get("size") or 0)
                side = "sell" if direction == "long" else "buy"
                try:
                    result = self.api.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="stop",
                        qty=size,
                        price=new_sl_rounded,
                        reduce_only=True
                    )
                except TypeError:
                    # на случай другой сигнатуры
                    result = self.api.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="stop",
                        qty=size,
                        stop_loss=new_sl_rounded
                    )

            if result:
                # Обновляем состояние
                old_sl = float(position.get("stop_loss") or 0)
                position["stop_loss"] = new_sl_rounded
                if hasattr(self.state, "update_position_stop_loss"):
                    self.state.update_position_stop_loss(new_sl_rounded)
                else:
                    self.state.set_position(position)

                print(f"[TRAIL] Stop loss updated to: {new_sl_rounded:.4f}")
                self._log_trailing_update(position, old_sl, new_sl_rounded)

        except Exception as e:
            print(f"Error updating stop loss: {e}")

    # ============== Логирование ==============

    def _log_trail(self, position: Dict, old_sl: float, new_sl: float, current_price: float, trigger: str):
        """Лог движения трейла для страницы аналитики (единообразные поля)."""
        try:
            direction = position.get("direction")
            entry_price = float(position.get("entry_price") or 0)
            qty = float(position.get("size") or 0)
            unrealized = 0.0
            if direction == "long":
                unrealized = (current_price - entry_price) * qty
            elif direction == "short":
                unrealized = (entry_price - current_price) * qty

            trail_distance = abs(new_sl - old_sl)
            self.trail_logger.log_trail_movement(
                position=position,
                old_sl=float(old_sl),
                new_sl=float(new_sl),
                current_price=float(current_price),
                trigger_type=trigger,
                lookback_value=int(getattr(self.config, "trail_lookback", 0) or 0),
                buffer_ticks=int(getattr(self.config, "trail_buf_ticks", 0) or 0),
                trail_distance=float(trail_distance),
                unrealized_pnl=float(unrealized),
                arm_status=bool(position.get("armed", False)),
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            print(f"Error logging trail movement: {e}")

    def _log_trailing_update(self, position: Dict, old_sl: float, new_sl: float):
        """Короткий вспомогательный лог для отладки (stdout)"""
        try:
            data = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": position.get("symbol"),
                "direction": position.get("direction"),
                "old_sl": float(old_sl),
                "new_sl": float(new_sl),
                "action": "trailing_update_applied"
            }
            print(f"Trailing log: {data}")
        except Exception as e:
            print(f"Error logging trailing update: {e}")

    # ============== Вспомогательное ==============

    def _tick_size(self) -> float:
        """Единый источник тик-сайза: config → state → дефолт."""
        tick = getattr(self.config, "tick_size", None)
        if tick and tick > 0:
            return float(tick)
        tick = getattr(self.state, "tick_size", None)
        if tick and tick > 0:
            return float(tick)
        return 0.01
