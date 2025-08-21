# trail_engine.py
import math
from typing import Dict, Optional
from datetime import datetime

from config import Config
from state_manager import StateManager
from analytics import TrailingLogger

# тик-хелперы (точное согласование с Pine)
from utils_round import round_price, floor_to_tick, ceil_to_tick


class TrailEngine:
    """
    Движок Smart Trailing для управления стоп-лоссами.
    Адаптирован под Pine-логику:
      • ARM после достижения RR (с поддержкой basis: extremum | last)
      • Якорение по экстремуму (anchor) — максимум/минимум против цены входа
      • % трейлинг от entry + offset относительно anchor
      • BarTrail (lowest/highest N закрытых баров без текущего) — опционально

    Совместим с вызовами из KWINStrategy:
      - on_entry(entry, sl, direction)
      - process_trailing(position, current_price)
    """

    def __init__(self, config: Config, state_manager: StateManager, bybit_api):
        self.config = config
        self.state = state_manager
        self.api = bybit_api

        # Логгер трейлинга
        self.trail_logger = TrailingLogger()

        # Внутреннее состояние (минимально необходимое)
        self._last_anchor_long: Optional[float] = None
        self._last_anchor_short: Optional[float] = None

    # ---------- События ----------

    def on_entry(self, entry_price: float, stop_loss: float, direction: str):
        """
        Инициализация трейла при входе (как в Pine):
        - сбрасываем/устанавливаем якорь на цене входа
        - не трогаем armed-флаг: им управляет стратегия (или ARM-блок ниже)
        """
        try:
            if direction == "long":
                self._last_anchor_long = float(entry_price)
            elif direction == "short":
                self._last_anchor_short = float(entry_price)
        except Exception:
            pass

    # ---------- Публичный вход ----------

    def process_trailing(self, position: Dict, current_price: float):
        """Основная логика обработки трейлинга (вызывается внешним циклом)."""
        if not position or current_price is None:
            return
        direction = position.get("direction")
        if direction == "long":
            self._process_long_trailing(position, float(current_price))
        elif direction == "short":
            self._process_short_trailing(position, float(current_price))

    # ---------- LONG ----------

    def _process_long_trailing(self, position: Dict, current_price: float):
        try:
            entry = float(position.get("entry_price") or 0)
            current_sl = float(position.get("stop_loss") or 0)
            if entry <= 0 or current_sl <= 0:
                return

            # ===== ARM (RR) =====
            armed = bool(position.get("armed", False))
            if getattr(self.config, "use_arm_after_rr", True) and not armed:
                risk = entry - current_sl
                if risk > 0:
                    # basis: 'extremum' -> считаем по anchor, 'last' -> по текущей цене
                    basis = (getattr(self.config, "arm_rr_basis", "extremum") or "extremum").lower()
                    anchor = float(position.get("trail_anchor") or self._last_anchor_long or entry)
                    # поддерживаем рост anchor по текущей цене (экстремум внутри бара)
                    if current_price > anchor:
                        anchor = current_price
                    rr_now = (anchor - entry) / risk if basis == "extremum" else (current_price - entry) / risk
                    need = float(getattr(self.config, "arm_rr", 0.5))
                    if rr_now >= need:
                        position["armed"] = True
                        armed = True
                        if hasattr(self.state, "update_position_armed"):
                            self.state.update_position_armed(True)
                        print(f"[TrailEngine ARM] long enabled: {rr_now:.3f}R (need {need}R, basis={basis})")

            if not armed:
                return

            # ===== Anchor (экстремум) =====
            anchor = float(position.get("trail_anchor") or self._last_anchor_long or entry)
            if current_price > anchor:
                anchor = current_price
                position["trail_anchor"] = anchor
                self._last_anchor_long = anchor
                if hasattr(self.state, "set_position"):
                    self.state.set_position(position)

            new_sl = current_sl

            # ===== BarTrail (lowest N закрытых баров, без текущего) =====
            if getattr(self.config, "use_bar_trail", False):
                bt_sl = self._calculate_bar_trail_long(current_sl)
                new_sl = max(new_sl, bt_sl)

            # ===== Smart Trailing (от entry) =====
            trail_perc = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            off_perc = float(getattr(self.config, "trailing_offset_perc", 0.4)) / 100.0
            if trail_perc > 0.0:
                trail_dist = entry * trail_perc
                off_dist = entry * off_perc
                # Pine: считаем кандидат строго от anchor
                candidate = floor_to_tick(anchor - trail_dist - off_dist, self._tick_size())
                new_sl = max(new_sl, candidate)

            # ===== Обновление SL =====
            if new_sl > current_sl:
                self.trail_logger.log_trail_movement(
                    position, current_sl, new_sl, current_price,
                    "Long Trail Update",
                    lookback_value=self.config.trail_lookback if getattr(self.config, "use_bar_trail", False) else 0,
                    buffer_ticks=self.config.trail_buf_ticks
                )
                self._update_stop_loss(position, new_sl)

        except Exception as e:
            print(f"Error in long trailing: {e}")

    # ---------- SHORT ----------

    def _process_short_trailing(self, position: Dict, current_price: float):
        try:
            entry = float(position.get("entry_price") or 0)
            current_sl = float(position.get("stop_loss") or 0)
            if entry <= 0 or current_sl <= 0:
                return

            # ===== ARM (RR) =====
            armed = bool(position.get("armed", False))
            if getattr(self.config, "use_arm_after_rr", True) and not armed:
                risk = current_sl - entry
                if risk > 0:
                    basis = (getattr(self.config, "arm_rr_basis", "extremum") or "extremum").lower()
                    anchor = float(position.get("trail_anchor") or self._last_anchor_short or entry)
                    if current_price < anchor:
                        anchor = current_price
                    rr_now = (entry - anchor) / risk if basis == "extremum" else (entry - current_price) / risk
                    need = float(getattr(self.config, "arm_rr", 0.5))
                    if rr_now >= need:
                        position["armed"] = True
                        armed = True
                        if hasattr(self.state, "update_position_armed"):
                            self.state.update_position_armed(True)
                        print(f"[TrailEngine ARM] short enabled: {rr_now:.3f}R (need {need}R, basis={basis})")

            if not armed:
                return

            # ===== Anchor (экстремум) =====
            anchor = float(position.get("trail_anchor") or self._last_anchor_short or entry)
            if current_price < anchor:
                anchor = current_price
                position["trail_anchor"] = anchor
                self._last_anchor_short = anchor
                if hasattr(self.state, "set_position"):
                    self.state.set_position(position)

            new_sl = current_sl

            # ===== BarTrail (highest N закрытых баров, без текущего) =====
            if getattr(self.config, "use_bar_trail", False):
                bt_sl = self._calculate_bar_trail_short(current_sl)
                new_sl = min(new_sl, bt_sl)

            # ===== Smart Trailing (от entry) =====
            trail_perc = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            off_perc = float(getattr(self.config, "trailing_offset_perc", 0.4)) / 100.0
            if trail_perc > 0.0:
                trail_dist = entry * trail_perc
                off_dist = entry * off_perc
                candidate = ceil_to_tick(anchor + trail_dist + off_dist, self._tick_size())
                new_sl = min(new_sl, candidate)

            # ===== Обновление SL =====
            if new_sl < current_sl:
                self.trail_logger.log_trail_movement(
                    position, current_sl, new_sl, current_price,
                    "Short Trail Update",
                    lookback_value=self.config.trail_lookback if getattr(self.config, "use_bar_trail", False) else 0,
                    buffer_ticks=self.config.trail_buf_ticks
                )
                self._update_stop_loss(position, new_sl)

        except Exception as e:
            print(f"Error in short trailing: {e}")

    # ---------- BarTrail Helpers ----------

    def _calculate_bar_trail_long(self, current_sl: float) -> float:
        """BarTrail для long: lowest(low, N)[1] по ЗАКРЫТЫМ барам (без текущего)."""
        try:
            lb = int(getattr(self.config, "trail_lookback", 50))
            if lb <= 0 or not hasattr(self.api, "get_klines"):
                return current_sl
            tf = self._tf()
            kl = self.api.get_klines(self.config.symbol, tf, lb + 2) or []
            if len(kl) < lb + 1:
                return current_sl
            lows = [float(c["low"]) for c in kl[1:lb + 1]]  # [0] — текущий, исключаем
            min_low = min(lows) if lows else current_sl
            buf = float(getattr(self.config, "trail_buf_ticks", 40)) * self._tick_size()
            bar_sl = min_low - buf
            return max(bar_sl, current_sl)  # стоп не опускаем
        except Exception as e:
            print(f"Error calculating bar trail long: {e}")
            return current_sl

    def _calculate_bar_trail_short(self, current_sl: float) -> float:
        """BarTrail для short: highest(high, N)[1] по ЗАКРЫТЫМ барам (без текущего)."""
        try:
            lb = int(getattr(self.config, "trail_lookback", 50))
            if lb <= 0 or not hasattr(self.api, "get_klines"):
                return current_sl
            tf = self._tf()
            kl = self.api.get_klines(self.config.symbol, tf, lb + 2) or []
            if len(kl) < lb + 1:
                return current_sl
            highs = [float(c["high"]) for c in kl[1:lb + 1]]
            max_high = max(highs) if highs else current_sl
            buf = float(getattr(self.config, "trail_buf_ticks", 40)) * self._tick_size()
            bar_sl = max_high + buf
            return min(bar_sl, current_sl)  # стоп не поднимаем
        except Exception as e:
            print(f"Error calculating bar trail short: {e}")
            return current_sl

    # ---------- Общие ----------

    def _update_stop_loss(self, position: Dict, new_sl: float) -> bool:
        """
        Единый апдейт SL c округлением к тику и синхронизацией state.
        Если биржа не поддерживает позиционный апдейт — используем modify_order.
        """
        try:
            tick = self._tick_size()
            new_sl = round_price(float(new_sl), tick)

            if not self.api:
                return False

            # приоритет: позиционный апдейт SL
            if hasattr(self.api, "update_position_stop_loss"):
                ok = self.api.update_position_stop_loss(self.config.symbol, new_sl)
                if ok:
                    position["stop_loss"] = new_sl
                    if hasattr(self.state, "set_position"):
                        self.state.set_position(position)
                    print(f"[TRAIL] SL -> {new_sl:.4f}")
                    return True

            # запасной путь: правим через trading-stop/modify_order
            if hasattr(self.api, "modify_order"):
                _ = self.api.modify_order(
                    symbol=position.get("symbol", self.config.symbol),
                    stop_loss=new_sl,
                    trigger_by_source=getattr(self.config, "trigger_price_source", "last"),
                )
                position["stop_loss"] = new_sl
                if hasattr(self.state, "set_position"):
                    self.state.set_position(position)
                print(f"[TRAIL] SL -> {new_sl:.4f}")
                return True

            # локально (если API нет)
            position["stop_loss"] = new_sl
            if hasattr(self.state, "set_position"):
                self.state.set_position(position)
            print(f"[TRAIL-LOCAL] SL -> {new_sl:.4f}")
            return True

        except Exception as e:
            print(f"Error updating stop loss: {e}")
            return False

    def _tick_size(self) -> float:
        try:
            return float(getattr(self.config, "tick_size", 0.01) or 0.01)
        except Exception:
            return 0.01

    def _tf(self) -> str:
        try:
            return str(getattr(self.config, "interval", "15") or "15")
        except Exception:
            return "15"
