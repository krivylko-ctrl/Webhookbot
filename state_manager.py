# state_manager.py — совместимая версия для KWINStrategy / TrailEngine
from __future__ import annotations

from typing import Dict, Optional, Any
from datetime import datetime, timezone
from threading import RLock

from database import Database


class StateManager:
    """
    Потокобезопасный стор состояния бота.
    Хранит:
      • equity
      • текущую позицию (direction, quantity/size, entry_price, stop_loss, take_profit,
        armed, trail_anchor, entry_time_ts, status)
      • статус бота
      • last_close_time_ms — метка времени (UTC, мс) последнего закрытия позиции (для кулдауна)
    Состояние периодически/при изменениях сохраняется в БД через Database.save_bot_state().
    """

    def __init__(self, db: Database, taker_fee_rate: float = 0.00055):
        self.db = db
        self._lock = RLock()

        self._equity: float = 100.0
        self._bot_status: str = "stopped"
        self._current_position: Optional[Dict[str, Any]] = None
        self._last_close_time_ms: Optional[int] = None  # для кулдауна входов

        # комиссия биржи (используем такую же, как и database.update_trade_exit)
        self._fee_rate: float = float(taker_fee_rate)

        self._load_state()

    # ----------------- Persist -----------------

    def _load_state(self) -> None:
        """Загрузка сохранённого состояния из БД."""
        try:
            state = self.db.get_bot_state() or {}
            with self._lock:
                self._current_position = state.get("position") or None
                self._equity = float(state.get("equity", 100.0))
                self._bot_status = str(state.get("status", "stopped"))
                lct = state.get("last_close_time_ms") or state.get("last_close_time")
                if lct is not None:
                    try:
                        self._last_close_time_ms = int(lct)
                    except Exception:
                        self._last_close_time_ms = None
        except Exception as e:
            print(f"[StateManager] Error loading state: {e}")

    def _save_state(self) -> None:
        """Сохранение текущего состояния в БД."""
        try:
            with self._lock:
                payload = {
                    "position": self._current_position,
                    "equity": float(self._equity),
                    "status": self._bot_status,
                    "last_close_time_ms": int(self._last_close_time_ms) if self._last_close_time_ms else None,
                    "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
                }
            self.db.save_bot_state(payload)
        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")

    # ----------------- Equity -----------------

    def get_equity(self) -> float:
        with self._lock:
            return float(self._equity)

    def set_equity(self, equity: float) -> None:
        with self._lock:
            self._equity = float(equity)
        self._save_state()

    # ----------------- Bot status -----------------

    def get_bot_status(self) -> str:
        with self._lock:
            return self._bot_status

    def set_bot_status(self, status: str) -> None:
        with self._lock:
            self._bot_status = str(status)
        self._save_state()

    # ----------------- Position (CRUD) -----------------

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._current_position is None else dict(self._current_position)

    def set_position(self, position: Optional[Dict[str, Any]]) -> None:
        """Полная установка позиции. None — закрыть локально (без сделки)."""
        with self._lock:
            if position is None:
                self._current_position = None
            else:
                pos = dict(position)

                # Дефолты/нормализация
                pos.setdefault("status", "open")
                pos.setdefault("armed", False)
                pos.setdefault("entry_time_ts", int(datetime.utcnow().timestamp() * 1000))
                if pos.get("trail_anchor") is None:
                    pos["trail_anchor"] = pos.get("entry_price")

                # Унификация объёма: size <-> quantity
                if "size" in pos and "quantity" not in pos:
                    try:
                        pos["quantity"] = float(pos["size"])
                    except Exception:
                        pos["quantity"] = pos["size"]
                if "quantity" in pos and "size" not in pos:
                    try:
                        pos["size"] = float(pos["quantity"])
                    except Exception:
                        pos["size"] = pos["quantity"]

                # Числовые поля (на всякий)
                for k in ("entry_price", "stop_loss", "take_profit"):
                    if k in pos and pos[k] is not None:
                        try:
                            pos[k] = float(pos[k])
                        except Exception:
                            pass

                self._current_position = pos
        self._save_state()

    def clear_position(self) -> None:
        with self._lock:
            self._current_position = None
        self._save_state()

    # Удобные апдейтеры — используются TrailEngine/стратегией
    def update_position_stop_loss(self, new_sl: float) -> None:
        with self._lock:
            if self._current_position:
                try:
                    self._current_position["stop_loss"] = float(new_sl)
                except Exception:
                    self._current_position["stop_loss"] = new_sl
        self._save_state()

    def update_position_armed(self, armed: bool) -> None:
        with self._lock:
            if self._current_position:
                self._current_position["armed"] = bool(armed)
        self._save_state()

    def update_trail_anchor(self, anchor: float) -> None:
        with self._lock:
            if self._current_position:
                try:
                    self._current_position["trail_anchor"] = float(anchor)
                except Exception:
                    self._current_position["trail_anchor"] = anchor
        self._save_state()

    # ----------------- Close position -----------------

    def close_position(self, exit_price: float, exit_reason: str = "manual") -> None:
        """
        Закрыть текущую позицию и синхронизировать это с БД.
        Net PnL считаем локально (как в Database.update_trade_exit) и прибавляем к equity.
        Затем обновляем запись сделки в БД (она посчитает то же самое).
        После закрытия — проставляем last_close_time_ms (UTC now, мс) и чистим локальную позицию.
        """
        pos = self.get_current_position()
        if not pos:
            return

        # --- 1) Локальный net PnL (без ожидания чтения из БД)
        try:
            entry = float(pos.get("entry_price") or 0.0)
            qty   = float(pos.get("size") or pos.get("quantity") or 0.0)
            side  = (pos.get("direction") or "").lower()
            exit_px = float(exit_price if exit_price is not None else entry)

            if qty > 0 and entry > 0:
                gross = (exit_px - entry) * qty if side == "long" else (entry - exit_px) * qty
                fee_in  = entry * qty * self._fee_rate
                fee_out = exit_px * qty * self._fee_rate
                net_pnl = gross - (fee_in + fee_out)
                with self._lock:
                    self._equity = float(self._equity) + float(net_pnl)
        except Exception as e:
            print(f"[StateManager] Error computing local PnL: {e}")

        # --- 2) Обновляем сделку в БД (БД тоже посчитает net PnL, что ок)
        try:
            self.db.update_trade_exit(
                {
                    "exit_price": float(exit_price),
                    "exit_time": datetime.utcnow(),
                    "exit_reason": exit_reason,
                    "status": "closed",
                },
                fee_rate=self._fee_rate,
            )
        except Exception as e:
            print(f"[StateManager] Error closing trade in DB: {e}")

        # --- 3) Обновляем last_close_time_ms (UTC now → ms)
        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        except Exception:
            now_ms = int(datetime.utcnow().timestamp() * 1000)
        self.set_last_close_time(now_ms)

        # --- 4) Локально чистим позицию
        self.clear_position()

    # ----------------- Cooldown helpers -----------------

    def get_last_close_time(self) -> Optional[int]:
        """Возвращает unix ms последнего закрытия позиции (или None)."""
        with self._lock:
            return None if self._last_close_time_ms is None else int(self._last_close_time_ms)

    def set_last_close_time(self, ts_ms: Optional[int]) -> None:
        """Явно проставить unix ms последнего закрытия позиции и сохранить состояние."""
        with self._lock:
            self._last_close_time_ms = None if ts_ms is None else int(ts_ms)
        self._save_state()

    # ----------------- Helpers -----------------

    def is_position_open(self) -> bool:
        with self._lock:
            return self._current_position is not None

    def get_position_direction(self) -> Optional[str]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("direction")

    def get_position_entry_price(self) -> Optional[float]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("entry_price")

    def get_position_stop_loss(self) -> Optional[float]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("stop_loss")

    def get_position_take_profit(self) -> Optional[float]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("take_profit")

    def get_position_quantity(self) -> Optional[float]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("quantity")

    # алиас для совместимости с кодом, где ожидали size
    def get_position_size(self) -> Optional[float]:
        with self._lock:
            return None if not self._current_position else self._current_position.get("size")

    def is_position_armed(self) -> bool:
        with self._lock:
            return False if not self._current_position else bool(self._current_position.get("armed", False))

    def get_state_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "equity": float(self._equity),
                "bot_status": self._bot_status,
                "position_open": self._current_position is not None,
                "position": None if self._current_position is None else dict(self._current_position),
                "last_close_time_ms": int(self._last_close_time_ms) if self._last_close_time_ms else None,
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),
            }
