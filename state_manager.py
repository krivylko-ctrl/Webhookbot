# state_manager.py — совместимая версия для KWINStrategy / TrailEngine
from __future__ import annotations

from typing import Dict, Optional, Any
from datetime import datetime, timezone
from threading import RLock

try:
    # тип только для аннотаций; сам модуль может отсутствовать при раннем импорте
    from database import Database  # type: ignore
except Exception:
    Database = None  # type: ignore


class StateManager:
    """
    Потокобезопасный стор состояния бота.
    Хранит:
      • equity
      • текущую позицию (direction, size/quantity, entry_price, stop_loss, take_profit,
        armed, trail_anchor, entry_time_ts, status)
      • статус бота
      • last_close_time_ms — UTC (мс) последнего закрытия
    Состояние по возможности сохраняется в БД (merge с существующим bot_state).
    """

    def __init__(
        self,
        db: Optional["Database"] = None,
        taker_fee_rate: float = 0.00055,
        initial_equity: float = 300.0,
    ):
        self.db: Optional["Database"] = db
        self._lock = RLock()

        self._equity: float = float(initial_equity)
        self._bot_status: str = "stopped"
        self._current_position: Optional[Dict[str, Any]] = None
        self._last_close_time_ms: Optional[int] = None  # для кулдауна входов

        # комиссия биржи (должна совпадать с database.update_trade_exit / стратегией)
        self._fee_rate: float = float(taker_fee_rate)

        self._load_state()

    # ----------------- DB attach/detach -----------------

    def attach_db(self, db: "Database") -> None:
        """Поздняя привязка БД (совместимо с текущим вебсокет-раннером)."""
        self.db = db
        # после привязки попробуем дозагрузить прошлое состояние
        self._load_state()

    # ----------------- Persist -----------------

    def _merge_and_save_state(self, payload: Dict[str, Any]) -> None:
        """Сливаем с текущим bot_state и сохраняем (не перетираем поля раннера)."""
        if not self.db:
            return
        try:
            existing = self.db.get_bot_state() or {}
        except Exception:
            existing = {}
        merged = dict(existing)
        merged.update(payload)
        try:
            self.db.save_bot_state(merged)
        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")

    def _load_state(self) -> None:
        """Загрузка сохранённого состояния из БД (если доступна)."""
        if not self.db:
            return
        try:
            state = self.db.get_bot_state() or {}
            with self._lock:
                pos = state.get("position") or None
                self._current_position = dict(pos) if isinstance(pos, dict) else None
                self._equity = float(state.get("equity", self._equity))
                self._bot_status = str(state.get("status", self._bot_status))
                lct = state.get("last_close_time_ms") or state.get("last_close_time")
                if lct is not None:
                    try:
                        self._last_close_time_ms = int(lct)
                    except Exception:
                        self._last_close_time_ms = None
        except Exception as e:
            print(f"[StateManager] Error loading state: {e}")

    def _save_state(self) -> None:
        """Сохранение текущего состояния (merge)."""
        payload = {
            "position": self.get_current_position(),
            "equity": float(self.get_equity()),
            "status": self.get_bot_status(),
            "last_close_time_ms": self.get_last_close_time(),
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }
        self._merge_and_save_state(payload)

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
                # таймстемп — всегда в мс
                ets = pos.get("entry_time_ts")
                if ets is None:
                    ets = int(datetime.utcnow().timestamp() * 1000)
                try:
                    ets = int(ets)
                except Exception:
                    ets = int(datetime.utcnow().timestamp() * 1000)
                if ets < 1_000_000_000_000:  # сек → мс
                    ets *= 1000
                pos["entry_time_ts"] = ets

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

                # Числовые поля
                for k in ("entry_price", "stop_loss", "take_profit", "trail_anchor"):
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

    # Удобные апдейтеры — используются трейлом/стратегией/GUI
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
        Считаем локальный net PnL (как database.update_trade_exit), прибавляем к equity,
        шлём апдейт в БД (если подключена), проставляем last_close_time_ms и чистим локальную позицию.
        """
        pos = self.get_current_position()
        if not pos:
            return

        # --- 1) Локальный net PnL
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

        # --- 2) Обновляем сделку в БД
        if self.db:
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

        # --- 3) last_close_time_ms (UTC now → ms)
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
