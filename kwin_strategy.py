# kwin_strategy.py  — Lux SFP only (oppos_prc + LTF volume) + SL from SFP extreme
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from database import Database
from utils_round import round_qty, round_to_tick, floor_to_tick, ceil_to_tick


class KWINStrategy:
    """Основная логика стратегии KWIN: входы ТОЛЬКО по Lux SFP + Smart/Bar trailing."""

    def __init__(
        self,
        config: Config,
        api: Any = None,
        state_manager: StateManager | None = None,
        db: Database | None = None,
        **kwargs: Any,
    ) -> None:
        # обратная совместимость
        if api is None and "bybit_api" in kwargs:
            api = kwargs.get("bybit_api")

        self.config = config
        self.api = api
        self.state = state_manager
        self.db = db

        # смарт-трейл движок
        self.trail_engine: Optional[TrailEngine] = None
        try:
            self.trail_engine = TrailEngine(config, state_manager, api)
        except TypeError:
            try:
                self.trail_engine = TrailEngine(config, state_manager=state_manager, bybit_api=api)
            except TypeError:
                try:
                    self.trail_engine = TrailEngine(config)
                except Exception:
                    self.trail_engine = None

        self.analytics = TradingAnalytics()

        # --------- Lux SFP режим (дефолты как на скрине) ---------
        self.lux_mode: bool = True
        self.lux_swings: int = int(getattr(config, "lux_swings", getattr(config, "sfp_len", 2)))
        # volume validation: "outside_gt" | "outside_lt" | "none"
        self.lux_volume_validation: str = str(getattr(config, "lux_volume_validation", "outside_gt")).lower()
        self.lux_volume_threshold_pct: float = float(getattr(config, "lux_volume_threshold_pct", 10.0))
        self.lux_auto: bool = bool(getattr(config, "lux_auto", False))
        self.lux_mlt: int = int(getattr(config, "lux_mlt", 10))
        self.lux_ltf: str = str(getattr(config, "lux_ltf", "1"))
        self.lux_premium: bool = bool(getattr(config, "lux_premium", False))
        self.lux_expire_bars: int = int(getattr(config, "lux_expire_bars", 500))

        # служебное состояние
        self.candles_15m: List[Dict] = []   # DESC: [0] — последний закрытый 15m бар
        self.candles_1m: List[Dict] = []    # DESC
        self.candles_1h: List[Dict] = []    # DESC
        self.last_processed_bar_ts: int = 0
        self.can_enter_long = True
        self.can_enter_short = True

        # активные Lux SFP (по одному на сторону)
        self._active_bull: Optional[Dict[str, Any]] = None
        self._active_bear: Optional[Dict[str, Any]] = None

        # инструмент / шаги
        self.symbol = str(getattr(config, "symbol", "ETHUSDT")).upper()
        self.tick_size = float(getattr(config, "tick_size", 0.01))
        self.qty_step = float(getattr(config, "qty_step", 0.01))
        self.min_order_qty = float(getattr(config, "min_order_qty", 0.01))

        # подтянуть фильтры инструмента
        self._init_instrument_info()

        # (необяз.) поддержка «cycle start» как в Pine:
        self.start_time_ms: Optional[int] = getattr(self.config, "start_time_ms", None)

        # одноразовое проигрывание истории 15m
        self._history_replayed: bool = False

    # ============ ВСПОМОГАТЕЛЬНОЕ ============

    def _align_15m_ms(self, ts_ms: int) -> int:
        return (int(ts_ms) // 900_000) * 900_000

    def _ensure_desc(self, arr: List[Dict], key: str = "timestamp"):
        if arr and arr[0].get(key) is not None:
            arr.sort(key=lambda x: x.get(key, 0), reverse=True)

    def _current_bar_ts_ms(self) -> int:
        return int(self.last_processed_bar_ts or 0)

    def _init_instrument_info(self):
        """Подтягиваем tick_size/qty_step/min_order_qty из API."""
        try:
            if not self.api or not hasattr(self.api, "get_instruments_info"):
                return
            info = self.api.get_instruments_info(self.symbol)
            if not info or not isinstance(info, dict):
                return
            pf = info.get("priceFilter") or {}
            ls = info.get("lotSizeFilter") or {}
            if pf.get("tickSize") is not None:
                self.tick_size = float(pf["tickSize"])
            if ls.get("qtyStep") is not None:
                self.qty_step = float(ls["qtyStep"])
            if ls.get("minOrderQty") is not None:
                self.min_order_qty = float(ls["minOrderQty"])
        except Exception:
            pass
        self.config.tick_size = self.tick_size or 0.01
        self.config.qty_step = self.qty_step or 0.01
        self.config.min_order_qty = self.min_order_qty or 0.01

    # ---------- Lux helpers ----------

    def _bar_window_ms(self, bar_ts_ms: int) -> Tuple[int, int]:
        """Возвращает [start,end] msec для 15m бара с указанным close-ts."""
        end_ms = self._align_15m_ms(bar_ts_ms)  # close aligned
        start_ms = end_ms - 900_000 + 1
        return start_ms, end_ms

    def _ltf_slices_for_bar(self, bar_ts_ms: int) -> List[Dict]:
        """LTF-свечи (обычно 1 мин), попадающие внутрь данного 15m бара."""
        ltf = str(self.lux_ltf or getattr(self.config, "intrabar_tf", "1"))
        if not self.candles_1m:
            return []
        start_ms, end_ms = self._bar_window_ms(bar_ts_ms)
        out = []
        for c in self.candles_1m:
            ts = int(c.get("timestamp") or 0)
            if start_ms <= ts <= end_ms:
                out.append(c)
            elif ts < start_ms:
                break
        return list(reversed(out))  # по времени вперёд

    def _ltf_outside_volume_ok(self, direction: str, swing_price: float, bar_ts_ms: int) -> bool:
        """Валидация объёма как в Lux (outside % против swing)."""
        mode = (self.lux_volume_validation or "outside_gt").lower()
        if mode == "none":
            return True

        chunks = self._ltf_slices_for_bar(bar_ts_ms)
        if not chunks:
            return True

        total_vol = float(sum(float(x.get("volume", 0.0)) for x in chunks)) or 0.0
        if total_vol <= 0:
            return True

        outside_vol = 0.0
        if direction == "bull":
            for x in chunks:
                cl = float(x.get("close"))
                if cl < swing_price:
                    outside_vol += float(x.get("volume", 0.0))
        else:  # bear
            for x in chunks:
                cl = float(x.get("close"))
                if cl > swing_price:
                    outside_vol += float(x.get("volume", 0.0))

        pct = 100.0 * outside_vol / total_vol
        thr = float(self.lux_volume_threshold_pct)

        if mode == "outside_gt":
            return pct > thr
        elif mode == "outside_lt":
            return pct < thr
        return True

    # ---------- Пивоты (для Lux) ----------

    def _pivot_low_value(self, left: int, right: int = 1) -> Optional[float]:
        n = int(left) + int(right) + 1
        if len(self.candles_15m) < n:
            return None
        lows = [float(self.candles_15m[i]["low"]) for i in range(0, n)]
        center = lows[int(right)]
        return center if center == min(lows) else None

    def _pivot_high_value(self, left: int, right: int = 1) -> Optional[float]:
        n = int(left) + int(right) + 1
        if len(self.candles_15m) < n:
            return None
        highs = [float(self.candles_15m[i]["high"]) for i in range(0, n)]
        center = highs[int(right)]
        return center if center == max(highs) else None

    # ---------- Lux SFP flow ----------

    def _lux_find_new_sfp(self) -> None:
        """На закрытии бара определяем SFP (wick + open/close за swing) и создаём active_*."""
        L = int(self.lux_swings)
        if len(self.candles_15m) < (L + 2):
            return

        curr = self.candles_15m[0]
        prev1 = self.candles_15m[1]  # центр пивота (right=1) — это именно [1]!

        # Bearish SFP:
        if self._pivot_high_value(L, 1) is not None:
            sw = float(prev1["high"])  # swing price = high[1]
            hi = float(curr["high"]); op = float(curr["open"]); cl = float(curr["close"])
            if hi > sw and op < sw and cl < sw:
                if self._ltf_outside_volume_ok("bear", sw, int(curr["timestamp"])):
                    self._active_bear = {
                        "active": True,
                        "confirmed": False,
                        "swing_prc": sw,
                        "sfp_extreme": hi,  # high SFP-бара (синяя точка)
                        "oppos_prc": sw,    # для right=1 oppos = swing
                        "created_bars_ago": 0,
                        "created_ts": int(curr.get("timestamp") or 0),
                    }

        # Bullish SFP:
        if self._pivot_low_value(L, 1) is not None:
            sw = float(prev1["low"])  # swing price = low[1]
            lo = float(curr["low"]); op = float(curr["open"]); cl = float(curr["close"])
            if lo < sw and op > sw and cl > sw:
                if self._ltf_outside_volume_ok("bull", sw, int(curr["timestamp"])):
                    self._active_bull = {
                        "active": True,
                        "confirmed": False,
                        "swing_prc": sw,
                        "sfp_extreme": lo,  # low SFP-бара (синяя точка)
                        "oppos_prc": sw,
                        "created_bars_ago": 0,
                        "created_ts": int(curr.get("timestamp") or 0),
                    }

    def _lux_update_active(self) -> None:
        """Обновляем состояние активных SFP: подтверждение, отмена, старение."""
        if not self.candles_15m:
            return

        cur = self.candles_15m[0]
        cl = float(cur["close"])

        # ---- Bear ----
        if self._active_bear and self._active_bear.get("active") and not self._active_bear.get("confirmed"):
            sw = float(self._active_bear["swing_prc"])
            oppos = float(self._active_bear["oppos_prc"])

            # подтверждение: close < oppos_prc (зелёная точка)
            if cl < oppos:
                tick = float(self.tick_size or 0.01)
                buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
                sl = ceil_to_tick(float(self._active_bear["sfp_extreme"]) + buf_ticks * tick, tick)
                if self.can_enter_short:
                    self._process_short_entry(entry_override=cl, sl_override=sl)
                    self._active_bear["confirmed"] = True

            # отмена: возврат выше swing_prc или протухание
            elif cl > sw:
                self._active_bear["active"] = False
            else:
                age = int(self._active_bear.get("created_bars_ago", 0)) + 1
                self._active_bear["created_bars_ago"] = age
                if age > int(self.lux_expire_bars):
                    self._active_bear["active"] = False

        # ---- Bull ----
        if self._active_bull and self._active_bull.get("active") and not self._active_bull.get("confirmed"):
            sw = float(self._active_bull["swing_prc"])
            oppos = float(self._active_bull["oppos_prc"])

            if cl > oppos:
                tick = float(self.tick_size or 0.01)
                buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
                sl = floor_to_tick(float(self._active_bull["sfp_extreme"]) - buf_ticks * tick, tick)
                if self.can_enter_long:
                    self._process_long_entry(entry_override=cl, sl_override=sl)
                    self._active_bull["confirmed"] = True

            elif cl < sw:
                self._active_bull["active"] = False
            else:
                age = int(self._active_bull.get("created_bars_ago", 0)) + 1
                self._active_bull["created_bars_ago"] = age
                if age > int(self.lux_expire_bars):
                    self._active_bull["active"] = False

    # ============ ОБРАБОТКА БАРОВ ============

    def on_bar_close_15m(self, candle: Dict):
        """Обработка закрытого 15m бара."""
        try:
            self.candles_15m.insert(0, candle)
            if len(self.candles_15m) > 200:
                self.candles_15m = self.candles_15m[:200]
            self._ensure_desc(self.candles_15m)

            bar_ts = int(candle.get("timestamp") or candle.get("open_time") or 0)
            if bar_ts and bar_ts < 1_000_000_000_000:
                bar_ts *= 1000
            aligned = self._align_15m_ms(bar_ts)
            if aligned == self.last_processed_bar_ts:
                return

            self.last_processed_bar_ts = aligned
            # Сбрасываем разрешения входов на новый 15м бар (не более 1 сделки на бар)
            self.can_enter_long = True
            self.can_enter_short = True
            self.run_cycle()
        except Exception as e:
            print(f"[on_bar_close_15m] {e}")

    def on_bar_close_60m(self, candle: Dict):
        try:
            self.candles_1h.insert(0, candle)
            if len(self.candles_1h) > 100:
                self.candles_1h = self.candles_1h[:100]
        except Exception as e:
            print(f"[on_bar_close_60m] {e}")

    def on_bar_close_1m(self, candle: Dict):
        """Интрабар — для Smart Trail и LTF объёма."""
        try:
            self.candles_1m.insert(0, candle)
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            if len(self.candles_1m) > lim:
                self.candles_1m = self.candles_1m[:lim]
            self._ensure_desc(self.candles_1m)

            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", False):
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[on_bar_close_1m] {e}")

    def update_candles(self):
        """Разовый пулл свечей; обработка закрытого 15m бара."""
        try:
            if not self.api or not hasattr(self.api, "get_klines"):
                return

            # объём истории по 15m
            need_15m = int(getattr(self.config, "days_back", 30)) * 96 + 50
            need_15m = max(200, min(5000, need_15m))

            # 15m
            kl15 = self.api.get_klines(self.symbol, "15", need_15m) or []
            for k in kl15:
                ts = k.get("timestamp")
                if ts is not None and ts < 1_000_000_000_000:
                    k["timestamp"] = int(ts * 1000)
            kl15.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            self.candles_15m = kl15

            # 1m (для LTF volume / трейлинга)
            tf = str(getattr(self.config, "intrabar_tf", "1"))
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            kl1 = self.api.get_klines(self.symbol, tf, min(1000, lim)) or []
            for k in kl1:
                ts = k.get("timestamp")
                if ts is not None and ts < 1_000_000_000_000:
                    k["timestamp"] = int(ts * 1000)
            kl1.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            self.candles_1m = kl1[:lim]

            # однократное «проигрывание» всей истории 15m
            if self.candles_15m and not self._history_replayed:
                bars_old_to_new = sorted(self.candles_15m, key=lambda x: x.get("timestamp", 0))
                self.candles_15m = []
                self.last_processed_bar_ts = 0
                for b in bars_old_to_new:
                    self.on_bar_close_15m(b)
                self._history_replayed = True
                return

            # стандартное поведение
            if self.candles_15m:
                ts = int(self.candles_15m[0].get("timestamp") or 0)
                aligned = self._align_15m_ms(ts)
                if aligned != self.last_processed_bar_ts:
                    self.last_processed_bar_ts = aligned
                    self.can_enter_long = True
                    self.can_enter_short = True
                    self.run_cycle()
                else:
                    pos = self.state.get_current_position() if self.state else None
                    if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", False):
                        self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[update_candles] {e}")

    # ========== ВХОДЫ ==========
    def _get_current_price(self) -> Optional[float]:
        try:
            if not self.api:
                return None
            src = str(getattr(self.config, "price_for_logic", "last")).lower()
            if hasattr(self.api, "get_price"):
                return float(self.api.get_price(self.symbol, source=src))
            t = self.api.get_ticker(self.symbol) or {}
            last = t.get("last_price") or t.get("lastPrice") or t.get("last")
            mark = t.get("mark_price") or t.get("markPrice")
            if src == "mark" and mark is not None:
                return float(mark)
            if last is not None:
                return float(last)
            if mark is not None:
                return float(mark)
        except Exception as e:
            print(f"[get_current_price] {e}")
        return None

    def _place_market_order(self, direction: str, quantity: float, stop_loss: Optional[float] = None):
        if not self.api or not hasattr(self.api, "place_order"):
            print("API not available for placing order")
            return None
        side_up = "Buy" if direction == "long" else "Sell"
        qty = float(quantity)

        sl_send = None
        if stop_loss is not None:
            if direction == "long":
                sl_send = floor_to_tick(stop_loss, self.tick_size)
            else:
                sl_send = ceil_to_tick(stop_loss, self.tick_size)

        try:
            return self.api.place_order(
                symbol=self.symbol,
                side=side_up,
                orderType="Market",
                qty=qty,
                stop_loss=sl_send,
                trigger_by_source=getattr(self.config, "trigger_price_source", "mark"),
            )
        except TypeError:
            # старые адаптеры
            side_lo = "buy" if direction == "long" else "sell"
            return self.api.place_order(
                symbol=self.symbol,
                side=side_lo,
                order_type="market",
                qty=qty,
                stop_loss=sl_send,
            )

    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        try:
            equity = self.state.get_equity() if self.state else None
            if equity is None or equity <= 0:
                return None
            risk_amount = float(equity) * (float(self.config.risk_pct) / 100.0)
            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None
            quantity = risk_amount / stop_size
            quantity = round_qty(quantity, self.qty_step)
            if getattr(self.config, "limit_qty_enabled", False):
                quantity = min(quantity, float(getattr(self.config, "max_qty_manual", quantity)))
            if quantity < float(self.min_order_qty):
                return None
            return float(quantity)
        except Exception as e:
            print(f"[calc_position_size] {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float,
                                    take_profit: Optional[float], quantity: float) -> bool:
    
        try:
            if quantity is None or float(quantity) <= 0:
                return False
            if float(quantity) < float(getattr(self.config, "min_order_qty", 0.01)):
                return False

            stop_size = abs(float(entry_price) - float(stop_loss))
            if stop_size <= 0:
                return False

        # (опционально) держим расчёт TP для последующих модулей,
        # но НЕ фильтруем по net profit
            rr = float(getattr(self.config, "risk_reward", 1.3))
            if entry_price >= stop_loss:  # long
                _ = float(entry_price) + stop_size * rr
            else:                         # short
                _ = float(entry_price) - stop_size * rr

            return True
        except Exception as e:
            print(f\"[validate_position] {e}")
            return False
    # ---------- Cooldown (отключён по ТЗ) ----------
    def _in_cooldown(self, _now_ts_ms: int) -> bool:
        return False

    def _process_long_entry(self, entry_override: Optional[float] = None, sl_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 2:
                return

            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close
            entry = round_to_tick(float(price), self.tick_size)

            # SL: строго из Lux (экстремум SFP‑свечи) + буфер
            if sl_override is None:
                return
            sl = float(sl_override)

            stop_size = entry - sl
            if stop_size <= 0:
                return

            tp_for_filter = round_to_tick(entry + stop_size * float(self.config.risk_reward), self.tick_size)

            qty = self._calculate_position_size(entry, sl, "long")
            if not qty or not self._validate_position_requirements(entry, sl, tp_for_filter, qty):
                return

            res = self._place_market_order("long", qty, stop_loss=sl)
            if res is None:
                return

            bar_ts_ms = self._current_bar_ts_ms()
            trade = {
                "symbol": self.symbol,
                "direction": "long",
                "entry_price": entry,
                "stop_loss": sl,
                "quantity": float(qty),
                "entry_time": datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                "status": "open",
                "take_profit": tp_for_filter,
            }
            if self.db:
                self.db.save_trade(trade)

            pos = {
                "symbol": self.symbol,
                "direction": "long",
                "size": float(qty),
                "entry_price": entry,
                "stop_loss": sl,
                "status": "open",
                "armed": not getattr(self.config, "use_arm_after_rr", True),
                "trail_anchor": entry,
                "entry_time_ts": bar_ts_ms,
                "take_profit": tp_for_filter,
            }
            if self.state:
                self.state.set_position(pos)

            if self.trail_engine:
                try:
                    self.trail_engine.on_entry(entry, sl, "long")
                except Exception:
                    pass

            self.can_enter_long = False
            self.can_enter_short = False
            print(f"[ENTRY LONG] {qty} @ {entry}, SL={sl}, TP(filt)={tp_for_filter}")
        except Exception as e:
            print(f"[process_long_entry] {e}")

    def _process_short_entry(self, entry_override: Optional[float] = None, sl_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 2:
                return

            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close
            entry = round_to_tick(float(price), self.tick_size)

            if sl_override is None:
                return
            sl = float(sl_override)

            stop_size = sl - entry
            if stop_size <= 0:
                return

            tp_for_filter = round_to_tick(entry - stop_size * float(self.config.risk_reward), self.tick_size)

            qty = self._calculate_position_size(entry, sl, "short")
            if not qty or not self._validate_position_requirements(entry, sl, tp_for_filter, qty):
                return

            res = self._place_market_order("short", qty, stop_loss=sl)
            if res is None:
                return

            bar_ts_ms = self._current_bar_ts_ms()
            trade = {
                "symbol": self.symbol,
                "direction": "short",
                "entry_price": entry,
                "stop_loss": sl,
                "quantity": float(qty),
                "entry_time": datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                "status": "open",
                "take_profit": tp_for_filter,
            }
            if self.db:
                self.db.save_trade(trade)

            pos = {
                "symbol": self.symbol,
                "direction": "short",
                "size": float(qty),
                "entry_price": entry,
                "stop_loss": sl,
                "status": "open",
                "armed": not getattr(self.config, "use_arm_after_rr", True),
                "trail_anchor": entry,
                "entry_time_ts": bar_ts_ms,
                "take_profit": tp_for_filter,
            }
            if self.state:
                self.state.set_position(pos)

            if self.trail_engine:
                try:
                    self.trail_engine.on_entry(entry, sl, "short")
                except Exception:
                    pass

            self.can_enter_short = False
            self.can_enter_long = False
            print(f"[ENTRY SHORT] {qty} @ {entry}, SL={sl}, TP(filt)={tp_for_filter}")
        except Exception as e:
            print(f"[process_short_entry] {e}")

    # ---------- Трейлинг ----------

    def _get_bar_extremes_for_trailing(self, current_price: float) -> Tuple[float, float]:
        try:
            if getattr(self.config, "use_intrabar", False) and self.candles_1m:
                last = self.candles_1m[0]
                return float(last["high"]), float(last["low"])
            elif self.candles_15m:
                last = self.candles_15m[0]
                return float(last["high"]), float(last["low"])
        except Exception:
            pass
        return float(current_price), float(current_price)

    def _update_smart_trailing(self, position: Dict):
        try:
            if not getattr(self.config, "enable_smart_trail", True):
                return

            direction = position.get("direction")
            entry = float(position.get("entry_price") or 0)
            sl = float(position.get("stop_loss") or 0)
            if not direction or entry <= 0 or sl <= 0:
                return

            price = self._get_current_price()
            if price is None:
                return
            price = float(price)

            bar_high, bar_low = self._get_bar_extremes_for_trailing(price)

            armed = bool(position.get("armed", not getattr(self.config, "use_arm_after_rr", True)))
            if not armed and getattr(self.config, "use_arm_after_rr", True):
                risk = abs(entry - sl)
                if risk > 0:
                    rr_ext = (bar_high - entry) / risk if direction == "long" else (entry - bar_low) / risk
                    rr_last = (price - entry) / risk   if direction == "long" else (entry - price) / risk
                    rr_need = float(getattr(self.config, "arm_rr", 0.5))
                    basis   = str(getattr(self.config, "arm_rr_basis", "extremum")).lower()
                    rr_now  = rr_ext if basis == "extremum" else rr_last
                    rr_alt  = rr_last if basis == "extremum" else rr_ext

                    if rr_now >= rr_need or rr_alt >= rr_need:
                        armed = True
                        position["armed"] = True
                        if self.state:
                            self.state.set_position(position)
                        print(f"[ARM] enabled at ≥{rr_need:.2f}R (basis={basis}, rr_now={rr_now:.3f}, rr_alt={rr_alt:.3f})")

            if not armed:
                return

            anchor = float(position.get("trail_anchor") or entry)
            anchor = max(anchor, bar_high) if direction == "long" else min(anchor, bar_low)
            if anchor != position.get("trail_anchor"):
                position["trail_anchor"] = anchor
                if self.state:
                    self.state.set_position(position)

            trail_perc  = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            offset_perc = float(getattr(self.config, "trailing_offset_perc", 0.4)) / 100.0
            trail_dist  = entry * trail_perc
            offset_dist = entry * offset_perc

            if direction == "long":
                candidate = floor_to_tick(anchor - trail_dist - offset_dist, self.tick_size)
                if candidate > sl:
                    self._update_stop_loss(position, candidate)
            else:
                candidate = ceil_to_tick(anchor + trail_dist + offset_dist, self.tick_size)
                if candidate < sl:
                    self._update_stop_loss(position, candidate)

        except Exception as e:
            print(f"[smart_trailing] {e}")

    def _update_stop_loss(self, position: Dict, new_sl: float) -> bool:
        try:
            direction = position.get("direction")
            if direction == "long":
                new_sl = floor_to_tick(float(new_sl), self.tick_size)
            else:
                new_sl = ceil_to_tick(float(new_sl), self.tick_size)

            if not self.api:
                position["stop_loss"] = float(new_sl)
                if self.state:
                    self.state.set_position(position)
                print(f"[TRAIL-LOCAL] SL -> {new_sl:.4f}")
                return True

            if hasattr(self.api, "update_position_stop_loss"):
                ok = self.api.update_position_stop_loss(
                    self.symbol,
                    new_sl,
                    trigger_by_source=getattr(self.config, "trigger_price_source", "mark"),
                )
                if ok:
                    position["stop_loss"] = float(new_sl)
                    if self.state:
                        self.state.set_position(position)
                    print(f"[TRAIL] SL -> {new_sl:.4f}")
                    return True

            if hasattr(self.api, "modify_order"):
                _ = self.api.modify_order(
                    symbol=position.get("symbol", self.symbol),
                    stop_loss=float(new_sl),
                    trigger_by_source=getattr(self.config, "trigger_price_source", "mark"),
                )
                position["stop_loss"] = float(new_sl)
                if self.state:
                    self.state.set_position(position)
                print(f"[TRAIL] SL -> {new_sl:.4f}")
                return True

            position["stop_loss"] = float(new_sl)
            if self.state:
                self.state.set_position(position)
            print(f"[TRAIL-LOCAL] SL -> {new_sl:.4f}")
            return True

        except Exception as e:
            print(f"[update_stop_loss] {e}")
            return False

    # ---------- Цикл/прочее ----------

    def process_trailing(self):
        try:
            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[process_trailing] {e}")

    def run_cycle(self):
        try:
            if not self.candles_15m:
                return

            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)
                return

            if len(self.candles_15m) < int(self.lux_swings) + 2:
                return

            ts = int(self.candles_15m[0]["timestamp"])
            if (not self._is_in_backtest_window_utc(ts)) or (not self._is_after_cycle_start(ts)):
                return

            # Lux‑поток
            self._lux_find_new_sfp()
            self._lux_update_active()
        except Exception as e:
            print(f"[run_cycle] {e}")

    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=int(getattr(self.config, "days_back", 30)))
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

    def _is_after_cycle_start(self, current_timestamp: int) -> bool:
        if self.start_time_ms is None:
            return True
        return int(current_timestamp) >= int(self.start_time_ms)

    def _update_equity(self):
        try:
            if not self.api or not hasattr(self.api, "get_wallet_balance"):
                return
            wallet = self.api.get_wallet_balance()
            if wallet and wallet.get("list"):
                for account in wallet["list"]:
                    if account.get("accountType") in ("SPOT", "UNIFIED"):
                        for coin in account.get("coin", []):
                            if coin.get("coin") == "USDT":
                                equity = float(coin.get("equity", 0))
                                if self.state:
                                    self.state.set_equity(equity)
                                if self.db:
                                    self.db.save_equity_snapshot(equity)
                                return
        except Exception as e:
            print(f"[update_equity] {e}")

    # ---------- Debug helper для дашборда ----------

    def get_trailing_debug(self) -> Dict[str, Any]:
        try:
            pos = self.state.get_current_position() if self.state else None
            if not pos:
                return {}
            return {
                "symbol": self.symbol,
                "entry": pos.get("entry_price"),
                "sl": pos.get("stop_loss"),
                "armed": pos.get("armed"),
                "anchor": pos.get("trail_anchor"),
                "tick_size": self.tick_size,
                "qty_step": self.qty_step,
            }
        except Exception:
            return {}
