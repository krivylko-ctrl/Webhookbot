# kwin_strategy.py  — обновлено: SL от SFP-свечи [0] / свинговый пивот, без ATR/prev[1] + invert_signals
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from database import Database
# точные хелперы округления к тику (1:1 с Pine)
from utils_round import round_price, round_qty, round_to_tick, floor_to_tick, ceil_to_tick


class KWINStrategy:
    """Основная логика стратегии KWIN (SFP + ARM + Smart trailing)."""

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

        # служебное состояние
        self.candles_15m: List[Dict] = []   # DESC: [0] — последний закрытый 15m бар
        self.candles_1m: List[Dict] = []    # DESC
        self.candles_1h: List[Dict] = []    # DESC
        self.last_processed_bar_ts: int = 0
        self.can_enter_long = True
        self.can_enter_short = True

        # инструмент / шаги
        self.symbol = str(getattr(config, "symbol", "ETHUSDT")).upper()
        self.tick_size = float(getattr(config, "tick_size", 0.01))
        self.qty_step = float(getattr(config, "qty_step", 0.01))
        self.min_order_qty = float(getattr(config, "min_order_qty", 0.01))

        # подтянуть фильтры инструмента
        self._init_instrument_info()

        # нормализация close_back_pct в [0..1]
        try:
            if self.config.close_back_pct is None:
                self.config.close_back_pct = 1.0
            if self.config.close_back_pct > 1.0:
                self.config.close_back_pct = float(self.config.close_back_pct) / 100.0
            if self.config.close_back_pct < 0.0:
                self.config.close_back_pct = 0.0
        except Exception:
            self.config.close_back_pct = 1.0

        # (необяз.) поддержка «cycle start» как в Pine:
        self.start_time_ms: Optional[int] = getattr(self.config, "start_time_ms", None)

        # одноразовое проигрывание истории 15m
        self._history_replayed: bool = False

        # инверсия сигналов (Long <-> Short)
        self.invert_signals: bool = bool(getattr(self.config, "invert_signals", False))
        if self.invert_signals:
            print("[KWIN] invert_signals = ON — long/short будут меняться местами.")

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

    # ---------- SL «зона» как в TV ----------
    def _calc_sl_with_zone(self, direction: str, entry_price: float) -> Optional[float]:
        """
        База SL выбирается в порядке приоритета:
          1) use_sfp_candle_sl=True  -> long: low[0];  short: high[0]
          2) use_swing_sl=True       -> long: min(low[1], low[sfp_len]); short: max(high[1], high[sfp_len])
          3) fallback                 -> long: low[1];  short: high[1]
        Затем добавляется буфер по тикам (sl_buf_ticks).
        Округление: long -> floor_to_tick, short -> ceil_to_tick.
        """
        try:
            L = int(getattr(self.config, "sfp_len", 2))
            if len(self.candles_15m) < max(L + 1, 2):
                return None

            cur = self.candles_15m[0]
            prev = self.candles_15m[1]

            use_sfp_candle = bool(getattr(self.config, "use_sfp_candle_sl", False))
            use_swing = bool(getattr(self.config, "use_swing_sl", True))

            tick = float(self.tick_size or 0.01)
            buf_ticks = int(getattr(self.config, "sl_buf_ticks", 40) or 0)
            buf_px = buf_ticks * tick

            # кандидаты от пивота (для свингового режима)
            pivot_low = None
            pivot_high = None
            if use_swing and len(self.candles_15m) > L:
                try:
                    pivot_low = float(self.candles_15m[L]["low"])
                    pivot_high = float(self.candles_15m[L]["high"])
                except Exception:
                    pivot_low = pivot_high = None

            # базовые уровни по приоритету
            if direction == "long":
                if use_sfp_candle:
                    base = float(cur["low"])
                elif use_swing and pivot_low is not None:
                    base = min(float(prev["low"]), float(pivot_low))
                else:
                    base = float(prev["low"])
                sl_raw = base - buf_px
                sl = floor_to_tick(sl_raw, tick)
                if sl >= entry_price:  # страхуемся
                    sl = floor_to_tick(entry_price - tick, tick)
            else:
                if use_sfp_candle:
                    base = float(cur["high"])
                elif use_swing and pivot_high is not None:
                    base = max(float(prev["high"]), float(pivot_high))
                else:
                    base = float(prev["high"])
                sl_raw = base + buf_px
                sl = ceil_to_tick(sl_raw, tick)
                if sl <= entry_price:
                    sl = ceil_to_tick(entry_price + tick, tick)

            return float(sl)
        except Exception as e:
            print(f"[calc_sl_with_zone] {e}")
            return None

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
            # Pine: canEnter* сбрасываются на закрытии нового 15m бара
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
        """Интрабар трейлинг + (опционально) интрабар-входы."""
        try:
            self.candles_1m.insert(0, candle)
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            if len(self.candles_1m) > lim:
                self.candles_1m = self.candles_1m[:lim]
            self._ensure_desc(self.candles_1m)

            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", False):
                self._update_smart_trailing(pos)

            if (not pos) and getattr(self.config, "use_intrabar_entries", False):
                self._try_intrabar_entry_from_m1(candle)

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

            # 1m
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

    # ---------- ХЕЛПЕР ВХОДА С УЧЁТОМ ИНВЕРСИИ ----------

    def _enter_by_signal(self, signal: str, entry_override: Optional[float] = None) -> None:
        """
        signal: "long" | "short" — тип СИГНАЛА (по условиям SFP на 15m).
        При invert_signals=True направление сделки меняется местами.
        """
        try:
            if not self.invert_signals:
                if signal == "long":
                    self._process_long_entry(entry_override=entry_override)
                else:
                    self._process_short_entry(entry_override=entry_override)
            else:
                if signal == "long":
                    self._process_short_entry(entry_override=entry_override)
                else:
                    self._process_long_entry(entry_override=entry_override)
        except Exception as e:
            print(f"[enter_by_signal] {e}")

    # ---------- ЛОГИКА ВХОДОВ (совместимость) ----------

    def on_bar_close(self):
        """Совместимость (основная точка — run_cycle())."""
        if len(self.candles_15m) < int(getattr(self.config, "sfp_len", 2)) + 2:
            return

        bull = self._detect_bull_sfp()
        bear = self._detect_bear_sfp()

        ts = int(self.candles_15m[0]["timestamp"])
        if not self._is_in_backtest_window_utc(ts) or not self._is_after_cycle_start(ts):
            return

        current_position = self.state.get_current_position() if self.state else None
        if current_position and current_position.get("status") == "open":
            return

        if bull and self.can_enter_long:
            self._enter_by_signal("long")
        elif bear and self.can_enter_short:
            self._enter_by_signal("short")

    # ---------- SFP (Pine-эквивалент) ----------

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

    def _detect_bull_sfp(self) -> bool:
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False

        # факт пивота
        if self._pivot_low_value(L, 1) is None:
            return False

        curr = self.candles_15m[0]
        ref = float(self.candles_15m[L]["low"])  # === low[sfpLen] ===

        lo = float(curr["low"])
        op = float(curr["open"])
        cl = float(curr["close"])

        cond_break = lo < ref
        cond_open  = op > ref
        cond_close = cl > ref
        if cond_break and cond_open and cond_close:
            return self._check_bull_sfp_quality(curr, ref) if getattr(self.config, "use_sfp_quality", True) else True
        return False

    def _detect_bear_sfp(self) -> bool:
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False

        if self._pivot_high_value(L, 1) is None:
            return False

        curr = self.candles_15m[0]
        ref = float(self.candles_15m[L]["high"])  # === high[sfpLen] ===

        hi = float(curr["high"])
        op = float(curr["open"])
        cl = float(curr["close"])

        cond_break = hi > ref
        cond_open  = op < ref
        cond_close = cl < ref
        if cond_break and cond_open and cond_close:
            return self._check_bear_sfp_quality(curr, ref) if getattr(self.config, "use_sfp_quality", True) else True
        return False

    def _check_bull_sfp_quality(self, current: dict, ref_low_val: float) -> bool:
        try:
            low = float(current["low"])
            close = float(current["close"])
            wick_depth = max(ref_low_val - low, 0.0)
            if wick_depth <= 0:
                return False
            m_tick = float(self.tick_size or 0.01)
            wick_ticks = wick_depth / m_tick
            if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
                return False
            cb = float(getattr(self.config, "close_back_pct", 1.0))
            return (close - low) >= (wick_depth * cb)
        except Exception:
            return False

    def _check_bear_sfp_quality(self, current: dict, ref_high_val: float) -> bool:
        try:
            high = float(current["high"])
            close = float(current["close"])
            wick_depth = max(high - ref_high_val, 0.0)
            if wick_depth <= 0:
                return False
            m_tick = float(self.tick_size or 0.01)
            wick_ticks = wick_depth / m_tick
            if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
                return False
            cb = float(getattr(self.config, "close_back_pct", 1.0))
            return (high - close) >= (wick_depth * cb)
        except Exception:
            return False

    # ---------- Интрабар-входы по M1 (опционально) ----------

    def _try_intrabar_entry_from_m1(self, m1: Dict) -> None:
        try:
            if not (self.candles_15m and self._is_in_backtest_window_utc(int(self.candles_15m[0]["timestamp"])) and self._is_after_cycle_start(int(self.candles_15m[0]["timestamp"]))):
                return
            if self.state and self.state.is_position_open():
                return

            L = int(getattr(self.config, "sfp_len", 2))
            if len(self.candles_15m) < (L + 2):
                return

            has_bull_pivot = self._pivot_low_value(L, 1)  is not None
            has_bear_pivot = self._pivot_high_value(L, 1) is not None

            if self.can_enter_long and has_bull_pivot:
                ref = float(self.candles_15m[L]["low"])  # low[sfpLen]
                lo = float(m1["low"]); cl = float(m1["close"])
                if (lo < ref) and (cl > ref):
                    if (not getattr(self.config, "use_sfp_quality", True)) or self._check_bull_sfp_quality({"low": lo, "close": cl}, ref):
                        # сигнал "long" → учесть инверсию направления в _enter_by_signal
                        self._enter_by_signal("long", entry_override=cl)
                        return

            if self.can_enter_short and has_bear_pivot:
                ref = float(self.candles_15m[L]["high"])  # high[sfpLen]
                hi = float(m1["high"]); cl = float(m1["close"])
                if (hi > ref) and (cl < ref):
                    if (not getattr(self.config, "use_sfp_quality", True)) or self._check_bear_sfp_quality({"high": hi, "close": cl}, ref):
                        # сигнал "short" → учесть инверсию направления в _enter_by_signal
                        self._enter_by_signal("short", entry_override=cl)
                        return
        except Exception as e:
            print(f"[intrabar_entry] {e}")

    # ---------- Ордера / вход ----------

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
        side_lo = "buy" if direction == "long" else "sell"
        qty = float(quantity)

        if stop_loss is not None:
            if direction == "long":
                sl_send = floor_to_tick(stop_loss, self.tick_size)
            else:
                sl_send = ceil_to_tick(stop_loss, self.tick_size)
        else:
            sl_send = None

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
            rr = float(getattr(self.config, "risk_reward", 1.3))
            if entry_price >= stop_loss:  # long
                tp_calc = float(entry_price) + stop_size * rr
            else:                         # short
                tp_calc = float(entry_price) - stop_size * rr

            taker = float(getattr(self.config, "taker_fee_rate", 0.00055))
            gross = abs(tp_calc - float(entry_price)) * float(quantity)
            fees = float(entry_price) * float(quantity) * taker * 2.0
            net = gross - fees
            min_net = float(getattr(self.config, "min_net_profit", 1.2))
            return net >= min_net
        except Exception as e:
            print(f"[validate_position] {e}")
            return False

    # ---------- Cooldown ----------
    def _in_cooldown(self, now_ts_ms: int) -> bool:
        """Если задан cooldown_minutes — запрещаем новые входы до истечения паузы после последнего закрытия."""
        try:
            cool_min = int(getattr(self.config, "cooldown_minutes", 0) or 0)
            if cool_min <= 0:
                return False
            last_close_ts = None
            # сперва спросим у StateManager (если он это хранит)
            if self.state and hasattr(self.state, "get_last_close_time"):
                try:
                    last_close_ts = self.state.get_last_close_time()  # ожидаем unix ms или datetime
                    if isinstance(last_close_ts, datetime):
                        last_close_ts = int(last_close_ts.timestamp() * 1000)
                except Exception:
                    last_close_ts = None
            # фоллбек: берём из БД последнюю закрытую сделку
            if (last_close_ts is None) and self.db and hasattr(self.db, "get_recent_trades"):
                try:
                    rec = self.db.get_recent_trades(1) or []
                    if rec and rec[0].get("exit_time"):
                        last_close_ts = int(pd.to_datetime(rec[0]["exit_time"]).timestamp() * 1000)
                except Exception:
                    last_close_ts = None

            if last_close_ts is None:
                return False

            return now_ts_ms < (int(last_close_ts) + cool_min * 60_000)
        except Exception:
            return False

    # ========== ВХОДЫ ==========
    def _process_long_entry(self, entry_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 2:
                return

            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close
            entry = round_to_tick(float(price), self.tick_size)

            # Cooldown-гейт
            if self._in_cooldown(self._current_bar_ts_ms()):
                return

            # Новый расчёт SL «зоной»
            sl = self._calc_sl_with_zone(direction="long", entry_price=entry)
            if sl is None:
                return

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
            print(f"[ENTRY LONG] {qty} @ {entry}, SL(zone)={sl}, TP(filt)={tp_for_filter}")
        except Exception as e:
            print(f"[process_long_entry] {e}")

    def _process_short_entry(self, entry_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 2:
                return

            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close
            entry = round_to_tick(float(price), self.tick_size)

            if self._in_cooldown(self._current_bar_ts_ms()):
                return

            sl = self._calc_sl_with_zone(direction="short", entry_price=entry)
            if sl is None:
                return

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
            print(f"[ENTRY SHORT] {qty} @ {entry}, SL(zone)={sl}, TP(filt)={tp_for_filter}")
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
                # Трейл обновляем вне зависимости от use_intrabar; источники экстремумов выбираются внутри
                self._update_smart_trailing(pos)
                return

            if len(self.candles_15m) < int(getattr(self.config, "sfp_len", 2)) + 2:
                return

            ts = int(self.candles_15m[0]["timestamp"])
            if (not self._is_in_backtest_window_utc(ts)) or (not self._is_after_cycle_start(ts)):
                return

            bull = self._detect_bull_sfp()
            bear = self._detect_bear_sfp()

            # Сначала проверяем "long"-сигнал (bull), затем "short"-сигнал (bear)
            if bull and self.can_enter_long and not self._in_cooldown(ts):
                self._enter_by_signal("long")
            elif bear and self.can_enter_short and not self._in_cooldown(ts):
                self._enter_by_signal("short")
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
