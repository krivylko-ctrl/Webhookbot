# kwin_strategy.py
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

        # смарт-трейл движок (без изменений)
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
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
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

            # 15m
            kl15 = self.api.get_klines(self.symbol, "15", 100) or []
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
                    if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                        self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[update_candles] {e}")

    # ---------- ЛОГИКА ВХОДОВ ----------

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
            self._process_long_entry()
        elif bear and self.can_enter_short:
            self._process_short_entry()

    # ---------- SFP (Pine-эквивалент) ----------

    def _pivot_low_value(self, left: int, right: int = 1) -> Optional[float]:
        """
        ta.pivotlow(low, left, right) на последнем закрытом 15m баре.
        Порядок свечей DESC: 0=текущий закрытый, 1=предыдущий, ...
        Окно -> индексы [0 .. left+right]; центр -> индекс 'right'.
        """
        n = int(left) + int(right) + 1
        if len(self.candles_15m) < n:
            return None
        lows = [float(self.candles_15m[i]["low"]) for i in range(0, n)]
        center = lows[int(right)]
        return center if center == min(lows) else None

    def _pivot_high_value(self, left: int, right: int = 1) -> Optional[float]:
        """ta.pivothigh(high, left, right) — см. комментарий выше."""
        n = int(left) + int(right) + 1
        if len(self.candles_15m) < n:
            return None
        highs = [float(self.candles_15m[i]["high"]) for i in range(0, n)]
        center = highs[int(right)]
        return center if center == max(highs) else None

    # >>> ДОБАВКА: булева проверка подтверждённого пивота (1:1 с Pine)
    def _pivot_confirmed(self, left: int, right: int = 1, kind: str = "low") -> bool:
        """
        True, если на баре [1] подтверждён pivot:
         - kind="low": ta.pivotlow(left, right)
         - kind="high": ta.pivothigh(left, right)
        Центр окна — индекс 'right' (у нас это 1).
        """
        n = int(left) + int(right) + 1
        if len(self.candles_15m) < n:
            return False
        if kind == "low":
            arr = [float(self.candles_15m[i]["low"]) for i in range(0, n)]
            center = arr[int(right)]
            return center == min(arr)
        else:
            arr = [float(self.candles_15m[i]["high"]) for i in range(0, n)]
            center = arr[int(right)]
            return center == max(arr)

    def _detect_bull_sfp(self) -> bool:
        """
        Бычий SFP по Pine:
          pivotlow(L,1) подтверждён И
          сравнение идёт с low[sfpLen] (а не с центром пивота),
          условия: low < low[sfpLen] И open > low[sfpLen] И close > low[sfpLen],
        + опциональная проверка качества (wick+closeBack).
        """
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False

        # факт пивота
        if not self._pivot_confirmed(L, 1, kind="low"):
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
        """
        Медвежий SFP по Pine:
          pivothigh(L,1) подтверждён И
          сравнение с high[sfpLen],
          условия: high > high[sfpLen] И open < high[sfpLen] И close < high[sfpLen],
        + опциональная проверка качества (wick+closeBack).
        """
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False

        if not self._pivot_confirmed(L, 1, kind="high"):
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
        """Качество бычьего SFP: wick depth в тиках и 'close-back %' от глубины фитиля."""
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
        """Качество медвежьего SFP."""
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
        """
        Имитация «calc_on_every_tick»: если внутри текущего 15m возникает SFP,
        то вход выполняется немедленно по цене закрытия M1.
        Требуем pivot(L,1) и сравниваем с low[sfpLen]/high[sfpLen] (как в Pine).
        """
        try:
            if not (self.candles_15m and self._is_in_backtest_window_utc(int(self.candles_15m[0]["timestamp"])) and self._is_after_cycle_start(int(self.candles_15m[0]["timestamp"]))):
                return
            if self.state and self.state.is_position_open():
                return

            L = int(getattr(self.config, "sfp_len", 2))
            if len(self.candles_15m) < (L + 2):
                return

            has_bull_pivot = self._pivot_confirmed(L, 1, kind="low")
            has_bear_pivot = self._pivot_confirmed(L, 1, kind="high")

            # бычий: low<low[L] и close>low[L] на минутке
            if self.can_enter_long and has_bull_pivot:
                ref = float(self.candles_15m[L]["low"])  # low[sfpLen]
                lo = float(m1["low"]); cl = float(m1["close"])
                if (lo < ref) and (cl > ref):
                    if not getattr(self.config, "use_sfp_quality", True) or self._check_bull_sfp_quality({"low": lo, "close": cl}, ref):
                        self._process_long_entry(entry_override=cl)
                        return

            # медвежий: high>high[L] и close<high[L] на минутке
            if self.can_enter_short and has_bear_pivot:
                ref = float(self.candles_15m[L]["high"])  # high[sfpLen]
                hi = float(m1["high"]); cl = float(m1["close"])
                if (hi > ref) and (cl < ref):
                    if not getattr(self.config, "use_sfp_quality", True) or self._check_bear_sfp_quality({"high": hi, "close": cl}, ref):
                        self._process_short_entry(entry_override=cl)
                        return
        except Exception as e:
            print(f"[intrabar_entry] {e}")

    # ---------- Ордера / вход ----------

    def _get_current_price(self) -> Optional[float]:
        """Единый источник цены: config.price_for_logic = 'last'|'mark'."""
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

        # корректное округление SL: long=floor, short=ceil
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
                trigger_by_source=getattr(self.config, "trigger_price_source", "last"),
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
        """Размер позиции в базовой валюте по risk_pct от equity и расстоянию до SL."""
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
        """
        Pine-эквивалент: требуем minOrderQty И expNetPnL >= minNetProfit.
        expNetPnL считается ВСЕГДА через TP = entry ± stopSize * riskReward,
        независимо от флага use_take_profit.
        """
        try:
            if quantity is None or float(quantity) <= 0:
                return False
            if float(quantity) < float(getattr(self.config, "min_order_qty", 0.01)):
                return False

            # TP по Pine: entry ± stopSize*riskReward
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

    def _process_long_entry(self, entry_override: Optional[float] = None):
        """Вход long — строго как в Pine: sl=low[1], entry=close(15m) (или m1 close при интрабаре)."""
        try:
            if len(self.candles_15m) < 2:
                return

            # Pine: entry = close текущего 15m
            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close

            raw_sl = float(self.candles_15m[1]["low"])

            entry = round_to_tick(float(price), self.tick_size)
            sl    = floor_to_tick(raw_sl, self.tick_size)
            stop_size = entry - sl
            if stop_size <= 0:
                return

            # TP по Pine, даже если TP отключён в лайве — для фильтра expNetPnL
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
            }
            # в БД TP хранить полезно, даже если фактический выход — трейлинг
            trade["take_profit"] = tp_for_filter
            if self.db:
                self.db.save_trade(trade)

            pos = {
                "symbol": self.symbol,
                "direction": "long",
                "size": float(qty),
                "entry_price": entry,
                "stop_loss": sl,
                "status": "open",
                # Pine: armed := not useArmAfterRR
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

            # как в Pine: запрет до следующего 15m бара
            self.can_enter_long = False
            self.can_enter_short = False
            print(f"[ENTRY LONG] {qty} @ {entry}, SL={sl}, TP(filt)={tp_for_filter}")
        except Exception as e:
            print(f"[process_long_entry] {e}")

    def _process_short_entry(self, entry_override: Optional[float] = None):
        """Вход short — строго как в Pine: sl=high[1], entry=close(15m) (или m1 close при интрабаре)."""
        try:
            if len(self.candles_15m) < 2:
                return

            bar_close = float(self.candles_15m[0]["close"])
            price = float(entry_override) if entry_override is not None else bar_close

            raw_sl = float(self.candles_15m[1]["high"])

            entry = round_to_tick(float(price), self.tick_size)
            sl    = ceil_to_tick(raw_sl, self.tick_size)
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

    # ---------- Трейлинг (без изменений механики) ----------

    def _get_bar_extremes_for_trailing(self, current_price: float) -> Tuple[float, float]:
        """Возвращает high/low для расчёта якоря: сначала 1м (если включено), иначе 15м."""
        try:
            if getattr(self.config, "use_intrabar", True) and self.candles_1m:
                last = self.candles_1m[0]
                return float(last["high"]), float(last["low"])
            elif self.candles_15m:
                last = self.candles_15m[0]
                return float(last["high"]), float(last["low"])
        except Exception:
            pass
        return float(current_price), float(current_price)

    def _update_smart_trailing(self, position: Dict):
        """Smart trailing: ARM по RR, якорь-экстремум и % трейл от entry + offset (как раньше)."""
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

            # ARM (RR)
            armed = bool(position.get("armed", not getattr(self.config, "use_arm_after_rr", True)))
            if not armed and getattr(self.config, "use_arm_after_rr", True):
                risk = abs(entry - sl)
                if risk > 0:
                    rr_ext = (bar_high - entry) / risk if direction == "long" else (entry - bar_low) / risk
                    rr_last = (price - entry) / risk   if direction == "long" else (entry - price) / risk
                    rr_need = float(getattr(self.config, "arm_rr", 0.5))
                    basis   = str(getattr(self.config, "arm_rr_basis", "extremum")).lower()
                    rr_now  = rr_ext if basis == "extremум" else rr_last
                    rr_alt  = rr_last if basis == "extremум" else rr_ext

                    if rr_now >= rr_need or rr_alt >= rr_need:
                        armed = True
                        position["armed"] = True
                        if self.state:
                            self.state.set_position(position)
                        print(f"[ARM] enabled at ≥{rr_need:.2f}R (basis={basis}, rr_now={rr_now:.3f}, rr_alt={rr_alt:.3f})")

            if not armed:
                return

            # Anchor (экстремум)
            anchor = float(position.get("trail_anchor") or entry)
            anchor = max(anchor, bar_high) if direction == "long" else min(anchor, bar_low)
            if anchor != position.get("trail_anchor"):
                position["trail_anchor"] = anchor
                if self.state:
                    self.state.set_position(position)

            # % трейл от entry + offset
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
        """Апдейт SL (long=floor, short=ceil) + sync state. Механика прежняя."""
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
                ok = self.api.update_position_stop_loss(self.symbol, new_sl)
                if ok:
                    position["stop_loss"] = float(new_sl)
                    if self.state:
                        self.state.set_position(position)
                    print(f"[TRAIL] SL -> {new_sl:.4f}")
                    return True

            if hasattr(self.api, "modify_order"):
                _ = self.api.modify_order(symbol=position.get("symbol", self.symbol), stop_loss=float(new_sl))
                position["stop_loss"] = float(new_sl)
                if self.state:
                    self.state.set_position(position)
                print(f"[TRAIL] SL -> {new_sl:.4f}")
                return True

            # фолбэк
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
        """Вызов на каждом тике (лайв-эмуляция)."""
        try:
            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[process_trailing] {e}")

    def run_cycle(self):
        """На закрытии 15m: если позиции нет — ищем вход; иначе трейлим."""
        try:
            if not self.candles_15m:
                return

            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)
                return

            if len(self.candles_15m) < int(getattr(self.config, "sfp_len", 2)) + 2:
                return

            ts = int(self.candles_15m[0]["timestamp"])
            if (not self._is_in_backtest_window_utc(ts)) or (not self._is_after_cycle_start(ts)):
                return

            if self._detect_bull_sfp() and self.can_enter_long:
                self._process_long_entry()
            elif self._detect_bear_sfp() and self.can_enter_short:
                self._process_short_entry()
        except Exception as e:
            print(f"[run_cycle] {e}")

    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        """Ограничение периода bt через days_back (UTC-полночь)."""
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=int(getattr(self.config, "days_back", 30)))
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

    def _is_after_cycle_start(self, current_timestamp: int) -> bool:
        """Фильтр «isActive»: если config.start_time_ms задан, требуем time >= start_time."""
        if self.start_time_ms is None:
            return True
        return int(current_timestamp) >= int(self.start_time_ms)

    def _update_equity(self):
        """Синхронизация equity из биржи (если API поддерживает)."""
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
