# kwin_strategy.py
from __future__ import annotations

import math
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
    """Основная логика стратегии KWIN (ARM + Smart trailing)."""

    # ---------- ИНИЦИАЛИЗАЦИЯ ----------

    def __init__(
        self,
        config: Config,
        api: Any = None,
        state_manager: StateManager | None = None,
        db: Database | None = None,
        **kwargs: Any,
    ) -> None:
        # обратная совместимость с bybit_api=...
        if api is None and "bybit_api" in kwargs:
            api = kwargs.get("bybit_api")

        self.config = config
        self.api = api
        self.state = state_manager
        self.db = db
        # смарт-трейл движок: поддерживаем несколько сигнатур
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
        self.candles_15m: List[Dict] = []
        self.candles_1m: List[Dict] = []
        self.candles_1h: List[Dict] = []
        self.last_processed_bar_ts: int = 0
        self.can_enter_long = True
        self.can_enter_short = True

        # инструменты/шаги
        self.symbol = str(getattr(config, "symbol", "ETHUSDT"))
        self.tick_size = float(getattr(config, "tick_size", 0.01))
        self.qty_step = float(getattr(config, "qty_step", 0.01))
        self.min_order_qty = float(getattr(config, "min_order_qty", 0.01))

        # на всякий случай подтянуть фильтры инструмента из API
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

    # ---------- ВСПОМОГАТЕЛЬНОЕ ----------

    def _align_15m_ms(self, ts_ms: int) -> int:
        return (int(ts_ms) // 900_000) * 900_000

    def _ensure_desc(self, arr: List[Dict], key: str = "timestamp"):
        if arr and arr[0].get(key) is not None:
            arr.sort(key=lambda x: x.get(key, 0), reverse=True)

    def _current_bar_ts_ms(self) -> int:
        return int(self.last_processed_bar_ts or 0)

    def _init_instrument_info(self):
        """Попробовать подтянуть tick_size/qty_step/min_order_qty из API (если доступно)."""
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
        # синхронизируем в конфиг, чтобы UI видел актуальные шаги
        self.config.tick_size = self.tick_size or 0.01
        self.config.qty_step = self.qty_step or 0.01
        self.config.min_order_qty = self.min_order_qty or 0.01

    # ---------- ОБРАБОТКА БАРОВ ----------

    def on_bar_close_15m(self, candle: Dict):
        """Обработка закрытого 15m бара (входы, первичный трейлинг)."""
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
        """Интрабар трейлинг (подтягиваем SL по минуткам)."""
        try:
            self.candles_1m.insert(0, candle)
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            if len(self.candles_1m) > lim:
                self.candles_1m = self.candles_1m[:lim]
            self._ensure_desc(self.candles_1m)

            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[on_bar_close_1m] {e}")

    # ---------- ПУЛЛИНГ СВЕЧЕЙ (если надо) ----------

    def update_candles(self):
        """Разовый пулл свечей; заботится о единичной обработке закрытого 15m бара."""
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

            # единичная обработка закрытого 15m
            if self.candles_15m:
                ts = int(self.candles_15m[0].get("timestamp") or 0)
                aligned = self._align_15m_ms(ts)
                if aligned != self.last_processed_bar_ts:
                    self.last_processed_bar_ts = aligned
                    self.can_enter_long = True
                    self.can_enter_short = True
                    self.on_bar_close()
                else:
                    pos = self.state.get_current_position() if self.state else None
                    if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                        self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[update_candles] {e}")

    # ---------- ЛОГИКА ВХОДОВ ----------

    def on_bar_close(self):
        if len(self.candles_15m) < int(getattr(self.config, "sfp_len", 2)) + 2:
            return

        bull = self._detect_bull_sfp()
        bear = self._detect_bear_sfp()

        ts = int(self.candles_15m[0]["timestamp"])
        if not self._is_in_backtest_window_utc(ts):
            return

        current_position = self.state.get_current_position() if self.state else None
        if current_position and current_position.get("status") == "open":
            return

        if bull and self.can_enter_long:
            self._process_long_entry()
        elif bear and self.can_enter_short:
            self._process_short_entry()

    # --- SFP (pine-exact) ---

    def _is_prev_pivot_low(self, left: int, right: int = 1) -> bool:
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        lows = [float(self.candles_15m[i]["low"]) for i in range(0, 1 + left + 1)]
        pivot = float(self.candles_15m[1]["low"])
        return pivot == min(lows)

    def _is_prev_pivot_high(self, left: int, right: int = 1) -> bool:
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        highs = [float(self.candles_15m[i]["high"]) for i in range(0, 1 + left + 1)]
        pivot = float(self.candles_15m[1]["high"])
        return pivot == max(highs)

    def _detect_bull_sfp(self) -> bool:
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False
        curr = self.candles_15m[0]
        ref_low = float(self.candles_15m[L]["low"])
        cond_pivot = self._is_prev_pivot_low(L, 1)
        cond_break = float(curr["low"]) < ref_low
        cond_close = float(curr["open"]) > ref_low and float(curr["close"]) > ref_low
        if cond_pivot and cond_break and cond_close:
            return self._check_bull_sfp_quality(curr, {"low": ref_low}) if getattr(self.config, "use_sfp_quality", True) else True
        return False

    def _detect_bear_sfp(self) -> bool:
        L = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (L + 2):
            return False
        curr = self.candles_15m[0]
        ref_high = float(self.candles_15m[L]["high"])
        cond_pivot = self._is_prev_pivot_high(L, 1)
        cond_break = float(curr["high"]) > ref_high
        cond_close = float(curr["open"]) < ref_high and float(curr["close"]) < ref_high
        if cond_pivot and cond_break and cond_close:
            return self._check_bear_sfp_quality(curr, {"high": ref_high}) if getattr(self.config, "use_sfp_quality", True) else True
        return False

    def _check_bull_sfp_quality(self, current: dict, pivot: dict) -> bool:
        ref_low = float(pivot["low"])
        low = float(current["low"])
        close = float(current["close"])
        wick_depth = max(ref_low - low, 0.0)
        m_tick = float(self.tick_size or 0.01)
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False
        cb = float(getattr(self.config, "close_back_pct", 1.0))
        required = wick_depth * cb
        return (close - low) >= required

    def _check_bear_sfp_quality(self, current: dict, pivot: dict) -> bool:
        ref_high = float(pivot["high"])
        high = float(current["high"])
        close = float(current["close"])
        wick_depth = max(high - ref_high, 0.0)
        m_tick = float(self.tick_size or 0.01)
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False
        cb = float(getattr(self.config, "close_back_pct", 1.0))
        required = wick_depth * cb
        return (high - close) >= required

    # ---------- ОРДЕРА / ВХОД ----------

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
        sl_send = round_price(stop_loss, self.tick_size) if stop_loss is not None else None
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
        try:
            equity = self.state.get_equity() if self.state else None
            if equity is None or equity <= 0:
                return None
            risk_amount = equity * (float(self.config.risk_pct) / 100.0)
            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None
            quantity = risk_amount / stop_size
            quantity = round_qty(quantity, self.qty_step)
            if getattr(self.config, "limit_qty_enabled", False):
                quantity = min(quantity, getattr(self.config, "max_qty_manual", quantity))
            if quantity < float(self.min_order_qty):
                return None
            return float(quantity)
        except Exception as e:
            print(f"[calc_position_size] {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float,
                                        take_profit: Optional[float], quantity: float) -> bool:
        """Если TP выключен — не требуем min_net_profit, оставляем базовые проверки."""
        try:
            if quantity is None or float(quantity) <= 0:
                return False
            if float(quantity) < float(getattr(self.config, "min_order_qty", 0.01)):
                return False

            if not getattr(self.config, "use_take_profit", True):
                # Без TP: проверяем только лоты/мин. размер.
                return True

            # С TP: проверяем ожидаемую чистую прибыль
            taker = float(getattr(self.config, "taker_fee_rate", 0.00055))
            if take_profit is None:
                return False
            gross = abs(float(take_profit) - float(entry_price)) * float(quantity)
            fees = float(entry_price) * float(quantity) * taker * 2.0
            net = gross - fees
            min_net = float(getattr(self.config, "min_net_profit", 1.2))
            return net >= min_net
        except Exception as e:
            print(f"[validate_position] {e}")
            return False

    def _process_long_entry(self):
        try:
            price = self._get_current_price()
            if not price or len(self.candles_15m) < 2:
                return

            raw_sl = float(self.candles_15m[1]["low"])

            # Pine-точное округление
            entry = round_to_tick(float(price), self.tick_size)
            sl    = floor_to_tick(raw_sl, self.tick_size)
            stop_size = entry - sl
            if stop_size <= 0:
                return

            # TP рассчитываем только если включён
            tp = None
            if getattr(self.config, "use_take_profit", True):
                tp = round_to_tick(entry + stop_size * float(self.config.risk_reward), self.tick_size)

            qty = self._calculate_position_size(entry, sl, "long")
            if not qty or not self._validate_position_requirements(entry, sl, tp, qty):
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
            if tp is not None:
                trade["take_profit"] = tp
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
            }
            if tp is not None:
                pos["take_profit"] = tp
            if self.state:
                self.state.set_position(pos)

            if self.trail_engine:
                try:
                    self.trail_engine.on_entry(entry, sl, "long")
                except Exception:
                    pass

            self.can_enter_long = False
            self.can_enter_short = False
            if tp is not None:
                print(f"[ENTRY LONG] {qty} @ {entry}, SL={sl}, TP={tp}")
            else:
                print(f"[ENTRY LONG] {qty} @ {entry}, SL={sl} (TP disabled)")
        except Exception as e:
            print(f"[process_long_entry] {e}")

    def _process_short_entry(self):
        try:
            price = self._get_current_price()
            if not price or len(self.candles_15m) < 2:
                return

            raw_sl = float(self.candles_15m[1]["high"])

            # Pine-точное округление
            entry = round_to_tick(float(price), self.tick_size)
            sl    = ceil_to_tick(raw_sl, self.tick_size)
            stop_size = sl - entry
            if stop_size <= 0:
                return

            # TP рассчитываем только если включён
            tp = None
            if getattr(self.config, "use_take_profit", True):
                tp = round_to_tick(entry - stop_size * float(self.config.risk_reward), self.tick_size)

            qty = self._calculate_position_size(entry, sl, "short")
            if not qty or not self._validate_position_requirements(entry, sl, tp, qty):
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
            }
            if tp is not None:
                trade["take_profit"] = tp
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
            }
            if tp is not None:
                pos["take_profit"] = tp
            if self.state:
                self.state.set_position(pos)

            if self.trail_engine:
                try:
                    self.trail_engine.on_entry(entry, sl, "short")
                except Exception:
                    pass

            self.can_enter_short = False
            self.can_enter_long = False
            if tp is not None:
                print(f"[ENTRY SHORT] {qty} @ {entry}, SL={sl}, TP={tp}")
            else:
                print(f"[ENTRY SHORT] {qty} @ {entry}, SL={sl} (TP disabled)")
        except Exception as e:
            print(f"[process_short_entry] {e}")

    # ---------- ТРЕЙЛИНГ ----------

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
        """Smart trailing: ARM по RR, затем якорь-экстремум и % трейл от entry + offset."""
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

            # --- ARM (RR) ---
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

                    try:
                        print(f"[ARM CHECK] risk={risk:.4f} rr_ext={rr_ext:.3f} rr_last={rr_last:.3f} need={rr_need} basis={basis}")
                    except Exception:
                        pass

                    if rr_now >= rr_need or rr_alt >= rr_need:
                        armed = True
                        position["armed"] = True
                        if self.state:
                            self.state.set_position(position)
                        print(f"[ARM] enabled (hit {rr_need}R; used={basis}, rr_now={rr_now:.3f}, rr_alt={rr_alt:.3f})")

            if not armed:
                return

            # --- Anchor (экстремум) ---
            anchor = float(position.get("trail_anchor") or entry)
            anchor = max(anchor, bar_high) if direction == "long" else min(anchor, bar_low)
            if anchor != position.get("trail_anchor"):
                position["trail_anchor"] = anchor
                if self.state:
                    self.state.set_position(position)

            # --- % трейл от entry + offset ---
            trail_perc  = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            offset_perc = float(getattr(self.config, "trailing_offset_perc", 0.4)) / 100.0
            trail_dist  = entry * trail_perc
            offset_dist = entry * offset_perc

            if direction == "long":
                candidate = floor_to_tick(anchor - trail_dist - offset_dist, self.tick_size)
                if candidate > sl:
                    self._update_stop_loss(position, candidate)
                else:
                    try:
                        print(f"[TRAIL SKIP] long: candidate={candidate:.4f} <= sl={sl:.4f} (anchor={anchor:.4f}, trail%={trail_perc*100:.2f}, off%={offset_perc*100:.2f})")
                    except Exception:
                        pass
            else:
                candidate = ceil_to_tick(anchor + trail_dist + offset_dist, self.tick_size)
                if candidate < sl:
                    self._update_stop_loss(position, candidate)
                else:
                    try:
                        print(f"[TRAIL SKIP] short: candidate={candidate:.4f} >= sl={sl:.4f} (anchor={anchor:.4f}, trail%={trail_perc*100:.2f}, off%={offset_perc*100:.2f})")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[smart_trailing] {e}")

    def _update_stop_loss(self, position: Dict, new_sl: float) -> bool:
        """Апдейт SL с округлением к тику и синхронизацией state."""
        try:
            new_sl = round_price(float(new_sl), self.tick_size)
            if not self.api:
                return False

            if hasattr(self.api, "update_position_stop_loss"):
                ok = self.api.update_position_stop_loss(self.symbol, new_sl)
                if ok:
                    position["stop_loss"] = new_sl
                    if self.state:
                        self.state.set_position(position)
                    print(f"[TRAIL] SL -> {new_sl:.4f}")
                    return True

            if hasattr(self.api, "modify_order"):
                _ = self.api.modify_order(symbol=position.get("symbol", self.symbol), stop_loss=new_sl)
                position["stop_loss"] = new_sl
                if self.state:
                    self.state.set_position(position)
                print(f"[TRAIL] SL -> {new_sl:.4f}")
                return True

            # Фолбэк: локально
            position["stop_loss"] = new_sl
            if self.state:
                self.state.set_position(position)
            print(f"[TRAIL-LOCAL] SL -> {new_sl:.4f}")
            return True

        except Exception as e:
            print(f"[update_stop_loss] {e}")
            return False

    def process_trailing(self):
        """Вызов на каждом тике (лайв-эмуляция)."""
        try:
            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"[process_trailing] {e}")

    def run_cycle(self):
        """Основной цикл на закрытии 15m: если позиции нет — ищем вход; иначе трейлим."""
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
            if not self._is_in_backtest_window_utc(ts):
                return

            if self._detect_bull_sfp() and self.can_enter_long:
                self._process_long_entry()
            elif self._detect_bear_sfp() and self.can_enter_short:
                self._process_short_entry()
        except Exception as e:
            print(f"[run_cycle] {e}")

    # ---------- ПРОЧЕЕ ----------

    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        """Ограничение периода bt через days_back (UTC-полночь)."""
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=int(getattr(self.config, "days_back", 30)))
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

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
