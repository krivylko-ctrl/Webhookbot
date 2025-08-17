import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from utils import price_round, qty_round
from database import Database


class KWINStrategy:
    """Основная логика стратегии KWIN (максимально приближена к Pine)."""

    def __init__(self, config: Config, bybit_api, state_manager: StateManager, db: Database):
        self.config = config
        self.api = bybit_api
        self.state = state_manager
        self.db = db
        self.trail_engine = TrailEngine(config, state_manager, bybit_api)
        self.analytics = TradingAnalytics()

        # История (newest-first, индекс 0 — текущий закрытый бар)
        self.candles_15m: List[Dict] = []
        self.candles_1m: List[Dict] = []
        self.candles_1h: List[Dict] = []
        self.last_processed_time = None
        self.last_candle_close_15m = None

        # бэктест: последний закрытый 15м бар
        self._bt_last_bar = None

        # Состояние входов (сбрасываем ТОЛЬКО на новом 15м баре)
        self.can_enter_long = True
        self.can_enter_short = True
        self.entered_this_bar = False  # ⚑ анти-спам: не более 1 входа на бар

        # Инструмент и фильтры
        self.symbol = self.config.symbol
        self.tick_size = 0.01
        self.qty_step = 0.01
        self.min_order_qty = 0.01
        self._init_instrument_info()

        # Нормализация close_back_pct в [0..1]
        if getattr(self.config, "close_back_pct", 0) > 1.0:
            self.config.close_back_pct = self.config.close_back_pct / 100.0
        elif self.config.close_back_pct < 0.0:
            self.config.close_back_pct = 0.0

        # === Drawdown / Risk Halt ===
        self.peak_equity: Optional[float] = None
        self.halted: bool = False  # при True — не входим
        # defaults при отсутствии в Config
        self.max_dd_pct = float(getattr(self.config, "max_dd_pct", 0.0))      # 0 — выключено
        self.max_dd_usd = float(getattr(self.config, "max_dd_usd", 0.0))      # 0 — выключено
        self.halt_on_dd = bool(getattr(self.config, "halt_on_dd", True))
        self.flat_on_halt = bool(getattr(self.config, "flat_on_halt", False))

        # Версионирование
        self.strategy_version = "2.3.0-A(Pine-like+DD)"

    # ==================== Pine-like helpers ====================

    @staticmethod
    def _ta_pivotlow(series_newest_first: List[float], left: int, right: int) -> bool:
        if len(series_newest_first) < left + right + 2:
            return False
        x = series_newest_first
        pivot = x[1]
        right_ok = pivot < min(x[0:1])  # с правой стороны bar[0]
        left_block = x[2:2 + left] if left > 0 else []
        left_ok = (len(left_block) == left) and (pivot < min(left_block)) if left > 0 else True
        return right_ok and left_ok

    @staticmethod
    def _ta_pivothigh(series_newest_first: List[float], left: int, right: int) -> bool:
        if len(series_newest_first) < left + right + 2:
            return False
        x = series_newest_first
        pivot = x[1]
        right_ok = pivot > max(x[0:1])  # bar[0]
        left_block = x[2:2 + left] if left > 0 else []
        left_ok = (len(left_block) == left) and (pivot > max(left_block)) if left > 0 else True
        return right_ok and left_ok

    def _series(self, field: str, tf: str) -> List[float]:
        if tf in ("15", "15m"):
            src = self.candles_15m
        elif tf in ("60", "60m", "1h"):
            src = self.candles_1h
        else:
            src = self.candles_1m
        if not src:
            return []
        return [float(b[field]) for b in src if field in b]

    def _normalize_klines(self, raw):
        if not raw:
            return []
        out = []
        for k in raw:
            ts = k.get("timestamp") or k.get("start") or k.get("open_time") or k.get("t") or 0
            if ts and ts < 10_000_000_000:
                ts = int(ts) * 1000
            out.append({
                "timestamp": int(ts),
                "open":  float(k.get("open",  k.get("o", 0.0))),
                "high":  float(k.get("high",  k.get("h", 0.0))),
                "low":   float(k.get("low",   k.get("l", 0.0))),
                "close": float(k.get("close", k.get("c", 0.0))),
            })
        out.sort(key=lambda x: x["timestamp"], reverse=True)
        return out

    # -------------------- Биржевые фильтры --------------------

    def _init_instrument_info(self):
        try:
            if self.api:
                if hasattr(self.api, "set_market_type") and hasattr(self.config, "market_type"):
                    self.api.set_market_type(self.config.market_type)
                if hasattr(self.api, "get_instruments_info"):
                    info = self.api.get_instruments_info(self.symbol) or {}
                    pf = info.get("priceFilter") or {}
                    lf = info.get("lotSizeFilter") or {}
                    ts = float(pf.get("tickSize") or 0.0)
                    qs = float(lf.get("qtyStep") or 0.0)
                    mq = float(lf.get("minOrderQty") or 0.0)
                    if ts > 0: self.tick_size = ts
                    if qs > 0: self.qty_step = qs
                    if mq > 0: self.min_order_qty = mq
        except Exception as e:
            print(f"Error initializing instrument info: {e}")

        if not self.tick_size or self.tick_size <= 0:
            self.tick_size = 0.01
        if not self.qty_step or self.qty_step <= 0:
            self.qty_step = 0.01
        if not self.min_order_qty or self.min_order_qty <= 0:
            self.min_order_qty = 0.01

        if hasattr(self.config, "min_order_qty"):
            self.config.min_order_qty = self.min_order_qty
            self.config.qty_step = self.qty_step

    # -------------------- Интеграция с рынком --------------------

    def on_bar_close_15m(self, candle: Dict):
        """ТОЛЬКО здесь обрабатываем закрытие 15м бара (как в Pine)."""
        try:
            norm = self._normalize_klines([candle])
            if not norm:
                return
            c = norm[0]
            self._bt_last_bar = c
            self.candles_15m.insert(0, c)
            if len(self.candles_15m) > 200:
                self.candles_15m = self.candles_15m[:200]

            current_bar_time = c["timestamp"]
            if self.last_candle_close_15m != current_bar_time:
                self.can_enter_long = True
                self.can_enter_short = True
                self.entered_this_bar = False
                self.last_candle_close_15m = current_bar_time

            # защита от двойного вызова на том же баре
            if self.entered_this_bar:
                return

            self.run_cycle()
        except Exception as e:
            print(f"Error in on_bar_close_15m: {e}")

    def on_bar_close_60m(self, candle: Dict):
        try:
            n = self._normalize_klines([candle])
            if n:
                self.candles_1h.insert(0, n[0])
                if len(self.candles_1h) > 100:
                    self.candles_1h = self.candles_1h[:100]
        except Exception as e:
            print(f"Error in on_bar_close_60m: {e}")

    def on_bar_close_1m(self, candle: Dict):
        # можно дергать трейл чаще, но ВХОДОВ тут нет
        pass

    def update_candles(self):
        try:
            if not self.api:
                return
            kl_15 = self.api.get_klines(self.symbol, "15", 100) or []
            self.candles_15m = self._normalize_klines(kl_15)
            kl_1 = self.api.get_klines(self.symbol, "1", 10) or []
            self.candles_1m = self._normalize_klines(kl_1)
        except Exception as e:
            print(f"Error updating candles: {e}")

    # -------------------- DD/risk контроль --------------------

    def _refresh_equity_and_dd(self):
        """Обновляем equity, ведём peak и проверяем DD-лимиты."""
        self._update_equity()
        eq = float(self.state.get_equity() or 0.0)
        if eq <= 0:
            return
        if self.peak_equity is None:
            self.peak_equity = eq
        else:
            self.peak_equity = max(self.peak_equity, eq)

        # абсолютный и процентный DD от пика
        dd_usd = self.peak_equity - eq
        dd_pct = (dd_usd / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        dd_hit_pct = (self.max_dd_pct > 0) and (dd_pct >= self.max_dd_pct)
        dd_hit_usd = (self.max_dd_usd > 0) and (dd_usd >= self.max_dd_usd)
        if self.halt_on_dd and (dd_hit_pct or dd_hit_usd):
            if not self.halted:
                print(f"[HALT] Max DD hit: dd_pct={dd_pct:.2f}% dd_usd={dd_usd:.2f} "
                      f"(limits pct={self.max_dd_pct}, usd={self.max_dd_usd})")
            self.halted = True
            if self.flat_on_halt:
                pos = self.state.get_current_position()
                if pos and pos.get("status") == "open" and self.api:
                    # закрываем по рынку
                    side = "sell" if pos.get("direction") == "long" else "buy"
                    qty = float(pos.get("size") or 0.0)
                    if qty > 0:
                        try:
                            self.api.place_order(symbol=self.symbol, side=side, order_type="market", qty=qty)
                            pos["status"] = "closed"
                            self.state.set_position(pos)
                            print("[HALT] Position force-closed due to DD limit.")
                        except Exception as e:
                            print(f"Error force closing on HALT: {e}")

    # -------------------- Основной цикл --------------------

    def run_cycle(self):
        try:
            if not self.candles_15m:
                return

            # обновляем equity и проверяем DD-лимиты на каждом цикле
            self._refresh_equity_and_dd()

            position = self.state.get_current_position()
            if position and position.get("status") == "open":
                self._update_smart_trailing(position)
                return

            # если стратегия остановлена из-за DD — не открываем новые сделки
            if self.halted:
                return

            self.on_bar_close()
        except Exception as e:
            print(f"Error in run_cycle: {e}")

    def on_bar_close(self):
        """Детекция сигналов + входы на закрытии 15м (строго Pine-like)."""
        sfp_len = int(getattr(self.config, "sfp_len", 2) or 2)
        if len(self.candles_15m) < sfp_len + 2:
            return

        current_ts = self.candles_15m[0]["timestamp"]
        if not self._is_in_backtest_window_utc(current_ts):
            return

        bull_sfp = self._detect_bull_sfp()
        bear_sfp = self._detect_bear_sfp()

        if bull_sfp and self.can_enter_long and not self.entered_this_bar:
            self._process_long_entry()
        if bear_sfp and self.can_enter_short and not self.entered_this_bar:
            self._process_short_entry()

    # -------------------- SFP (точная имитация Pine) --------------------

    def _detect_bull_sfp(self) -> bool:
        sfpLen = int(getattr(self.config, "sfp_len", 2) or 2)
        lows15 = self._series("low", "15")
        if len(lows15) < sfpLen + 2:
            return False
        if not self._ta_pivotlow(lows15, sfpLen, 1):
            return False
        prev_ref_low = lows15[sfpLen]
        cur_open = self.candles_15m[0]["open"]
        cur_close = self.candles_15m[0]["close"]
        cur_low = self.candles_15m[0]["low"]
        cond = (cur_low < prev_ref_low) and (cur_open > prev_ref_low) and (cur_close > prev_ref_low)
        if not cond:
            return False
        if getattr(self.config, "use_sfp_quality", True):
            return self._check_bull_sfp_quality_new(self.candles_15m[0], prev_ref_low)
        return True

    def _detect_bear_sfp(self) -> bool:
        sfpLen = int(getattr(self.config, "sfp_len", 2) or 2)
        highs15 = self._series("high", "15")
        if len(highs15) < sfpLen + 2:
            return False
        if not self._ta_pivothigh(highs15, sfpLen, 1):
            return False
        prev_ref_high = highs15[sfpLen]
        cur_open = self.candles_15m[0]["open"]
        cur_close = self.candles_15m[0]["close"]
        cur_high = self.candles_15m[0]["high"]
        cond = (cur_high > prev_ref_high) and (cur_open < prev_ref_high) and (cur_close < prev_ref_high)
        if not cond:
            return False
        if getattr(self.config, "use_sfp_quality", True):
            return self._check_bear_sfp_quality_new(self.candles_15m[0], prev_ref_high)
        return True

    def _check_bull_sfp_quality_new(self, current_bar: Dict, prev_ref_low: float) -> bool:
        wick_depth = prev_ref_low - current_bar["low"]
        min_tick = float(self.tick_size) if self.tick_size else 0.01
        if (wick_depth / min_tick) < float(getattr(self.config, "wick_min_ticks", 0)):
            return False
        close_back = current_bar["close"] - current_bar["low"]
        required_close_back = wick_depth * float(getattr(self.config, "close_back_pct", 1.0))
        return close_back >= required_close_back

    def _check_bear_sfp_quality_new(self, current_bar: Dict, prev_ref_high: float) -> bool:
        wick_depth = current_bar["high"] - prev_ref_high
        min_tick = float(self.tick_size) if self.tick_size else 0.01
        if (wick_depth / min_tick) < float(getattr(self.config, "wick_min_ticks", 0)):
            return False
        close_back = current_bar["high"] - current_bar["close"]
        required_close_back = wick_depth * float(getattr(self.config, "close_back_pct", 1.0))
        return close_back >= required_close_back

    # -------------------- Входы (1:1 Pine) --------------------

    def _process_long_entry(self):
        try:
            if self.entered_this_bar or self.halted:
                return

            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]
            stop_loss = float(prev["low"])

            if bool(getattr(self.config, "use_stop_guards", False)):
                max_stop_pct = float(getattr(self.config, "max_stop_pct", 0.08))
                stop_size = entry_price - stop_loss
                if stop_size <= 0 or stop_size > entry_price * max_stop_pct:
                    print(f"[GUARD] Aborting long: abnormal SL ({stop_loss}) vs entry ({entry_price})")
                    return
            else:
                stop_size = entry_price - stop_loss
                if stop_size <= 0:
                    return

            quantity = self._calculate_position_size(entry_price, stop_loss, "long")
            if not quantity:
                return

            take_profit = entry_price + stop_size * float(self.config.risk_reward)

            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return
            if not self.api:
                return

            res = self.api.place_order(
                symbol=self.symbol,
                side="buy",
                order_type="market",
                qty=quantity,
                stop_loss=price_round(stop_loss, self.tick_size),
            )
            if res:
                entry_ts = (self._bt_last_bar["timestamp"] if self._bt_last_bar else None)
                entry_dt = (datetime.fromtimestamp(entry_ts/1000, tz=timezone.utc)
                            if entry_ts else datetime.now(timezone.utc))

                trade = {
                    "symbol": self.symbol,
                    "direction": "long",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "quantity": quantity,
                    "entry_time": entry_dt,
                    "status": "open",
                }
                self.db.save_trade(trade)
                self.state.set_position({
                    "symbol": self.symbol,
                    "direction": "long",
                    "size": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "armed": (not getattr(self.config, "use_arm_after_rr", True)),
                    "peak": entry_price,
                    "trough": None,
                    "status": "open",
                })
                self.can_enter_long = False
                self.can_enter_short = False
                self.entered_this_bar = True
                print(f"[{self.symbol}] LONG entry={entry_price:.2f} prev_low={prev['low']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing long entry: {e}")

    def _process_short_entry(self):
        try:
            if self.entered_this_bar or self.halted:
                return

            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]
            stop_loss = float(prev["high"])

            if bool(getattr(self.config, "use_stop_guards", False)):
                max_stop_pct = float(getattr(self.config, "max_stop_pct", 0.08))
                stop_size = stop_loss - entry_price
                if stop_size <= 0 or stop_size > entry_price * max_stop_pct:
                    print(f"[GUARD] Aborting short: abnormal SL ({stop_loss}) vs entry ({entry_price})")
                    return
            else:
                stop_size = stop_loss - entry_price
                if stop_size <= 0:
                    return

            quantity = self._calculate_position_size(entry_price, stop_loss, "short")
            if not quantity:
                return

            take_profit = entry_price - stop_size * float(self.config.risk_reward)

            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return
            if not self.api:
                return

            res = self.api.place_order(
                symbol=self.symbol,
                side="sell",
                order_type="market",
                qty=quantity,
                stop_loss=price_round(stop_loss, self.tick_size),
            )
            if res:
                entry_ts = (self._bt_last_bar["timestamp"] if self._bt_last_bar else None)
                entry_dt = (datetime.fromtimestamp(entry_ts/1000, tz=timezone.utc)
                            if entry_ts else datetime.now(timezone.utc))

                trade = {
                    "symbol": self.symbol,
                    "direction": "short",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "quantity": quantity,
                    "entry_time": entry_dt,
                    "status": "open",
                }
                self.db.save_trade(trade)
                self.state.set_position({
                    "symbol": self.symbol,
                    "direction": "short",
                    "size": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "armed": (not getattr(self.config, "use_arm_after_rr", True)),
                    "peak": None,
                    "trough": entry_price,
                    "status": "open",
                })
                self.can_enter_short = False
                self.can_enter_long = False
                self.entered_this_bar = True
                print(f"[{self.symbol}] SHORT entry={entry_price:.2f} prev_high={prev['high']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing short entry: {e}")

    # -------------------- Подсистемы --------------------

    def _get_current_price(self) -> Optional[float]:
        try:
            if self._bt_last_bar:
                return float(self._bt_last_bar["close"])
            if not self.api:
                return None
            t = self.api.get_ticker(self.symbol) or {}
            lp = t.get("last_price")
            return float(lp) if lp is not None else None
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None

    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        try:
            equity = float(self.state.get_equity() or 0.0)  # USDT
            risk_amount = equity * (float(getattr(self.config, "risk_pct", 3.0)) / 100.0)

            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None

            qty = risk_amount / stop_size
            qty = qty_round(qty, self.qty_step)

            if getattr(self.config, "limit_qty_enabled", True):
                qty = min(qty, float(getattr(self.config, "max_qty_manual", 50.0)))

            if qty < float(self.min_order_qty):
                return None
            return qty
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float,
                                        take_profit: float, quantity: float) -> bool:
        try:
            if quantity < float(getattr(self.config, "min_order_qty", self.min_order_qty)):
                return False

            if bool(getattr(self.config, "use_stop_guards", False)):
                stop_size = abs(entry_price - stop_loss)
                min_stop_size = float(self.tick_size) * 5.0
                if stop_size < min_stop_size:
                    return False

            gross = abs(take_profit - entry_price) * quantity
            fee_rate = float(getattr(self.config, "taker_fee_rate", 0.00055))
            fee_in = entry_price * quantity * fee_rate
            fee_out = take_profit * quantity * fee_rate
            net = gross - (fee_in + fee_out)
            return net >= float(getattr(self.config, "min_net_profit", 1.2))
        except Exception as e:
            print(f"Error validating position: {e}")
            return False

    def _is_in_backtest_window_utc(self, current_timestamp_ms: int) -> bool:
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = utc_midnight - timedelta(days=int(getattr(self.config, "days_back", 60)))
        current_dt = datetime.utcfromtimestamp(current_timestamp_ms / 1000.0)
        return current_dt >= start_dt.replace(tzinfo=None)

    # -------------------- Smart Trailing (Arm + Percent+Offset + Bar) --------------------

    def _update_smart_trailing(self, position: Dict):
        try:
            if not getattr(self.config, "enable_smart_trail", True):
                return

            direction = position.get("direction")
            entry_price = float(position.get("entry_price", 0) or 0)
            current_sl = float(position.get("stop_loss", 0) or 0)
            if not direction or entry_price <= 0 or current_sl <= 0:
                return

            current_price = self._get_current_price()
            if current_price is None:
                return

            # === 1) Экстремумы со входа (для percent+offset трейла) — всегда ===
            if direction == "long":
                peak = float(position.get("peak") or entry_price)
                if self.candles_15m:
                    peak = max(peak, float(self.candles_15m[0]["high"]))
                position["peak"] = peak
            else:
                trough = float(position.get("trough") or entry_price)
                if self.candles_15m:
                    trough = min(trough, float(self.candles_15m[0]["low"]))
                position["trough"] = trough
            self.state.set_position(position)

            # === 2) ПРОЦЕНТНЫЙ ТРЕЙЛ (с offset) — до ARM, как в Pine ===
            percent_sl = self._calculate_percentage_trailing_stop(
                direction, current_price, current_sl, position
            )
            new_sl_after_percent = current_sl
            if percent_sl is not None:
                if direction == "long":
                    new_sl_after_percent = max(current_sl, percent_sl)
                else:
                    new_sl_after_percent = min(current_sl, percent_sl)

            # === 3) ARM по RR — только после взвода разрешаем баровый трейл ===
            armed = bool(position.get("armed", not getattr(self.config, "use_arm_after_rr", True)))
            if not armed and getattr(self.config, "use_arm_after_rr", True):
                arm_rr = float(getattr(self.config, "arm_rr", 0.5))
                if direction == "long":
                    moved = current_price - entry_price
                    need = (entry_price - current_sl) * arm_rr
                    armed = moved >= need
                else:
                    moved = entry_price - current_price
                    need = (current_sl - entry_price) * arm_rr
                    armed = moved >= need
                if armed:
                    position["armed"] = True
                    self.state.set_position(position)

            candidate_sl = new_sl_after_percent

            # === 4) БАРОВЫЙ — после ARM; берём «жёстче» ===
            if armed and getattr(self.config, "use_bar_trail", True):
                bar_sl = self._calculate_bar_trailing_stop(direction, current_sl=candidate_sl)
                if bar_sl is not None:
                    if direction == "long":
                        candidate_sl = max(candidate_sl, bar_sl)
                    else:
                        candidate_sl = min(candidate_sl, bar_sl)

            # === 5) Применяем, если улучшили ===
            if (direction == "long" and candidate_sl > current_sl) or (direction == "short" and candidate_sl < current_sl):
                self._update_stop_loss(position, price_round(candidate_sl, self.tick_size))

        except Exception as e:
            print(f"Error in smart trailing: {e}")

    def _calculate_bar_trailing_stop(self, direction: str, current_sl: float) -> Optional[float]:
        try:
            lookback = int(getattr(self.config, "trail_lookback", 50) or 50)
            buf_ticks = int(getattr(self.config, "trail_buf_ticks", 0) or 0)
            buf = float(self.tick_size) * buf_ticks

            if len(self.candles_15m) < lookback + 1:
                return current_sl

            hist = self.candles_15m[1:lookback + 1]  # без текущего
            if direction == "long":
                lb_low = min(b["low"] for b in hist)
                return max(lb_low - buf, current_sl)
            else:
                lb_high = max(b["high"] for b in hist)
                return min(lb_high + buf, current_sl)
        except Exception as e:
            print(f"Error calculating bar trailing stop: {e}")
        return current_sl

    def _calculate_percentage_trailing_stop(self, direction: str, current_price: float, current_sl: float, position: Dict) -> Optional[float]:
        """Эмуляция strategy.exit(..., trail_points=entry*perc, trail_offset=entry*offset) с учётом пика/дна."""
        try:
            entry_price = float(position.get("entry_price", 0) or 0)
            trail_pct  = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            offset_pct = float(getattr(self.config, "trailing_offset_perc", 0.4)) / 100.0
            if trail_pct <= 0 or entry_price <= 0:
                return current_sl

            if direction == "long":
                peak = float(position.get("peak") or entry_price)
                trail_points = entry_price * trail_pct
                offset_pts   = entry_price * offset_pct
                cand = peak - trail_points
                cand = min(cand, current_price - offset_pts)  # не ближе чем offset от цены
                return max(cand, current_sl)
            else:
                trough = float(position.get("trough") or entry_price)
                trail_points = entry_price * trail_pct
                offset_pts   = entry_price * offset_pct
                cand = trough + trail_points
                cand = max(cand, current_price + offset_pts)
                return min(cand, current_sl)
        except Exception as e:
            print(f"Error calculating percentage trailing stop: {e}")
            return current_sl

    def _update_stop_loss(self, position: Dict, new_sl: float):
        try:
            if not self.api:
                return
            res = self.api.modify_order(symbol=position["symbol"], stop_loss=new_sl)
            if res:
                position["stop_loss"] = new_sl
                self.state.set_position(position)
                print(f"[TRAIL] SL → {new_sl}")
        except Exception as e:
            print(f"Error updating stop loss: {e}")

    # -------------------- Equity (Bybit: UNIFIED / CONTRACT) --------------------

    def _update_equity(self):
        """Берём equity только из деривативного/UNIFIED-аккаунта (SPOT отключён)."""
        try:
            if not self.api:
                return
            equity = 0.0
            if hasattr(self.api, "get_unified_balance"):
                bal = self.api.get_unified_balance() or {}
                equity = float(bal.get("totalEquity") or bal.get("equity") or 0.0)
            if equity == 0.0 and hasattr(self.api, "get_wallet_balance"):
                wallet = self.api.get_wallet_balance() or {}
                for acc in wallet.get("list", []):
                    if acc.get("accountType") in ("UNIFIED", "CONTRACT", "DERIVATIVES", "LINEAR"):
                        for coin in acc.get("coin", []):
                            if coin.get("coin") in ("USDT", "USD"):
                                equity = max(equity, float(coin.get("equity", 0)))
            if equity > 0.0:
                self.state.set_equity(equity)
                self.db.save_equity_snapshot(equity)
        except Exception as e:
            print(f"Error updating equity: {e}")
