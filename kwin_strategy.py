import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

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

        # История (новейший бар — индекс 0, как в Pine-модели)
        self.candles_15m = []
        self.candles_1m = []
        self.candles_1h = []
        self.last_processed_time = None
        self.last_candle_close_15m = None

        # Состояние входов (сбрасываем ТОЛЬКО на новом 15м баре)
        self.can_enter_long = True
        self.can_enter_short = True

        # Инструмент и фильтры
        self.symbol = self.config.symbol
        self.tick_size = 0.01
        self.qty_step = 0.01
        self.min_order_qty = 0.01
        self._init_instrument_info()  # подтянуть реальные фильтры инструмента

        # Нормализация close_back_pct в [0..1] (в Pine у тебя это доля, не проценты)
        if getattr(self.config, "close_back_pct", 0) > 1.0:
            self.config.close_back_pct = self.config.close_back_pct / 100.0
        elif self.config.close_back_pct < 0.0:
            self.config.close_back_pct = 0.0

        # Версионирование
        self.strategy_version = "2.1.1"

    # -------------------- УТИЛИТЫ ДАННЫХ --------------------

    def _normalize_klines(self, raw):
        """
        Приводим свечи к унифицированному виду и правильному порядку:
        - поля: timestamp(ms), open, high, low, close (float)
        - newest-first (индекс 0 — текущий закрытый бар)
        """
        if not raw:
            return []
        out = []
        for k in raw:
            ts = k.get("timestamp") or k.get("start") or k.get("open_time") or k.get("t") or 0
            # если пришли секунды — переводим в мс
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
        """Тянем tickSize/qtyStep/minOrderQty с биржи. Жёсткие фолбэки на всякий случай."""
        try:
            if self.api:
                if hasattr(self.api, "set_market_type") and hasattr(self.config, "market_type"):
                    # ожидается 'linear' / 'contract' и т.п.
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

        # Фолбэки, если биржа вернула мусор
        if not self.tick_size or self.tick_size <= 0:
            self.tick_size = 0.01
        if not self.qty_step or self.qty_step <= 0:
            self.qty_step = 0.01
        if not self.min_order_qty or self.min_order_qty <= 0:
            self.min_order_qty = 0.01

        # Проталкиваем в конфиг
        if hasattr(self.config, "min_order_qty"):
            self.config.min_order_qty = self.min_order_qty
            self.config.qty_step = self.qty_step

    # -------------------- Интеграция с рынком --------------------

    def on_bar_close_15m(self, candle: Dict):
        """ТОЛЬКО здесь триггерим обработку нового 15м бара (как в Pine)."""
        try:
            # Нормализуем одиночную свечу и ставим в начало
            norm = self._normalize_klines([candle])
            if not norm:
                return
            c = norm[0]
            self.candles_15m.insert(0, c)
            if len(self.candles_15m) > 200:
                self.candles_15m = self.candles_15m[:200]

            # Сброс флагов входа на новом баре
            current_bar_time = c["timestamp"]
            if self.last_candle_close_15m != current_bar_time:
                self.can_enter_long = True
                self.can_enter_short = True
                self.last_candle_close_15m = current_bar_time

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
        # Можно использовать для более частого трейлинга (входы НЕ даём)
        pass

    def update_candles(self):
        """Обновляем локальные свечи (без триггера входов)."""
        try:
            if not self.api:
                return
            # 15m
            kl_15 = self.api.get_klines(self.symbol, "15", 100) or []
            self.candles_15m = self._normalize_klines(kl_15)
            # 1m
            kl_1 = self.api.get_klines(self.symbol, "1", 10) or []
            self.candles_1m = self._normalize_klines(kl_1)
        except Exception as e:
            print(f"Error updating candles: {e}")

    # -------------------- Основной цикл --------------------

    def run_cycle(self):
        """Если позиция открыта — ведём smart-trailing; если flat — ищем вход 1:1 к Pine."""
        try:
            if not self.candles_15m:
                return

            position = self.state.get_current_position()
            if position and position.get("status") == "open":
                self._update_smart_trailing(position)
                return

            self.on_bar_close()  # проверка сигнала + вход
        except Exception as e:
            print(f"Error in run_cycle: {e}")

    def on_bar_close(self):
        """Детекция сигналов + входы на закрытии 15м (1:1 к Pine)."""
        sfp_len = int(getattr(self.config, "sfp_len", 2) or 2)
        if len(self.candles_15m) < sfp_len + 2:
            return

        # окно бэктеста по UTC полуночи
        current_ts = self.candles_15m[0]["timestamp"]  # ms
        if not self._is_in_backtest_window_utc(current_ts):
            return

        # SFP
        bull_sfp = self._detect_bull_sfp()
        bear_sfp = self._detect_bear_sfp()

        if bull_sfp and self.can_enter_long:
            self._process_long_entry()
        if bear_sfp and self.can_enter_short:
            self._process_short_entry()

    # -------------------- SFP (как в твоём Pine) --------------------

    def _detect_bull_sfp(self) -> bool:
        """Bull SFP: pivotlow(sfpLen,1) на баре [1] + cur low<low[sfpLen] и open/close>low[sfpLen]."""
        sfpLen = int(getattr(self.config, "sfp_len", 2) or 2)
        if len(self.candles_15m) < sfpLen + 2:
            return False

        lows = [b["low"] for b in self.candles_15m]
        if len(lows) < 2 + sfpLen:
            return False

        is_pivot_low = (lows[1] < lows[0]) and (lows[1] < min(lows[2:2 + sfpLen]))
        if not is_pivot_low:
            return False

        prev_ref_low = lows[sfpLen]
        cur = self.candles_15m[0]
        cond = (cur["low"] < prev_ref_low) and (cur["open"] > prev_ref_low) and (cur["close"] > prev_ref_low)
        if not cond:
            return False

        if getattr(self.config, "use_sfp_quality", True):
            return self._check_bull_sfp_quality_new(cur, prev_ref_low)
        return True

    def _detect_bear_sfp(self) -> bool:
        """Bear SFP: pivothigh(sfpLen,1) на баре [1] + cur high>high[sfpLen] и open/close<high[sfpLen]."""
        sfpLen = int(getattr(self.config, "sfp_len", 2) or 2)
        if len(self.candles_15m) < sfpLen + 2:
            return False

        highs = [b["high"] for b in self.candles_15m]
        if len(highs) < 2 + sfpLen:
            return False

        is_pivot_high = (highs[1] > highs[0]) and (highs[1] > max(highs[2:2 + sfpLen]))
        if not is_pivot_high:
            return False

        prev_ref_high = highs[sfpLen]
        cur = self.candles_15m[0]
        cond = (cur["high"] > prev_ref_high) and (cur["open"] < prev_ref_high) and (cur["close"] < prev_ref_high)
        if not cond:
            return False

        if getattr(self.config, "use_sfp_quality", True):
            return self._check_bear_sfp_quality_new(cur, prev_ref_high)
        return True

    def _check_bull_sfp_quality_new(self, current_bar: Dict, prev_ref_low: float) -> bool:
        """Фильтр качества bull SFP: глубина вика + close-back доля."""
        wick_depth = prev_ref_low - current_bar["low"]
        min_tick = float(self.tick_size) if self.tick_size else 0.01
        if (wick_depth / min_tick) < float(getattr(self.config, "wick_min_ticks", 0)):
            return False
        close_back = current_bar["close"] - current_bar["low"]
        required_close_back = wick_depth * float(getattr(self.config, "close_back_pct", 1.0))
        return close_back >= required_close_back

    def _check_bear_sfp_quality_new(self, current_bar: Dict, prev_ref_high: float) -> bool:
        """Фильтр качества bear SFP: глубина вика + close-back доля."""
        wick_depth = current_bar["high"] - prev_ref_high
        min_tick = float(self.tick_size) if self.tick_size else 0.01
        if (wick_depth / min_tick) < float(getattr(self.config, "wick_min_ticks", 0)):
            return False
        close_back = current_bar["high"] - current_bar["close"]
        required_close_back = wick_depth * float(getattr(self.config, "close_back_pct", 1.0))
        return close_back >= required_close_back

    # -------------------- Входы --------------------

    def _process_long_entry(self):
        try:
            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]                   # ПРЕДЫДУЩАЯ 15m свеча
            stop_loss = float(prev["low"])               # SL как в Pine: за LOW предыдущей

            # sanity-guard от битых свечей (можно убрать, если хочешь 1:1 без защиты)
            max_stop_pct = float(getattr(self.config, "max_stop_pct", 0.08))
            stop_size = entry_price - stop_loss
            if stop_size <= 0 or stop_size > entry_price * max_stop_pct:
                print(f"[GUARD] Aborting long: abnormal SL ({stop_loss}) vs entry ({entry_price})")
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
                trade = {
                    "symbol": self.symbol,
                    "direction": "long",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "quantity": quantity,
                    "entry_time": datetime.now(timezone.utc),
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
                    "status": "open",
                })
                self.can_enter_long = False
                print(f"[{self.symbol}] LONG entry={entry_price:.2f} prev_low={prev['low']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing long entry: {e}")

    def _process_short_entry(self):
        try:
            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]                   # ПРЕДЫДУЩАЯ 15m свеча
            stop_loss = float(prev["high"])              # SL как в Pine: за HIGH предыдущей

            max_stop_pct = float(getattr(self.config, "max_stop_pct", 0.08))
            stop_size = stop_loss - entry_price
            if stop_size <= 0 or stop_size > entry_price * max_stop_pct:
                print(f"[GUARD] Aborting short: abnormal SL ({stop_loss}) vs entry ({entry_price})")
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
                trade = {
                    "symbol": self.symbol,
                    "direction": "short",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "quantity": quantity,
                    "entry_time": datetime.now(timezone.utc),
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
                    "status": "open",
                })
                self.can_enter_short = False
                print(f"[{self.symbol}] SHORT entry={entry_price:.2f} prev_high={prev['high']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing short entry: {e}")

    # -------------------- Подсистемы --------------------

    def _get_current_price(self) -> Optional[float]:
        try:
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
            equity = float(self.state.get_equity() or 0.0)
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

            stop_size = abs(entry_price - stop_loss)
            min_stop_size = float(self.tick_size) * 5.0  # минимум 5 тиков
            if stop_size < min_stop_size:
                return False

            gross = abs(take_profit - entry_price) * quantity
            fee_in = entry_price * quantity * float(getattr(self.config, "taker_fee_rate", 0.00055))
            fee_out = take_profit * quantity * float(getattr(self.config, "taker_fee_rate", 0.00055))
            net = gross - (fee_in + fee_out)
            return net >= float(getattr(self.config, "min_net_profit", 1.2))
        except Exception as e:
            print(f"Error validating position: {e}")
            return False

    def _is_in_backtest_window_utc(self, current_timestamp_ms: int) -> bool:
        """Совпадение с Pine: от полуночи UTC минус N дней."""
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = utc_midnight - timedelta(days=int(getattr(self.config, "days_back", 60)))
        current_dt = datetime.utcfromtimestamp(current_timestamp_ms / 1000.0)
        return current_dt >= start_dt.replace(tzinfo=None)

    # -------------------- Smart Trailing (Arm + Bar/Percent) --------------------

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

            if not armed:
                return

            if getattr(self.config, "use_bar_trail", True):
                new_sl = self._calculate_bar_trailing_stop(direction, current_sl)
            else:
                new_sl = self._calculate_percentage_trailing_stop(direction, current_price, current_sl)

            if new_sl is None:
                return

            if (direction == "long" and new_sl > current_sl) or (direction == "short" and new_sl < current_sl):
                self._update_stop_loss(position, price_round(new_sl, self.tick_size))
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

    def _calculate_percentage_trailing_stop(self, direction: str, current_price: float, current_sl: float) -> Optional[float]:
        try:
            trail_pct = float(getattr(self.config, "trailing_perc", 0.5)) / 100.0
            if trail_pct <= 0:
                return current_sl
            if direction == "long":
                return max(current_price - current_price * trail_pct, current_sl)
            else:
                return min(current_price + current_price * trail_pct, current_sl)
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

            # Предпочтительно: единый метод для UNIFIED
            if hasattr(self.api, "get_unified_balance"):
                bal = self.api.get_unified_balance() or {}
                equity = float(bal.get("totalEquity") or bal.get("equity") or 0.0)

            # Альтернативно: общий кошелёк, но учитываем только деривативные аккаунты
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
