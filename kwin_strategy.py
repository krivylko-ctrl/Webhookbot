
# kwin_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from config import Config
from state_manager import StateManager
from trail_engine import TrailEngine
from analytics import TradingAnalytics
from database import Database

# округление цен/количеств по тик-сайзу и шагу
from utils_round import round_price, round_qty


class KWINStrategy:
    """Основная логика стратегии KWIN"""

    def __init__(self, config: Config, bybit_api, state_manager: StateManager, db: Database):
        self.config = config
        self.api = bybit_api
        self.state = state_manager
        self.db = db
        self.trail_engine = TrailEngine(config, state_manager, bybit_api)
        self.analytics = TradingAnalytics()

        # Внутренние данные (crash-safe state)
        self.candles_15m: List[Dict] = []
        self.candles_1m: List[Dict] = []
        self.candles_1h: List[Dict] = []
        self.last_processed_time = None
        self.last_processed_bar_ts = 0  # Для восстановления после crash

        # Trade state
        self.entry_price: Optional[float] = None
        self.entry_sl: Optional[float] = None
        self.trade_id: Optional[int] = None
        self.armed: bool = False  # ArmRR статус

        self.strategy_version = "2.0.1"

        # Состояние входов
        self.can_enter_long = True
        self.can_enter_short = True
        self.last_candle_close_15m: Optional[int] = None

        # Инструмент
        self.symbol = self.config.symbol
        self.tick_size = 0.01
        self.qty_step = 0.01
        self.min_order_qty = 0.01

        self._init_instrument_info()

        # Нормализация close_back_pct к [0..1]
        if self.config.close_back_pct > 1.0:
            self.config.close_back_pct = self.config.close_back_pct / 100.0
        elif self.config.close_back_pct < 0.0:
            self.config.close_back_pct = 0.0

    # ====== утилиты выравнивания и порядка ======
    def _align_15m_ms(self, ts_ms: int) -> int:
        return (int(ts_ms) // 900_000) * 900_000

    def _ensure_15m_desc(self):
        if self.candles_15m and self.candles_15m[0].get("timestamp") is not None:
            self.candles_15m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    def _ensure_1m_desc(self):
        if self.candles_1m and self.candles_1m[0].get("timestamp") is not None:
            self.candles_1m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    def _current_bar_ts_ms(self) -> int:
        return int(self.last_processed_bar_ts or 0)
    # ============================================

    def _init_instrument_info(self):
        """Инициализация информации об инструменте"""
        try:
            if self.api:
                if hasattr(self.api, 'set_market_type') and hasattr(self.config, 'market_type'):
                    self.api.set_market_type(self.config.market_type)
                if hasattr(self.api, 'get_instruments_info'):
                    info = self.api.get_instruments_info(self.symbol)
                    if info:
                        # Bybit v5 формат может быть вложенным (строки); страхуемся
                        pf = info.get('priceFilter') if isinstance(info, dict) else {}
                        ls = info.get('lotSizeFilter') if isinstance(info, dict) else {}
                        if pf and pf.get('tickSize') is not None:
                            self.tick_size = float(pf['tickSize'])
                        if ls:
                            if ls.get('qtyStep') is not None:
                                self.qty_step = float(ls['qtyStep'])
                            if ls.get('minOrderQty') is not None:
                                self.min_order_qty = float(ls['minOrderQty'])
        except Exception as e:
            print(f"Error initializing instrument info: {e}")

        # Синхронизируем в конфиг
        if self.qty_step and self.qty_step > 0:
            self.config.qty_step = self.qty_step
        if self.min_order_qty and self.min_order_qty > 0:
            self.config.min_order_qty = self.min_order_qty

        # Fallback
        if not self.tick_size or self.tick_size <= 0:
            self.tick_size = 0.01
        if not self.qty_step or self.qty_step <= 0:
            self.qty_step = 0.01
        if not self.min_order_qty or self.min_order_qty <= 0:
            self.min_order_qty = 0.01

    # ==================== Приём закрытых баров ====================

    def on_bar_close_15m(self, candle: Dict):
        """ТОЧНАЯ синхронизация с Pine Script: обработка только закрытых 15м баров"""
        try:
            self.candles_15m.insert(0, candle)
            max_history = 200
            if len(self.candles_15m) > max_history:
                self.candles_15m = self.candles_15m[:max_history]

            self._ensure_15m_desc()

            current_bar_time = candle.get('start') or candle.get('open_time') or candle.get('timestamp')
            if not current_bar_time:
                return
            aligned_ts = self._align_15m_ms(int(current_bar_time))
            if self.last_processed_bar_ts == aligned_ts:
                return  # этот бар уже обработан

            # сброс флагов входа на новом закрытом баре
            self.can_enter_long = True
            self.can_enter_short = True
            self.last_candle_close_15m = aligned_ts
            self.last_processed_bar_ts = aligned_ts
            try:
                print(f"[STRATEGY] New 15m bar: {float(candle['close']):.2f} at {aligned_ts}")
            except Exception:
                pass

            self.run_cycle()
        except Exception as e:
            print(f"Error in on_bar_close_15m: {e}")

    def on_bar_close_60m(self, candle: Dict):
        try:
            self.candles_1h.insert(0, candle)
            if len(self.candles_1h) > 100:
                self.candles_1h = self.candles_1h[:100]
        except Exception as e:
            print(f"Error in on_bar_close_60m: {e}")

    def on_bar_close_1m(self, candle: Dict):
        """
        Обработка 1м баров (интрабар трейлинг). Входы — только на закрытии 15м.
        """
        try:
            self.candles_1m.insert(0, candle)
            max_history = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            if len(self.candles_1m) > max_history:
                self.candles_1m = self.candles_1m[:max_history]
            self._ensure_1m_desc()

            pos = self.state.get_current_position()
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                self._update_smart_trailing(pos)
        except Exception as e:
            print(f"Error in on_bar_close_1m: {e}")

    # ==================== Пуллинг свечей с биржи ====================

    def update_candles(self):
        try:
            if not self.api:
                return

            # 15m
            klines_15m = self.api.get_klines(self.symbol, "15", 100)
            if klines_15m:
                for k in klines_15m:
                    ts = k.get("timestamp")
                    if ts is not None and ts < 1_000_000_000_000:
                        k["timestamp"] = int(ts * 1000)
                klines_15m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                self.candles_15m = klines_15m

            # 1m
            intrabar_tf = str(getattr(self.config, "intrabar_tf", "1"))
            intrabar_lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            klines_1m = self.api.get_klines(self.symbol, intrabar_tf, min(1000, intrabar_lim))
            if klines_1m:
                for k in klines_1m:
                    ts = k.get("timestamp")
                    if ts is not None and ts < 1_000_000_000_000:
                        k["timestamp"] = int(ts * 1000)
                klines_1m.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                self.candles_1m = klines_1m[:intrabar_lim]

            # «одно срабатывание» на закрытый 15m бар
            if self.candles_15m:
                current_candle = self.candles_15m[0]
                current_timestamp = int(current_candle.get('timestamp', 0))
                if current_timestamp and current_timestamp < 1_000_000_000_000:
                    current_timestamp *= 1000
                aligned_timestamp = self._align_15m_ms(current_timestamp)

                if self.last_processed_bar_ts == aligned_timestamp:
                    # если позиция открыта — можно подтрейлить интрабар
                    pos = self.state.get_current_position()
                    if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                        self._update_smart_trailing(pos)
                    return

                self.last_processed_bar_ts = aligned_timestamp
                self.can_enter_long = True
                self.can_enter_short = True
                self.on_bar_close()
        except Exception as e:
            print(f"Error updating candles: {str(e) if e else 'Unknown error'}")

    # ==================== Логика входов ====================

    def on_bar_close(self):
        if len(self.candles_15m) < self.config.sfp_len + 2:
            return

        bull_sfp = self._detect_bull_sfp()
        bear_sfp = self._detect_bear_sfp()

        current_ts = self.candles_15m[0]['timestamp']  # ms
        if not self._is_in_backtest_window_utc(current_ts):
            return

        current_position = self.state.get_current_position()
        if current_position and current_position.get('status') == 'open':
            return

        if bull_sfp and self.can_enter_long:
            self._process_long_entry()
        elif bear_sfp and self.can_enter_short:
            self._process_short_entry()

    # ======== PINE-EXACT SFP DETECTION ========

    def _is_prev_pivot_low(self, left: int, right: int = 1) -> bool:
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        window_lows = [float(self.candles_15m[i]['low']) for i in range(0, 1 + left + 1)]
        pivot_val = float(self.candles_15m[1]['low'])
        return pivot_val == min(window_lows)

    def _is_prev_pivot_high(self, left: int, right: int = 1) -> bool:
        need = left + right + 1
        if len(self.candles_15m) < (need + 1):
            return False
        window_highs = [float(self.candles_15m[i]['high']) for i in range(0, 1 + left + 1)]
        pivot_val = float(self.candles_15m[1]['high'])
        return pivot_val == max(window_highs)

    def _detect_bull_sfp(self) -> bool:
        sfpLen = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (sfpLen + 2):
            return False

        curr = self.candles_15m[0]
        ref_low = float(self.candles_15m[sfpLen]['low'])

        cond_pivot = self._is_prev_pivot_low(sfpLen, right=1)
        cond_break = float(curr['low']) < ref_low
        cond_close = float(curr['open']) > ref_low and float(curr['close']) > ref_low

        if cond_pivot and cond_break and cond_close:
            if getattr(self.config, "use_sfp_quality", True):
                return self._check_bull_sfp_quality_new(curr, {"low": ref_low})
            return True
        return False

    def _detect_bear_sfp(self) -> bool:
        sfpLen = int(getattr(self.config, "sfp_len", 2))
        if len(self.candles_15m) < (sfpLen + 2):
            return False

        curr = self.candles_15m[0]
        ref_high = float(self.candles_15m[sfpLen]['high'])

        cond_pivot = self._is_prev_pivot_high(sfpLen, right=1)
        cond_break = float(curr['high']) > ref_high
        cond_close = float(curr['open']) < ref_high and float(curr['close']) < ref_high

        if cond_pivot and cond_break and cond_close:
            if getattr(self.config, "use_sfp_quality", True):
                return self._check_bear_sfp_quality_new(curr, {"high": ref_high})
            return True
        return False

    def _check_bull_sfp_quality_new(self, current: dict, pivot: dict) -> bool:
        ref_low = float(pivot['low'])
        low     = float(current['low'])
        close   = float(current['close'])

        wick_depth = max(ref_low - low, 0.0)
        m_tick = float(getattr(self, "tick_size", 0.01)) or 0.01
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False

        close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
        required_close_back = wick_depth * close_back_pct
        return (close - low) >= required_close_back

    def _check_bear_sfp_quality_new(self, current: dict, pivot: dict) -> bool:
        ref_high = float(pivot['high'])
        high     = float(current['high'])
        close    = float(current['close'])

        wick_depth = max(high - ref_high, 0.0)
        m_tick = float(getattr(self, "tick_size", 0.01)) or 0.01
        wick_ticks = wick_depth / m_tick
        if wick_ticks < float(getattr(self.config, "wick_min_ticks", 7)):
            return False

        close_back_pct = float(getattr(self.config, "close_back_pct", 1.0))
        required_close_back = wick_depth * close_back_pct
        return (high - close) >= required_close_back

    # ==================== Ордеры / вход ====================

    def _place_market_order(self, direction: str, quantity: float, stop_loss: Optional[float] = None):
        if not self.api or not hasattr(self.api, 'place_order'):
            print("API not available for placing order")
            return None
        side_up = "Buy" if direction == "long" else "Sell"
        side_lo = "buy" if direction == "long" else "sell"
        qty = float(quantity)
        sl_send = round_price(stop_loss, self.tick_size) if stop_loss is not None else None
        try:
            # современный v5-метод; прокидываем источник триггера из конфига
            return self.api.place_order(
                symbol=self.symbol,
                side=side_up,
                orderType="Market",
                qty=qty,
                stop_loss=sl_send,
                trigger_by_source=getattr(self.config, "trigger_price_source", "last"),
            )
        except TypeError:
            # совместимость со старым snake_case
            return self.api.place_order(
                symbol=self.symbol,
                side=side_lo,
                order_type="market",
                qty=qty,
                stop_loss=sl_send,
            )

    def _process_long_entry(self):
        try:
            current_price = self._get_current_price()
            if not current_price or len(self.candles_15m) < 2:
                return

            raw_sl = float(self.candles_15m[1]['low'])
            entry_price = round_price(float(current_price), self.tick_size)
            stop_loss   = round_price(raw_sl, self.tick_size)

            stop_size = entry_price - stop_loss
            if stop_size <= 0:
                return
            take_profit = round_price(entry_price + stop_size * self.config.risk_reward, self.tick_size)

            quantity = self._calculate_position_size(entry_price, stop_loss, "long")
            if not quantity:
                return
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return

            order_result = self._place_market_order("long", quantity, stop_loss=stop_loss)
            if order_result is None:
                return

            bar_ts_ms = self._current_bar_ts_ms()

            trade_data = {
                'symbol': self.symbol,
                'direction': 'long',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': float(quantity),
                'entry_time': datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                'status': 'open'
            }
            self.db.save_trade(trade_data)

            self.state.set_position({
                'symbol': self.symbol,
                'direction': 'long',
                'size': float(quantity),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open',
                'armed': not self.config.use_arm_after_rr,
                'trail_anchor': entry_price,            # стартовый якорь
                'entry_time_ts': bar_ts_ms
            })

            # синхронизируем трейл-движок
            try:
                self.trail_engine.on_entry(entry_price, stop_loss, "long")
            except Exception:
                pass

            self.can_enter_long = False
            self.can_enter_short = False
            print(f"Long entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            print(f"Error processing long entry: {str(e) if e else 'Unknown error'}")

    def _process_short_entry(self):
        try:
            current_price = self._get_current_price()
            if not current_price or len(self.candles_15m) < 2:
                return

            raw_sl = float(self.candles_15m[1]['high'])
            entry_price = round_price(float(current_price), self.tick_size)
            stop_loss   = round_price(raw_sl, self.tick_size)

            stop_size = stop_loss - entry_price
            if stop_size <= 0:
                return
            take_profit = round_price(entry_price - stop_size * self.config.risk_reward, self.tick_size)

            quantity = self._calculate_position_size(entry_price, stop_loss, "short")
            if not quantity:
                return
            if not self._validate_position_requirements(entry_price, stop_loss, take_profit, quantity):
                return

            order_result = self._place_market_order("short", quantity, stop_loss=stop_loss)
            if order_result is None:
                return

            bar_ts_ms = self._current_bar_ts_ms()

            trade_data = {
                'symbol': self.symbol,
                'direction': 'short',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': float(quantity),
                'entry_time': datetime.utcfromtimestamp(bar_ts_ms / 1000) if bar_ts_ms else datetime.utcnow(),
                'status': 'open'
            }
            self.db.save_trade(trade_data)

            self.state.set_position({
                'symbol': self.symbol,
                'direction': 'short',
                'size': float(quantity),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open',
                'armed': not self.config.use_arm_after_rr,
                'trail_anchor': entry_price,
                'entry_time_ts': bar_ts_ms
            })

            try:
                self.trail_engine.on_entry(entry_price, stop_loss, "short")
            except Exception:
                pass

            self.can_enter_short = False
            self.can_enter_long = False
            print(f"Short entry: {quantity} @ {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            print(f"Error processing short entry: {str(e) if e else 'Unknown error'}")

    # ==================== Поддержка позиции / трейлинг ====================

    def _get_current_price(self) -> Optional[float]:
        """
        Единый источник цены для логики — из конфига:
        config.price_for_logic: "last" | "mark"
        """
        try:
            if not self.api:
                return None
            src = str(getattr(self.config, "price_for_logic", "last")).lower()
            # унифицированный метод API (из bybit_api.py)
            if hasattr(self.api, "get_price"):
                return float(self.api.get_price(self.symbol, source=src))

            # фолбэк: тикер
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
            print(f"Error getting current price: {str(e) if e else 'Unknown error'}")
        return None

    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        try:
            equity = self.state.get_equity()
            if equity is None or equity <= 0:
                return None
            risk_amount = equity * (self.config.risk_pct / 100.0)
            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None
            quantity = risk_amount / stop_size
            quantity = round_qty(quantity, self.qty_step)
            if getattr(self.config, "limit_qty_enabled", False):
                quantity = min(quantity, getattr(self.config, "max_qty_manual", quantity))
            if quantity < self.min_order_qty:
                return None
            return float(quantity)
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float,
                                        take_profit: float, quantity: float) -> bool:
        try:
            if quantity is None:
                return False

            qty = float(quantity)
            if qty <= 0:
                return False

            min_order_qty = float(getattr(self.config, "min_order_qty", 0.01))
            if qty < min_order_qty:
                return False

            taker = float(getattr(self.config, "taker_fee_rate", 0.00055))
            gross = abs(float(take_profit) - float(entry_price)) * qty
            fees  = float(entry_price) * qty * taker * 2.0  # entry + exit
            net   = gross - fees

            min_net_profit = float(getattr(self.config, "min_net_profit", 1.2))
            return net >= min_net_profit
        except Exception as e:
            print(f"Error validating position: {e}")
            return False

    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=self.config.days_back)
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

    def _get_bar_extremes_for_trailing(self, fallback_price: float) -> Tuple[float, float]:
        try:
            if getattr(self.config, "use_intrabar", True) and self.candles_1m:
                last_bar = self.candles_1m[0]
                return float(last_bar['high']), float(last_bar['low'])
            elif self.candles_15m:
                last_bar = self.candles_15m[0]
                return float(last_bar['high']), float(last_bar['low'])
        except Exception:
            pass
        return float(fallback_price), float(fallback_price)

    def _update_smart_trailing(self, position: Dict):
        """
        Процентный Smart Trailing с ARM по RR.
        """
        try:
            if not getattr(self.config, "enable_smart_trail", True):
                return

            direction = position.get('direction')
            entry = float(position.get('entry_price') or 0)
            sl = float(position.get('stop_loss') or 0)
            if not direction or entry <= 0 or sl <= 0:
                return

            # 1) Текущая цена для самой стратегии
            price = self._get_current_price()
            if price is None:
                return
            price = float(price)

            # 2) Экстремумы последнего бара — для якоря и (опционально) для ARM
            bar_high, bar_low = self._get_bar_extremes_for_trailing(price)

            # 2a) ARM по RR с учётом базы: 'extremum' | 'last'
            armed = bool(position.get('armed', not getattr(self.config, 'use_arm_after_rr', True)))
            if not armed and getattr(self.config, 'use_arm_after_rr', True):
                risk = abs(entry - sl)
                if risk > 0:
                    basis = str(getattr(self.config, 'arm_rr_basis', 'extremum')).lower()
                    rr_px = (bar_high if direction == 'long' else bar_low) if basis == "extremum" else price
                    rr = (rr_px - entry) / risk if direction == 'long' else (entry - rr_px) / risk
                    if rr >= float(getattr(self.config, 'arm_rr', 0.5)):
                        armed = True
                        position['armed'] = True
                        self.state.set_position(position)
                        print(f"Position ARMED at {self.config.arm_rr}R (basis={basis})")
            if not armed:
                return

            # 3) Якорь экстремума (как в Pine) — с момента входа
            anchor = float(position.get('trail_anchor') or entry)
            anchor = max(anchor, bar_high) if direction == 'long' else min(anchor, bar_low)
            if anchor != position.get('trail_anchor'):
                position['trail_anchor'] = anchor
                self.state.set_position(position)

            # 4) Процентный трейл от entry + offset
            trail_perc  = float(getattr(self.config, 'trailing_perc', 0.5)) / 100.0
            offset_perc = float(getattr(self.config, 'trailing_offset_perc', 0.4)) / 100.0
            trail_dist  = entry * trail_perc
            offset_dist = entry * offset_perc

            if direction == 'long':
                candidate = round_price(anchor - trail_dist - offset_dist, self.tick_size)
                if candidate > sl:
                    self._update_stop_loss(position, candidate)
            else:
                candidate = round_price(anchor + trail_dist + offset_dist, self.tick_size)
                if candidate < sl:
                    self._update_stop_loss(position, candidate)

        except Exception as e:
            print(f"Error in smart trailing: {e}")

    def _update_stop_loss(self, position: Dict, new_sl: float):
        try:
            if not self.api:
                return False
            new_sl = round_price(float(new_sl), self.tick_size)

            # сперва правильный v5-метод (деривативы)
            if hasattr(self.api, "update_position_stop_loss"):
                ok = self.api.update_position_stop_loss(self.symbol, new_sl)
                if ok:
                    position['stop_loss'] = new_sl
                    self.state.set_position(position)
                    print(f"[TRAIL] SL -> {new_sl:.4f}")
                    return True

            # фолбэк — modify_order (работает в paper API)
            if hasattr(self.api, "modify_order"):
                _ = self.api.modify_order(symbol=position['symbol'], stop_loss=new_sl)
                position['stop_loss'] = new_sl
                self.state.set_position(position)
                print(f"[TRAIL] SL -> {new_sl:.4f}")
                return True

            return False
        except Exception as e:
            print(f"Error updating stop loss: {e}")
            return False

    def process_trailing(self):
        """LEGACY метод для обратной совместимости"""
        try:
            current_position = self.state.get_current_position()
            if current_position and current_position.get('status') == 'open':
                self._update_smart_trailing(current_position)
        except Exception as e:
            print(f"Error processing trailing: {e}")

    def run_cycle(self):
        """
        Основной цикл обработки (вызов из on_bar_close_15m).
        Если позиции нет — ищем SFP и входим; иначе обновляем трейлинг.
        """
        try:
            if not self.candles_15m:
                return

            current_position = self.state.get_current_position()
            if current_position and current_position.get('status') == 'open':
                self._update_smart_trailing(current_position)
                return

            if len(self.candles_15m) < int(getattr(self.config, "sfp_len", 2)) + 2:
                return

            current_ts = int(self.candles_15m[0]['timestamp'])
            if not self._is_in_backtest_window_utc(current_ts):
                return

            if self._detect_bull_sfp() and self.can_enter_long:
                self._process_long_entry()
            elif self._detect_bear_sfp() and self.can_enter_short:
                self._process_short_entry()

        except Exception as e:
            print(f"[run_cycle ERROR] {e}")

    # ==================== Equity (live) ====================

    def _update_equity(self):
        try:
            if not self.api:
                return
            wallet = self.api.get_wallet_balance()
            # формат зависит от accountType; примерная совместимость:
            if wallet and wallet.get("list"):
                for account in wallet["list"]:
                    if account.get("accountType") in ("SPOT", "UNIFIED"):
                        for coin in account.get("coin", []):
                            if coin.get("coin") == "USDT":
                                equity = float(coin.get("equity", 0))
                                self.state.set_equity(equity)
                                self.db.save_equity_snapshot(equity)
                                break
        except Exception as e:
            print(f"Error updating equity: {str(e) if e else 'Unknown error'}")
