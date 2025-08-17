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

        # Версионирование
        self.strategy_version = "2.2.0-A(Pine-like)"

    # ==================== Pine-like helpers ====================

    @staticmethod
    def _ta_pivotlow(series_newest_first: List[float], left: int, right: int) -> bool:
        """
        Эмуляция ta.pivotlow(left,right) на баре [1] для массивов newest-first.
        Для right=1 сравниваем бар[1] с баром[0] (право) и с left барами слева: [2..2+left-1].
        """
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
        """Эмуляция ta.pivothigh(left,right) на баре [1] (newest-first)."""
        if len(series_newest_first) < left + right + 2:
            return False
        x = series_newest_first
        pivot = x[1]
        right_ok = pivot > max(x[0:1])  # bar[0]
        left_block = x[2:2 + left] if left > 0 else []
        left_ok = (len(left_block) == left) and (pivot > max(left_block)) if left > 0 else True
        return right_ok and left_ok

    def _series(self, field: str, tf: str) -> List[float]:
        """
        Pine-like series accessor: request.security(syminfo.tickerid, tf, <field>)
        Возвращает массив newest-first.
        tf: "1" | "15" | "60"
        """
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
        """Приводим свечи к единому виду; newest-first."""
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

    # -------------------- Основной цикл --------------------

    def run_cycle(self):
        try:
            if not self.candles_15m:
                return

            position = self.state.get_current_position()
            if position and position.get("status") == "open":
                # 1) сначала обновляем трейл на закрывшемся баре
                self._update_smart_trailing(position)
                # 2) затем проверяем срабатывание SL (или трейл-SL) на этом баре
                self._check_and_close_position(position)
                return

            self.on_bar_close()
        except Exception as e:
            print(f"Error in run_cycle: {e}")

    def _check_and_close_position(self, position: Dict):
        """Проверка закрытия позиции по SL/TP на закрытом баре (как в Pine)"""
        try:
            if not self.candles_15m or len(self.candles_15m) < 1:
                return

            # Берем последний закрытый бар
            current_bar = self.candles_15m[0]
            direction = position.get('direction')
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            
            if not direction or not stop_loss:
                return

            # Проверяем касание SL по high/low закрытого бара
            sl_hit = False
            tp_hit = False
            exit_price = None
            exit_reason = None

            if direction == 'long':
                # Лонг: SL касание если low <= stop_loss
                if current_bar['low'] <= stop_loss:
                    sl_hit = True
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif take_profit and current_bar['high'] >= take_profit:
                    tp_hit = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'
            
            elif direction == 'short':
                # Шорт: SL касание если high >= stop_loss  
                if current_bar['high'] >= stop_loss:
                    sl_hit = True
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif take_profit and current_bar['low'] <= take_profit:
                    tp_hit = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'

            # Закрываем позицию если был тач
            if sl_hit or tp_hit:
                print(f"Position closed: {exit_reason} at {exit_price}")
                
                # Записываем выход в БД с комиссией
                trade_data = {
                    'trade_id': position.get('trade_id'),
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_time': datetime.now(),
                    'status': 'closed'
                }
                
                # Обновляем в БД с автоматическим расчетом PnL/RR
                fee_rate = float(getattr(self.config, "taker_fee_rate", 0.00055))
                self.db.update_trade_exit(trade_data, fee_rate=fee_rate)
                
                # Очищаем позицию в состоянии
                self.state.clear_position()
                
                # Сбрасываем состояние входов для следующего бара
                self.can_enter_long = True
                self.can_enter_short = True
                self.entered_this_bar = False

        except Exception as e:
            print(f"Error in _check_and_close_position: {e}")

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

        # pivotlow(sfpLen,1) на баре [1]
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

        # pivothigh(sfpLen,1) на баре [1]
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
            if self.entered_this_bar:
                return

            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]              # предыдущая 15m свеча
            stop_loss = float(prev["low"])          # SL за LOW предыдущей (Pine)

            # Гварды — опционально (по умолчанию выключены, как в Pine)
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
                trade_id = self.db.save_trade(trade)  # фиксируем в БД и сохраняем id

                # сохраняем экстремум со входа для percent+offset трейла
                self.state.set_position({
                    "trade_id": trade_id,
                    "symbol": self.symbol,
                    "direction": "long",
                    "size": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "armed": (not getattr(self.config, "use_arm_after_rr", True)),
                    "peak": entry_price,   # максимум со входа
                    "trough": None,
                    "status": "open",
                })
                # анти-спам на баре
                self.can_enter_long = False
                self.can_enter_short = False
                self.entered_this_bar = True
                print(f"[{self.symbol}] LONG entry={entry_price:.2f} prev_low={prev['low']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing long entry: {e}")

    def _process_short_entry(self):
        try:
            if self.entered_this_bar:
                return

            entry_price = self._get_current_price()
            if entry_price is None or len(self.candles_15m) < 2:
                return

            prev = self.candles_15m[1]
            stop_loss = float(prev["high"])         # SL за HIGH предыдущей (Pine)

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
                trade_id = self.db.save_trade(trade)

                self.state.set_position({
                    "trade_id": trade_id,
                    "symbol": self.symbol,
                    "direction": "short",
                    "size": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "armed": (not getattr(self.config, "use_arm_after_rr", True)),
                    "peak": None,
                    "trough": entry_price,  # минимум со входа
                    "status": "open",
                })

                self.can_enter_long = False
                self.can_enter_short = False
                self.entered_this_bar = True
                print(f"[{self.symbol}] SHORT entry={entry_price:.2f} prev_high={prev['high']:.2f} "
                      f"SL={stop_loss:.2f} TP={take_profit:.2f} qty={quantity:.4f}")
        except Exception as e:
            print(f"Error processing short entry: {e}")

    # -------------------- Утилиты позиций --------------------

    def _get_current_price(self) -> Optional[float]:
        try:
            if not self.candles_15m:
                return None
            return float(self.candles_15m[0]["close"])
        except Exception:
            return None

    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        try:
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance <= 0:
                return None

            equity = float(self.state.get_equity() or 0)
            if equity <= 0:
                return None

            risk_amount = equity * (float(self.config.risk_pct) / 100.0)
            base_qty = risk_amount / stop_distance

            # Применяем фильтры
            base_qty = min(base_qty, float(self.config.max_qty_manual))
            base_qty = max(base_qty, float(self.min_order_qty))

            return qty_round(base_qty, self.qty_step)
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None

    def _validate_position_requirements(self, entry_price: float, stop_loss: float, 
                                        take_profit: float, quantity: float) -> bool:
        try:
            # Базовые проверки
            if quantity < float(self.min_order_qty):
                return False

            # Проверка минимальной прибыли
            min_profit = getattr(self.config, "min_profit_usd", 0.0)
            if min_profit > 0:
                expected_profit = abs(take_profit - entry_price) * quantity
                if expected_profit < min_profit:
                    print(f"Expected profit {expected_profit:.2f} < min required {min_profit}")
                    return False

            return True
        except Exception:
            return False

    # -------------------- Временные окна (UTC) --------------------

    def _is_in_backtest_window_utc(self, bar_timestamp_ms: int) -> bool:
        try:
            start_date = getattr(self.config, "backtest_start_date", None)
            end_date = getattr(self.config, "backtest_end_date", None)

            if not start_date or not end_date:
                return True

            bar_dt = datetime.fromtimestamp(bar_timestamp_ms / 1000, tz=timezone.utc)
            bar_date = bar_dt.date()

            # Парсинг дат
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

            return start_date <= bar_date <= end_date
        except Exception:
            return True

    # -------------------- Smart Trailing --------------------

    def _update_smart_trailing(self, position: Dict):
        try:
            if not getattr(self.config, "enable_smart_trail", True):
                return

            current_price = self._get_current_price()
            if current_price is None:
                return

            # Проверяем арм
            if not position.get("armed", False):
                if self._check_arm_condition(position, current_price):
                    position["armed"] = True
                    self.state.set_position(position)
                    print(f"Position ARMED at price {current_price}")
                return

            # Обновляем экстремумы
            direction = position["direction"]
            if direction == "long":
                old_peak = position.get("peak", position["entry_price"])
                if current_price > old_peak:
                    position["peak"] = current_price
            else:
                old_trough = position.get("trough", position["entry_price"])
                if current_price < old_trough:
                    position["trough"] = current_price

            # Рассчитываем новый SL
            new_sl = self._calculate_trailing_stop(position, current_price)
            old_sl = position["stop_loss"]

            # Проверяем, нужно ли обновить
            should_update = False
            if direction == "long" and new_sl > old_sl:
                should_update = True
            elif direction == "short" and new_sl < old_sl:
                should_update = True

            if should_update:
                position["stop_loss"] = new_sl
                self.state.set_position(position)
                print(f"Trailing SL updated: {old_sl:.2f} -> {new_sl:.2f}")

        except Exception as e:
            print(f"Error in smart trailing: {e}")

    def _check_arm_condition(self, position: Dict, current_price: float) -> bool:
        if not getattr(self.config, "use_arm_after_rr", True):
            return True

        try:
            entry_price = position["entry_price"]
            initial_sl = position.get("initial_stop_loss", position["stop_loss"])
            direction = position["direction"]
            arm_rr = getattr(self.config, "arm_rr", 0.5)

            if direction == "long":
                stop_distance = entry_price - initial_sl
                required_move = stop_distance * arm_rr
                return current_price >= entry_price + required_move
            else:
                stop_distance = initial_sl - entry_price
                required_move = stop_distance * arm_rr
                return current_price <= entry_price - required_move
        except Exception:
            return False

    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> float:
        direction = position["direction"]
        entry_price = position["entry_price"]

        try:
            # Процентный трейлинг
            trailing_perc = getattr(self.config, "trailing_perc", 0.5) / 100.0
            trailing_offset = getattr(self.config, "trailing_offset_perc", 0.4) / 100.0

            if direction == "long":
                peak = position.get("peak", entry_price)
                percent_sl = peak * (1.0 - trailing_perc - trailing_offset)
                
                # Bar-based трейлинг
                bar_sl = self._get_bar_trailing_stop("long")
                
                return max(percent_sl, bar_sl) if bar_sl else percent_sl
            else:
                trough = position.get("trough", entry_price)
                percent_sl = trough * (1.0 + trailing_perc + trailing_offset)
                
                bar_sl = self._get_bar_trailing_stop("short")
                
                return min(percent_sl, bar_sl) if bar_sl else percent_sl

        except Exception as e:
            print(f"Error calculating trailing stop: {e}")
            return position["stop_loss"]

    def _get_bar_trailing_stop(self, direction: str) -> Optional[float]:
        if not getattr(self.config, "use_bar_trail", True):
            return None

        try:
            lookback = getattr(self.config, "trail_lookback", 50)
            buffer_ticks = getattr(self.config, "trail_buf_ticks", 0)
            
            if len(self.candles_15m) < lookback + 1:
                return None

            bars = self.candles_15m[1:lookback+1]  # исключаем текущий бар
            
            if direction == "long":
                trail_level = min(bar["low"] for bar in bars)
                return trail_level - (buffer_ticks * self.tick_size)
            else:
                trail_level = max(bar["high"] for bar in bars)
                return trail_level + (buffer_ticks * self.tick_size)

        except Exception:
            return None

    def _check_and_close_position(self, position: Dict):
        """Проверяем срабатывание SL на текущем баре"""
        try:
            if not self.candles_15m:
                return

            current_bar = self.candles_15m[0]
            direction = position["direction"]
            stop_loss = position["stop_loss"]

            should_close = False
            exit_price = stop_loss

            if direction == "long" and current_bar["low"] <= stop_loss:
                should_close = True
            elif direction == "short" and current_bar["high"] >= stop_loss:
                should_close = True

            if should_close:
                self._close_position(position, exit_price, "stop_loss")

        except Exception as e:
            print(f"Error checking position close: {e}")

    def _close_position(self, position: Dict, exit_price: float, exit_reason: str):
        try:
            trade_id = position.get("trade_id")
            if trade_id:
                # Обновляем запись в БД
                trade_update = {
                    "exit_price": exit_price,
                    "exit_time": datetime.now(timezone.utc),
                    "exit_reason": exit_reason,
                    "status": "closed"
                }
                
                # Рассчитываем PnL
                entry_price = position["entry_price"]
                quantity = position["size"]
                direction = position["direction"]
                
                if direction == "long":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity
                
                trade_update["pnl"] = pnl
                # Обновление записи о сделке в БД (если метод существует)
                if hasattr(self.db, 'update_trade'):
                    self.db.update_trade(trade_id, trade_update)
                
                # Обновляем equity
                current_equity = self.state.get_equity() or 0
                new_equity = current_equity + pnl
                self.state.set_equity(new_equity)

            # Закрываем позицию
            position["status"] = "closed"
            position["exit_price"] = exit_price
            position["exit_time"] = datetime.now(timezone.utc)
            self.state.set_position(position)

            direction = position.get("direction", "unknown")
            print(f"Position closed: {direction} @ {exit_price:.2f} | Reason: {exit_reason}")

        except Exception as e:
            print(f"Error closing position: {e}")

    # -------------------- Обновление equity --------------------

    def _update_equity(self):
        try:
            if not self.api:
                return
            
            balance = self.api.get_balance()
            if balance:
                equity = float(balance.get("equity", 0) or balance.get("totalWalletBalance", 0))
                if equity > 0:
                    self.state.set_equity(equity)
        except Exception as e:
            print(f"Error updating equity: {e}")