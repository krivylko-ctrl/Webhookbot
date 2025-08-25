# kwin_strategy.py — Lux SFP only (SFP wick + close-back, SL = swing level)
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from decimal import Decimal, ROUND_HALF_UP, getcontext

from config import Config
from state_manager import StateManager
from analytics import TradingAnalytics
from database import Database
from utils_round import round_qty, round_to_tick, floor_to_tick, ceil_to_tick

# Высокая точность для HALF_UP округления qty (сохраняем для других мест, но qty раундинг — как в Pine)
getcontext().prec = 28


class KWINStrategy:
    """Основная логика стратегии KWIN: входы ТОЛЬКО по Lux SFP + Smart/Bar trailing.
       Самодостаточный файл (TrailEngine НЕ используется)."""

    def __init__(
        self,
        config: Config,
        api: Any = None,
        state_manager: StateManager | None = None,
        db: Database | None = None,
        **kwargs: Any,
    ) -> None:
        if api is None and "bybit_api" in kwargs:
            api = kwargs.get("bybit_api")

        self.config = config
        self.api = api
        self.state = state_manager
        self.db = db

        # TrailEngine убран — логика трейлинга встроена
        self.analytics = TradingAnalytics()

        # --------- Lux SFP режим ---------
        self.lux_mode: bool = True
        self.lux_swings: int = int(getattr(config, "lux_swings", getattr(config, "sfp_len", 2)))

        # LTF-volume ВЫКЛЮЧЕН ЖЁСТКО (паритет Pine-обёртки)
        self.lux_volume_validation: str = "none"
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

        # --- PATCH C: periodChoice → days_back (30/60/180), как в Pine ---
        try:
            days_map = {"30": 30, "60": 60, "180": 180}
            period_choice = str(getattr(self.config, "period_choice", "30"))
            self.config.days_back = int(days_map.get(period_choice, int(getattr(self.config, "days_back", 30))))
        except Exception:
            pass

        # --- PATCH E: принудительная нижняя планка мин. объёма для паритета с Pine (опционально) ---
        if bool(getattr(self.config, "force_pine_min_qty", False)):
            self.min_order_qty = max(self.min_order_qty, 0.01)

        self.start_time_ms: Optional[int] = getattr(self.config, "start_time_ms", None)
        self._history_replayed: bool = False

        # ================== ПОВЕДЕНЧЕСКИЕ ФЛАГИ (паритет с Pine) ==================
        self.use_one_per_swing: bool = bool(getattr(self.config, "use_once_per_swing", True))
        self.use_one_per_bar:   bool = bool(getattr(self.config, "use_one_per_bar", True))
        # нормализация строк, как в Pine ("Prefer Bear" | "Prefer Bull" | "Skip")
        bp = str(getattr(self.config, "bar_priority", "skip")).strip().lower()
        aliases = {
            "prefer bear": "prefer_bear",
            "prefer_bear": "prefer_bear",
            "prefer bull": "prefer_bull",
            "prefer_bull": "prefer_bull",
            "skip": "skip",
        }
        self.bar_priority: str = aliases.get(bp, "skip")
        self.use_dir_lock:      bool = bool(getattr(self.config, "use_dir_lock", True))

        # ================== РИСК / КОМИССИИ / РЕИНВЕСТ ==================
        self.risk_pct:        float = float(getattr(self.config, "risk_pct", 3.0))
        self.use_fee_filter:  bool  = bool(getattr(self.config, "use_fee_filter", True))
        self.taker_fee_rate:  float = float(getattr(self.config, "taker_fee_rate", 0.00055))
        self.min_net_profit:  float = float(getattr(self.config, "min_net_profit", 2.0))

        # Лимиты объёма — как в Pine (включены по умолчанию)
        self.limit_qty_enabled: bool = bool(getattr(self.config, "limit_qty_enabled", True))
        self.max_qty_manual:    float = float(getattr(self.config, "max_qty_manual", 50.0))

        # ================== SMART TRAIL + BAR-TRAIL ==================
        self.enable_smart_trail:    bool  = bool(getattr(self.config, "enable_smart_trail", True))
        self.use_arm_after_rr:      bool  = bool(getattr(self.config, "use_arm_after_rr", True))
        self.arm_rr:                float = float(getattr(self.config, "arm_rr", 0.5))
        self.trailing_perc:         float = float(getattr(self.config, "trailing_perc", 0.5))          # %
        self.trailing_offset_perc:  float = float(getattr(self.config, "trailing_offset_perc", 0.4))   # %
        self.use_bar_trail:         bool  = bool(getattr(self.config, "use_bar_trail", True))
        self.trail_lookback:        int   = int(getattr(self.config, "trail_lookback", 50))
        self.trail_buf_ticks:       int   = int(getattr(self.config, "trail_buf_ticks", 40))

        # ARM база — принудительно от close (паритет Pine)
        self.arm_rr_basis: str = "close"

        # Swing-locks / per-bar
        # ID свинга = timestamp пивота (как уникальный идентификатор), строго как в Pine-подходе (bar_index)
        self.traded_bull_swing_id: Optional[int] = None
        self.traded_bear_swing_id: Optional[int] = None
        self._bar_index_15m: int = 0

        # ================== FIX-флажки для паритета с Pine ==================
        # источник эквити и режим (жёстко задаём дефолты "как в Pine")
        self.equity_source: str = str(getattr(self.config, "equity_source", "local")).lower()  # "local"|"wallet"
        self.equity_mode:   str = str(getattr(self.config, "equity_mode", "wallet_plus_upnl")).lower()  # "wallet"|"wallet_plus_upnl"
        self.wallet_includes_upnl: bool = bool(getattr(self.config, "wallet_includes_upnl", False))
        # квантование стартового SL — ВЫКЛ по умолчанию (для 1:1 с Pine); включай для реального исполнения
        self.quantize_initial_sl: bool = bool(getattr(self.config, "quantize_initial_sl", False))

        # One-per-bar защита: отметка ts 1m-бара, где уже был вход
        self._last_enter_ts_1m: Optional[int] = None
        # Доп. защита: одна сделка на 15m-окно
        self._last_entered_15m_start: Optional[int] = None

    # ============ ЛОГИ ============
    def _log(self, level: str, msg: str):
        try:
            if self.db and hasattr(self.db, "save_log"):
                self.db.save_log(level, msg, module="KWINStrategy")
        except Exception:
            pass
        print(f"[{level.upper()}] {msg}")

    # ============ ВСПОМОГАТЕЛЬНОЕ ============
    def _align_15m_ms(self, ts_ms: int) -> int:
        """Возвращает НАЧАЛО 15-минутного бара (timestamp округлён вниз)."""
        return (int(ts_ms) // 900_000) * 900_000

    def _ensure_desc(self, arr: List[Dict], key: str = "timestamp"):
        if arr and arr[0].get(key) is not None:
            arr.sort(key=lambda x: x.get(key, 0), reverse=True)

    def _current_bar_ts_ms(self) -> int:
        return int(self.last_processed_bar_ts or 0)

    def _init_instrument_info(self):
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
        start_ms = self._align_15m_ms(bar_ts_ms)
        end_ms = start_ms + 900_000 - 1
        return start_ms, end_ms

    def _ltf_slices_for_bar(self, bar_ts_ms: int) -> List[Dict]:
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
        mode = (self.lux_volume_validation or "none").lower()  # ВСЕГДА "none"
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
        else:
            for x in chunks:
                cl = float(x.get("close"))
                if cl > swing_price:
                    outside_vol += float(x.get("volume", 0.0))
        pct = 100.0 * outside_vol / total_vol
        thr = float(self.lux_volume_threshold_pct)
        ok = (pct > thr) if mode == "outside_gt" else (pct < thr)
        if not ok:
            self._log("debug", f"LTF-volume reject [{direction}] pct={pct:.2f}% thr={thr:.2f}%")
        return ok

    # ---------- Пивоты 15m (right=1, как в Pine) ----------
    def _pivot_low_value(self, left: int, right: int = 1) -> Optional[float]:
        L, R = int(left), int(right)
        arr = self.candles_15m
        need = L + R
        if len(arr) < (need + 1):
            return None
        center_idx = R  # центр пивота — бар с индексом R (у нас [0] — текущий)
        center_low = float(arr[center_idx]["low"])
        left_lows = [float(arr[i]["low"]) for i in range(center_idx + 1, center_idx + 1 + L)]
        right_lows = [float(arr[i]["low"]) for i in range(0, R)]
        if all(center_low <= x for x in left_lows + right_lows):
            return center_low
        return None

    def _pivot_high_value(self, left: int, right: int = 1) -> Optional[float]:
        L, R = int(left), int(right)
        arr = self.candles_15m
        need = L + R
        if len(arr) < (need + 1):
            return None
        center_idx = R
        center_high = float(arr[center_idx]["high"])
        left_highs = [float(arr[i]["high"]) for i in range(center_idx + 1, center_idx + 1 + L)]
        right_highs = [float(arr[i]["high"]) for i in range(0, R)]
        if all(center_high >= x for x in left_highs + right_highs):
            return center_high
        return None

    # ---------- Пивоты intrabar (локальные) + локальный SFP ----------
    # PATCH F: возвращаем (value, timestamp, local_index)
    def _pivot_low_value_tf(self, arr: List[Dict], left: int, right: int = 1) -> Optional[Tuple[float, int, int]]:
        L, R = int(left), int(right)
        if len(arr) < (L + R + 1):
            return None
        center_idx = R
        center_low = float(arr[center_idx]["low"])
        left_lows  = [float(arr[i]["low"]) for i in range(center_idx + 1, center_idx + 1 + L)]
        right_lows = [float(arr[i]["low"]) for i in range(0, R)]
        if all(center_low <= x for x in left_lows + right_lows):
            return center_low, int(arr[center_idx].get("timestamp") or 0), int(center_idx)
        return None

    def _pivot_high_value_tf(self, arr: List[Dict], left: int, right: int = 1) -> Optional[Tuple[float, int, int]]:
        L, R = int(left), int(right)
        if len(arr) < (L + R + 1):
            return None
        center_idx = R
        center_high = float(arr[center_idx]["high"])
        left_highs  = [float(arr[i]["high"]) for i in range(center_idx + 1, center_idx + 1 + L)]
        right_highs = [float(arr[i]["high"]) for i in range(0, R)]
        if all(center_high >= x for x in left_highs + right_highs):
            return center_high, int(arr[center_idx].get("timestamp") or 0), int(center_idx)
        return None

    def _find_local_sfp_intrabar(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Локальные (M1/M5) SFP: wick за swing и close-back через swing. Возвращает контексты bear/bull."""
        if not self.candles_1m:
            return None, None
        L = int(self.lux_swings)
        cur = self.candles_1m[0]
        op, cl, hi, lo = map(float, (cur["open"], cur["close"], cur["high"], cur["low"]))
        tick = float(self.tick_size or 0.01)

        bear_ctx = None
        bull_ctx = None

        ph = self._pivot_high_value_tf(self.candles_1m, L, 1)
        if ph is not None:
            swH, swing_ts, swing_idx = ph
            if hi > swH and op < swH and cl < swH:
                if bool(getattr(self.config, "use_sfp_quality", False)):
                    min_ticks = int(getattr(self.config, "wick_min_ticks", 0) or 0)
                    wick_ticks = max(0.0, (hi - swH) / tick)
                    if wick_ticks < min_ticks:
                        return None, None
                use_buf   = bool(getattr(self.config, "use_sl_buffer", False))
                buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
                sl = float(swH + (buf_ticks * tick if use_buf else 0.0))  # RAW, без квантования
                bear_ctx = {"sl": sl, "sid_local": swing_ts, "sid_local_idx": swing_idx, "source": "local"}

        pl = self._pivot_low_value_tf(self.candles_1m, L, 1)
        if pl is not None:
            swL, swing_ts, swing_idx = pl
            if lo < swL and op > swL and cl > swL:
                if bool(getattr(self.config, "use_sfp_quality", False)):
                    min_ticks = int(getattr(self.config, "wick_min_ticks", 0) or 0)
                    wick_ticks = max(0.0, (swL - lo) / tick)
                    if wick_ticks < min_ticks:
                        return None, None
                use_buf   = bool(getattr(self.config, "use_sl_buffer", False))
                buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
                sl = float(swL - (buf_ticks * tick if use_buf else 0.0))  # RAW, без квантования
                bull_ctx = {"sl": sl, "sid_local": swing_ts, "sid_local_idx": swing_idx, "source": "local"}

        return bear_ctx, bull_ctx

    # --- окр. к ближайшему шагу — PATCH D: pine-like (round) вместо Decimal HALF_UP для qty ---
    def _round_to_step_nearest(self, v: float, step: float) -> float:
        # Pine math.round эквивалентен округлению к ближайшему; 0.5→вверх на практике для положительных qty
        return round(v / step) * step

    # --- 15m-SFP intrabar detector (anti-repaint like Pine) ---
    def _find_15m_sfp_intrabar(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Возвращает bear_ctx/bull_ctx для 15m-SFP, подтверждённого внутри текущего 15m окна.
          Bear: hi15_cur > swH И 1m open < swH И 1m close < swH
          Bull: lo15_cur < swL И 1m open > swL И 1m close > swL
        SL = nz(local_swing, экстремум 1m бара).
        """
        if not self.candles_15m or not self.candles_1m:
            return None, None

        L = int(self.lux_swings)
        if len(self.candles_15m) < (L + 2):
            return None, None

        cur15 = self.candles_15m[0]

        # swing = 15m[L], а не prev1
        swH = None
        swL = None
        if self._pivot_high_value(L, 1) is not None and len(self.candles_15m) > L:
            swH = float(self.candles_15m[L]["high"])
        if self._pivot_low_value(L, 1) is not None and len(self.candles_15m) > L:
            swL = float(self.candles_15m[L]["low"])

        cur_ts = int(cur15.get("timestamp") or 0)
        slices = self._ltf_slices_for_bar(cur_ts)
        if not slices:
            return None, None

        hi15_cur = max(float(x["high"]) for x in slices if x.get("high") is not None)
        lo15_cur = min(float(x["low" ]) for x in slices if x.get("low" ) is not None)

        last1 = self.candles_1m[0]
        op, cl = float(last1["open"]), float(last1["close"])

        # локальные свинги на 1m для SL-first логики
        ph_loc = self._pivot_high_value_tf(self.candles_1m, L, 1)
        pl_loc = self._pivot_low_value_tf (self.candles_1m, L, 1)
        swH_loc = float(ph_loc[0]) if ph_loc else None
        swL_loc = float(pl_loc[0]) if pl_loc else None

        bear_ctx = None
        bull_ctx = None

        if swH is not None and hi15_cur > swH and op < swH and cl < swH:
            sl = self._sl_local_first("short", swH_loc, last1)  # RAW
            bear_ctx = {"sl": sl, "sid_15m": int(self.candles_15m[1].get("timestamp") or 0) if len(self.candles_15m) > 1 else 0, "source": "15m_intrabar"}

        if swL is not None and lo15_cur < swL and op > swL and cl > swL:
            sl = self._sl_local_first("long", swL_loc, last1)  # RAW
            bull_ctx = {"sl": sl, "sid_15m": int(self.candles_15m[1].get("timestamp") or 0) if len(self.candles_15m) > 1 else 0, "source": "15m_intrabar"}

        return bear_ctx, bull_ctx

    # ---------- SL fallback как в Pine (если свинг отсутствует) ----------
    def _sl_with_fallback(self, side: str, swing_price: Optional[float], bar: Dict) -> float:
        tick = float(self.tick_size or 0.01)
        use_buf   = bool(getattr(self.config, "use_sl_buffer", False))
        buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
        if side == "long":
            base = swing_price if swing_price is not None else float(bar["low"])
            sl   = base - (buf_ticks * tick if use_buf and swing_price is not None else 0.0)
            return float(sl)  # RAW
        else:
            base = swing_price if swing_price is not None else float(bar["high"])
            sl   = base + (buf_ticks * tick if use_buf and swing_price is not None else 0.0)
            return float(sl)  # RAW

    # ---------- SL с приоритетом локального свинга ----------
    def _sl_local_first(self, side: str, local_swing: Optional[float], bar: Dict) -> float:
        tick = float(self.tick_size or 0.01)
        use_buf   = bool(getattr(self.config, "use_sl_buffer", False))
        buf_ticks = int(getattr(self.config, "sl_buf_ticks", 0) or 0)
        if side == "long":
            base = float(local_swing) if local_swing is not None else float(bar["low"])
            if local_swing is not None and use_buf:
                base -= buf_ticks * tick
            return float(base)  # RAW
        else:
            base = float(local_swing) if local_swing is not None else float(bar["high"])
            if local_swing is not None and use_buf:
                base += buf_ticks * tick
            return float(base)  # RAW

    # ---------- Fee filter ----------
    def _expected_net_ok(self, entry: float, sl: float, qty: float) -> bool:
        if not self.use_fee_filter:
            return True
        stop_size = abs(entry - sl)
        exp_gross = stop_size * float(qty)
        exp_fees  = float(entry) * float(qty) * float(self.taker_fee_rate) * 2.0
        return (exp_gross - exp_fees) >= float(self.min_net_profit)

    # ======= One-per-bar helpers =======
    def _mark_entered_this_bar(self, ts_1m: Optional[int]) -> None:
        if ts_1m:
            self._last_enter_ts_1m = int(ts_1m)

    def _already_entered_this_bar(self, ts_1m: Optional[int]) -> bool:
        if ts_1m is None:
            return False
        return self._last_enter_ts_1m == int(ts_1m)

    # --------- PATCH A: локальный учёт реализованной PnL и реинвест ---------
    def _apply_realized_pnl(self, exit_price: float, reason: str = "close"):
        pos = self.state.get_current_position() if self.state else None
        if not pos or str(pos.get("status")) != "open":
            return
        side = str(pos["direction"])
        qty  = float(pos["size"] or 0.0)
        ent  = float(pos["entry_price"] or 0.0)
        if qty <= 0 or ent <= 0:
            return

        # валовая PnL
        gross = (exit_price - ent) * qty if side == "long" else (ent - exit_price) * qty
        # комиссии — вход + выход
        fees  = (ent * qty * self.taker_fee_rate) + (exit_price * qty * self.taker_fee_rate)
        pnl   = gross - fees

        # Pine-parity: обновляем локальную equity (если source=local)
        equity_local = self.state.get_equity() if self.state else None
        if equity_local is None:
            equity_local = float(getattr(self.config, "initial_capital", 300.0))
        new_eq = float(equity_local) + float(pnl)

        if self.state:
            self.state.set_equity(new_eq)

        if self.db and hasattr(self.db, "save_equity_snapshot"):
            try:
                self.db.save_equity_snapshot(new_eq)
            except Exception:
                pass

        # закрыть позицию в локальном состоянии
        pos["status"] = "closed"
        if self.state:
            self.state.set_position(None)

        self._log("info", f"[CLOSE {side.upper()}] @ {exit_price:.6f} PnL={pnl:.2f} equity->{new_eq:.2f} reason={reason}")

    # --------- PATCH A: детектор удара стопа на текущем баре ---------
    def _check_stop_hit(self):
        pos = self.state.get_current_position() if self.state else None
        if not pos or str(pos.get("status")) != "open":
            return
        sl = float(pos.get("stop_loss") or 0.0)
        if sl <= 0:
            return

        price = self._get_current_price()
        price = float(price) if price is not None else sl
        hi, lo = self._get_bar_extremes_for_trailing(price)
        side = str(pos.get("direction"))

        if side == "long" and lo <= sl:
            self._apply_realized_pnl(sl, reason="stop")
        elif side == "short" and hi >= sl:
            self._apply_realized_pnl(sl, reason="stop")

    # ============ ОБРАБОТЧИКИ БАРОВ ============
    def on_bar_close_15m(self, candle: Dict):
        try:
            self.candles_15m.insert(0, candle)
            if len(self.candles_15m) > 200:
                self.candles_15m = self.candles_15m[:200]
            self._ensure_desc(self.candles_15m)

            bar_ts = int(candle.get("timestamp") or candle.get("open_time") or 0)
            if bar_ts and bar_ts < 1_000_000_000_000:
                bar_ts *= 1000
            aligned_start = self._align_15m_ms(bar_ts)
            if aligned_start == self.last_processed_bar_ts:
                return

            self.last_processed_bar_ts = aligned_start
            self.can_enter_long = True
            self.can_enter_short = True
            self._bar_index_15m += 1
            self._log("debug", f"15m close @ts(start)={aligned_start}")
            self.run_cycle()

            # после цикла проверить удар SL
            self._check_stop_hit()
        except Exception as e:
            self._log("error", f"on_bar_close_15m: {e}")

    def on_bar_close_60m(self, candle: Dict):
        try:
            self.candles_1h.insert(0, candle)
            if len(self.candles_1h) > 100:
                self.candles_1h = self.candles_1h[:100]
        except Exception as e:
            self._log("error", f"on_bar_close_60m: {e}")

    def _flip_position(self, new_dir: str) -> bool:
        """Разворот при use_dir_lock=False: закрыть текущую позицию и открыть новую сторону."""
        pos = self.state.get_current_position() if self.state else None
        if not pos or pos.get("status") != "open":
            return True
        cur_dir = str(pos.get("direction"))
        if cur_dir == new_dir:
            return True

        size = float(pos.get("size") or 0)
        if size <= 0:
            return True

        try:
            # 1) закрыть текущую на рынке (или API)
            if self.api and hasattr(self.api, "close_position"):
                self.api.close_position(self.symbol)
                # эквити нужно обновить по текущей цене (parity с Pine)
                px = self._get_current_price()
                if px is not None:
                    self._apply_realized_pnl(float(px), reason="flip(api)")
            else:
                close_side = "short" if cur_dir == "long" else "long"
                # исполним локально и пересчитаем equity
                px = self._get_current_price()
                if px is None:
                    # fallback: используем стоп как worst-case
                    px = float(pos.get("stop_loss") or pos.get("entry_price"))
                self._apply_realized_pnl(float(px), reason="flip(local)")
                self._place_market_order(close_side, size)

            return True
        except Exception as e:
            self._log("error", f"flip_position failed: {e}")
            return False

    def on_bar_close_1m(self, candle: Dict):
        """Интрабар — локальные входы (Dual-SFP) и Smart Trail."""
        try:
            self.candles_1m.insert(0, candle)
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            if len(self.candles_1m) > lim:
                self.candles_1m = self.candles_1m[:lim]
            self._ensure_desc(self.candles_1m)

            cur_ts = int(candle.get("timestamp") or 0)

            # --- reset once-per-swing по АКТУАЛЬНОМУ timestamp пивота (как в Pine) ---
            L = int(self.lux_swings)
            ph = self._pivot_high_value_tf(self.candles_1m, L, 1)  # -> (value, ts, idx) или None
            pl = self._pivot_low_value_tf (self.candles_1m, L, 1)

            cur_bear_sid_ts = int(ph[1]) if ph else None  # timestamp текущего H-свинга
            cur_bull_sid_ts = int(pl[1]) if pl else None  # timestamp текущего L-свинга

            # если активный свинг сменился/исчез — снимаем лок
            if self.traded_bear_swing_id is not None:
                if (cur_bear_sid_ts is None) or (self.traded_bear_swing_id != cur_bear_sid_ts):
                    self.traded_bear_swing_id = None

            if self.traded_bull_swing_id is not None:
                if (cur_bull_sid_ts is None) or (self.traded_bull_swing_id != cur_bull_sid_ts):
                    self.traded_bull_swing_id = None

            # --- Dual-SFP: локальные + 15m-intrabar, one-per-bar с приоритетами ---
            if bool(getattr(self.config, "use_intrabar", True)):
                bear_loc, bull_loc = self._find_local_sfp_intrabar()
                bear_15m, bull_15m = self._find_15m_sfp_intrabar()

                bear_raw = bool(bear_loc) or bool(bear_15m)
                bull_raw = bool(bull_loc) or bool(bull_15m)

                # once-per-swing блок даже для 15m сигналов (если локальный свинг уже торговался)
                cur_bear_sid_ts_chk = int(ph[1]) if ph else 0
                cur_bull_sid_ts_chk = int(pl[1]) if pl else 0
                if self.use_one_per_swing and bool(bear_15m) and cur_bear_sid_ts_chk and self.traded_bear_swing_id == cur_bear_sid_ts_chk:
                    bear_raw = False
                if self.use_one_per_swing and bool(bull_15m) and cur_bull_sid_ts_chk and self.traded_bull_swing_id == cur_bull_sid_ts_chk:
                    bull_raw = False

                # one-per-bar конфликт + приоритет
                if self.use_one_per_bar and bear_raw and bull_raw:
                    if self.bar_priority == "prefer_bull":
                        bear_raw = False
                    elif self.bar_priority == "prefer_bear":
                        bull_raw = False
                    else:
                        bear_raw = False
                        bull_raw = False

                # Если уже входили в этот 1m-бар — блокируем повторный вход
                if self.use_one_per_bar and self._already_entered_this_bar(cur_ts):
                    bear_raw = False
                    bull_raw = False

                # Метим текущий 1m-бар для блокировки 15m-ветки в run_cycle()
                if self.use_one_per_bar and (bear_raw or bull_raw):
                    self._mark_entered_this_bar(cur_ts)

                cl = float(self.candles_1m[0]["close"])

                executed = False
                # SHORT
                if bear_raw and self.can_enter_short:
                    pos = self.state.get_current_position() if self.state else None
                    if pos and pos.get("status") == "open":
                        if not self.use_dir_lock and str(pos.get("direction")) == "long":
                            if not self._flip_position("short"):
                                pass
                        else:
                            bear_raw = False

                    if bear_raw:
                        ctx = bear_loc or bear_15m
                        sl  = float(ctx["sl"])  # RAW

                        # ID текущего свинга = timestamp локального H-свинга (если есть)
                        sid_local_ts = int((bear_loc or {}).get("sid_local") or 0)
                        if sid_local_ts == 0 and self.candles_1m:
                            ph_now = self._pivot_high_value_tf(self.candles_1m, int(self.lux_swings), 1)
                            sid_local_ts = int(ph_now[1]) if ph_now else 0

                        # once-per-swing: пропускаем, если уже торговали ЭТОТ свинг (по ts)
                        if (not self.use_one_per_swing) or (self.traded_bear_swing_id != sid_local_ts):
                            qty = self._calculate_position_size(cl, sl, "short")
                            if qty and qty >= self.min_order_qty and self._expected_net_ok(cl, sl, qty):
                                self._process_short_entry(entry_override=cl, sl_override=sl)
                                if self.use_one_per_swing and sid_local_ts:
                                    self.traded_bear_swing_id = sid_local_ts
                                self._last_entered_15m_start = self.last_processed_bar_ts
                                self.can_enter_short = False
                                executed = True

                # LONG
                if bull_raw and self.can_enter_long and not (executed and self.use_one_per_bar):
                    pos = self.state.get_current_position() if self.state else None
                    if pos and pos.get("status") == "open":
                        if not self.use_dir_lock and str(pos.get("direction")) == "short":
                            if not self._flip_position("long"):
                                pass
                        else:
                            bull_raw = False

                    if bull_raw:
                        ctx = bull_loc or bull_15m
                        sl  = float(ctx["sl"])  # RAW

                        # ID текущего свинга = timestamp локального L-свинга (если есть)
                        sid_local_ts = int((bull_loc or {}).get("sid_local") or 0)
                        if sid_local_ts == 0 and self.candles_1m:
                            pl_now = self._pivot_low_value_tf(self.candles_1m, int(self.lux_swings), 1)
                            sid_local_ts = int(pl_now[1]) if pl_now else 0

                        # once-per-swing: пропускаем, если уже торговали ЭТОТ свинг (по ts)
                        if (not self.use_one_per_swing) or (self.traded_bull_swing_id != sid_local_ts):
                            qty = self._calculate_position_size(cl, sl, "long")
                            if qty and qty >= self.min_order_qty and self._expected_net_ok(cl, sl, qty):
                                self._process_long_entry(entry_override=cl, sl_override=sl)
                                if self.use_one_per_swing and sid_local_ts:
                                    self.traded_bull_swing_id = sid_local_ts
                                self._last_entered_15m_start = self.last_processed_bar_ts
                                self.can_enter_long = False

            # Smart Trailing по интрабар-цене, если позиция открыта
            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                self._update_smart_trailing(pos)

            # после обработки 1m — проверить удар SL
            self._check_stop_hit()

        except Exception as e:
            self._log("error", f"on_bar_close_1m: {e}")

    # ========== ВХОДЫ ==========
    def _get_current_price(self) -> Optional[float]:
        """Паритет Pine: по умолчанию используем close рабочего ТФ; при intrabar — close последней M1."""
        try:
            src = str(getattr(self.config, "price_for_logic", "close")).lower()
            if src == "close":
                if getattr(self.config, "use_intrabar", True) and self.candles_1m:
                    return float(self.candles_1m[0]["close"])
                return float(self.candles_15m[0]["close"]) if self.candles_15m else None

            # Явно задан "last"/"mark" — используем API
            if self.api and hasattr(self.api, "get_price"):
                px = self.api.get_price(self.symbol, source=src)
                if px:
                    return float(px)
            if self.api and hasattr(self.api, "get_ticker"):
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
            self._log("error", f"get_current_price: {e}")

        # Fallback: intrabar close if available, else 15m close
        if getattr(self.config, "use_intrabar", True) and self.candles_1m:
            return float(self.candles_1m[0]["close"])
        return float(self.candles_15m[0]["close"]) if self.candles_15m else None

    # --- ордера без изменений логики исполнения ---
    def _place_market_order(self, direction: str, quantity: float, stop_loss: Optional[float] = None):
        if not self.api or not hasattr(self.api, "place_order"):
            self._log("warn", "API not available for placing order (local mode)")
            return {"ok": True, "filled": True}
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
                trigger_by_source=getattr(self.config, "trigger_price_source", "last"),
            )
        except TypeError:
            side_lo = "buy" if direction == "long" else "sell"
            return self.api.place_order(
                symbol=self.symbol,
                side=side_lo,
                order_type="market",
                qty=qty,
                stop_loss=sl_send,
            )

    # ---------- Pine-like equity ----------
    def _get_equity_like_pine(self) -> Optional[float]:
        equity_local = self.state.get_equity() if self.state else None
        if equity_local is None:
            equity_local = float(getattr(self.config, "initial_capital", 300.0))

        equity = float(equity_local)

        if self.equity_mode == "wallet_plus_upnl":
            pos = self.state.get_current_position() if self.state else None
            price = self._get_current_price()
            if pos and price is not None:
                sz    = float(pos.get("size") or 0.0)
                entry = float(pos.get("entry_price") or 0.0)
                if sz > 0 and entry > 0:
                    if str(pos.get("direction")) == "long":
                        equity += (float(price) - entry) * sz
                    else:
                        equity += (entry - float(price)) * sz
        return float(equity)

    # --- pine-like округление qty (PATCH D уже внедрён выше) ---
    def _calculate_position_size(self, entry_price: float, stop_loss: float, direction: str) -> Optional[float]:
        try:
            equity = self._get_equity_like_pine()
            if equity is None or equity <= 0:
                return None
            risk_amount = float(equity) * (float(self.risk_pct) / 100.0)
            stop_size = (entry_price - stop_loss) if direction == "long" else (stop_loss - entry_price)
            if stop_size <= 0:
                return None
            quantity = risk_amount / stop_size
            quantity = self._round_to_step_nearest(quantity, self.qty_step)
            if self.limit_qty_enabled:
                quantity = min(quantity, float(self.max_qty_manual))
            if quantity < float(self.min_order_qty):
                return None
            return float(quantity)
        except Exception as e:
            self._log("error", f"calc_position_size: {e}")
            return None

    # ========== ВХОДЫ ==========
    def _process_long_entry(self, entry_override: Optional[float] = None, sl_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 1 or sl_override is None:
                return
            bar_close = float(self.candles_15m[0]["close"])
            entry = float(entry_override) if entry_override is not None else float(bar_close)

            sl_raw = float(sl_override)  # RAW
            stop_size = entry - sl_raw
            if stop_size <= 0:
                return

            rr = float(getattr(self.config, "risk_reward", 3.0))
            tp_calc = round_to_tick(entry + stop_size * rr, self.tick_size)

            qty = self._calculate_position_size(entry, sl_raw, "long")
            if not qty:
                self._log("debug", "Skip entry long: qty calc failed")
                return
            if not self._expected_net_ok(entry, sl_raw, qty):
                self._log("debug", "Skip entry long: fee filter failed (exp_net < MIN_NET_PROFIT)")
                return

            sl_send = floor_to_tick(sl_raw, self.tick_size) if self.quantize_initial_sl else sl_raw
            res = self._place_market_order("long", qty, stop_loss=sl_send)
            if res is None:
                self._log("warn", "Order failed (long)")
                return

            bar_ts_ms = self._current_bar_ts_ms()
            pos = {
                "symbol": self.symbol,
                "direction": "long",
                "size": float(qty),
                "entry_price": entry,
                "stop_loss": float(sl_send),
                "sl_calc": float(sl_raw),
                "status": "open",
                "armed": (not self.use_arm_after_rr),
                "entry_time_ts": bar_ts_ms,
                "take_profit": tp_calc,
                "trail_points": float(entry) * (self.trailing_perc / 100.0),
                "trail_offset": float(entry) * (self.trailing_offset_perc / 100.0),
                "trail_active": False,
                "trail_anchor": float(entry),
                # PATCH B: экстремум с момента активации
                "trail_extreme": float(entry),
            }
            if self.state:
                self.state.set_position(pos)

            self.can_enter_long = False
            self._last_entered_15m_start = self.last_processed_bar_ts
            if self.candles_1m:
                self._mark_entered_this_bar(int(self.candles_1m[0].get("timestamp") or 0))

            self._log("info", f"[ENTRY LONG] qty={qty} @ {entry:.6f} SL(raw)={sl_raw:.6f} SL(api)={sl_send:.6f} TP≈{tp_calc:.6f}")
        except Exception as e:
            self._log("error", f"process_long_entry: {e}")

    def _process_short_entry(self, entry_override: Optional[float] = None, sl_override: Optional[float] = None):
        try:
            if len(self.candles_15m) < 1 or sl_override is None:
                return
            bar_close = float(self.candles_15m[0]["close"])
            entry = float(entry_override) if entry_override is not None else float(bar_close)

            sl_raw = float(sl_override)  # RAW
            stop_size = sl_raw - entry
            if stop_size <= 0:
                return

            rr = float(getattr(self.config, "risk_reward", 3.0))
            tp_calc = round_to_tick(entry - stop_size * rr, self.tick_size)

            qty = self._calculate_position_size(entry, sl_raw, "short")
            if not qty:
                self._log("debug", "Skip entry short: qty calc failed")
                return
            if not self._expected_net_ok(entry, sl_raw, qty):
                self._log("debug", "Skip entry short: fee filter failed (exp_net < MIN_NET_PROFIT)")
                return

            sl_send = ceil_to_tick(sl_raw, self.tick_size) if self.quantize_initial_sl else sl_raw
            res = self._place_market_order("short", qty, stop_loss=sl_send)
            if res is None:
                self._log("warn", "Order failed (short)")
                return

            bar_ts_ms = self._current_bar_ts_ms()
            pos = {
                "symbol": self.symbol,
                "direction": "short",
                "size": float(qty),
                "entry_price": entry,
                "stop_loss": float(sl_send),
                "sl_calc": float(sl_raw),
                "status": "open",
                "armed": (not self.use_arm_after_rr),
                "entry_time_ts": bar_ts_ms,
                "take_profit": tp_calc,
                "trail_points": float(entry) * (self.trailing_perc / 100.0),
                "trail_offset": float(entry) * (self.trailing_offset_perc / 100.0),
                "trail_active": False,
                "trail_anchor": float(entry),
                # PATCH B
                "trail_extreme": float(entry),
            }
            if self.state:
                self.state.set_position(pos)

            self.can_enter_short = False
            self._last_entered_15m_start = self.last_processed_bar_ts
            if self.candles_1m:
                self._mark_entered_this_bar(int(self.candles_1m[0].get("timestamp") or 0))

            self._log("info", f"[ENTRY SHORT] qty={qty} @ {entry:.6f} SL(raw)={sl_raw:.6f} SL(api)={sl_send:.6f} TP≈{tp_calc:.6f}")
        except Exception as e:
            self._log("error", f"process_short_entry: {e}")

    # ---------- Трейлинг ----------
    def _get_bar_extremes_for_trailing(self, current_price: float) -> Tuple[float, float]:
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

    # >>> ДОБАВЛЕНО: корректное обновление SL (используется Smart/Bar-трейлингом)
    def _update_stop_loss(self, position: Dict, new_sl: float):
        try:
            direction = str(position.get("direction"))
            old_sl = float(position.get("stop_loss") or 0.0)

            # Квантование в сторону "ужесточения"
            if direction == "long":
                candidate = floor_to_tick(float(new_sl), self.tick_size)
                if candidate <= old_sl:
                    return
            else:
                candidate = ceil_to_tick(float(new_sl), self.tick_size)
                if candidate >= old_sl:
                    return

            api_updated = False
            if self.api:
                try:
                    # популярные методы обновления стопа в API-обёртках
                    if hasattr(self.api, "update_stop_loss"):
                        self.api.update_stop_loss(self.symbol, stop_loss=candidate)
                        api_updated = True
                    elif hasattr(self.api, "set_trading_stop"):
                        self.api.set_trading_stop(self.symbol, stop_loss=candidate)
                        api_updated = True
                except Exception as e:
                    self._log("warn", f"API SL update failed: {e}")

            position["stop_loss"] = float(candidate)
            if self.state:
                self.state.set_position(position)
            self._log("info", f"[TRAIL MOVE] SL {old_sl:.6f} -> {candidate:.6f}{' (api)' if api_updated else ''}")
        except Exception as e:
            self._log("error", f"_update_stop_loss: {e}")

    def _bar_trail_update(self, position: Dict):
        """Баровый трейл включается ПОСЛЕ ARM. Всегда стремимся к более “тугому” стопу."""
        if not self.use_bar_trail or self.trail_lookback <= 0:
            return
        mt  = float(self.tick_size or 0.01)
        buf = float(self.trail_buf_ticks) * mt
        sl  = float(position.get("stop_loss") or 0.0)

        use_intrabar = getattr(self.config, "use_intrabar", True)
        src = self.candles_1m if (use_intrabar and self.candles_1m) else self.candles_15m
        if not src or len(src) < 2:
            return

        if position.get("direction") == "long":
            lows = [float(c.get("low")) for c in src[1 : 1 + self.trail_lookback] if c.get("low") is not None]
            if not lows:
                return
            lb_low = min(lows)
            candidate = max(lb_low - buf, sl)
            candidate = floor_to_tick(candidate, mt)
            if candidate > sl:
                self._update_stop_loss(position, candidate)
        elif position.get("direction") == "short":
            highs = [float(c.get("high")) for c in src[1 : 1 + self.trail_lookback] if c.get("high") is not None]
            if not highs:
                return
            lb_high = max(highs)
            candidate = min(lb_high + buf, sl)
            candidate = ceil_to_tick(candidate, mt)
            if candidate < sl:
                self._update_stop_loss(position, candidate)

    def _update_smart_trailing(self, position: Dict):
        """Pine-parity Smart Trailing (PATCH B: используем trail_extreme как highest/lowest since activation)."""
        try:
            if not self.enable_smart_trail:
                return

            direction = position.get("direction")
            entry = float(position.get("entry_price") or 0)
            sl_api = float(position.get("stop_loss") or 0)
            sl_calc = float(position.get("sl_calc", sl_api))
            if not direction or entry <= 0 or sl_api <= 0:
                return

            price = self._get_current_price()
            if price is None:
                return
            price = float(price)

            # ARM только после достижения R от close
            armed = bool(position.get("armed", not self.use_arm_after_rr))
            if not armed and self.use_arm_after_rr:
                risk = abs(entry - sl_calc)
                if risk > 0:
                    rr_close = ((price - entry) / risk) if direction == "long" else ((entry - price) / risk)
                    if rr_close >= float(self.arm_rr):
                        armed = True
                        position["armed"] = True
                        if self.state:
                            self.state.set_position(position)
                        self._log("info", f"[ARM] enabled ≥{self.arm_rr:.2f}R (basis=close, rr={rr_close:.3f})")

            # trail params
            trail_points = float(position.get("trail_points") or (entry * (self.trailing_perc / 100.0)))
            trail_offset = float(position.get("trail_offset") or (entry * (self.trailing_offset_perc / 100.0)))

            # ---- 1) АКТИВАЦИЯ ПО OFFSET ----
            trail_active = bool(position.get("trail_active", False))
            if not trail_active:
                hit = (price >= entry + trail_offset) if direction == "long" else (price <= entry - trail_offset)
                if hit:
                    bar_high, bar_low = self._get_bar_extremes_for_trailing(price)
                    anchor = float(bar_high) if direction == "long" else float(bar_low)
                    position["trail_active"] = True
                    position["trail_anchor"] = float(anchor)
                    # PATCH B: инициализируем экстремум с момента активации
                    position["trail_extreme"] = float(anchor)
                    if self.state:
                        self.state.set_position(position)
                    trail_active = True
                    self._log("info", f"[TRAIL ACTIVATE] offset reached @ {price:.6f}")

            # ---- 2) БАЗОВЫЙ ТРЕЙЛИНГ (ТОЛЬКО ПОСЛЕ АКТИВАЦИИ) ----
            if trail_active:
                bar_high, bar_low = self._get_bar_extremes_for_trailing(price)

                # PATCH B: обновляем running extreme
                if direction == "long":
                    new_extreme = max(float(position.get("trail_extreme", entry)), float(bar_high))
                    position["trail_extreme"] = new_extreme
                    baseline = new_extreme - trail_points
                    candidate = max(baseline, sl_api)
                    candidate = floor_to_tick(candidate, self.tick_size)
                    if candidate > sl_api:
                        self._update_stop_loss(position, candidate)
                else:
                    new_extreme = min(float(position.get("trail_extreme", entry)), float(bar_low))
                    position["trail_extreme"] = new_extreme
                    baseline = new_extreme + trail_points
                    candidate = min(baseline, sl_api)
                    candidate = ceil_to_tick(candidate, self.tick_size)
                    if candidate < sl_api:
                        self._update_stop_loss(position, candidate)

                # anchor обновляем для совместимости (визуал/лог)
                anchor = float(position.get("trail_anchor") or entry)
                if direction == "long":
                    anchor = max(anchor, float(bar_high))
                else:
                    anchor = min(anchor, float(bar_low))
                if anchor != float(position.get("trail_anchor") or anchor):
                    position["trail_anchor"] = float(anchor)
                    if self.state:
                        self.state.set_position(position)

            # ---- 3) BAR-TRAIL: ТОЛЬКО ПОСЛЕ ARM ----
            if armed:
                self._bar_trail_update(position)

        except Exception as e:
            self._log("error", f"smart_trailing: {e}")

    def update_candles(self):
        """Разовый пулл свечей; обработка закрытого 15m бара."""
        try:
            if not self.api or not hasattr(self.api, "get_klines"):
                return

            need_15m = int(getattr(self.config, "days_back", 30)) * 96 + 50
            need_15m = max(200, min(5000, need_15m))

            kl15 = self.api.get_klines(self.symbol, "15", need_15m) or []
            for k in kl15:
                ts = k.get("timestamp")
                if ts is not None and ts < 1_000_000_000_000:
                    k["timestamp"] = int(ts * 1000)
            kl15.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            self.candles_15m = kl15

            tf = str(getattr(self.config, "intrabar_tf", "1"))
            lim = int(getattr(self.config, "intrabar_pull_limit", 1000) or 1000)
            kl1 = self.api.get_klines(self.symbol, tf, min(1000, lim)) or []
            for k in kl1:
                ts = k.get("timestamp")
                if ts is not None and ts < 1_000_000_000_000:
                    k["timestamp"] = int(ts * 1000)
            kl1.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            self.candles_1m = kl1[:lim]

            # -------- ИСТОРИЧЕСКИЙ ПРОИГРЫШ: сначала 1m внутри окна, потом 15m-логика --------
            if self.candles_15m and not self._history_replayed:
                bars_old_to_new = sorted(self.candles_15m, key=lambda x: x.get("timestamp", 0))
                # обнулим, будем наращивать пошагово
                self.candles_15m = []
                self.last_processed_bar_ts = 0

                for b in bars_old_to_new:
                    # 1) вставляем 15m бар как текущий
                    self.candles_15m.insert(0, b)
                    if len(self.candles_15m) > 200:
                        self.candles_15m = self.candles_15m[:200]

                    # 2) открываем новое 15m окно — сброс флагов как при закрытии 15m
                    bar_ts = int(b.get("timestamp") or b.get("open_time") or 0)
                    if bar_ts and bar_ts < 1_000_000_000_000:
                        bar_ts *= 1000
                    aligned_start = self._align_15m_ms(bar_ts)
                    self.last_processed_bar_ts = aligned_start
                    self.can_enter_long = True
                    self.can_enter_short = True
                    self._bar_index_15m += 1

                    # 3) проигрываем все 1m свечи внутри этого окна (локальные входы/трейл)
                    for m in self._ltf_slices_for_bar(bar_ts):
                        self.on_bar_close_1m(m)

                    # 4) применяем 15m немедленные входы на закрытии окна
                    self.run_cycle()

                self._history_replayed = True
                self._log("info", "History replay finished")
                return

            # -------- Обычный поток --------
            if self.candles_15m:
                ts = int(self.candles_15m[0].get("timestamp") or 0)
                aligned = self._align_15m_ms(ts)
                if aligned != self.last_processed_bar_ts:
                    self.last_processed_bar_ts = aligned
                    self.can_enter_long = True
                    self.can_enter_short = True
                    self._bar_index_15m += 1
                    self.run_cycle()
                else:
                    pos = self.state.get_current_position() if self.state else None
                    if pos and pos.get("status") == "open" and getattr(self.config, "use_intrabar", True):
                        self._update_smart_trailing(pos)
        except Exception as e:
            self._log("error", f"update_candles: {e}")

    # >>> ДОБАВЛЕНО: служебные визуал-хелперы Lux SFP (лог/совместимость с Pine), вызываются в run_cycle()
    def _lux_find_new_sfp(self):
        try:
            L = int(self.lux_swings)
            if len(self.candles_15m) < L + 2:
                return
            cur15 = self.candles_15m[0]
            prev1 = self.candles_15m[1]
            op15, cl15, hi15, lo15 = map(float, (cur15["open"], cur15["close"], cur15["high"], cur15["low"]))

            ph15 = self._pivot_high_value(L, 1)
            pl15 = self._pivot_low_value (L, 1)

            sid_prev_ts = int(prev1.get("timestamp") or 0)

            new_bear = None
            new_bull = None

            if ph15 is not None and len(self.candles_15m) > L:
                swH_val = float(self.candles_15m[L]["high"])
                if hi15 > swH_val and op15 < swH_val and cl15 < swH_val:
                    new_bear = {
                        "sid_15m": sid_prev_ts,
                        "swing": swH_val,
                        "ts_bar": int(cur15.get("timestamp") or 0),
                        "created_index": int(self._bar_index_15m),
                        "expire_after": int(self.lux_expire_bars),
                        "source": "15m",
                    }

            if pl15 is not None and len(self.candles_15m) > L:
                swL_val = float(self.candles_15m[L]["low"])
                if lo15 < swL_val and op15 > swL_val and cl15 > swL_val:
                    new_bull = {
                        "sid_15m": sid_prev_ts,
                        "swing": swL_val,
                        "ts_bar": int(cur15.get("timestamp") or 0),
                        "created_index": int(self._bar_index_15m),
                        "expire_after": int(self.lux_expire_bars),
                        "source": "15m",
                    }

            if new_bear:
                self._active_bear = new_bear
            if new_bull:
                self._active_bull = new_bull
        except Exception as e:
            self._log("error", f"_lux_find_new_sfp: {e}")

    def _lux_update_active(self):
        try:
            idx = int(self._bar_index_15m)

            def alive(ctx: Optional[Dict]) -> bool:
                if not ctx:
                    return False
                created = int(ctx.get("created_index", idx))
                exp = int(ctx.get("expire_after", self.lux_expire_bars))
                return (idx - created) <= exp

            if not alive(self._active_bear):
                self._active_bear = None
            if not alive(self._active_bull):
                self._active_bull = None
        except Exception as e:
            self._log("error", f"_lux_update_active: {e}")

    def run_cycle(self):
        """15m логика входа (немедленно на закрытии окна) + подтяжка трейла."""
        try:
            if not self.candles_15m:
                return

            # Обновляем equity перед расчётами (реинвест)
            self._update_equity()

            # Трейл для открытой позиции
            pos = self.state.get_current_position() if self.state else None
            if pos and pos.get("status") == "open":
                self._update_smart_trailing(pos)

            if len(self.candles_15m) < int(self.lux_swings) + 2:
                return

            ts = int(self.candles_15m[0]["timestamp"])
            if (not self._is_in_backtest_window_utc(ts)) or (not self._is_after_cycle_start(ts)):
                return

            # Защита: одна сделка на 15m окно
            if self._last_entered_15m_start is not None and self._last_entered_15m_start == self.last_processed_bar_ts:
                return

            # ===== 15m IMMEDIATE ENTRY (без «confirmed»), как в Pine =====
            L = int(self.lux_swings)
            cur15 = self.candles_15m[0]
            prev1 = self.candles_15m[1] if len(self.candles_15m) > 1 else None
            if not prev1:
                return

            op15, cl15, hi15, lo15 = map(float, (cur15["open"], cur15["close"], cur15["high"], cur15["low"]))
            ph15 = self._pivot_high_value(L, 1)
            pl15 = self._pivot_low_value (L, 1)

            # локальные свинги на 1m для SL-first
            last1 = self.candles_1m[0] if self.candles_1m else cur15
            ph_loc = self._pivot_high_value_tf(self.candles_1m, L, 1) if self.candles_1m else None
            pl_loc = self._pivot_low_value_tf (self.candles_1m, L, 1) if self.candles_1m else None
            swH_loc = float(ph_loc[0]) if ph_loc else None
            swL_loc = float(pl_loc[0]) if pl_loc else None

            bear_ctx = None
            bull_ctx = None

            # swing = 15m[L], как в Pine (high15[len]/low15[len])
            swH_15 = float(self.candles_15m[L]["high"]) if ph15 is not None and len(self.candles_15m) > L else None
            swL_15 = float(self.candles_15m[L]["low" ]) if pl15 is not None and len(self.candles_15m) > L else None

            if swH_15 is not None and hi15 > swH_15 and op15 < swH_15 and cl15 < swH_15:
                sl = self._sl_local_first("short", swH_loc, last1)  # RAW
                bear_ctx = {"sl": sl, "sid_15m": int(prev1.get("timestamp") or 0), "source": "15m"}

            if swL_15 is not None and lo15 < swL_15 and op15 > swL_15 and cl15 > swL_15:
                sl = self._sl_local_first("long", swL_loc, last1)  # RAW
                bull_ctx = {"sl": sl, "sid_15m": int(prev1.get("timestamp") or 0), "source": "15m"}

            # ===== once-per-swing по timestamp локального свинга =====
            if self.use_one_per_swing:
                if bear_ctx:
                    ph_l = self._pivot_high_value_tf(self.candles_1m, int(self.lux_swings), 1) if self.candles_1m else None
                    sid_ts = int(ph_l[1]) if ph_l else 0
                    if sid_ts and self.traded_bear_swing_id == sid_ts:
                        bear_ctx = None
                if bull_ctx:
                    pl_l = self._pivot_low_value_tf(self.candles_1m, int(self.lux_swings), 1) if self.candles_1m else None
                    sid_ts = int(pl_l[1]) if pl_l else 0
                    if sid_ts and self.traded_bull_swing_id == sid_ts:
                        bull_ctx = None

            # конфликт сигналов — one-per-bar + приоритет
            if self.use_one_per_bar and bear_ctx and bull_ctx:
                if self.bar_priority == "prefer_bull":
                    bear_ctx = None
                elif self.bar_priority == "prefer_bear":
                    bull_ctx = None
                else:
                    bear_ctx = None
                    bull_ctx = None

            # Если уже был вход в текущем 1m-баре — блокируем 15m-вход
            if self.use_one_per_bar and self.candles_1m:
                cur_1m_ts = int(self.candles_1m[0].get("timestamp") or 0)
                if self._already_entered_this_bar(cur_1m_ts):
                    bear_ctx = None
                    bull_ctx = None

            executed = False
            # SHORT по 15m
            if bear_ctx and self.can_enter_short:
                pos = self.state.get_current_position() if self.state else None
                if pos and pos.get("status") == "open":
                    if not self.use_dir_lock and str(pos.get("direction")) == "long":
                        if not self._flip_position("short"):
                            bear_ctx = None
                    else:
                        bear_ctx = None

            if bear_ctx:
                entry = cl15
                sl    = float(bear_ctx["sl"])  # RAW
                qty   = self._calculate_position_size(entry, sl, "short")
                if qty and qty >= self.min_order_qty and self._expected_net_ok(entry, sl, qty):
                    self._process_short_entry(entry_override=entry, sl_override=sl)
                    self.can_enter_short = False
                    executed = True

            # LONG по 15m
            if bull_ctx and self.can_enter_long and not (executed and self.use_one_per_bar):
                pos = self.state.get_current_position() if self.state else None
                if pos and pos.get("status") == "open":
                    if not self.use_dir_lock and str(pos.get("direction")) == "short":
                        if not self._flip_position("long"):
                            bull_ctx = None
                    else:
                        bull_ctx = None

            if bull_ctx and not (executed and self.use_one_per_bar):
                entry = cl15
                sl    = float(bull_ctx["sl"])  # RAW
                qty   = self._calculate_position_size(entry, sl, "long")
                if qty and qty >= self.min_order_qty and self._expected_net_ok(entry, sl, qty):
                    self._process_long_entry(entry_override=entry, sl_override=sl)
                    self.can_enter_long = False

            # Логи активных SFP (визуал/совместимость)
            self._lux_find_new_sfp()
            self._lux_update_active()

        except Exception as e:
            self._log("error", f"run_cycle: {e}")

    # ---------- Окно теста ----------
    def _is_in_backtest_window_utc(self, current_timestamp: int) -> bool:
        """Плавающее окно дней назад + опционная стартовая дата (как в Pine-обёртке)."""
        days_back = int(getattr(self.config, "days_back", 30))
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = utc_midnight - timedelta(days=days_back)
        current_time = datetime.utcfromtimestamp(current_timestamp / 1000)
        return current_time >= start_date.replace(tzinfo=None)

    def _is_after_cycle_start(self, current_timestamp: int) -> bool:
        if self.start_time_ms is None:
            return True
        return int(current_timestamp) >= int(self.start_time_ms)

    def _update_equity(self):
        """Если задано equity_source='wallet' — обновляем equity из кошелька, иначе не трогаем локальную метрику."""
        try:
            if str(self.equity_source) != "wallet":
                return
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
            self._log("error", f"update_equity: {e}")

    # ---------- Debug helper ----------
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
                "extreme": pos.get("trail_extreme"),
                "tick_size": self.tick_size,
                "qty_step": self.qty_step,
            }
        except Exception:
            return {}
